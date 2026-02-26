#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------- Limites de threads internas (ANTES de importar numpy/cv/tf) ----------
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
_os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
_os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

# ------------------------------------------------------------------------------
import os
import time
import json
import argparse
import numpy as np
import cv2 as cv
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image
from itertools import product
from matplotlib import cm
import tensorflow as tf
import multiprocessing as mp

TRASH_HOLD = 0.9

# -----------------------------
# PREPROCESSING (funções)
# -----------------------------
def _apply_blur_filter(image_bgr: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    return cv.GaussianBlur(image_bgr, (kernel_size, kernel_size), 0)

def _variance_of_laplacian(image_bgr: np.ndarray) -> float:
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(gray, cv.CV_64F).var()

def _macenko_normalization(image_bgr: np.ndarray, template_path: str) -> np.ndarray:
    """
    Implementação simplificada (LAB mean/std matching nos canais a/b).
    % Observação: isso não é o Macenko clássico (OD+SVD), mas mantive exatamente sua lógica.
    """
    try:
        template = cv.imread(str(template_path))
        if template is None:
            print(f"[WARNING] Template não encontrado: {template_path}")
            return image_bgr

        image_lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2LAB)
        template_lab = cv.cvtColor(template, cv.COLOR_BGR2LAB)

        for channel in [1, 2]:
            image_channel = image_lab[:, :, channel].astype(np.float32)
            template_channel = template_lab[:, :, channel].astype(np.float32)

            image_mean = np.mean(image_channel)
            image_std = np.std(image_channel)
            template_mean = np.mean(template_channel)
            template_std = np.std(template_channel)

            if image_std > 0:
                image_channel = (image_channel - image_mean) * (template_std / image_std) + template_mean
                image_lab[:, :, channel] = np.clip(image_channel, 0, 255).astype(np.uint8)

        normalized = cv.cvtColor(image_lab, cv.COLOR_LAB2BGR)
        return normalized
    except Exception as e:
        print(f"[WARNING] Erro na normalização Macenko: {e}")
        return image_bgr


# -----------------------------
# Config do preprocessing (globais simples)
# -----------------------------
PREPROC_ENABLE_BLUR = False
PREPROC_BLUR_KERNEL = 5

PREPROC_ENABLE_LAPLACIAN_GATE = False
PREPROC_LAPLACIAN_MIN_VAR = 0.0  # se var < min_var, descarta patch

PREPROC_ENABLE_MACENKO = False
PREPROC_MACENKO_TEMPLATE_PATH = None  # caminho str


# Estado do processo principal
patch_size = None
width = None
hight = None
dz_level = None
dz_cols = None
dz_rows = None
mask = None
indices = None

# Estado por worker
_worker_tiles = None
_worker_interpreter = None
_worker_input_details = None
_worker_output_details = None
_worker_patch_size = None
_worker_dz_level = None
_worker_dz_cols = None
_worker_dz_rows = None

# Preproc por worker
_worker_preproc = None  # dict com configs


def processing_time(start_time, end_time):
    total_time = end_time - start_time
    minutos, segundos = divmod(total_time, 60)
    milisegundos = int((segundos - int(segundos)) * 1000)
    print(f"Tempo de execução:\n00:{int(minutos):02d}:{int(segundos):02d}.{milisegundos:03d}")


def _pick_mp_context():
    try:
        ctx = mp.get_context("forkserver")
        return ctx, "forkserver"
    except Exception:
        ctx = mp.get_context("spawn")
        return ctx, "spawn"


def _init_worker(
    image_path,
    tflite_path,
    _patch_size,
    _dz_level,
    preproc_cfg: dict,
):
    global _worker_tiles, _worker_interpreter, _worker_input_details, _worker_output_details
    global _worker_patch_size, _worker_dz_level, _worker_dz_cols, _worker_dz_rows
    global _worker_preproc

    try:
        cv.setNumThreads(1)
    except Exception:
        pass

    _worker_patch_size = int(_patch_size)
    _worker_dz_level = int(_dz_level)

    # guarda cfg do preproc no worker
    _worker_preproc = dict(preproc_cfg or {})

    # OpenSlide/DeepZoom por processo
    wsi = openslide.OpenSlide(image_path)
    _worker_tiles = DeepZoomGenerator(wsi, tile_size=_worker_patch_size, overlap=0)
    _worker_dz_cols, _worker_dz_rows = _worker_tiles.level_tiles[_worker_dz_level]

    # Interpreter por processo (threads internas = 1)
    try:
        import tflite_runtime.interpreter as tflite
        _worker_interpreter = tflite.Interpreter(model_path=tflite_path, num_threads=1)
    except Exception:
        _worker_interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=1)

    _worker_interpreter.allocate_tensors()
    _worker_input_details = _worker_interpreter.get_input_details()
    _worker_output_details = _worker_interpreter.get_output_details()


def _process_block(y, x):
    # RODA NO PROCESSO PRINCIPAL (usa mask/indices globais)
    if indices is None or mask is None:
        raise RuntimeError("indices/mask não inicializados no processo principal.")

    coords = []
    for m, n in product(indices, indices):
        yy = y + m
        xx = x + n
        if yy >= hight or xx >= width:
            continue
        y1 = min(yy + patch_size, hight)
        x1 = min(xx + patch_size, width)
        region = mask[yy:y1, xx:x1]
        if region.size == 0:
            continue
        if np.mean(region) > 16:
            coords.append((int(yy), int(xx)))
    return coords


def _apply_preprocessing_pipeline(image_bgr: np.ndarray):
    """
    Aplica blur + macenko + gate por laplacian variance.
    Retorna (image_bgr, keep: bool).
    """
    cfg = _worker_preproc or {}

    # blur
    if cfg.get("enable_blur", False):
        k = int(cfg.get("blur_kernel", 5))
        if k % 2 == 0:
            k += 1
        if k < 3:
            k = 3
        image_bgr = _apply_blur_filter(image_bgr, kernel_size=k)

    # macenko (na sua versão LAB mean/std)
    if cfg.get("enable_macenko", False):
        tpl = cfg.get("macenko_template_path", None)
        if tpl:
            image_bgr = _macenko_normalization(image_bgr, template_path=tpl)

    # laplacian gate (descarta patch borrado)
    if cfg.get("enable_laplacian_gate", False):
        minv = float(cfg.get("laplacian_min_var", 0.0))
        v = _variance_of_laplacian(image_bgr)
        if v < minv:
            return image_bgr, False

    return image_bgr, True


def _process_patch(abs_y, abs_x):
    # RODA NO WORKER
    tile_row = abs_y // _worker_patch_size
    tile_col = abs_x // _worker_patch_size
    if tile_row >= _worker_dz_rows or tile_col >= _worker_dz_cols:
        return None

    # get tile (RGB PIL) -> numpy BGR uint8 para preprocessing
    img_rgb = _worker_tiles.get_tile(_worker_dz_level, (tile_col, tile_row)).convert("RGB")
    img_bgr = cv.cvtColor(np.array(img_rgb, dtype=np.uint8), cv.COLOR_RGB2BGR)

    # -----------------------------
    # PREPROCESSING AQUI (ANTES da inferência)
    # -----------------------------
    img_bgr, keep = _apply_preprocessing_pipeline(img_bgr)
    if not keep:
        return None

    # prepara input do modelo (mantendo seu padrão BGR float32)
    img_bgr = img_bgr.astype(np.float32)
    img_bgr = cv.resize(img_bgr, (224, 224), interpolation=cv.INTER_LINEAR)
    img_bgr = np.expand_dims(img_bgr, axis=0)

    _worker_interpreter.set_tensor(_worker_input_details[0]["index"], img_bgr)
    _worker_interpreter.invoke()
    pred = _worker_interpreter.get_tensor(_worker_output_details[0]["index"])[0][1]

    if pred > TRASH_HOLD:
        return (int(tile_row), int(tile_col), float(pred))
    return None


def run(
    image_path,
    tflite_path,
    output_dir,
    threshold,
    patch_multiplier,
    overlay_level,
    processes=None,
    chunksize=256,
    # --- preprocessing flags ---
    enable_blur=False,
    blur_kernel=5,
    enable_laplacian_gate=False,
    laplacian_min_var=0.0,
    enable_macenko=False,
    macenko_template_path=None,
):
    global TRASH_HOLD, patch_size, width, hight, dz_level, dz_cols, dz_rows, mask, indices

    TRASH_HOLD = float(threshold)
    if patch_multiplier <= 0:
        raise ValueError("patch_multiplier deve ser positivo")

    # ====== principal: abre WSI para metadata + máscara ======
    wsi = openslide.OpenSlide(image_path)
    w, h = wsi.dimensions
    patch_size = int(224 * patch_multiplier)

    width = ((w + patch_size - 1) // patch_size) * patch_size
    hight = ((h + patch_size - 1) // patch_size) * patch_size

    tiles_main = DeepZoomGenerator(wsi, tile_size=patch_size, overlap=0)
    dz_level = tiles_main.level_count - 1
    dz_cols, dz_rows = tiles_main.level_tiles[dz_level]

    level = len(wsi.level_dimensions) - 1
    img_low = wsi.read_region((0, 0), level, wsi.level_dimensions[level]).convert("RGB")
    gray_img = cv.cvtColor(np.array(img_low), cv.COLOR_RGB2GRAY)
    _, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    mask = cv.resize(mask, (width, hight), interpolation=cv.INTER_NEAREST)

    c = 4
    slide_dim = patch_size * c
    indices = np.arange(0, slide_dim, patch_size, dtype=np.int64)

    y_indices = np.arange(0, hight, slide_dim, dtype=np.int64)
    x_indices = np.arange(0, width, slide_dim, dtype=np.int64)

    block_list = []
    for y, x in product(y_indices, x_indices):
        y = int(y); x = int(x)
        y1 = min(y + slide_dim, hight)
        x1 = min(x + slide_dim, width)
        region = mask[y:y1, x:x1]
        if region.size and np.mean(region) > 16:
            block_list.append((y, x))

    jet_heatmap_matrix = np.zeros((int(hight / patch_size), int(width / patch_size)), dtype=np.float16)

    wsi.close()

    # ====== multiprocess context ======
    ctx, ctx_name = _pick_mp_context()
    print(f"[INFO] multiprocessing context: {ctx_name}")

    if processes is None:
        processes = min(8, os.cpu_count() or 8)
    processes = max(1, int(processes))

    # cfg do preprocessing (passado para cada worker via initargs)
    preproc_cfg = dict(
        enable_blur=bool(enable_blur),
        blur_kernel=int(blur_kernel),
        enable_laplacian_gate=bool(enable_laplacian_gate),
        laplacian_min_var=float(laplacian_min_var),
        enable_macenko=bool(enable_macenko),
        macenko_template_path=macenko_template_path,
    )

    start_time = time.time()

    # ====== 1) block -> patch_coords (NO PRINCIPAL) ======
    patch_coords_nested = [_process_block(y, x) for (y, x) in block_list]
    patch_coords = [xy for sub in patch_coords_nested for xy in sub]
    print(f"Quantidade de imagens: {len(patch_coords)}")

    # ====== 2) patch -> inferência (NO POOL) ======
    with ctx.Pool(
        processes=processes,
        initializer=_init_worker,
        initargs=(image_path, tflite_path, patch_size, dz_level, preproc_cfg),
        maxtasksperchild=200,
    ) as pool:
        results = pool.starmap(_process_patch, patch_coords, chunksize=chunksize)

    results = [r for r in results if r is not None]
    if results:
        arr = np.array(results, dtype=np.float32)
        ys = arr[:, 0].astype(np.int64)
        xs = arr[:, 1].astype(np.int64)
        vals = arr[:, 2].astype(np.float16)
        jet_heatmap_matrix[ys, xs] = vals

    end_time = time.time()
    processing_time(start_time, end_time)

    # ====== salvar ======
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    heatmap_path = os.path.join(output_dir, f"hm_{base}_qat.jpg")
    superimposed_path = os.path.join(output_dir, f"superimposed_{base}_qat.jpg")

    #exportar metricas em json
    metrics = {
        "image": base,
        "patches_processed": len(results),
        "processing_time": end_time - start_time,
        "threshold": threshold,
        "patch_size": patch_size,
        "overlay_level": overlay_level
    }
    metrics_path = os.path.join(output_dir, f"metrics_{base}_qat.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    heatmap_matrix = np.uint8(255 * jet_heatmap_matrix)
    jet_colors = cm.get_cmap("jet")(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_matrix]
    jet_heatmap_img = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap_img.save(heatmap_path, quality=100)

    wsi = openslide.OpenSlide(image_path)
    img = wsi.read_region((0, 0), overlay_level, wsi.level_dimensions[overlay_level]).convert("RGB")
    img_array = np.array(img, dtype=np.float32)

    jet_heatmap_img = Image.open(heatmap_path).resize((img_array.shape[1], img_array.shape[0]))
    superimposed_img = np.array(jet_heatmap_img, dtype=np.float32) + img_array
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)
    superimposed_img.save(superimposed_path, quality=100)
    wsi.close()

    print(f"Quantidade de patches processados: {len(results)}")
    print(f"Heatmap salvo em: {heatmap_path}")
    print(f"Superimposed salvo em: {superimposed_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", required=True)
    parser.add_argument("--tflite_path", required=True)
    parser.add_argument("--output_dir", default="heat_maps_images")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--patch_multiplier", type=int, default=4)
    parser.add_argument("--overlay_level", type=int, default=2)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=256)

    # -------- flags preprocessing --------
    parser.add_argument("--enable_blur", action="store_true", help="Aplica Gaussian blur antes da inferência")
    parser.add_argument("--blur_kernel", type=int, default=5, help="Kernel (ímpar) do blur")
    parser.add_argument("--enable_laplacian_gate", action="store_true", help="Descarta patch se Laplacian var < min")
    parser.add_argument("--laplacian_min_var", type=float, default=0.0, help="Threshold de Laplacian var")
    parser.add_argument("--enable_macenko", action="store_true", help="Aplica normalização (LAB mean/std) usando template")
    parser.add_argument("--macenko_template_path", type=str, default=None, help="Caminho para imagem template (BGR)")

    args = parser.parse_args()

    run(
        image_path=args.image_path,
        tflite_path=args.tflite_path,
        output_dir=args.output_dir,
        threshold=args.threshold,
        patch_multiplier=args.patch_multiplier,
        overlay_level=args.overlay_level,
        processes=args.processes,
        chunksize=args.chunksize,
        enable_blur=args.enable_blur,
        blur_kernel=args.blur_kernel,
        enable_laplacian_gate=args.enable_laplacian_gate,
        laplacian_min_var=args.laplacian_min_var,
        enable_macenko=args.enable_macenko,
        macenko_template_path=args.macenko_template_path,
    )


if __name__ == "__main__":
    main()