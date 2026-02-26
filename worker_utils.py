# worker_utils.py
import numpy as np
import cv2 as cv
import openslide
from openslide.deepzoom import DeepZoomGenerator
import tensorflow as tf

# Estado global interno de cada worker
_worker_vars = {}

def _apply_blur_filter(image_bgr, kernel_size=5):
    return cv.GaussianBlur(image_bgr, (kernel_size, kernel_size), 0)

def _variance_of_laplacian(image_bgr):
    gray = cv.cvtColor(image_bgr, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(gray, cv.CV_64F).var()

def _macenko_normalization(image_bgr, template_path):
    try:
        template = cv.imread(str(template_path))
        if template is None: return image_bgr
        image_lab = cv.cvtColor(image_bgr, cv.COLOR_BGR2LAB)
        template_lab = cv.cvtColor(template, cv.COLOR_BGR2LAB)
        for channel in [1, 2]:
            img_c = image_lab[:, :, channel].astype(np.float32)
            tpl_c = template_lab[:, :, channel].astype(np.float32)
            img_std = np.std(img_c)
            if img_std > 0:
                img_c = (img_c - np.mean(img_c)) * (np.std(tpl_c) / img_std) + np.mean(tpl_c)
                image_lab[:, :, channel] = np.clip(img_c, 0, 255).astype(np.uint8)
        return cv.cvtColor(image_lab, cv.COLOR_LAB2BGR)
    except:
        return image_bgr

def init_worker(image_path, tflite_path, patch_size, dz_level, preproc_cfg):
    """Inicializa o modelo e o acesso ao slide em cada núcleo da CPU"""
    global _worker_vars
    _worker_vars['patch_size'] = int(patch_size)
    _worker_vars['dz_level'] = int(dz_level)
    _worker_vars['preproc'] = preproc_cfg
    
    wsi = openslide.OpenSlide(image_path)
    _worker_vars['tiles'] = DeepZoomGenerator(wsi, tile_size=int(patch_size), overlap=0)
    
    try:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(model_path=tflite_path, num_threads=1)
    except:
        interp = tf.lite.Interpreter(model_path=tflite_path, num_threads=1)
    
    interp.allocate_tensors()
    _worker_vars['interp'] = interp
    _worker_vars['in_idx'] = interp.get_input_details()[0]["index"]
    _worker_vars['out_idx'] = interp.get_output_details()[0]["index"]

def process_patch(coords):
    """Processa um patch individual enviado pelo Pool"""
    abs_y, abs_x = coords
    cfg = _worker_vars['preproc']
    ps = _worker_vars['patch_size']
    
    tile_row, tile_col = abs_y // ps, abs_x // ps
    
    # Extração do Slide
    img_rgb = _worker_vars['tiles'].get_tile(_worker_vars['dz_level'], (tile_col, tile_row)).convert("RGB")
    img_bgr = cv.cvtColor(np.array(img_rgb), cv.COLOR_RGB2BGR)

    # Pré-processamento
    if cfg.get("enable_blur"):
        img_bgr = _apply_blur_filter(img_bgr, cfg["blur_kernel"])
    if cfg.get("enable_macenko") and cfg.get("macenko_template_path"):
        img_bgr = _macenko_normalization(img_bgr, cfg["macenko_template_path"])
    
    # Filtro de Nitidez (Laplacian)
    if cfg.get("enable_laplacian_gate"):
        if _variance_of_laplacian(img_bgr) < cfg.get("laplacian_min_var", 0):
            return None

    # Inferência
    input_data = cv.resize(img_bgr.astype(np.float32), (224, 224))
    input_data = np.expand_dims(input_data, axis=0)
    
    _worker_vars['interp'].set_tensor(_worker_vars['in_idx'], input_data)
    _worker_vars['interp'].invoke()
    pred = _worker_vars['interp'].get_tensor(_worker_vars['out_idx'])[0][1]

    # Retorno se acima do threshold
    if pred > cfg.get("threshold", 0.9):
        return (int(tile_row), int(tile_col), float(pred))
    return None