import boto3
import openslide
import os
import cv2 as cv
import numpy as np
from urllib.parse import urlparse

import tflite_runtime.interpreter as tflite
from itertools import product
from multiprocessing import Pool

import tempfile, json
from urllib.parse import urlparse

EFS_MOUNT = "/mnt/efs-tcga"
TRASH_HOLD = 0.8

def _process_block(y, x):
    """
    Retorna uma lista de coordenadas (y+m, x+n) para patches
    que valem a pena processar, baseado na média do 'mask'.
    """
    return [
        (y + m, x + n)
        for m, n in product(indices, indices)
        if np.mean(mask[y + m : y + m + patch_size, x + n : x + n + patch_size]) > 16
    ]

def _process_patch(abs_y, abs_x):
    """
    Lê um patch da WSI em (abs_x, abs_y), redimensiona para 224x224
    e roda a inferência TFLite. Retorna (gridY, gridX, score) se passar do threshold.
    """

    # Ler o patch em resolução 0
    img_block = wsi.read_region((abs_x, abs_y), 0, (patch_size, patch_size)).convert('RGB')

    # Converter para array float32 e trocar RGB -> BGR
    img_block = np.array(img_block, dtype=np.float32)[:, :, ::-1]

    # Redimensionar para 224x224
    img_block = cv.resize(img_block, (224, 224))

    # Adicionar dimensão batch (1, 224, 224, 3)
    img_block = np.expand_dims(img_block, axis=0)

    # Fazer inferência no modelo TFLite
    interpreter.set_tensor(input_details[0]['index'], img_block)
    interpreter.invoke()
    prediction_xy = interpreter.get_tensor(output_details[0]['index'])[0][1]

    if prediction_xy > TRASH_HOLD:
        # Convertemos as coordenadas absolutos para índice de patch
        return (int(abs_y / patch_size), int(abs_x / patch_size), prediction_xy)
        

def main(event):
    # 1) Extrair path do body
    body = json.loads(event["body"])
    image_path = body.get("image_path")

    if not image_path:
        raise ValueError("É necessário passar image_path no body")

    # 2) Decidir que a imagem será lida diretamente do EFS
    local_filename = image_path

    # ----------------------------------
    # 2) Abrir a WSI e criar a 'mask'
    # ----------------------------------
    global wsi, mask, patch_size, width, hight
    wsi = openslide.OpenSlide(local_filename)

    # Vamos pegar o menor nível (para gerar a mask)
    level = len(wsi.level_dimensions) - 1

    # Lê a imagem nesse menor nível e converte p/ RGB
    img_low = wsi.read_region((0, 0), level, wsi.level_dimensions[level]).convert('RGB')

    # Dimensões reais no nível 0 (resolução máxima)
    w, h = wsi.dimensions

    # Definimos patch_size = 224 * 4 = 896
    patch_size = 224 * 4

    # Ajustar width e hight (sim, sem "e" -> hight) para serem múltiplos de patch_size
    width = w + (patch_size - w % patch_size)
    hight = h + (patch_size - h % patch_size)

    # Converte img_low em numpy (RGBA->GRAY)
    gray_img = cv.cvtColor(np.array(img_low), cv.COLOR_RGBA2GRAY)

    # Threshold + invert + OTSU
    _, thresh = cv.threshold(gray_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # Morph. close (remover buracos etc.)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    mask = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

    # Redimensionar a mask para o tamanho (width, hight)
    mask = cv.resize(mask, (width, hight))

    # ----------------------------------
    # 3) Carregar modelo TFLite
    # ----------------------------------
    global interpreter, input_details, output_details
    TF_LITE_QA_MODEL_FILE_NAME = 'tubular-bin-4x-v2.tflite'
    interpreter = tflite.Interpreter(model_path=TF_LITE_QA_MODEL_FILE_NAME)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # ----------------------------------
    # 4) Dividir em blocos e processar em paralelo
    # ----------------------------------
    global indices
    c = 4  # Cada bloco = c * patch_size = 4 * 896 = 3584
    slide_dim = patch_size * c

    indices = np.arange(0, slide_dim, patch_size)
    y_indices = np.arange(0, hight, slide_dim)
    x_indices = np.arange(0, width, slide_dim)

    # Matriz final de predição (cada célula -> pontuação)
    jet_heatmap_matrix = np.zeros((int(hight / patch_size), int(width / patch_size)), dtype=np.float16)

    # Filtrar blocos que têm média > 16 na mask
    block_coords = [
        (y, x)
        for y, x in product(y_indices, x_indices)
        if np.mean(mask[y : y + slide_dim, x : x + slide_dim]) > 16
    ]

    # 4.1) Primeiro chama _process_block(...) em paralelo
    with Pool() as pool:
        results_block = pool.starmap(_process_block, block_coords)

    # "results_block" é lista de listas, então "achatar" com list comprehension
    results_block = [sub for lista_sub in results_block for sub in lista_sub]

    print(f"Quantidade de patches selecionados: {len(results_block)}")

    # 4.2) Depois chama _process_patch(...) em paralelo
    with Pool() as pool:
        results_patch = pool.starmap(_process_patch, results_block)

    # Remove None
    results_patch = [item for item in results_patch if item is not None]

    if len(results_patch) == 0:
        print("Nenhum patch excedeu o threshold.")
    else:
        # Converter para numpy e separar colunas
        results_array = np.array(results_patch)
        ys = np.int_(results_array[:, 0])
        xs = np.int_(results_array[:, 1])
        vals = results_array[:, 2]

        # Preencher a jet_heatmap_matrix
        jet_heatmap_matrix[ys, xs] = vals

        print(f"jet_heatmap_matrix shape: {jet_heatmap_matrix.shape}")
        print(f"jet_heatmap_matrix max: {jet_heatmap_matrix.max()}")

    # Fecha o WSI e remove o arquivo local
    wsi.close()
    os.remove(local_filename)

    print("Inference completed, matrix shape:", jet_heatmap_matrix.shape)

    # Transformar os valores conforme necessário
    # Multiplica por 10 e converte para inteiros
    matriz_transformada = (jet_heatmap_matrix * 10).astype(int)

    # Imprimir a matriz transformada
    for linha in matriz_transformada:
        print(" ".join(map(str, linha)))
    
    # Retornar a matriz de predições
    return jet_heatmap_matrix
