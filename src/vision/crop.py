"""
Recorte con mascara, fondo neutro y normalizacion de tamano.

La salida de este modulo es la unica entrada de las tres vias, para que la
comparacion entre representaciones no dependa de diferencias de preprocesado.
"""

from __future__ import annotations

import cv2
import numpy as np

import config

FONDO_NEUTRO = 128   # gris medio: no sesga ningun canal de color


def crop_to_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    size: int = config.CROP_SIZE,
    margen: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    """Recorta el objeto a un cuadrado normalizado.

    Pasos:
      1. caja envolvente de la mascara, expandida un `margen` relativo
      2. cuadrado centrado en esa caja (preserva la relacion de aspecto, que
         es una de las features de la via A)
      3. fondo sustituido por gris neutro donde la mascara es cero
      4. redimensionado a (size, size)

    Devuelve (recorte BGR uint8, mascara uint8 0/255), ambos (size, size).
    """
    alto, ancho = mask.shape[:2]
    ys, xs = np.nonzero(mask)
    if ys.size == 0:                       # sin objeto: centro de la imagen
        y0, x0, y1, x1 = 0, 0, alto, ancho
    else:
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

    lado = max(y1 - y0, x1 - x0)
    lado = int(round(lado * (1.0 + 2.0 * margen)))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2

    # Recorte con relleno explicito: si la caja se sale de la imagen, se
    # rellena con gris neutro en vez de desplazar el centro (desplazarlo
    # falsearia la posicion del objeto dentro del recorte).
    mitad = lado // 2
    recorte = np.full((lado, lado, 3), FONDO_NEUTRO, np.uint8)
    recorte_mask = np.zeros((lado, lado), np.uint8)

    sy0, sy1 = max(0, cy - mitad), min(alto, cy - mitad + lado)
    sx0, sx1 = max(0, cx - mitad), min(ancho, cx - mitad + lado)
    dy0, dx0 = sy0 - (cy - mitad), sx0 - (cx - mitad)

    recorte[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = image_bgr[sy0:sy1, sx0:sx1]
    recorte_mask[dy0:dy0 + (sy1 - sy0), dx0:dx0 + (sx1 - sx0)] = mask[sy0:sy1, sx0:sx1]

    recorte[recorte_mask == 0] = FONDO_NEUTRO

    recorte = cv2.resize(recorte, (size, size), interpolation=cv2.INTER_AREA)
    recorte_mask = cv2.resize(recorte_mask, (size, size),
                              interpolation=cv2.INTER_NEAREST)
    return recorte, recorte_mask


def to_gray_vector(crop_bgr: np.ndarray, size: int = config.EIGEN_SIZE) -> np.ndarray:
    """Recorte -> vector fila de la via B.

    Convierte a gris con la fila de luminancia de la matriz YCbCr, redimensiona
    a size x size y aplana a R^(size*size). Normalizado a [0, 1] para que la
    media y la SVD posteriores no dependan de la escala de uint8.
    """
    gris = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gris = cv2.resize(gris, (size, size), interpolation=cv2.INTER_AREA)
    return gris.astype(np.float64).ravel() / 255.0
