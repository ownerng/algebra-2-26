"""
Segmentacion objeto/fondo en el espacio YCbCr.

Por que YCbCr y no HSV
----------------------
La conversion RGB -> HSV **no es una transformacion lineal**: H y S se definen
a partir de max(R,G,B), min(R,G,B) y divisiones entre ellos, operaciones que no
se pueden escribir como una matriz. RGB -> YCbCr si lo es. Con la recomendacion
ITU-R BT.601:

    [Y ]   [ 0.299000   0.587000   0.114000] [R]   [  0]
    [Cb] = [-0.168736  -0.331264   0.500000] [G] + [128]
    [Cr]   [ 0.500000  -0.418688  -0.081312] [B]   [128]

es decir  y = M x + b  con M una matriz 3x3 exacta y b un desplazamiento
constante (transformacion afin cuya parte lineal es M). Todo el proyecto usa
esta matriz para que la etapa de segmentacion sea tambien algebra lineal.
"""

from __future__ import annotations

import cv2
import numpy as np

import config

# Parte lineal de la conversion (ITU-R BT.601), en orden R, G, B.
YCBCR_MATRIX = np.array([
    [0.299000,  0.587000,  0.114000],
    [-0.168736, -0.331264,  0.500000],
    [0.500000, -0.418688, -0.081312],
], dtype=np.float64)

YCBCR_OFFSET = np.array([0.0, 128.0, 128.0], dtype=np.float64)

# Peso de la luminancia dentro de la distancia al fondo. La croma sola no
# distingue un objeto oscuro sobre fondo negro (COIL-100), donde Cb y Cr valen
# ~128 en ambos; la luma si, pero pesa menos porque cambia con la iluminacion.
PESO_LUMA = 0.5


def rgb_to_ycbcr(image_bgr: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR uint8 a YCbCr float64 con la matriz 3x3.

    Formula:  y = M x + b,  aplicada pixel a pixel como producto
    matriz-vector. Se implementa con un unico producto tensorial
    (H, W, 3) @ (3, 3)^T para no iterar en Python.
    """
    rgb = image_bgr[..., ::-1].astype(np.float64)
    return rgb @ YCBCR_MATRIX.T + YCBCR_OFFSET


def _color_de_fondo(ycbcr: np.ndarray, borde: int = 6) -> np.ndarray:
    """Estima el color del fondo con la mediana del marco exterior.

    La mediana resiste que un objeto toque el borde: mientras ocupe menos de
    la mitad del marco, el valor devuelto sigue siendo el del fondo.
    """
    marco = np.concatenate([
        ycbcr[:borde].reshape(-1, 3),
        ycbcr[-borde:].reshape(-1, 3),
        ycbcr[:, :borde].reshape(-1, 3),
        ycbcr[:, -borde:].reshape(-1, 3),
    ])
    return np.median(marco, axis=0)


def distancia_al_fondo(image_bgr: np.ndarray) -> np.ndarray:
    """Mapa (H, W) de distancia de cada pixel al color de fondo en YCbCr.

    d(p) = || diag(w) (ycbcr(p) - fondo) ||_2 ,  w = (PESO_LUMA, 1, 1)
    """
    ycbcr = rgb_to_ycbcr(image_bgr)
    fondo = _color_de_fondo(ycbcr)
    delta = ycbcr - fondo
    delta[..., 0] *= PESO_LUMA
    return np.linalg.norm(delta, axis=2)


def segment(
    image_bgr: np.ndarray,
    umbral: float = config.CHROMA_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Segmenta el objeto principal.

    Devuelve (mascara uint8 0/255, contorno o None). El contorno es el de
    mayor area por encima de config.MIN_CONTOUR_AREA; si ninguno lo supera se
    devuelve None y la mascara queda vacia, lo que el disparador de estabilidad
    interpreta como "no hay objeto".
    """
    dist = distancia_al_fondo(image_bgr)
    mascara = (dist > umbral).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return np.zeros_like(mascara), None

    mayor = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(mayor) < config.MIN_CONTOUR_AREA:
        return np.zeros_like(mascara), None

    limpia = np.zeros_like(mascara)
    cv2.drawContours(limpia, [mayor], -1, 255, thickness=cv2.FILLED)
    return limpia, mayor
