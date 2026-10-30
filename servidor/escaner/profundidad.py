"""Profundidad Z estimada a partir de la silueta y de los gradientes de la foto.

Una sola foto no trae profundidad, así que se supone que el objeto está a una
distancia Z₀ de la cámara y se "infla" su silueta:

  • d(u, v): distancia de cada píxel al borde de la silueta (transformada de
    distancia; cumple |∇d| = 1). Con s = d / d_max, el perfil de un cuarto de
    círculo  h = R·√(1 − (1 − s)²)  da secciones redondeadas, como un cilindro.
  • Detalle: el gradiente de la luminancia (Sobel) se integra de forma
    aproximada con un filtro pasa-altos: las zonas más claras que su entorno
    suelen estar más cerca de la luz y de la cámara.

La cara frontal queda en Z₀ − g − h y la trasera en Z₀ + g + ε·h (g: medio
grosor de la extrusión, ε: cuánto se repite el relieve atrás).
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Profundidad:
    Z_frente: np.ndarray   # mm, por píxel (solo tiene sentido dentro de la máscara)
    Z_atras: np.ndarray
    altura: np.ndarray     # h(u, v) en mm
    radio_mm: float        # d_max convertido a mm
    gradiente: np.ndarray  # |∇L| normalizado (0–1), para mostrar


def _suavizar(campo: np.ndarray, mascara: np.ndarray, sigma: float) -> np.ndarray:
    """Convolución gaussiana normalizada: promedia solo los píxeles del objeto."""
    m = mascara.astype(np.float32)
    num = cv2.GaussianBlur(campo * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return np.where(m > 0, num / np.maximum(den, 1e-6), 0).astype(np.float32)


def estimar(
    mascara: np.ndarray,
    gris: np.ndarray,
    Z0: float,
    mm_por_px: float,
    relieve: float = 0.75,
    detalle: float = 0.06,
    grosor: float = 0.10,
    espejo: float = 0.6,
) -> Profundidad:
    m = mascara.astype(np.uint8)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    d_max = float(d.max()) or 1.0
    s = d / d_max
    perfil = np.sqrt(np.clip(1 - (1 - s) ** 2, 0, 1)).astype(np.float32)
    radio_mm = d_max * mm_por_px

    # Gradiente de la luminancia (para mostrarlo) y detalle por sombreado
    L = gris.astype(np.float32) / 255
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    magnitud = np.sqrt(gx * gx + gy * gy) * m
    gradiente = magnitud / (float(magnitud.max()) or 1.0)

    L_suave = _suavizar(L, m, sigma=max(2.0, d_max / 3))
    pasa_altos = (L - L_suave) * m
    desv = float(pasa_altos[m > 0].std()) or 1.0
    textura = np.clip(pasa_altos / (2.5 * desv), -1, 1)

    h = radio_mm * (relieve * perfil + detalle * textura * perfil)
    h = _suavizar(np.maximum(h, 0), m, sigma=max(1.0, d_max / 40))

    g = grosor * radio_mm
    Z_frente = (Z0 - g - h).astype(np.float32)
    Z_atras = (Z0 + g + espejo * h).astype(np.float32)
    return Profundidad(Z_frente, Z_atras, h, radio_mm, gradiente.astype(np.float32))
