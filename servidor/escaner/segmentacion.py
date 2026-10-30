"""Segmentación: separar el objeto del fondo sin necesidad de un fondo neutro.

1. rembg (red U²-Net / IS-Net) estima una máscara alfa del objeto principal.
2. Si rembg no está disponible o no encuentra nada, se usa GrabCut de OpenCV
   inicializado con el recuadro de encuadre (el objeto está dentro de él).
3. La máscara se limpia: apertura y cierre morfológicos, se queda la región
   conectada que más se superpone al recuadro y se rellenan sus huecos.
"""

import os
import threading

import cv2
import numpy as np

MODELO_REMBG = os.environ.get("ESCANER_MODELO", "u2net")

_sesion = None
_candado = threading.Lock()


def _sesion_rembg():
    global _sesion
    with _candado:
        if _sesion is None:
            from rembg import new_session

            _sesion = new_session(MODELO_REMBG)
        return _sesion


def precargar() -> None:
    """Carga la red al iniciar el servidor (la primera vez se descarga)."""
    try:
        _sesion_rembg()
    except Exception as e:  # sin red: quedará GrabCut
        print(f"[segmentación] rembg no disponible: {e}")


def recuadro_px(marco: tuple[float, float, float, float], ancho: int, alto: int) -> tuple[int, int, int, int]:
    """Recuadro de encuadre en fracciones (x0, y0, x1, y1) → píxeles (x, y, w, h)."""
    x0, y0 = int(marco[0] * ancho), int(marco[1] * alto)
    x1, y1 = int(marco[2] * ancho), int(marco[3] * alto)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def con_rembg(bgr: np.ndarray) -> np.ndarray:
    from rembg import remove

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    alfa = remove(rgb, session=_sesion_rembg(), only_mask=True, post_process_mask=True)
    return np.asarray(alfa, dtype=np.uint8)


def con_grabcut(bgr: np.ndarray, recuadro: tuple[int, int, int, int]) -> np.ndarray:
    mascara = np.zeros(bgr.shape[:2], np.uint8)
    fondo, frente = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mascara, recuadro, fondo, frente, 5, cv2.GC_INIT_WITH_RECT)
    return np.where((mascara == cv2.GC_FGD) | (mascara == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def limpiar(alfa: np.ndarray, recuadro: tuple[int, int, int, int]) -> np.ndarray:
    """Alfa (0–255) → máscara binaria (0/1) de una sola pieza y sin huecos."""
    m = (alfa > 127).astype(np.uint8)
    lado = max(3, (min(m.shape) // 160) | 1)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, nucleo)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, nucleo)

    # Región conectada con más píxeles dentro del recuadro de encuadre
    n, etiquetas = cv2.connectedComponents(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    x, y, w, h = recuadro
    dentro = np.bincount(etiquetas[y:y + h, x:x + w].ravel(), minlength=n)
    dentro[0] = 0
    if dentro.max() == 0:  # nada dentro del recuadro: la región más grande
        dentro = np.bincount(etiquetas.ravel(), minlength=n)
        dentro[0] = 0
    m = (etiquetas == int(dentro.argmax())).astype(np.uint8)

    # Huecos: el contorno exterior relleno
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    lleno = np.zeros_like(m)
    cv2.drawContours(lleno, contornos, -1, 1, thickness=cv2.FILLED)
    return lleno


def segmentar(bgr: np.ndarray, marco: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, str]:
    """Devuelve (máscara 0/1, alfa 0–255, método usado)."""
    alto, ancho = bgr.shape[:2]
    recuadro = recuadro_px(marco, ancho, alto)
    area_min = 0.002 * ancho * alto
    try:
        alfa = con_rembg(bgr)
        mascara = limpiar(alfa, recuadro)
        if mascara.sum() >= area_min:
            return mascara, alfa, f"rembg ({MODELO_REMBG})"
    except Exception as e:
        print(f"[segmentación] rembg falló, se usa GrabCut: {e}")
    alfa = con_grabcut(bgr, recuadro)
    return limpiar(alfa, recuadro), alfa, "GrabCut (OpenCV)"
