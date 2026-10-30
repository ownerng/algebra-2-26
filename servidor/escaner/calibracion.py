"""Calibración de la cámara con el tablero de marcadores (método de Zhang).

La matriz K de proceso.py sale de un ángulo de visión supuesto (65°) y no
tiene en cuenta la distorsión del lente. Con varias fotos del tablero desde
ángulos distintos, cada una aporta una homografía H = K·[r₁ r₂ t] del plano
del tablero a la imagen; como r₁ y r₂ son ortonormales,

    h₁ᵀ·K⁻ᵀK⁻¹·h₂ = 0,    h₁ᵀ·K⁻ᵀK⁻¹·h₁ = h₂ᵀ·K⁻ᵀK⁻¹·h₂,

dos ecuaciones lineales por foto en los elementos de B = K⁻ᵀK⁻¹. Con 3 o más
fotos se resuelve B (mínimos cuadrados), de B sale K, y todo se refina junto
con la distorsión radial y tangencial (cv2.calibrateCamera).

K escala con la imagen: si la foto se reduce por s, K' = diag(s, s, 1)·K. La
distorsión está en coordenadas normalizadas y no cambia.
"""

import json
import os
from pathlib import Path

import cv2
import numpy as np

from . import tablero
from .geometria import matriz_K

RUTA = Path(os.environ.get("ESCANER_CALIBRACION", Path.home() / ".cache" / "escaner" / "calibracion.json"))
MIN_FOTOS = 5
MIN_MARCADORES = 4

capturas: list[tuple[np.ndarray, np.ndarray]] = []   # (X del tablero, uv) de cada foto
tamano: tuple[int, int] | None = None                 # (ancho, alto) de las fotos capturadas
actual: dict | None = None                            # la calibración en uso (se carga al iniciar)


def cargar() -> None:
    global actual
    try:
        actual = json.loads(RUTA.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        actual = None


def _escala(ancho: int, alto: int) -> float | None:
    """Factor de la foto respecto a la calibración, o None si no aplica (otra proporción)."""
    if actual is None or abs(ancho / alto - actual["ancho"] / actual["alto"]) > 0.01:
        return None
    return ancho / actual["ancho"]


def aplica(ancho: int, alto: int) -> bool:
    """¿Hay calibración para una foto de este tamaño?"""
    return _escala(ancho, alto) is not None


def K(ancho: int, alto: int, fov: float) -> np.ndarray:
    """La K calibrada, escalada a esta foto; si no hay, la del ángulo de visión."""
    s = _escala(ancho, alto)
    if s is None:
        return matriz_K(ancho, alto, fov)
    return np.diag([s, s, 1.0]) @ np.array(actual["K"])


def corregir(bgr: np.ndarray) -> np.ndarray:
    """Quita la distorsión del lente (la K de la foto corregida es la misma)."""
    alto, ancho = bgr.shape[:2]
    if _escala(ancho, alto) is None:
        return bgr
    return cv2.undistort(bgr, K(ancho, alto, 0), np.array(actual["distorsion"]))


def agregar(bgr: np.ndarray) -> int:
    """Guarda las esquinas del tablero de una foto; devuelve cuántos marcadores vio (0 si no sirve)."""
    global tamano
    alto, ancho = bgr.shape[:2]
    det = tablero.detectar(bgr)
    if det is None or len(det[0]) < 4 * MIN_MARCADORES:
        return 0
    if tamano != (ancho, alto):   # cambió la cámara o su resolución: se empieza de nuevo
        capturas.clear()
        tamano = (ancho, alto)
    capturas.append(det)
    return len(det[0]) // 4


def calcular() -> dict:
    """Calibra con las fotos capturadas y guarda el resultado."""
    global actual
    if len(capturas) < MIN_FOTOS:
        raise ValueError(f"hacen falta al menos {MIN_FOTOS} fotos del tablero ({len(capturas)} hasta ahora)")
    # calibrateCamera quiere el patrón en el plano z = 0: (X, 0, Z) → (X, Z, 0)
    objetos = [np.column_stack([X[:, 0], X[:, 2], np.zeros(len(X))]).astype(np.float32) for X, _ in capturas]
    imagenes = [uv.astype(np.float32) for _, uv in capturas]
    error, Km, dist, _, _ = cv2.calibrateCamera(objetos, imagenes, tamano, None, None)
    actual = {"ancho": tamano[0], "alto": tamano[1], "K": Km.tolist(), "distorsion": dist.ravel().tolist(),
              "error_px": float(error), "fotos": len(capturas)}
    RUTA.parent.mkdir(parents=True, exist_ok=True)
    RUTA.write_text(json.dumps(actual, indent=2), encoding="utf-8")
    return actual


def borrar() -> None:
    global actual, tamano
    capturas.clear()
    tamano = None
    actual = None
    RUTA.unlink(missing_ok=True)


def estado() -> dict:
    a = actual
    return {
        "fotos": len(capturas), "minimo": MIN_FOTOS, "calibrada": a is not None,
        "f_px": round(a["K"][0][0], 1) if a else None, "error_px": round(a["error_px"], 3) if a else None,
        "resolucion": f'{a["ancho"]}×{a["alto"]}' if a else None,
    }
