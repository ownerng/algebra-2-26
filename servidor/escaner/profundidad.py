"""Profundidad Z estimada a partir de la silueta y de los gradientes de la foto.

Una sola foto no trae profundidad, así que se supone que el objeto está a una
distancia Z₀ de la cámara y se "infla" su silueta:

  • d(u, v): distancia de cada píxel al borde de la silueta (transformada de
    distancia; cumple |∇d| = 1). Con s = d / d_max, el perfil de un cuarto de
    círculo  h = R·√(1 − (1 − s)²)  da secciones redondeadas, como un cilindro.
  • Forma real (si está la red): Depth Anything V2 estima la disparidad D
    (relativa: más grande = más cerca) de cada píxel. Se mide respecto al borde
    de la silueta, r = (D − D_borde) / (D_alto − D_borde), y se mezcla con el
    perfil redondeado, que garantiza que el sólido cierre en el contorno.
  • Detalle (sin la red): el gradiente de la luminancia (Sobel) se integra de
    forma aproximada con un filtro pasa-altos: las zonas más claras que su
    entorno suelen estar más cerca de la luz y de la cámara.

La cara frontal queda en Z₀ − g − h y la trasera en Z₀ + g + ε·h (g: medio
grosor de la extrusión, ε: cuánto se repite el relieve atrás).
"""

import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

URL_IA = "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
RUTA_IA = Path(os.environ.get("ESCANER_PROFUNDIDAD", Path.home() / ".cache" / "escaner" / "depth_anything_v2_small.onnx"))
LADO_IA = 518   # lado mayor de la entrada de la red (múltiplo de 14, como en su entrenamiento)
MEDIA = np.array([0.485, 0.456, 0.406], np.float32)   # normalización de ImageNet (RGB)
DESV = np.array([0.229, 0.224, 0.225], np.float32)

_sesion = None
_candado = threading.Lock()


@dataclass
class Profundidad:
    Z_frente: np.ndarray   # mm, por píxel (solo tiene sentido dentro de la máscara)
    Z_atras: np.ndarray
    altura: np.ndarray     # h(u, v) en mm
    radio_mm: float        # d_max convertido a mm
    gradiente: np.ndarray  # |∇L| normalizado (0–1), para mostrar


def precargar() -> None:
    """Descarga la red la primera vez (~100 MB) y la carga al iniciar el servidor."""
    try:
        if not RUTA_IA.exists():
            print(f"[profundidad] descargando Depth Anything V2 en {RUTA_IA}…")
            RUTA_IA.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(URL_IA, RUTA_IA.with_suffix(".parcial"))
            RUTA_IA.with_suffix(".parcial").replace(RUTA_IA)
        _sesion_ia()
    except Exception as e:  # sin red: queda el perfil de la silueta
        print(f"[profundidad] Depth Anything no disponible: {e}")


def _sesion_ia():
    global _sesion
    with _candado:
        if _sesion is None:
            import onnxruntime as ort

            _sesion = ort.InferenceSession(str(RUTA_IA), providers=["CPUExecutionProvider"])
        return _sesion


def disparidad(bgr: np.ndarray, mascara: np.ndarray) -> np.ndarray | None:
    """Disparidad de Depth Anything V2, del tamaño de la foto (None si la red no
    está). La red ve solo un recorte alrededor del objeto, así le dedica más píxeles."""
    if _sesion is None and not RUTA_IA.exists():
        return None
    try:
        sesion = _sesion_ia()
    except Exception as e:
        print(f"[profundidad] Depth Anything falló: {e}")
        return None
    alto, ancho = mascara.shape
    vs, us = np.nonzero(mascara)
    m = int(0.15 * max(us.max() - us.min(), vs.max() - vs.min()))
    u0, u1 = max(0, us.min() - m), min(ancho, us.max() + m + 1)
    v0, v1 = max(0, vs.min() - m), min(alto, vs.max() + m + 1)
    recorte = bgr[v0:v1, u0:u1]
    h, w = recorte.shape[:2]
    e = LADO_IA / max(h, w)
    H, W = max(14, round(h * e / 14) * 14), max(14, round(w * e / 14) * 14)
    x = cv2.resize(recorte, (W, H), interpolation=cv2.INTER_CUBIC)[:, :, ::-1].astype(np.float32) / 255
    x = ((x - MEDIA) / DESV).transpose(2, 0, 1)[None]
    d = sesion.run(None, {"pixel_values": np.ascontiguousarray(x)})[0][0]
    salida = np.zeros((alto, ancho), np.float32)
    salida[v0:v1, u0:u1] = cv2.resize(d.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    return salida


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
    disparidad: np.ndarray | None = None,
    ia: float = 0.7,
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

    if disparidad is not None:
        dentro = m > 0
        D_borde = float(np.median(disparidad[dentro & (d <= 2)]))
        D_alto = float(np.percentile(disparidad[dentro], 99))
        # Si el objeto es plano, D_alto − D_borde es puro ruido: el denominador
        # mínimo (15 % del rango de toda la escena) evita amplificarlo
        escena = float(np.percentile(disparidad, 99) - np.percentile(disparidad, 1))
        r = np.clip((disparidad - D_borde) / max(D_alto - D_borde, 0.15 * escena, 1e-6), 0, 1)
        h = radio_mm * relieve * (ia * r + (1 - ia) * perfil)
    else:
        h = radio_mm * (relieve * perfil + detalle * textura * perfil)
    h = _suavizar(np.maximum(h, 0), m, sigma=max(1.0, d_max / 40))

    g = grosor * radio_mm
    Z_frente = (Z0 - g - h).astype(np.float32)
    Z_atras = (Z0 + g + espejo * h).astype(np.float32)
    return Profundidad(Z_frente, Z_atras, h, radio_mm, gradiente.astype(np.float32))
