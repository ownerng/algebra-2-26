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

Para habitaciones (habitacion.py) se usa la versión métrica de la misma red,
entrenada con interiores (Hypersim): da Z directamente en metros; SegFormer
(segmentación de escenas, ADE20K) para saber qué es cada píxel, y LaMa (una
red de inpainting) para inventar lo que hay detrás de los muebles.
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
# Depth Anything V2 Metric Indoor Small exportada a ONNX (entrada fija de 518 × 518)
URL_INTERIOR = "https://huggingface.co/ogasurfproject/yakumo-depth-metric-indoor-small/resolve/main/onnx/model.onnx"
RUTA_INTERIOR = Path(os.environ.get("ESCANER_PROFUNDIDAD_INTERIOR",
                                    Path.home() / ".cache" / "escaner" / "depth_anything_v2_metric_indoor_small.onnx"))

# LaMa (inpainting) del repositorio de modelos de OpenCV (entrada fija de 512 × 512)
URL_RELLENO = "https://huggingface.co/opencv/inpainting_lama/resolve/main/inpainting_lama_2025jan.onnx"
RUTA_RELLENO = Path(os.environ.get("ESCANER_RELLENO", Path.home() / ".cache" / "escaner" / "inpainting_lama.onnx"))
LADO_RELLENO = 512
# SegFormer-B2 entrenada con ADE20K (150 clases: pared, piso, techo, sofá, mesa…), cuantizada
URL_ESCENA = "https://huggingface.co/Xenova/segformer-b2-finetuned-ade-512-512/resolve/main/onnx/model_quantized.onnx"
RUTA_ESCENA = Path(os.environ.get("ESCANER_ESCENA", Path.home() / ".cache" / "escaner" / "segformer_b2_ade.onnx"))
LADO_ESCENA = 768

_sesiones: dict[Path, object] = {}
_candado = threading.Lock()


@dataclass
class Profundidad:
    Z_frente: np.ndarray   # mm, por píxel (solo tiene sentido dentro de la máscara)
    Z_atras: np.ndarray
    altura: np.ndarray     # h(u, v) en mm
    radio_mm: float        # d_max convertido a mm
    gradiente: np.ndarray  # |∇L| normalizado (0–1), para mostrar


def precargar() -> None:
    """Descarga las redes la primera vez (~100 MB cada una) y carga la de objetos;
    las de habitaciones se cargan al usarlas por primera vez."""
    for url, ruta in ((URL_IA, RUTA_IA), (URL_INTERIOR, RUTA_INTERIOR), (URL_RELLENO, RUTA_RELLENO), (URL_ESCENA, RUTA_ESCENA)):
        try:
            if not ruta.exists():
                print(f"[profundidad] descargando {ruta.name}…")
                ruta.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, ruta.with_suffix(".parcial"))
                ruta.with_suffix(".parcial").replace(ruta)
        except Exception as e:  # sin red: objetos con el perfil de la silueta, sin habitaciones
            print(f"[profundidad] {ruta.name} no disponible: {e}")
    try:
        _sesion_ia()
    except Exception as e:
        print(f"[profundidad] Depth Anything no disponible: {e}")


def _sesion_ia(ruta: Path = RUTA_IA):
    with _candado:
        if ruta not in _sesiones:
            import onnxruntime as ort

            opciones = ort.SessionOptions()
            opciones.log_severity_level = 3   # LaMa avisa de cada inicializador sin usar
            _sesiones[ruta] = ort.InferenceSession(str(ruta), opciones, providers=["CPUExecutionProvider"])
        return _sesiones[ruta]


def _entrada(bgr: np.ndarray, W: int, H: int) -> np.ndarray:
    """La foto a W × H, en RGB normalizado como en ImageNet, con forma [1, 3, H, W]."""
    x = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_CUBIC)[:, :, ::-1].astype(np.float32) / 255
    return np.ascontiguousarray(((x - MEDIA) / DESV).transpose(2, 0, 1)[None])


def metros(bgr: np.ndarray) -> np.ndarray | None:
    """Profundidad Z en metros de cada píxel de la foto (None si la red no está).
    La entrada de esta red es fija: la foto se estira a 518 × 518 y el resultado
    vuelve a su forma."""
    if RUTA_INTERIOR not in _sesiones and not RUTA_INTERIOR.exists():
        return None
    alto, ancho = bgr.shape[:2]
    z = _sesion_ia(RUTA_INTERIOR).run(None, {"pixel_values": _entrada(bgr, LADO_IA, LADO_IA)})[0][0]
    return cv2.resize(z.astype(np.float32), (ancho, alto), interpolation=cv2.INTER_CUBIC)


def etiquetas(bgr: np.ndarray) -> np.ndarray | None:
    """Clase de ADE20K de cada píxel (None si la red no está). La red da los
    logits a 1/4 de la entrada; se amplían solo los de las clases presentes."""
    if RUTA_ESCENA not in _sesiones and not RUTA_ESCENA.exists():
        return None
    alto, ancho = bgr.shape[:2]
    H = max(32, round(alto * LADO_ESCENA / ancho / 32) * 32)
    logits = _sesion_ia(RUTA_ESCENA).run(None, {"pixel_values": _entrada(bgr, LADO_ESCENA, H)})[0][0]
    clases = np.unique(logits.argmax(axis=0))
    grande = np.stack([cv2.resize(logits[c], (ancho, alto), interpolation=cv2.INTER_LINEAR) for c in clases])
    return clases[grande.argmax(axis=0)].astype(np.int32)


def rellenar(bgr: np.ndarray, mascara: np.ndarray) -> np.ndarray | None:
    """La foto con la zona de la máscara inventada por LaMa (None si la red no
    está). La red trabaja a 512 × 512; fuera de la máscara queda la foto original."""
    if RUTA_RELLENO not in _sesiones and not RUTA_RELLENO.exists():
        return None
    alto, ancho = bgr.shape[:2]
    m = cv2.dilate(mascara.astype(np.uint8), np.ones((7, 7), np.uint8))   # que no queden restos del borde
    lado = (LADO_RELLENO, LADO_RELLENO)
    x = cv2.resize(bgr, lado, interpolation=cv2.INTER_AREA).astype(np.float32).transpose(2, 0, 1)[None] / 255
    mm = (cv2.resize(m, lado, interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)[None, None]
    y = _sesion_ia(RUTA_RELLENO).run(None, {"image": np.ascontiguousarray(x), "mask": mm})[0][0]
    y = cv2.resize(np.clip(y.transpose(1, 2, 0), 0, 255).astype(np.uint8), (ancho, alto), interpolation=cv2.INTER_CUBIC)
    return np.where(m[..., None] > 0, y, bgr)


def disparidad(bgr: np.ndarray, mascara: np.ndarray) -> np.ndarray | None:
    """Disparidad de Depth Anything V2, del tamaño de la foto (None si la red no
    está). La red ve solo un recorte alrededor del objeto, así le dedica más píxeles."""
    if RUTA_IA not in _sesiones and not RUTA_IA.exists():
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
    d = sesion.run(None, {"pixel_values": _entrada(recorte, W, H)})[0][0]
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
