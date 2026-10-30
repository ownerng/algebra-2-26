"""Del cuadro de la cámara al sólido 3D, en cuatro pasos:

  1. Segmentación  → máscara alfa del objeto (rembg / GrabCut).
  2. Profundidad   → Z(u, v) a partir de la silueta y los gradientes.
  3. Retroproyección → P = Z·K⁻¹·[u, v, 1]ᵀ.
  4. Malla         → V ∈ R^(N×3), F ∈ Z^(M×3), con extrusión trasera y paredes.
"""

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import malla as malla_mod
from . import profundidad as prof_mod
from .geometria import inversa_K, matriz_K
from .segmentacion import segmentar

ANCHO_PROCESO = 640                     # la foto se reduce a este ancho
MARCO = (0.2, 0.15, 0.8, 0.85)          # recuadro de encuadre (fracciones), igual que en la web


@dataclass
class Parametros:
    distancia_mm: float = 400.0   # Z₀: distancia supuesta entre la cámara y el objeto
    fov: float = 65.0             # ángulo de visión horizontal de la cámara (grados)
    relieve: float = 0.75
    detalle: float = 0.06
    marco: tuple[float, float, float, float] = MARCO


@dataclass
class Resultado:
    foto: np.ndarray        # BGR reducida
    mascara: np.ndarray     # 0/1
    alfa: np.ndarray        # 0–255
    K: np.ndarray
    malla: malla_mod.Malla
    metodo: str
    tamano_mm: tuple[float, float, float]
    volumen_mm3: float
    tiempos: dict[str, float] = field(default_factory=dict)


class SinObjeto(Exception):
    pass


def reducir(bgr: np.ndarray) -> np.ndarray:
    alto, ancho = bgr.shape[:2]
    if ancho <= ANCHO_PROCESO:
        return bgr
    escala = ANCHO_PROCESO / ancho
    return cv2.resize(bgr, (ANCHO_PROCESO, round(alto * escala)), interpolation=cv2.INTER_AREA)


def escanear(bgr: np.ndarray, p: Parametros | None = None) -> Resultado:
    p = p or Parametros()
    tiempos: dict[str, float] = {}
    t = time.perf_counter()

    def marca(nombre: str) -> None:
        nonlocal t
        ahora = time.perf_counter()
        tiempos[nombre] = round((ahora - t) * 1000)
        t = ahora

    foto = reducir(bgr)
    alto, ancho = foto.shape[:2]

    mascara, alfa, metodo = segmentar(foto, p.marco)
    area = int(mascara.sum())
    if area < 0.002 * ancho * alto or area > 0.95 * ancho * alto:
        raise SinObjeto("no se distinguió el objeto; ponlo dentro del recuadro y que ocupe buena parte de él")
    marca("segmentacion")

    K = matriz_K(ancho, alto, p.fov)
    mm_por_px = p.distancia_mm / K[0, 0]  # un píxel a la distancia Z₀ mide Z₀ / f
    gris = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
    prof = prof_mod.estimar(mascara, gris, p.distancia_mm, mm_por_px, relieve=p.relieve, detalle=p.detalle)
    marca("profundidad")

    m = malla_mod.construir(mascara, prof.Z_frente, prof.Z_atras, K, p.distancia_mm)
    marca("malla")

    tam = m.V.max(axis=0) - m.V.min(axis=0)
    return Resultado(
        foto=foto, mascara=mascara, alfa=alfa, K=K, malla=m, metodo=metodo,
        tamano_mm=(float(tam[0]), float(tam[1]), float(tam[2])),
        volumen_mm3=malla_mod.volumen(m.V, m.F), tiempos=tiempos,
    )


__all__ = ["Parametros", "Resultado", "SinObjeto", "escanear", "inversa_K", "MARCO"]
