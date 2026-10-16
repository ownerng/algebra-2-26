"""
Inferencia de la via D: clasificacion de fruta con TTA, calibracion y rechazo.

Tres decisiones que importan mas que la arquitectura:

  1. Calibracion por temperatura. Una red afinada sale sobreconfiada (dice 99 %
     y se equivoca). Se ajusta un escalar T en validacion para que la
     confianza signifique algo: el rechazo y la barra de confianza dependen
     de eso.
  2. Promedio de vistas (TTA): original + espejo horizontal + espejo vertical
     + giro de 90 grados. Una fruta no tiene orientacion, asi que promediar
     solo quita varianza; cuesta 4 pasadas y casi siempre suma exactitud ante
     imagenes de camara.
  3. Rechazo por confianza calibrada y por entropia. Los umbrales se leen del
     propio conjunto de validacion (percentil, config.REJECT_PERCENTILE), igual
     que en las vias A-C: nada escrito a mano.

Uso:
    from src.fruit.infer import FrutaNet
    fn = FrutaNet.cargar()
    r = fn.predecir(imagen_bgr)        # {'clase', 'confianza', 'conocido', 'top': [...]}
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import config

log = logging.getLogger(__name__)


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = np.asarray(z, np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def entropia_normalizada(p: np.ndarray) -> np.ndarray:
    """H(p) / log(C) en [0, 1]: 0 = seguro, 1 = uniforme."""
    h = -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=-1)
    return h / np.log(p.shape[-1])


def ajustar_temperatura(logits: np.ndarray, y: np.ndarray) -> float:
    """T que minimiza la log-verosimilitud negativa en validacion (Guo et al. 2017).

    Busqueda en rejilla logaritmica + refinado: un escalar no necesita LBFGS y
    asi no se depende de torch para calibrar.
    """
    logits = np.asarray(logits, np.float64)
    y = np.asarray(y)

    def nll(T: float) -> float:
        p = softmax(logits, T)
        return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean())

    gruesa = np.exp(np.linspace(np.log(0.3), np.log(10.0), 80))
    T0 = float(gruesa[int(np.argmin([nll(t) for t in gruesa]))])
    fina = np.linspace(T0 * 0.9, T0 * 1.1, 60)
    return float(fina[int(np.argmin([nll(t) for t in fina]))])


def fijar_umbrales(logits: np.ndarray, y: np.ndarray, T: float,
                   percentil: float = config.REJECT_PERCENTILE) -> dict:
    """Umbrales de rechazo leidos de validacion (solo con las muestras ACERTADAS).

    Se usan las acertadas porque el umbral debe describir "como luce estar
    seguro y tener razon"; mezclar los errores subiria la tolerancia justo
    donde se quiere exigir mas.
    """
    p = softmax(logits, T)
    ok = p.argmax(1) == np.asarray(y)
    conf = p.max(1)[ok]
    ent = entropia_normalizada(p)[ok]
    return {
        "confianza_min": float(np.percentile(conf, 100.0 - percentil)),
        "entropia_max": float(np.percentile(ent, percentil)),
        "percentil": float(percentil),
    }


def es_conocido(conf: np.ndarray, ent: np.ndarray, umbrales: dict) -> np.ndarray:
    return (np.asarray(conf) >= umbrales["confianza_min"]) & \
           (np.asarray(ent) <= umbrales["entropia_max"])


class FrutaNet:
    """Red afinada lista para inferir sobre imagenes BGR."""

    def __init__(self, red, meta: dict) -> None:
        self.red = red
        self.clases: list[str] = meta["clases"]
        self.lado: int = meta["lado"]
        self.T: float = meta["temperatura"]
        self.umbrales: dict = meta["umbrales"]
        self.metricas: dict = meta.get("metricas", {})
        self.arch: str = meta["arch"]

    @staticmethod
    def disponible(ruta: Path = config.FRUIT_MODEL_PATH) -> bool:
        return Path(ruta).exists()

    @staticmethod
    def cargar(ruta: Path = config.FRUIT_MODEL_PATH) -> "FrutaNet":
        from src.fruit import model
        red, meta = model.cargar(ruta)
        log.info("via D lista: %s, %d clases, T=%.3f, umbrales %s",
                 meta["arch"], len(meta["clases"]), meta["temperatura"],
                 meta["umbrales"])
        return FrutaNet(red, meta)

    # -- nucleo ---------------------------------------------------------
    def logits(self, imagenes_bgr: list[np.ndarray], tta: bool = config.FRUIT_TTA,
               batch: int = 32) -> np.ndarray:
        """Logits medios (m, C). Con TTA se promedian los logits de 4 vistas."""
        import torch
        from src.fruit.data import a_tensor

        salida = []
        with torch.inference_mode():
            for ini in range(0, len(imagenes_bgr), batch):
                x = torch.stack([a_tensor(im, self.lado)
                                 for im in imagenes_bgr[ini:ini + batch]])
                z = self.red(x)
                if tta:
                    z = z + self.red(torch.flip(x, dims=[3]))
                    z = z + self.red(torch.flip(x, dims=[2]))
                    z = z + self.red(torch.rot90(x, 1, dims=[2, 3]))
                    z = z / 4.0
                salida.append(z.numpy())
        return np.concatenate(salida).astype(np.float64)

    def predecir_lote(self, imagenes_bgr: list[np.ndarray], top: int = 3,
                      tta: bool = config.FRUIT_TTA) -> list[dict]:
        p = softmax(self.logits(imagenes_bgr, tta), self.T)
        ent = entropia_normalizada(p)
        conocido = es_conocido(p.max(1), ent, self.umbrales)
        resultados = []
        for i in range(len(p)):
            orden = np.argsort(-p[i])[:top]
            resultados.append({
                "clase": self.clases[int(orden[0])],
                "confianza": float(p[i, orden[0]]),
                "entropia": float(ent[i]),
                "conocido": bool(conocido[i]),
                "top": [(self.clases[int(j)], float(p[i, j])) for j in orden],
            })
        return resultados

    def predecir(self, imagen_bgr: np.ndarray, top: int = 3,
                 tta: bool = config.FRUIT_TTA) -> dict:
        return self.predecir_lote([imagen_bgr], top, tta)[0]
