"""
Via C - embeddings de una red convolucional preentrenada.

Este es el UNICO punto del proyecto donde entra torch, y solo como extractor
de caracteristicas: el modelo llega preentrenado en ImageNet, se congela
(requires_grad = False, model.eval()) y ninguna capa se entrena. La PCA que
reduce los embeddings y el clasificador que decide son implementacion propia
en NumPy.

Nota sobre la dimension del embedding
-------------------------------------
MobileNetV3-Small termina en  Linear(576 -> 1024) -> Hardswish -> Dropout ->
Linear(1024 -> 1000). La penultima capa entrega por tanto 1024 componentes,
no 1280; 1280 es el ancho de MobileNetV3-Large. El codigo no fija el numero:
lo mide del modelo cargado y lo expone en EMBED_DIM.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np

import config

# Los pesos preentrenados se cachean dentro del proyecto, no en el perfil del
# usuario, para que el proyecto sea autocontenido.
os.environ.setdefault("TORCH_HOME", str(config.RAW_DIR / "torch"))

_MODELO = None
_DIM = None

# Pesos preentrenados puestos a mano, para redes que bloquean
# download.pytorch.org. Si el archivo existe se usa en vez de descargar.
WEIGHTS_LOCAL = config.RAW_DIR / "torch" / "mobilenet_v3_small.pth"
WEIGHTS_URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_model():
    """Carga MobileNetV3-Small preentrenado y CONGELADO.

    Congelar significa dos cosas distintas y ambas hacen falta:
      - requires_grad = False en todos los parametros: no acumulan gradiente
      - model.eval(): BatchNorm usa sus estadisticos guardados y Dropout se
        desactiva, asi el embedding de una imagen es deterministico

    Se cachea a nivel de modulo: cargar los pesos cuesta segundos y el modelo
    no cambia nunca.
    """
    global _MODELO, _DIM
    if _MODELO is not None:
        return _MODELO

    import torch
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    if WEIGHTS_LOCAL.exists():
        modelo = mobilenet_v3_small(weights=None)
        modelo.load_state_dict(torch.load(WEIGHTS_LOCAL, map_location="cpu"))
    else:
        try:
            modelo = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        except Exception as exc:
            raise RuntimeError(
                "No se pudieron descargar los pesos preentrenados (%s).\n"
                "Descarga manualmente %s y guardalo como %s"
                % (exc, WEIGHTS_URL, WEIGHTS_LOCAL)
            ) from exc

    for p in modelo.parameters():
        p.requires_grad = False
    modelo.eval()

    _DIM = modelo.classifier[-1].in_features     # 1024 en MobileNetV3-Small
    _MODELO = modelo
    return modelo


def embed_dim() -> int:
    """Dimension del embedding de la penultima capa del modelo cargado."""
    if _DIM is None:
        load_model()
    return int(_DIM)


def preprocess(crops_bgr: np.ndarray) -> np.ndarray:
    """Recortes BGR uint8 -> tensor NCHW float32 normalizado a ImageNet.

    (m, H, W, 3) uint8  ->  (m, 3, 224, 224) float32
    """
    lote = np.stack([
        cv2.resize(c, (config.CNN_INPUT, config.CNN_INPUT),
                   interpolation=cv2.INTER_AREA)
        for c in crops_bgr
    ])
    rgb = lote[..., ::-1].astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))


def _forward(crops_bgr: np.ndarray, modelo, hasta_penultima: bool,
             batch: int = 64) -> np.ndarray:
    import torch

    salidas = []
    with torch.no_grad():                      # sin grafo: nada que entrenar
        for ini in range(0, len(crops_bgr), batch):
            x = torch.from_numpy(preprocess(crops_bgr[ini:ini + batch]))
            h = modelo.features(x)
            h = modelo.avgpool(h)
            h = torch.flatten(h, 1)
            capas = modelo.classifier[:-1] if hasta_penultima else modelo.classifier
            salidas.append(capas(h).numpy())
    return np.concatenate(salidas).astype(np.float64)


def extract_embeddings(images: np.ndarray, model=None, batch: int = 64) -> np.ndarray:
    """Recortes BGR -> embeddings de la penultima capa, (m, embed_dim()).

    El modelo esta congelado: esta funcion solo hace pasadas hacia adelante.
    """
    modelo = model if model is not None else load_model()
    return _forward(np.asarray(images), modelo, hasta_penultima=True, batch=batch)


def imagenet_logits(images: np.ndarray, model=None, batch: int = 64) -> np.ndarray:
    """Salida de la cabeza original de ImageNet, (m, 1000).

    Se usa solo para el baseline "MobileNetV3 completo": mide el techo del
    modelo preentrenado tal como viene, sin reemplazar su cabeza.
    """
    modelo = model if model is not None else load_model()
    return _forward(np.asarray(images), modelo, hasta_penultima=False, batch=batch)


def demo() -> None:
    """Verifica que el modelo esta congelado y que el embedding es estable."""
    import torch

    modelo = load_model()
    assert not any(p.requires_grad for p in modelo.parameters()), "modelo NO congelado"
    assert not modelo.training, "modelo NO esta en modo eval"

    rng = np.random.default_rng(0)
    crops = rng.integers(0, 255, (4, 128, 128, 3), dtype=np.uint8)

    z1 = extract_embeddings(crops)
    z2 = extract_embeddings(crops)
    assert z1.shape == (4, embed_dim())
    # eval() + no_grad => deterministico
    assert np.abs(z1 - z2).max() < 1e-6
    assert imagenet_logits(crops[:2]).shape == (2, 1000)

    # Ningun parametro guarda gradiente tras la pasada.
    assert all(p.grad is None for p in modelo.parameters())
    _ = torch  # el import documenta la dependencia del check

    print("cnn demo ok, embed_dim = %d" % embed_dim())


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    demo()
