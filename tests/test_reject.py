"""
El rechazo fuera de dominio es un contrato, no un adorno.

El clasificador decide con argmax(x W) sobre un conjunto cerrado de clases, asi
que sin rechazo devuelve una fruta para cualquier entrada: un celular, una cara
o ruido. Estos tests fijan las dos propiedades que hacen util el rechazo:

  1. no rechaza el dominio que aprendio (si no, la aplicacion no sirve)
  2. si rechaza lo que es visiblemente ajeno    (si no, el rechazo no existe)

y que los umbrales salen de los datos, no de constantes escritas a mano.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.features import physical  # noqa: E402
from src.model import reject  # noqa: E402
from src.pipeline import novedad_via, predict_via, train_via  # noqa: E402
from src.vision.crop import to_gray_vector  # noqa: E402


@pytest.fixture
def dominio():
    """Dos clases sinteticas separables: circulos grandes rojos y chicos verdes."""
    rng = config.set_seed(config.SEED)
    crops, masks, y = [], [], []
    for etiqueta in (0, 1):
        for _ in range(40):
            img = np.full((128, 128, 3), 128, np.uint8)
            m = np.zeros((128, 128), np.uint8)
            r = 40 if etiqueta == 0 else 20
            color = (40, 40, 220) if etiqueta == 0 else (60, 200, 60)
            cx, cy = rng.integers(50, 78, 2)
            cv2.circle(img, (int(cx), int(cy)), r, color, -1)
            cv2.circle(m, (int(cx), int(cy)), r, 255, -1)
            crops.append(img)
            masks.append(m)
            y.append(etiqueta)

    crops = np.array(crops)
    masks = np.array(masks)
    return {
        "y": np.array(y),
        "A": physical.extract_batch(masks, crops),
        "B": np.stack([to_gray_vector(c) for c in crops]),
    }


@pytest.fixture
def ajeno():
    """Imagenes que no se parecen a nada del dominio."""
    rng = np.random.default_rng(7)
    crops = rng.integers(0, 255, (20, 128, 128, 3), dtype=np.uint8)
    masks = np.full((20, 128, 128), 255, np.uint8)
    return {
        "A": physical.extract_batch(masks, crops),
        "B": np.stack([to_gray_vector(c) for c in crops]),
    }


@pytest.fixture(params=["A", "B"])
def modelo(request, dominio):
    idx = np.arange(len(dominio["y"]))
    return train_via(request.param, dominio[request.param], dominio["y"], idx,
                     k=10, lam=config.LAMBDA_DEFAULT, n_clases=2)


def test_umbrales_salen_del_entrenamiento(modelo):
    """Ningun umbral es una constante del codigo: los dos se ajustan."""
    assert np.isfinite(modelo["umbral_novedad"]) and modelo["umbral_novedad"] > 0
    assert 0.0 < modelo["umbral_confianza"] <= 1.0


def test_no_rechaza_su_propio_dominio(modelo, dominio):
    """El percentil pactado acota el falso rechazo sobre datos del dominio."""
    raw = dominio[modelo["via"]]
    nov = novedad_via(modelo, raw)
    falso_rechazo = (nov > 1.0).mean()
    assert falso_rechazo <= 0.05, "rechaza %.1f%% de su entrenamiento" % (
        100 * falso_rechazo)


def test_rechaza_lo_ajeno(modelo, ajeno):
    """Una muestra que no vive en el dominio tiene que superar el umbral."""
    nov = novedad_via(modelo, ajeno[modelo["via"]])
    assert (nov > 1.0).all(), "novedad minima %.2f" % nov.min()


def test_sin_rechazo_el_argmax_siempre_inventa_una_clase(modelo, ajeno):
    """La razon de existir del rechazo, fijada como test.

    Sin el criterio, predict_via devuelve una de las clases entrenadas para
    cualquier entrada, incluidas las que no tienen nada que ver.
    """
    pred, _ = predict_via(modelo, ajeno[modelo["via"]])
    assert set(np.unique(pred)).issubset({0, 1})
    assert len(pred) == len(ajeno[modelo["via"]])


def test_las_dos_condiciones_son_necesarias():
    """Novedad y confianza rechazan cosas distintas; se exigen ambas."""
    assert reject.es_conocido([0.5], [0.9], 0.2)[0]
    assert not reject.es_conocido([1.5], [0.9], 0.2)[0]
    assert not reject.es_conocido([0.5], [0.1], 0.2)[0]


def test_umbral_respeta_el_percentil():
    valores = np.linspace(0.0, 1.0, 1001)
    assert reject.umbral(valores, 99.0) == pytest.approx(0.99, abs=1e-3)
    assert reject.umbral(valores, 1.0) == pytest.approx(0.01, abs=1e-3)
