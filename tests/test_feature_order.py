"""
El orden del vector de features de la via A es un contrato.

Si el orden cambia entre entrenamiento e inferencia, cada peso aprendido pasa
a multiplicar una magnitud distinta y el clasificador falla en silencio, que
es la peor forma de fallar. Estos tests fijan ese contrato.
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


@pytest.fixture
def objeto():
    mask = np.zeros((128, 128), np.uint8)
    cv2.circle(mask, (64, 64), 30, 255, -1)
    img = np.full((128, 128, 3), 128, np.uint8)
    img[..., 2] = 200
    return mask, img


def test_orden_declarado_es_el_orden_producido(objeto):
    """FEATURE_NAMES describe exactamente lo que devuelve extract_physical."""
    mask, img = objeto
    v = physical.extract_physical(mask, img, use_weight=False)
    nombres = physical.feature_names(use_weight=False)

    assert len(v) == len(nombres)
    assert nombres[:7] == ("area", "perimetro", "aspecto", "circularidad",
                           "R_med", "G_med", "B_med")
    assert nombres[7:14] == tuple("hu%d" % i for i in range(1, 8))
    assert nombres[14:] == ("eje_mayor", "eje_menor")

    # Cada nombre apunta a la magnitud correcta.
    assert v[nombres.index("area")] == float((mask > 0).sum())
    assert abs(v[nombres.index("R_med")] - 200.0) < 1.0
    assert abs(v[nombres.index("B_med")] - 128.0) < 1.0


def test_peso_desactivado_no_entra_al_vector(objeto):
    """Con USE_WEIGHT en False el peso se ignora aunque se pase."""
    mask, img = objeto
    sin = physical.extract_physical(mask, img, use_weight=False)
    con_peso_ignorado = physical.extract_physical(mask, img, peso_g=180.0,
                                                  use_weight=False)
    assert len(sin) == 16
    assert np.array_equal(sin, con_peso_ignorado)
    assert "peso_g" not in physical.feature_names(use_weight=False)


def test_peso_activado_va_al_final(objeto):
    """Con USE_WEIGHT en True el peso es la ULTIMA componente, no otra."""
    mask, img = objeto
    v = physical.extract_physical(mask, img, peso_g=180.0, use_weight=True)
    nombres = physical.feature_names(use_weight=True)

    assert len(v) == 17
    assert nombres[-1] == "peso_g"
    assert v[-1] == 180.0
    # Las 16 anteriores no se mueven al activar el peso.
    assert np.array_equal(v[:-1], physical.extract_physical(mask, img,
                                                            use_weight=False))


def test_peso_activado_sin_medicion_falla_fuerte(objeto):
    """Mejor un error que un vector mas corto que el de entrenamiento."""
    mask, img = objeto
    with pytest.raises(ValueError):
        physical.extract_physical(mask, img, peso_g=None, use_weight=True)


def test_lote_e_individual_coinciden(objeto):
    """extract_batch no reordena nada respecto a la llamada individual."""
    mask, img = objeto
    lote = physical.extract_batch(np.stack([mask, mask]), np.stack([img, img]))
    uno = physical.extract_physical(mask, img)
    assert lote.shape == (2, len(physical.feature_names()))
    assert np.array_equal(lote[0], uno)


def test_config_declara_peso_desactivado():
    """Los datasets publicos no traen masa: la via A corre sin esa componente."""
    assert config.USE_WEIGHT is False
