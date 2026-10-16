"""
Contrato de la via D: fondos aleatorios, calibracion, rechazo y un humo
de extremo a extremo con un dataset sintetico (sin red preentrenada).
"""

import random

import cv2
import numpy as np
import pytest

from src.fruit import data, infer


def _fruta_sobre_blanco(color=(30, 40, 200), lado=100) -> np.ndarray:
    img = np.full((lado, lado, 3), 255, np.uint8)
    cv2.circle(img, (lado // 2, lado // 2), lado // 3, color, -1)
    # un reflejo BLANCO dentro de la fruta: no debe contarse como fondo
    cv2.circle(img, (lado // 2, lado // 2), 5, (255, 255, 255), -1)
    return img


def test_mascara_solo_toma_el_fondo_conectado_al_borde():
    img = _fruta_sobre_blanco()
    m = data.mascara_fondo_blanco(img)
    assert m[0, 0] and m[-1, -1]
    assert not m[50, 50], "el reflejo blanco interior no es fondo"
    assert not m[50, 70], "la fruta no es fondo"


def test_random_background_conserva_la_fruta_y_cambia_el_fondo():
    random.seed(1)
    img = _fruta_sobre_blanco()
    rb = data.RandomBackground(p=1.0)
    cambios = 0
    for _ in range(20):
        out = rb(img)
        assert out.shape == img.shape and out.dtype == np.uint8
        # el centro de la fruta (lejos del borde suavizado) queda intacto
        assert np.abs(out[40:45, 62:66].astype(int) - img[40:45, 62:66]).max() < 3
        cambios += int(np.abs(out[0:5, 0:5].astype(int) - 255).max() > 20)
    assert cambios >= 15, "el fondo casi nunca cambia"


def test_degradar_camara_no_cambia_forma_ni_tipo():
    random.seed(2)
    np.random.seed(2)
    out = data.degradar_camara(_fruta_sobre_blanco())
    assert out.shape == (100, 100, 3) and out.dtype == np.uint8


def test_temperatura_corrige_sobreconfianza():
    rng = np.random.default_rng(0)
    n, c = 4000, 5
    y = rng.integers(0, c, n)
    z = rng.normal(0, 1, (n, c))
    z[np.arange(n), y] += 2.0            # la clase correcta gana a menudo...
    z *= 6.0                              # ...pero con logits inflados (sobreconfiado)
    T = infer.ajustar_temperatura(z, y)
    assert T > 2.0, "una red sobreconfiada debe producir T > 1"

    def nll(T):
        p = infer.softmax(z, T)
        return -np.log(p[np.arange(n), y]).mean()

    assert nll(T) < nll(1.0)


def test_umbrales_vienen_de_los_datos_y_rechazan_lo_incierto():
    rng = np.random.default_rng(1)
    n, c = 2000, 6
    y = rng.integers(0, c, n)
    z = rng.normal(0, 1, (n, c))
    z[np.arange(n), y] += 6.0
    um = infer.fijar_umbrales(z, y, T=1.0)
    plano = np.zeros((1, c))              # logits iguales: ni idea de que es
    p = infer.softmax(plano)
    assert not infer.es_conocido(p.max(1), infer.entropia_normalizada(p), um)[0]
    seguro = np.zeros((1, c))
    seguro[0, 2] = 12.0
    p = infer.softmax(seguro)
    assert infer.es_conocido(p.max(1), infer.entropia_normalizada(p), um)[0]


def test_particion_de_validacion_es_estratificada_y_disjunta():
    y = [0] * 100 + [1] * 50 + [2] * 20
    tr, va = data.partir_validacion(y, 0.1)
    assert set(tr).isdisjoint(va) and len(tr) + len(va) == len(y)
    assert {y[i] for i in va} == {0, 1, 2}


# -- humo de extremo a extremo (necesita torch) ---------------------------

torch = pytest.importorskip("torch")


def _dataset_sintetico(raiz, clases=("Roja", "Verde", "Azul"), n_tr=24, n_te=8):
    colores = {"Roja": (30, 30, 210), "Verde": (40, 190, 50), "Azul": (200, 60, 30)}
    for split, n in (("Training", n_tr), ("Test", n_te)):
        for c in clases:
            d = raiz / split / c
            d.mkdir(parents=True)
            for i in range(n):
                img = _fruta_sobre_blanco(colores[c])
                img = np.clip(img.astype(int) + np.random.randint(-8, 8, img.shape),
                              0, 255).astype(np.uint8)
                cv2.imwrite(str(d / ("%d.jpg" % i)), img)


def test_entrenamiento_guardado_y_carga_extremo_a_extremo(tmp_path):
    import argparse

    from src.fruit import train
    from src.fruit.infer import FrutaNet

    _dataset_sintetico(tmp_path / "fruits")
    salida = tmp_path / "out" / "fruit_model.pt"
    args = argparse.Namespace(
        arch="mobilenet_v3_large", data=str(tmp_path / "fruits"), real=None,
        fondos=None, epocas=3, calentamiento=1, batch=8, lr=1e-3, lado=64,
        workers=0, por_clase=0, salida=str(salida), rapido=False,
        sin_preentrenar=True, sin_agrupar=False, con_verduras=False)
    metricas = train.entrenar(args)

    assert salida.exists()
    assert metricas["n_clases"] == 3
    assert "test_oficial" in metricas and "test_dominio_roto" in metricas
    assert (salida.parent / "metricas.json").exists()

    fn = FrutaNet.cargar(salida)
    assert fn.clases == ["Azul", "Roja", "Verde"]
    r = fn.predecir(_fruta_sobre_blanco((30, 30, 210)))
    assert set(r) >= {"clase", "confianza", "entropia", "conocido", "top"}
    assert 0.0 <= r["confianza"] <= 1.0 and len(r["top"]) == 3
    # el modelo guardado esta congelado y es deterministico
    assert not any(p.requires_grad for p in fn.red.parameters())
    r2 = fn.predecir(_fruta_sobre_blanco((30, 30, 210)))
    assert r2["confianza"] == pytest.approx(r["confianza"])


def test_cuadrar_no_deforma_y_es_cuadrada():
    from src.pipeline import cuadrar
    img = np.random.randint(0, 255, (48, 64, 3), np.uint8)
    out = cuadrar(img)
    assert out.shape == (64, 64, 3)
    assert np.array_equal(out[8:56], img)          # el contenido queda intacto
    assert cuadrar(out) is out
