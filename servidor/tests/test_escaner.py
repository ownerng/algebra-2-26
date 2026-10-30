import numpy as np
import pytest

from escaner import segmentacion
from escaner.geometria import inversa_K, matriz_K, proyectar, retroproyectar
from escaner.malla import construir, volumen
from escaner.proceso import escanear


def test_inversa_de_K():
    K = matriz_K(640, 480, 65)
    assert np.allclose(K @ inversa_K(K), np.eye(3))
    assert np.allclose(inversa_K(K), np.linalg.inv(K))


def test_retroproyectar_y_proyectar_son_inversas():
    K = matriz_K(640, 480, 65)
    u, v, Z = np.array([0.0, 320, 600]), np.array([10.0, 240, 470]), np.array([300.0, 400, 900])
    P = retroproyectar(inversa_K(K), u, v, Z)
    assert P.shape == (3, 3)
    assert np.allclose(P[:, 2], Z)
    assert np.allclose(proyectar(K, P), np.stack([u, v], 1))


def _disco(ancho=200, alto=160, r=50):
    v, u = np.mgrid[:alto, :ancho]
    return ((u - ancho / 2) ** 2 + (v - alto / 2) ** 2 <= r * r).astype(np.uint8)


def test_malla_cerrada_y_con_volumen_positivo():
    m = _disco()
    K = matriz_K(m.shape[1], m.shape[0], 60)
    frente = np.full(m.shape, 390, np.float32)
    atras = np.full(m.shape, 410, np.float32)
    malla = construir(m, frente, atras, K, 400)
    assert malla.V.shape[1] == 3 and malla.F.shape[1] == 3
    assert malla.F.max() < len(malla.V)
    # Sólido cerrado: cada arista la comparten exactamente dos caras, en sentidos opuestos
    F = malla.F.astype(np.int64)
    dirigidas = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    n = len(malla.V)
    claves = dirigidas[:, 0] * n + dirigidas[:, 1]
    assert len(np.unique(claves)) == len(claves)
    opuestas = dirigidas[:, 1] * n + dirigidas[:, 0]
    assert np.isin(opuestas, claves).all()
    # Un disco de 20 mm de grosor: volumen ≈ π·r²·20 (r en mm a 400 mm)
    r_mm = 50 * 400 / K[0, 0]
    assert volumen(malla.V, malla.F) == pytest.approx(np.pi * r_mm**2 * 20, rel=0.08)
    # La cara trasera no se sale de la silueta (x, y del contorno a 400 mm)
    atras = malla.V[malla.n_frente:]
    assert np.abs(atras[:, 0]).max() <= r_mm * 1.02


def test_escanear_con_fondo_texturizado(monkeypatch):
    # Fondo con ruido de colores (sin cartulina) y un objeto rojo en el centro
    rng = np.random.default_rng(0)
    foto = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    foto = (foto // 2 + 60).astype(np.uint8)
    v, u = np.mgrid[:240, :320]
    objeto = (u - 160) ** 2 / 60**2 + (v - 120) ** 2 / 45**2 <= 1
    foto[objeto] = (40, 40, 220)

    def sin_rembg(_):
        raise RuntimeError("prueba sin red")

    monkeypatch.setattr(segmentacion, "con_rembg", sin_rembg)
    r = escanear(foto)
    assert r.metodo.startswith("GrabCut")
    acierto = (r.mascara.astype(bool) == objeto).mean()
    assert acierto > 0.97
    assert r.volumen_mm3 > 0
    ancho, alto, prof = r.tamano_mm
    assert ancho > alto > 0 and prof > 0


def _silueta(R, puntos, ancho=480, alto=360, distancia=400.0):
    """Proyecta una nube de puntos densa con una cámara a `distancia` del centro."""
    import cv2
    from escaner.multivista import Camara

    cam = Camara(matriz_K(ancho, alto, 65), R, np.array([0.0, 0.0, distancia]))
    uv, _ = cam.proyectar(puntos)
    m = np.zeros((alto, ancho), np.uint8)
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    m[v.clip(0, alto - 1), u.clip(0, ancho - 1)] = 1
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def test_envolvente_visual_de_una_caja():
    from escaner.multivista import ROTACIONES, VISTAS, Vista, reconstruir

    medidas = np.array([100.0, 60.0, 40.0])   # ancho (X), alto (Y), fondo (Z) en mm
    rng = np.random.default_rng(1)
    puntos = (rng.random((200_000, 3)) - 0.5) * medidas
    vistas = {}
    for n in VISTAS:
        m = _silueta(ROTACIONES[n], puntos)
        vs, us = np.nonzero(m)
        foto = np.full((*m.shape, 3), 128, np.uint8)
        vistas[n] = Vista(n, foto, m, m * 255, (us.min(), vs.min(), us.max(), vs.max()), "prueba")
    r = reconstruir(vistas)
    # Ejes del objeto: x = X, y = −Y, z = −Z → las medidas no cambian. La escala
    # absoluta depende de la distancia supuesta Z₀; las proporciones no.
    tam = np.array(r.tamano_mm)
    assert np.allclose(tam / tam[1], medidas / medidas[1], rtol=0.05)
    assert np.allclose(tam, medidas, rtol=0.15)
    assert r.volumen_mm3 == pytest.approx(medidas.prod(), rel=0.3)
    assert r.malla.colores.shape == (len(r.malla.V), 3)
