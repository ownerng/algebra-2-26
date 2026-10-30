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


def test_profundidad_con_disparidad():
    from escaner.profundidad import estimar

    m = _disco()
    gris = np.full(m.shape, 128, np.uint8)
    v, u = np.mgrid[:m.shape[0], :m.shape[1]]
    centro = (u - 100) ** 2 + (v - 80) ** 2 <= 15 ** 2
    plana = estimar(m, gris, 400, 1.0, disparidad=np.ones(m.shape, np.float32))
    bulto = estimar(m, gris, 400, 1.0, disparidad=np.where(centro, 2.0, 1.0).astype(np.float32))
    # Disparidad plana: solo queda la parte del perfil redondeado (1 − ia)
    sin_ia = estimar(m, gris, 400, 1.0, detalle=0)
    assert plana.altura.max() == pytest.approx(0.3 * sin_ia.altura.max(), rel=0.1)
    # Lo que está más cerca (mayor disparidad) sobresale más
    assert bulto.altura[centro].mean() > 2 * plana.altura[centro].mean()
    assert (bulto.Z_frente[m > 0] <= 400).all()


def test_borde_subpixel_con_alfa():
    # Disco con borde antialias: el contorno exacto está en r = 40,4 px
    r, ancho, alto = 40.4, 200, 160
    v, u = np.mgrid[:alto, :ancho]
    dist = np.hypot(u - 100, v - 80)
    alfa = (np.clip(r - dist + 0.5, 0, 1) * 255).astype(np.uint8)
    mascara = (alfa > 127).astype(np.uint8)
    K = matriz_K(ancho, alto, 60)
    Z = np.full(mascara.shape, 400, np.float32)

    def error_contorno(malla):
        # Vértices de las paredes = contorno; su radio en píxeles a 400 mm
        F = malla.F[-malla.caras["paredes"]:].astype(np.int64)
        P = malla.V[np.unique(F)]
        radio = np.hypot(P[:, 0], P[:, 1]) * K[0, 0] / 400
        return np.abs(radio - r).mean()

    escalones = error_contorno(construir(mascara, Z, Z + 10, K, 400))
    fino = error_contorno(construir(mascara, Z, Z + 10, K, 400, alfa))
    assert fino < 0.15 and fino < escalones / 3


def _foto_con_tablero(azimut, elevacion, radio=30.0, dist=420.0, ancho=640, alto=480, K=None, mira=(0.0, 0.0)):
    """Foto sintética: la hoja impresa en perspectiva y una esfera roja encima.
    La cámara apunta al punto (mira[0], −radio, mira[1]): corrido, el tablero
    sale fuera del centro de la imagen."""
    import cv2
    from escaner import tablero as T

    a, e = np.radians(azimut), np.radians(elevacion)
    C = dist * np.array([np.cos(e) * np.sin(a), -np.sin(e), -np.cos(e) * np.cos(a)])
    z = np.array([mira[0], -radio, mira[1]]) - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, -1.0, 0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    t = -R @ C
    K = matriz_K(ancho, alto, 65) if K is None else K
    # La hoja se dibuja a 3× y se reduce promediando, como integra la luz un píxel real
    # (muestrear la hoja directo a la resolución final corre las esquinas detectadas)
    ss = 3
    S = np.array([[ss, 0, (ss - 1) / 2], [0, ss, (ss - 1) / 2], [0, 0, 1]])   # píxel → centro de su bloque ss × ss
    H = S @ K @ np.column_stack([R[:, 0], R[:, 2], t]) @ T.A_HOJA
    gris = np.full((alto * ss, ancho * ss), 110, np.uint8)
    hoja = cv2.warpPerspective(T.HOJA, H, (ancho * ss, alto * ss))
    sobre = cv2.warpPerspective(np.full_like(T.HOJA, 255), H, (ancho * ss, alto * ss)) > 127
    gris[sobre] = hoja[sobre]
    foto = cv2.cvtColor(cv2.resize(gris, (ancho, alto), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    # La esfera: su contorno proyectado es una elipse; se rellena con sus puntos
    u = np.random.default_rng(0).normal(size=(200_000, 3))
    P = radio * u / np.linalg.norm(u, axis=1, keepdims=True) + [0, -radio, 0]
    q = (P @ R.T + t) @ K.T
    uv = np.round(q[:, :2] / q[:, 2:]).astype(int)
    esfera = np.zeros((alto, ancho), np.uint8)
    esfera[uv[:, 1].clip(0, alto - 1), uv[:, 0].clip(0, ancho - 1)] = 1
    # Se cierra solo la esfera: cerrar toda la foto redondearía las esquinas de los marcadores
    if radio > 0:
        foto[cv2.morphologyEx(esfera, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0] = (40, 50, 210)
    return foto, R, t


def test_tablero_da_la_pose_y_la_escala_real(monkeypatch):
    from escaner import tablero
    from escaner.multivista import preparar_vista, reconstruir
    from escaner.proceso import MARCO

    def sin_rembg(_):
        raise RuntimeError("prueba sin red")

    monkeypatch.setattr(segmentacion, "con_rembg", sin_rembg)
    vistas = {}
    for k, (az, el) in enumerate([(0, 30), (100, 35), (200, 30), (290, 40), (0, 80)]):
        foto, R, t = _foto_con_tablero(az, el)
        X, uv = tablero.detectar(foto)
        R2, t2, error = tablero.pose(X, uv, matriz_K(640, 480, 65))
        assert error < 2 and np.allclose(R2, R, atol=0.01) and np.allclose(t2, t, atol=3)
        v = preparar_vista(f"extra-{k}", foto, MARCO)
        assert v.camara is not None
        vistas[v.nombre] = v
    r = reconstruir(vistas)
    # Esfera de 60 mm: la escala sale del tablero, no de una distancia supuesta
    assert np.allclose(r.tamano_mm, 60, rtol=0.12)
    assert r.volumen_mm3 == pytest.approx(4 / 3 * np.pi * 30**3, rel=0.25)


def test_color_mezcla_las_vistas_sin_costura():
    from escaner.multivista import ROTACIONES, Camara, Vista, colorear

    K = matriz_K(200, 200, 60)
    vistas, cams = {}, {}
    for n, bgr in (("frontal", (0, 0, 255)), ("derecha", (255, 0, 0))):   # roja y azul
        foto = np.zeros((200, 200, 3), np.uint8)
        foto[:] = bgr
        m = np.ones((200, 200), np.uint8)
        vistas[n] = Vista(n, foto, m, m * 255, (0, 0, 199, 199), "prueba")
        cams[n] = Camara(K, ROTACIONES[n], np.array([0.0, 0.0, 400.0]))
    # Tres vértices en el centro: mirando a la frontal (−Z), a 45° y a la derecha (+X)
    V = np.zeros((3, 3))
    N = np.array([[0, 0, -1], [1, 0, -1], [1, 0, 0]], dtype=np.float64)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    c = colorear(V, N, vistas, cams).astype(int)
    assert c[0].tolist() == [255, 0, 0] and c[2].tolist() == [0, 0, 255]
    assert abs(c[1, 0] - c[1, 2]) < 10 and c[1, 0] > 100   # a 45°: mitad y mitad, sin salto


def test_taubin_quita_escalones_sin_encoger():
    from skimage.measure import marching_cubes
    from escaner.malla import taubin

    # Esfera voxelizada (binaria): la superficie sale en escalones
    r = 12.3
    z, y, x = np.mgrid[-16:17, -16:17, -16:17]
    V, F, _, _ = marching_cubes((x * x + y * y + z * z <= r * r).astype(np.float32), 0.5)
    V = V - 16
    S = taubin(V, F)
    radio = lambda P: np.linalg.norm(P, axis=1)  # noqa: E731
    assert radio(S).std() < 0.6 * radio(V).std()        # más redonda (lo que queda es de la voxelización)
    assert abs(volumen(S, F) / volumen(V, F) - 1) < 0.03  # y del mismo tamaño


def test_calibracion_recupera_K(tmp_path, monkeypatch):
    from escaner import calibracion

    monkeypatch.setattr(calibracion, "RUTA", tmp_path / "calibracion.json")
    # Una cámara que no es la supuesta: focal 560 px y centro corrido
    K = np.array([[560.0, 0, 300], [0, 560.0, 250], [0, 0, 1]])
    # Como en una calibración real: inclinado, cerca, lejos y en distintas zonas de la imagen
    tomas = [(0, 35, 380, (0, 0)), (40, 50, 420, (60, 40)), (-50, 40, 400, (-60, 30)), (120, 30, 450, (40, -60)),
             (200, 60, 360, (-50, -50)), (300, 45, 400, (70, 0)), (10, 75, 380, (0, 70)), (80, 55, 420, (-70, -20))]
    for az, el, d, mira in tomas:
        foto, _, _ = _foto_con_tablero(az, el, radio=0.0, dist=d, K=K, mira=mira)
        assert calibracion.agregar(foto) >= calibracion.MIN_MARCADORES
    c = calibracion.calcular()
    try:
        assert c["error_px"] < 0.6
        Kc = np.array(c["K"])
        # La focal (la que fija la escala) queda al 2 %; el centro óptico es lo
        # peor condicionado de una calibración: décimas de píxel lo mueven ~10 px
        assert np.allclose(Kc[[0, 1], [0, 1]], 560, rtol=0.02)
        assert np.allclose(Kc[[0, 1], [2, 2]], [300, 250], atol=15)
        # Escala con la foto: a la mitad de tamaño, la mitad de focal
        assert calibracion.K(320, 240, 65)[0, 0] == pytest.approx(c["K"][0][0] / 2)
        assert calibracion.K(400, 400, 65)[0, 0] == pytest.approx(matriz_K(400, 400, 65)[0, 0])   # otra proporción
    finally:
        calibracion.borrar()
    assert calibracion.actual is None and not (tmp_path / "calibracion.json").exists()
