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


def test_habitacion_endereza_el_piso_y_corta_los_bordes():
    """Cuarto de 5 × 6 m con la cámara a 1.4 m, inclinada 15° hacia abajo y 5°
    de lado, y una caja delante de la pared del fondo."""
    import cv2
    from escaner.habitacion import LADO, alinear, capa_oculta, construir, maqueta, nitida

    ancho, alto = 480, 360
    K = matriz_K(ancho, alto, 65)
    cab, lad = np.radians(-15), np.radians(5)   # cab < 0: mira hacia abajo (y apunta abajo)
    Rx = np.array([[1, 0, 0], [0, np.cos(cab), -np.sin(cab)], [0, np.sin(cab), np.cos(cab)]])
    Rz = np.array([[np.cos(lad), -np.sin(lad), 0], [np.sin(lad), np.cos(lad), 0], [0, 0, 1]])
    R = Rz @ Rx                                    # de la cámara al mundo (y hacia abajo)
    u, v = np.meshgrid(np.arange(ancho), np.arange(alto))
    r = (inversa_K(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])).T   # rayos con z = 1
    w = r @ R.T
    # Paredes del cuarto (eje, posición): la cámara está dentro, gana el corte positivo más cercano
    t = np.full(len(r), np.inf)
    clase = np.zeros(len(r), np.int32)              # etiquetas de ADE20K: pared 0, piso 3, techo 5, sofá 23
    for eje, pos, c in ((0, -2.5, 0), (0, 2.5, 0), (1, 1.4, 3), (1, -1.2, 5), (2, 5.0, 0), (2, -1.0, 0)):
        s = pos / w[:, eje]
        clase = np.where((s > 0) & (s < t), c, clase)
        t = np.where((s > 0) & (s < t), s, t)
    vacio_real = t.reshape(alto, ancho).copy()
    s = 3.0 / w[:, 2]                              # caja: cara de enfrente en z = 3
    P = s[:, None] * w
    caja = (s > 0) & (np.abs(P[:, 0]) < 0.6) & (P[:, 1] > 0.5)
    clase = np.where(caja & (s < t), 23, clase).reshape(alto, ancho)
    t = np.where(caja & (s < t), s, t)
    piso_real = np.isclose((t[:, None] * w)[:, 1], 1.4)
    Z = t.reshape(alto, ancho).astype(np.float32)   # z del rayo es 1: Z = t (m)

    malla, vista, altura = construir(Z, K)
    assert abs(altura - 1400) < 30
    paso = int(np.ceil(ancho / LADO))
    nodos = piso_real.reshape(alto, ancho)[::paso, ::paso].ravel()
    y = malla.V[nodos, 1]
    assert y.std() < 10 and abs(y.mean() + 1400) < 30      # piso plano y horizontal
    filas, cols = np.ceil(alto / paso), np.ceil(ancho / paso)
    cortadas = 2 * (filas - 1) * (cols - 1) - len(malla.F)
    assert 0 < cortadas < 0.05 * 2 * (filas - 1) * (cols - 1)   # solo en el borde de la caja
    assert vista["fov_v"] > 40

    # Capa de atrás: detrás de la caja (a 3 m) está el fondo, más lejos
    Z_suave = cv2.GaussianBlur(Z, (0, 0), 2)    # la red suaviza los saltos así
    Z_nitida = nitida(Z_suave, K[0, 0])
    assert np.abs(Z_nitida - Z)[en_caja_borde := np.abs(Z_suave - Z) > 0.3].mean() < 0.3 * np.abs(Z_suave - Z)[en_caja_borde].mean()
    _, detras, Z_fondo = capa_oculta(Z_nitida, K[0, 0])
    en_caja = caja.reshape(alto, ancho) & (Z < 3.2)
    junto_caja = cv2.dilate(en_caja.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
    assert detras[en_caja].mean() > 0.15 and junto_caja[detras].mean() > 0.9   # franja del borde de la caja
    assert np.median(Z_fondo[detras] / Z[detras]) > 1.2
    con_fondo, _, _ = construir(Z, K, Z_fondo, detras)
    assert len(con_fondo.V) > len(malla.V) and len(con_fondo.uv) == len(con_fondo.V)

    # Maqueta: con ruido de 2 %, el piso vuelve a su plano, y detrás de la caja
    # sale el cuarto vacío
    ruido = Z * (1 + 0.02 * np.random.default_rng(1).standard_normal(Z.shape)).astype(np.float32)
    plano, vacio = maqueta(ruido, K, clase, np.random.default_rng(0))
    piso_px = piso_real.reshape(alto, ancho)
    assert np.abs(plano[piso_px] / Z[piso_px] - 1).mean() < 0.005
    assert np.median(np.abs(vacio[en_caja] / vacio_real[en_caja] - 1)) < 0.02
    # Bordes según los objetos: la franja difuminada vuelve al lado de su clase
    alineada = alinear(Z_suave, clase, K[0, 0])
    assert np.abs(alineada - Z)[en_caja_borde].mean() < 0.3 * np.abs(Z_suave - Z)[en_caja_borde].mean()
    # Dentro de un mueble solo se cortan los saltos grandes: la caja (3 m)
    # contra el piso de atrás (~3,3 m) queda unida; contra la pared (5 m), no
    todo_mueble, _, _ = construir(Z, K, mueble=np.ones_like(Z, bool))
    assert len(malla.F) < len(todo_mueble.F) < 2 * (filas - 1) * (cols - 1)


def test_generativo_prepara_la_foto_y_cuenta_el_glb():
    """Sin red: la imagen que va a fal.ai y la lectura del GLB que vuelve."""
    import base64
    import json

    import cv2
    from escaner.generativo import contar, imagen

    foto = np.zeros((300, 400, 3), np.uint8)
    alfa = np.zeros((300, 400), np.uint8)
    alfa[100:200, 150:220] = 255
    png = cv2.imdecode(np.frombuffer(base64.b64decode(imagen(foto, alfa).split(",")[1]), np.uint8), cv2.IMREAD_COLOR)
    assert png.shape[0] == png.shape[1] == int(100 * 1.15)        # cuadrado con margen
    assert png[0, 0].tolist() == [255, 255, 255] and png[57, 57].tolist() == [0, 0, 0]

    g = json.dumps({"accessors": [{"count": 8}, {"count": 36}], "meshes": [
        {"primitives": [{"attributes": {"POSITION": 0}, "indices": 1}]}]}).encode()
    g += b" " * (-len(g) % 4)
    glb = b"glTF" + (2).to_bytes(4, "little") + (20 + len(g)).to_bytes(4, "little") + len(g).to_bytes(4, "little") + b"JSON" + g
    assert contar(glb) == (8, 12)

    # La caja pasa por la transformación del nodo: +90° en x lleva z a y
    from escaner.generativo import caja
    g = json.dumps({"accessors": [{"count": 8, "min": [-1, -2, -0.3], "max": [1, 2, 0]}],
                    "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                    "nodes": [{"mesh": 0, "rotation": [0.7071068, 0, 0, 0.7071068]}]}).encode()
    g += b" " * (-len(g) % 4)
    glb = b"glTF" + (2).to_bytes(4, "little") + (20 + len(g)).to_bytes(4, "little") + len(g).to_bytes(4, "little") + b"JSON" + g
    cmin, cmax = caja(glb)
    assert np.allclose(cmin, [-1, 0, -2], atol=1e-5) and np.allclose(cmax, [1, 0.3, 2], atol=1e-5)


def test_colocar_mueble_en_la_maqueta():
    """Un GLB de 2 × 1 × 1 (frente hacia +z) va donde se ve el mueble: apoyado en
    el piso, con su alto real y el frente mirando a la cámara (en el origen)."""
    from escaner.habitacion import colocar

    cmin, cmax = np.array([-1.0, -0.5, -0.5]), np.array([1.0, 0.5, 0.5])
    for cx in (0.0, 2000.0):
        y = np.linspace(-1400, -500, 50)
        V = np.stack([np.full(50, cx), y, np.full(50, -3000.0)], axis=1)   # lo que se ve: su frente
        M = colocar(V, -1400.0, cmin, cmax)
        frente = M @ [0, -0.5, 0.5, 1]
        assert np.allclose(frente[:3], [cx, -1400, -3000], atol=1)          # frente abajo, sobre lo visto
        assert abs((M @ [0, 0.5, 0, 1])[1] + 500) < 20                       # alto real: 900 mm
        d = M[:3, :3] @ [0, 0, 1]
        assert np.dot(d / np.linalg.norm(d), -np.array([cx, 0, -3000]) / np.hypot(cx, 3000)) > 0.999   # mira a la cámara
