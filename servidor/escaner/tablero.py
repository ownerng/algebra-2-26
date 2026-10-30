"""Tablero de marcadores ArUco: la pose exacta de la cámara en cada foto.

Se imprime a escala real (180 × 180 mm) y el objeto va en el centro libre.
Tiene 16 marcadores de 30 mm en el perímetro y cada uno aporta sus 4 esquinas,
cuya posición en el mundo se conoce. Los ejes son los de la reconstrucción: X a
la derecha, Y hacia abajo (el tablero es el plano Y = 0 y el objeto queda en
Y < 0) y Z hacia el fondo, alejándose del lado marcado FRENTE.

Con n ≥ 4 correspondencias Xᵢ ↔ (uᵢ, vᵢ), solvePnP halla R y t tales que

    λᵢ·[uᵢ, vᵢ, 1]ᵀ = K·(R·Xᵢ + tᵢ)

minimizando el error de reproyección. Como los puntos son coplanares, se parte
de la homografía del plano (método IPPE) y se refina con Levenberg-Marquardt.
"""

import os

import cv2
import numpy as np

LADO_TABLERO = 180.0
# Si la impresora escala la hoja, mide un marcador con regla y ajusta este valor
LADO_MARCADOR = float(os.environ.get("ESCANER_MARCADOR_MM", 30.0))
POR_LADO = 5   # marcadores por lado (las esquinas se comparten): 16 en total
DICCIONARIO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _posiciones() -> list[tuple[float, float]]:
    """Esquina superior izquierda (x, y) de cada marcador en la hoja, en mm."""
    hueco = (LADO_TABLERO - POR_LADO * LADO_MARCADOR) / (POR_LADO - 1)
    paso = LADO_MARCADOR + hueco
    return [(i * paso, j * paso) for j in range(POR_LADO) for i in range(POR_LADO)
            if i in (0, POR_LADO - 1) or j in (0, POR_LADO - 1)]


def _a_mundo(x: float, y: float) -> list[float]:
    """Hoja (x a la derecha, y hacia el borde FRENTE) → mundo (X, 0, Z), origen en el centro."""
    m = LADO_TABLERO / 2
    return [x - m, 0.0, m - y]


def _tablero() -> cv2.aruco.Board:
    s = LADO_MARCADOR
    esquinas = [np.array([_a_mundo(x, y), _a_mundo(x + s, y), _a_mundo(x + s, y + s), _a_mundo(x, y + s)], np.float32)
                for x, y in _posiciones()]
    return cv2.aruco.Board(esquinas, DICCIONARIO, np.arange(len(esquinas)))


TABLERO = _tablero()
_param = cv2.aruco.DetectorParameters()
_param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
DETECTOR = cv2.aruco.ArucoDetector(DICCIONARIO, _param)


def detectar(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Correspondencias (X: n × 3 en mm, uv: n × 2 en píxeles), o None si se
    ven menos de 2 marcadores (8 esquinas)."""
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    esquinas, ids, _ = DETECTOR.detectMarkers(gris)
    if ids is None or len(ids) < 2:
        return None
    X, uv = TABLERO.matchImagePoints(esquinas, ids)
    return X.reshape(-1, 3).astype(np.float64), uv.reshape(-1, 2).astype(np.float64)


def pose(X: np.ndarray, uv: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(R, t, error medio de reproyección en px) de la cámara que vio X en uv."""
    _, rvec, tvec = cv2.solvePnP(X, uv, K, None, flags=cv2.SOLVEPNP_IPPE)
    rvec, tvec = cv2.solvePnPRefineLM(X, uv, K, None, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    proy, _ = cv2.projectPoints(X, rvec, tvec, K, None)
    error = float(np.linalg.norm(proy.reshape(-1, 2) - uv, axis=1).mean())
    return R, tvec.ravel(), error


def svg() -> str:
    """El tablero para imprimir: SVG en milímetros (imprimir al 100 %, sin ajustar a la página)."""
    celda = LADO_MARCADOR / 6   # 4 × 4 bits más el borde negro
    rects = []
    for k, (x, y) in enumerate(_posiciones()):
        bits = DICCIONARIO.generateImageMarker(k, 6)
        for j, i in zip(*np.nonzero(bits == 0)):
            rects.append(f'<rect x="{x + i * celda:.3f}" y="{y + j * celda:.3f}" width="{celda:.3f}" height="{celda:.3f}"/>')
    m, alto = LADO_TABLERO / 2, LADO_TABLERO + 14
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{LADO_TABLERO:g}mm" height="{alto:g}mm" '
        f'viewBox="0 0 {LADO_TABLERO:g} {alto:g}">'
        f'<rect width="{LADO_TABLERO:g}" height="{alto:g}" fill="#fff"/>'
        f'<g fill="#000" shape-rendering="crispEdges">{"".join(rects)}</g>'
        f'<path d="M{m - 4} {m}H{m + 4}M{m} {m - 4}V{m + 4}" stroke="#999" stroke-width="0.3"/>'
        f'<text x="{m}" y="{LADO_TABLERO + 6}" font-family="sans-serif" font-size="4.5" text-anchor="middle">'
        f'▼ FRENTE ▼ (aquí va la foto frontal)</text>'
        f'<text x="{m}" y="{LADO_TABLERO + 11.5}" font-family="sans-serif" font-size="3" text-anchor="middle" fill="#555">'
        f'Imprime al 100 %: cada marcador debe medir {LADO_MARCADOR:g} mm. Pon el objeto en el centro.</text>'
        '</svg>'
    )


PX_MM = 4   # resolución de la hoja rasterizada para compararla con la foto


def _hoja() -> np.ndarray:
    """La hoja impresa en gris (0 negro, 255 blanco), a PX_MM píxeles por mm."""
    n = round(LADO_TABLERO * PX_MM)
    hoja = np.full((n, n), 255, np.uint8)
    lado = round(LADO_MARCADOR * PX_MM)
    for k, (x, y) in enumerate(_posiciones()):
        bits = cv2.resize(DICCIONARIO.generateImageMarker(k, 6), (lado, lado), interpolation=cv2.INTER_NEAREST)
        i, j = round(x * PX_MM), round(y * PX_MM)
        hoja[j:j + lado, i:i + lado] = bits
    return hoja


HOJA = _hoja()


def quitar(bgr: np.ndarray, mascara: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Quita de la silueta la hoja impresa (la segmentación suele tomarla como
    parte del objeto). Un punto (X, 0, Z) del tablero se ve en

        λ·[u, v, 1]ᵀ = K·(X·r₁ + Z·r₃ + t) = H·[X, Z, 1]ᵀ,    H = K·[r₁ r₃ t],

    así que H (compuesta con la escala de la hoja) dice qué se vería en cada
    píxel si ahí estuviera la hoja. Lo que coincide con eso, en brillo y sin
    color, es hoja. Límite: un objeto blanco o negro sobre la parte de la hoja
    de su mismo color se confunde con ella."""
    alto, ancho = mascara.shape
    m = LADO_TABLERO / 2
    A = np.array([[1 / PX_MM, 0, -m], [0, -1 / PX_MM, m], [0, 0, 1]])   # píxel de la hoja → (X, Z, 1)
    H = K @ np.column_stack([R[:, 0], R[:, 2], t]) @ A
    esperado = cv2.warpPerspective(HOJA, H, (ancho, alto), flags=cv2.INTER_AREA).astype(np.float32) / 255
    esperado = cv2.GaussianBlur(esperado, (0, 0), 1.0)   # la cámara desenfoca un poco
    dentro = cv2.warpPerspective(np.full_like(HOJA, 255), H, (ancho, alto)) > 250
    claro, oscuro = dentro & (esperado > 0.9), dentro & (esperado < 0.1)
    if claro.sum() < 100 or oscuro.sum() < 100:
        return mascara
    gris = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blanco, negro = float(np.median(gris[claro])), float(np.median(gris[oscuro]))
    prevista = negro + (blanco - negro) * esperado
    saturacion = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    es_hoja = dentro & (np.abs(gris - prevista) < 0.3 * abs(blanco - negro)) & (saturacion < 70)
    es_hoja = cv2.morphologyEx(es_hoja.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
    return (mascara.astype(bool) & ~es_hoja).astype(np.uint8)
