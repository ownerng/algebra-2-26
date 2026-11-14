"""Una habitación completa desde una sola foto:

  1. Profundidad métrica → Z(u, v) en metros con Depth Anything V2 Metric Indoor
     (una red entrenada con interiores; profundidad.metros), y la clase de cada
     píxel (pared, piso, sofá, mesa…) con SegFormer (profundidad.etiquetas).
  2. Maqueta → piso, techo y paredes se ajustan a planos (RANSAC + SVD) y su Z
     se reemplaza por la del plano: un rayo r = K⁻¹·[u, v, 1]ᵀ corta n·P = d en
     Z = d / (n·r). Con los planos sale también el cuarto vacío: dentro de un
     cuarto convexo, lo que hay detrás de un mueble es el corte más cercano.
  3. Bordes según los objetos → la red de profundidad difumina los saltos; en
     esas franjas cada píxel toma la Z de su propio objeto (el de su clase).
  4. Retroproyección → P = Z·K⁻¹·[u, v, 1]ᵀ sobre una rejilla de toda la foto,
     y cada celda se parte en dos triángulos con la foto como textura.
  5. Cortes → donde un mueble tapa la pared, Z salta (la red lo suaviza en
     varios píxeles); los triángulos de ese salto serían una "cortina" estirada
     que la cámara ve casi de canto. Se quitan si el rayo r y la normal
     n = (p₁ − p₀) × (p₂ − p₀) son casi perpendiculares (|n̂·r̂| < sen 5°) y
     además Z salta (Z_max / Z_min > 1 + CORTE): el costado curvo de un sofá
     se ve de canto pero no salta, y queda. Dentro de los muebles tampoco se
     corta (salvo saltos de más de 1 + GRANDE, una silla delante del sofá): el
     cojín y el asiento quedan unidos, estirando sus propios píxeles, en lugar
     de abrir un agujero que deja ver el cuarto vacío.
  6. Capa de atrás → al mover la vista se vería lo que la foto no tiene (detrás
     de los muebles). Ahí se arma una segunda malla con la Z del cuarto vacío
     (o, sin planos, el promedio de los píxeles de fondo alrededor) y la
     textura que inventa LaMa (una red de inpainting) al borrar los muebles.
  7. Piso horizontal → el celular casi nunca está derecho. RANSAC busca el plano
     n·P = d con más puntos en la parte baja de la foto, se ajusta por mínimos
     cuadrados (SVD: n es el vector singular del menor valor singular de los
     puntos centrados) y la rotación de Rodrigues R lleva n a la vertical:

         R = I + [v]ₓ + [v]ₓ² / (1 + c),   v = n × m,  c = n·m   (R·n = m)

Procesamiento alto: SAM 3 separa cada mueble (una máscara por sofá, mesa…),
Hunyuan3D lo genera completo en fal.ai (también por detrás) y se coloca en la
maqueta con una matriz 4 × 4: escala por su altura real, giro en y para que su
frente mire a la cámara y apoyado en el piso. El mueble plano de la foto se quita.
"""

import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import calibracion
from . import generativo as gen
from . import profundidad as prof_mod
from .geometria import inversa_K, retroproyectar
from .malla import R_OBJETO, Malla
from .proceso import Parametros, Resultado, SinObjeto, reducir

LADO = 480           # nodos de la rejilla a lo largo del lado mayor de la foto
CANTO = 5.0          # grados: un triángulo visto con menos inclinación que esta es un salto
CORTE = 0.12         # cuánto más lejos (relativo) tiene que estar el fondo para armar la capa de atrás
GRANDE = 0.5        # entre dos muebles solo se corta si Z salta más que esto
ALCANCE = 0.06       # hasta dónde (fracción del ancho) se rellena detrás de cada borde
ARRIBA = np.array([0.0, -1.0, 0.0])   # la vertical en ejes de la cámara (y apunta hacia abajo)
# Clases de ADE20K que son el cuarto: piso; techo; pared, ventana, puerta, espejo (hasta 4 paredes)
ESTRUCTURA = (((3,), 1), ((5,), 1), ((0, 8, 14, 27), 4))
# Lo que se le pide a SAM 3 en el procesamiento alto (US$0,005 y ~3 s cada uno, en paralelo)
TEXTOS = ("sofa", "chair", "table", "bed", "cabinet")
MUEBLES_MAX = 5
CARAS_MUEBLE = 150_000   # menos que las 500 000 por defecto: varios muebles juntos en el navegador


def cruz(v: np.ndarray) -> np.ndarray:
    """[v]ₓ: la matriz con [v]ₓ·w = v × w."""
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def girar_a(n: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Rotación de Rodrigues que lleva el vector unitario n al m (no opuestos)."""
    v, c = np.cross(n, m), float(n @ m)
    return np.eye(3) + cruz(v) + cruz(v) @ cruz(v) / (1 + c)


def ransac(P: np.ndarray, rng: np.random.Generator, valido=None, minimo: float = 0.2,
           intentos: int = 300) -> tuple[np.ndarray, float, np.ndarray] | None:
    """El plano n·P = d con más puntos a menos de un 2 % de la distancia
    (RANSAC: planos por tríos al azar, n = (b − a) × (c − a)), ajustado por
    mínimos cuadrados (SVD), y qué puntos están en él. None si ninguno junta
    `minimo` de los puntos. `valido(n, d)` descarta planos (n apunta hacia arriba)."""
    tol = 0.02 * float(np.median(P[:, 2]))
    muestra = P[rng.choice(len(P), min(len(P), 4000), replace=False)]
    i = rng.integers(0, len(P), (intentos, 3))
    n = np.cross(P[i[:, 1]] - P[i[:, 0]], P[i[:, 2]] - P[i[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    n *= np.where(n @ ARRIBA < 0, -1.0, 1.0)[:, None]
    d = (n * P[i[:, 0]]).sum(axis=1)
    if valido is not None:
        ok = valido(n, d)
        n, d = n[ok], d[ok]
    if not len(n):
        return None
    votos = (np.abs(muestra @ n.T - d) < tol).sum(axis=0)
    k = int(np.argmax(votos))
    if votos[k] < minimo * len(muestra):
        return None
    dentro = np.abs(P @ n[k] - d[k]) < tol
    c = P[dentro].mean(axis=0)
    n = np.linalg.svd(P[dentro] - c, full_matrices=False)[2][2]
    n *= -1.0 if n @ ARRIBA < 0 else 1.0
    return n, float(n @ c), np.abs(P @ n - n @ c) < tol


def piso(P: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float] | None:
    """Plano del piso (n hacia arriba, n·P = d), o None: a menos de 40° de la
    horizontal y debajo de la cámara."""
    plano = ransac(P, rng, lambda n, d: (n @ ARRIBA > np.cos(np.radians(40))) & (d < 0))
    return None if plano is None else plano[:2]


def maqueta(Z: np.ndarray, K: np.ndarray, L: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(Z con piso, techo y paredes planos, Z del cuarto vacío).

    Por cada clase de ESTRUCTURA se buscan planos uno tras otro (RANSAC sobre
    los píxeles que quedan). Cada rayo r = K⁻¹·[u, v, 1]ᵀ (r_z = 1) corta el
    plano n·P = d en Z = d / (n·r); los píxeles de la clase a menos de un 10 %
    de ese valor pasan a estar exactamente en el plano. El cuarto vacío es, en
    cada píxel, el corte positivo más cercano con todos los planos (∞ si no hay)."""
    alto, ancho = Z.shape
    u, v = np.meshgrid(np.arange(ancho), np.arange(alto))
    r = (inversa_K(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])).T
    z = Z.ravel().astype(np.float64)
    P = r * z[:, None]
    plano_z, vacio = z.copy(), np.full(z.size, np.inf)
    for clases, veces in ESTRUCTURA:
        resto = np.flatnonzero(np.isin(L.ravel(), clases))
        for _ in range(veces):
            if len(resto) < 0.01 * z.size:
                break
            sub = resto[rng.choice(len(resto), min(len(resto), 20000), replace=False)]
            plano = ransac(P[sub], rng, minimo=0.15)
            if plano is None:
                break
            n, d, _ = plano
            with np.errstate(divide="ignore", invalid="ignore"):
                t = d / (r @ n)
            vacio = np.where(t > 0.1, np.minimum(vacio, t), vacio)
            cerca = (t[resto] > 0) & (np.abs(np.log(np.maximum(t[resto], 1e-6) / z[resto])) < 0.1)
            plano_z[resto[cerca]] = t[resto[cerca]]
            resto = resto[~cerca]
    return plano_z.reshape(alto, ancho).astype(np.float32), vacio.reshape(alto, ancho)


def de_canto(Z: np.ndarray, f: float) -> np.ndarray:
    """Píxeles donde la superficie se ve casi de canto: la pendiente de Z por
    píxel comparada con lo que mide un píxel a esa distancia (Z / f):

        tan(ángulo entre superficie y rayo) ≈ (Z / f) / |∇Z|  =  1 / (f·|∇ log Z|)

    Un piso o una pared nunca se ven así de inclinados; el salto entre un
    mueble y lo que está detrás (que la red suaviza en varios píxeles), sí."""
    L = np.log(Z)
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3) / 8
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3) / 8
    return f * np.hypot(gx, gy) > 1 / np.tan(np.radians(CANTO))


def nitida(Z: np.ndarray, f: float) -> np.ndarray:
    """La red difumina cada salto en unos 8 píxeles. En esas franjas (vistas de
    canto), cada píxel toma el mínimo o el máximo de Z de su ventana de 11 × 11,
    el más cercano en escala logarítmica: el salto queda en un solo paso, a
    mitad de camino, y el corte sigue el borde del mueble."""
    franja = cv2.dilate(de_canto(Z, f).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    k = np.ones((11, 11), np.uint8)
    Zmin, Zmax = cv2.erode(Z, k), cv2.dilate(Z, k)
    cerca = np.log(Z / Zmin) < np.log(Zmax / Z)
    return np.where(franja, np.where(cerca, Zmin, Zmax), Z).astype(np.float32)


def alinear(Z: np.ndarray, L: np.ndarray, f: float) -> np.ndarray:
    """Bordes según los objetos: en la franja de cada salto, cada píxel toma la
    Z de su propio objeto, el promedio gaussiano de los píxeles de su misma
    clase fuera de la franja (convolución normalizada). Así el corte sigue el
    contorno que ve SegFormer. Sin datos de su clase cerca, queda `nitida`."""
    franja = cv2.dilate(de_canto(Z, f).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    salida = nitida(Z, f)
    for c in np.unique(L[franja]):
        fuente = ((L == c) & ~franja).astype(np.float32)
        num = cv2.GaussianBlur(Z * fuente, (0, 0), 4)
        den = cv2.GaussianBlur(fuente, (0, 0), 4)
        sel = franja & (L == c) & (den > 0.05)
        salida[sel] = num[sel] / den[sel]
    return salida


def capa_oculta(Z: np.ndarray, f: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(máscara para LaMa, píxeles con capa de atrás, Z del fondo).

    Lo que se puede destapar al mover la vista es el propio salto y lo que está
    a menos de ALCANCE de él y más cerca que el fondo del salto. Ahí la Z del
    fondo es el promedio gaussiano de los píxeles de fondo (convolución
    normalizada: Σ w·Z·fondo / Σ w·fondo)."""
    alto, ancho = Z.shape
    W = max(2, int(ALCANCE * ancho))
    disco = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * W + 1, 2 * W + 1))
    salto = de_canto(Z, f)
    salto = cv2.dilate(salto.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    fondo_salto = cv2.dilate(np.where(salto, Z, 0).astype(np.float32), disco)
    oculto = ((cv2.dilate(salto.astype(np.uint8), disco) > 0) & (Z * (1 + CORTE) < fondo_salto)) | salto

    fondo = (~oculto).astype(np.float32)
    num = cv2.GaussianBlur(Z * fondo, (0, 0), W / 2)
    den = cv2.GaussianBlur(fondo, (0, 0), W / 2)
    Z_fondo = np.where(oculto, np.where(den > 0.02, num / np.maximum(den, 1e-6), fondo_salto), Z)
    Z_fondo = np.maximum(Z_fondo, Z).astype(np.float32)   # el fondo nunca queda delante
    # Donde el fondo queda a la misma Z (el piso delante de un mueble) la capa
    # de atrás sobraría y pelearía con la de adelante
    return oculto, (oculto & (Z_fondo > Z * (1 + CORTE / 2))) | salto, Z_fondo


def rejilla(Z: np.ndarray, K: np.ndarray, paso: int, usar: np.ndarray | None = None, mueble: np.ndarray | None = None):
    """Nodos de la rejilla (u, v), sus puntos P (mm) y los triángulos que no
    son un salto: vistos de canto (|n̂·r̂| < sen CANTO, n̂: normal del triángulo,
    r̂: rayo a su centro) y con Z_max / Z_min > 1 + CORTE. Con `mueble`, un
    triángulo con los tres nodos en muebles solo se corta si salta más de 1 + GRANDE.
    Con `usar`, solo los nodos donde es verdadera (y las celdas con los tres)."""
    alto, ancho = Z.shape
    U, Vg = np.meshgrid(np.arange(0, ancho, paso), np.arange(0, alto, paso))
    dentro = np.ones(U.shape, bool) if usar is None else usar[Vg, U]
    indice = np.full(U.shape, -1, np.int64)
    indice[dentro] = np.arange(int(dentro.sum()))
    u, v = U[dentro], Vg[dentro]
    P = retroproyectar(inversa_K(K), u, v, Z[v, u] * 1000)

    a, b = indice[:-1, :-1].ravel(), indice[:-1, 1:].ravel()
    d, e = indice[1:, :-1].ravel(), indice[1:, 1:].ravel()
    F = np.concatenate([np.stack([a, d, b], 1), np.stack([b, d, e], 1)])
    F = F[(F >= 0).all(axis=1)]
    t = P[F]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    r = t.sum(axis=1)
    coseno = np.abs((n * r).sum(axis=1)) / np.maximum(np.linalg.norm(n, axis=1) * np.linalg.norm(r, axis=1), 1e-12)
    razon = t[:, :, 2].max(axis=1) / t[:, :, 2].min(axis=1)
    salto = (coseno < np.sin(np.radians(CANTO))) & (razon > 1 + CORTE)
    if mueble is not None:
        en_mueble = mueble[v, u][F].all(axis=1)
        salto &= ~en_mueble | (razon > 1 + GRANDE)
    return u, v, P, F[~salto]


def construir(Z: np.ndarray, K: np.ndarray, Z_fondo: np.ndarray | None = None, detras: np.ndarray | None = None,
              mueble: np.ndarray | None = None, quitar: np.ndarray | None = None,
              semilla: int = 0) -> tuple[Malla, dict, float | None]:
    """Malla de la habitación (mm, ejes x derecha, y arriba, z hacia la cámara),
    la vista de la foto para el visor y la altura de la cámara sobre el piso (mm).
    Con capa de atrás, la textura es la foto arriba y el relleno abajo. Los
    píxeles de `quitar` (muebles que se reemplazan) no van en la capa de adelante."""
    alto, ancho = Z.shape
    paso = max(1, int(np.ceil(max(ancho, alto) / LADO)))
    u, v, P, F = rejilla(Z, K, paso, mueble=mueble)
    if quitar is not None:
        F = F[~quitar[v, u][F].any(axis=1)]
    uv = np.stack([(u + 0.5) / ancho, 1 - (v + 0.5) / alto], axis=1)
    n_frente = len(P)
    if Z_fondo is not None:
        # Un nodo más alrededor, para que la capa de atrás no deje rendijas
        usar = cv2.dilate(detras.astype(np.uint8), np.ones((2 * paso + 1, 2 * paso + 1), np.uint8)) > 0
        u2, v2, P2, F2 = rejilla(Z_fondo, K, paso, usar)
        uv2 = np.stack([(u2 + 0.5) / ancho, 1 - (v2 + 0.5) / alto], axis=1)
        uv = np.vstack([uv * [1, 0.5] + [0, 0.5], uv2 * [1, 0.5]])
        P, F = np.vstack([P, P2]), np.vstack([F, F2 + n_frente])

    # Piso: entre los puntos del tercio de abajo de la foto que no son muebles (solo la capa de adelante)
    R, altura = np.eye(3), None
    abajo = v > 0.65 * alto
    if mueble is not None:   # sin muebles: una mesa de centro también es un plano horizontal
        libre = abajo & ~mueble[v, u]
        abajo = libre if libre.sum() > 0.05 * abajo.sum() else abajo
    plano = piso(P[:n_frente][abajo], np.random.default_rng(semilla))
    if plano is not None:
        n, dist = plano
        R, altura = girar_a(n, ARRIBA), -dist
    G = R_OBJETO @ R
    V = P @ G.T

    eje = G @ np.array([0.0, 0.0, 1.0])
    vista = {
        "origen": [0.0, 0.0, 0.0],
        "mira": (eje * 1000).round(1).tolist(),   # se gira alrededor de un punto a 1 m: casi sin moverse del sitio
        "fov_v": round(float(np.degrees(2 * np.arctan(alto / 2 / K[1, 1]))), 2),
        "ejes": G.tolist(),   # P (cámara) → V = G·P (maqueta)
    }
    malla = Malla(
        V=V.astype(np.float32), F=F.astype(np.uint32), uv=uv.astype(np.float32), colores=None,
        n_frente=n_frente, caras={"frente": len(F), "atras": 0, "paredes": 0},
        centro=np.zeros(3),
    )
    return malla, vista, altura


def colocar(V: np.ndarray, y_piso: float, cmin: np.ndarray, cmax: np.ndarray) -> np.ndarray:
    """Matriz 4 × 4 que lleva un mueble generado (ejes del GLB: y arriba, frente
    hacia +z, caja [cmin, cmax]) a su lugar en la maqueta, dados los puntos V que
    se ven de él en la foto (la cámara está en el origen):

        M = T(p)·R_y(θ)·S(s)·T(−base),   s = alto real / alto del GLB,
        R_y(θ)·e_z = d (d: horizontal, del mueble hacia la cámara),

    con base = el centro de abajo de su caja y p = ese punto en la maqueta: en
    el piso, detrás de lo que se ve la mitad de su fondo."""
    tam = cmax - cmin
    s = (np.percentile(V[:, 1], 98) - y_piso) / max(float(tam[1]), 1e-9)
    c = np.median(V, axis=0)
    d = -np.array([c[0], 0.0, c[2]])
    d /= max(float(np.linalg.norm(d)), 1e-9)
    R = np.array([[d[2], 0, d[0]], [0, 1, 0], [-d[0], 0, d[2]]])   # θ = atan2(d_x, d_z): R·e_z = d
    base = np.array([(cmin[0] + cmax[0]) / 2, cmin[1], (cmin[2] + cmax[2]) / 2])
    p = np.array([c[0], y_piso, c[2]]) - d * (s * tam[2] / 2)
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = p - s * R @ base
    return M


def _muebles(bgr: np.ndarray, foto: np.ndarray) -> list[tuple[np.ndarray, object]]:
    """Máscaras de los muebles más grandes (SAM 3, sin repetidos) y, para cada
    una, la generación en 3D ya pedida a fal.ai (un Future con el GLB)."""
    with ThreadPoolExecutor(len(TEXTOS)) as hilos:
        encontradas = [m for lista in hilos.map(lambda t: gen.instancias(foto, t), TEXTOS) for m in lista]
    mascaras: list[np.ndarray] = []
    for m in sorted(encontradas, key=lambda m: -m.sum()):
        if m.mean() > 0.01 and all((m & o).sum() < 0.5 * min(m.sum(), o.sum()) for o in mascaras):
            mascaras.append(m)
    mascaras = mascaras[:MUEBLES_MAX]
    hilos = ThreadPoolExecutor(max(1, len(mascaras)))
    alto, ancho = bgr.shape[:2]

    def pedir(m: np.ndarray) -> bytes:
        alfa = cv2.resize((m * 255).astype(np.uint8), (ancho, alto), interpolation=cv2.INTER_LINEAR)
        return gen.generar(gen.imagen(bgr, alfa), caras=CARAS_MUEBLE)

    return [(m, hilos.submit(pedir, m)) for m in mascaras]


def reconstruir(bgr: np.ndarray, p: Parametros | None = None, alto_proceso: bool = False) -> Resultado:
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
    Z = prof_mod.metros(foto)
    if Z is None:
        raise SinObjeto("falta la red de profundidad para interiores; reinicia el servidor con internet para descargarla")
    marca("profundidad")

    K = calibracion.K(ancho, alto, p.fov)
    f = K[0, 0]
    L = prof_mod.etiquetas(foto)
    if L is None:
        Z, Z_vacio = nitida(Z, f), None
    else:
        Z, Z_vacio = maqueta(Z, K, L, np.random.default_rng(0))
        Z = alinear(Z, L, f)
    marca("objetos")

    # Procesamiento alto: los muebles se piden a fal.ai ahora y se generan
    # mientras se arma la maqueta
    piezas, aviso = [], ""
    if alto_proceso and L is not None:
        try:
            piezas = _muebles(bgr, foto)
        except Exception as e:   # sin clave o sin red: queda la maqueta del modo bajo
            aviso = f" · sin muebles 3D ({e})"

    para_lama, detras, Z_fondo = capa_oculta(Z, f)
    mueble = None if L is None else ~np.isin(L, [c for clases, _ in ESTRUCTURA for c in clases])
    if Z_vacio is not None:
        # Detrás de cada mueble, el cuarto vacío (si sus planos quedan detrás)
        hueco = mueble & np.isfinite(Z_vacio) & (Z_vacio > Z * (1 + CORTE / 2))
        Z_fondo = np.where(hueco, Z_vacio, Z_fondo).astype(np.float32)
        detras |= hueco
        para_lama |= mueble
    # Donde va un mueble generado, la capa de atrás cubre toda su silueta (con un
    # margen) y LaMa la borra entera: sin sombras del mueble de la foto
    zona = None
    if piezas:
        zona = cv2.dilate(np.any([m for m, _ in piezas], axis=0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        detras |= zona
        para_lama |= zona
        if Z_vacio is not None:
            Z_fondo = np.where(zona & np.isfinite(Z_vacio) & (Z_vacio > Z), Z_vacio, Z_fondo).astype(np.float32)
    relleno = prof_mod.rellenar(foto, para_lama) if detras.any() else None
    if relleno is not None:
        # Por la rendija de cada corte se ve la capa de atrás: ahí lleva la foto,
        # y desde el punto de la foto no se nota el corte
        k = np.ones((3, 3), np.uint8)
        rendija = cv2.dilate(Z, k) > cv2.erode(Z, k) * (1 + CORTE)
        rendija = cv2.dilate(rendija.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        if zona is not None:
            rendija &= ~zona
        relleno[rendija] = foto[rendija]
    marca("relleno")

    muebles = []
    quitar = None
    if piezas:
        listas = []
        for mascara, futuro in piezas:
            try:
                listas.append((mascara, futuro.result()))
            except Exception as e:
                print(f"[habitacion] un mueble no se generó: {e}")
        if listas:
            quitar = cv2.dilate(np.any([m for m, _ in listas], axis=0).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        piezas = listas
        marca("muebles 3D")

    if relleno is None:
        m, vista, altura = construir(Z, K, quitar=quitar)
        textura = foto
    else:
        m, vista, altura = construir(Z, K, Z_fondo, detras, mueble, quitar)
        textura = np.vstack([foto, relleno])
    marca("malla")

    if piezas:
        G = np.array(vista["ejes"])
        K_inv = inversa_K(K)
        for mascara, glb in piezas:
            vs, us = np.nonzero(mascara)
            V = retroproyectar(K_inv, us, vs, Z[vs, us] * 1000) @ G.T
            y_piso = -altura if altura else float(np.percentile(V[:, 1], 2))
            M = colocar(V, y_piso, *gen.caja(glb))
            muebles.append((glb, M.T.ravel().round(5).tolist()))   # por columnas, como Matrix4 de three.js

    metodo = "Depth Anything V2 Metric Indoor" + (" · SegFormer" if L is not None else "")
    if relleno is not None:
        metodo += " · relleno LaMa"
    metodo += f" · cámara a {altura / 1000:.2f} m del piso" if altura else " · sin piso a la vista"
    if muebles:
        metodo += f" · {len(muebles)} muebles con SAM 3 + Hunyuan3D"
    metodo += aviso
    if calibracion.aplica(ancho, alto):
        metodo += " · K calibrada"
    # El alfa marca solo la foto (arriba de la textura): la miniatura no muestra el relleno
    alfa = np.zeros(textura.shape[:2], np.uint8)
    alfa[:alto] = 255
    tam = m.V[:m.n_frente].max(axis=0) - m.V[:m.n_frente].min(axis=0)
    return Resultado(
        foto=textura, mascara=(alfa > 0).astype(np.uint8), alfa=alfa, K=K, malla=m,
        metodo=metodo, tamano_mm=(float(tam[0]), float(tam[1]), float(tam[2])), volumen_mm3=0.0,
        tiempos=tiempos, vista=vista, muebles=muebles or None,
    )
