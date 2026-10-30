"""Reconstrucción con varias fotos: frontal, lateral izquierda, trasera, lateral
derecha y superior, por talla de voxeles (envolvente visual).

Ejes del mundo (los de la cámara frontal): X a la derecha, Y hacia abajo, Z
alejándose de la cámara frontal; el origen es el centro del objeto. Cada vista
es una cámara con matriz de proyección

    P_v = K_v·[R_v | t_v],      λ·[u, v, 1]ᵀ = K_v·(R_v·X + t_v),

donde las filas de R_v son los ejes de esa cámara escritos en el mundo:

    frontal    R = I
    izquierda  la cámara está en −X y mira hacia +X   (su derecha es −Z)
    derecha    la cámara está en +X y mira hacia −X   (su derecha es +Z)
    trasera    la cámara está en +Z y mira hacia −Z   (su derecha es −X)
    superior   la cámara está arriba (−Y) y mira hacia abajo; la parte de
               arriba de la foto apunta hacia el fondo (+Z)

Las fotos se toman a pulso, así que no se conoce la distancia de cada una. Se
fija con la frontal (Z₀) y las demás se escalan para que coincidan las medidas
compartidas: las laterales comparten la altura, la superior el ancho y la
trasera ambos. El
centro del objeto está sobre el rayo del centro de su silueta:
t_v = d_v·K⁻¹·[u_c, v_c, 1]ᵀ.

Un voxel X pertenece al objeto si su proyección cae dentro de la silueta en
todas las vistas. La superficie se extrae con marching cubes y cada vértice se
colorea con la foto que lo ve más de frente (mayor n̂·ĉ).
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.measure import marching_cubes

from .geometria import inversa_K, matriz_K
from .malla import Malla, R_OBJETO, volumen
from .proceso import Parametros, Resultado, SinObjeto, reducir
from .segmentacion import segmentar

# En el orden en que se toman: una vuelta alrededor del objeto y al final desde arriba
VISTAS = ("frontal", "izquierda", "trasera", "derecha", "superior")

ROTACIONES = {
    "frontal": np.eye(3),
    "izquierda": np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),
    "derecha": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),
    "trasera": np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64),
    "superior": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),
}

LADO_VOXELES = 120   # voxeles a lo largo del lado mayor de la caja
MARGEN = 1.3         # la caja de voxeles es un 30 % más grande que el objeto


@dataclass
class Vista:
    nombre: str
    foto: np.ndarray       # BGR reducida
    mascara: np.ndarray    # 0/1
    alfa: np.ndarray
    caja: tuple[int, int, int, int]  # u0, v0, u1, v1 de la silueta
    metodo: str

    @property
    def ancho_px(self) -> float:
        return self.caja[2] - self.caja[0] + 1

    @property
    def alto_px(self) -> float:
        return self.caja[3] - self.caja[1] + 1


@dataclass
class Camara:
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    @property
    def centro(self) -> np.ndarray:
        """Posición de la cámara en el mundo: C = −Rᵀ·t."""
        return -self.R.T @ self.t

    def proyectar(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """N × 3 puntos del mundo → (N × 2 píxeles, N profundidades)."""
        x = X @ self.R.T + self.t          # ejes de la cámara
        q = x @ self.K.T                   # λ·[u, v, 1]
        z = q[:, 2]
        return q[:, :2] / np.maximum(z, 1e-9)[:, None], z


def preparar_vista(nombre: str, bgr: np.ndarray, marco) -> Vista:
    if nombre not in VISTAS:
        raise ValueError(f"vista desconocida: {nombre}")
    foto = reducir(bgr)
    mascara, alfa, metodo = segmentar(foto, marco)
    alto, ancho = mascara.shape
    area = int(mascara.sum())
    if area < 0.002 * ancho * alto or area > 0.95 * ancho * alto:
        raise SinObjeto("no se distinguió el objeto; ponlo dentro del recuadro y que ocupe buena parte de él")
    vs, us = np.nonzero(mascara)
    return Vista(nombre, foto, mascara, alfa, (int(us.min()), int(vs.min()), int(us.max()), int(vs.max())), metodo)


def camaras(vistas: dict[str, Vista], p: Parametros) -> tuple[dict[str, Camara], np.ndarray]:
    """Calibra cada vista con las medidas compartidas. Devuelve las cámaras y
    las medidas del objeto (ancho, alto, fondo) en mm."""
    fr = vistas["frontal"]
    K = {n: matriz_K(v.foto.shape[1], v.foto.shape[0], p.fov) for n, v in vistas.items()}
    f = {n: k[0, 0] for n, k in K.items()}

    # La frontal fija la escala: ancho y alto del objeto a la distancia Z₀
    d = {"frontal": p.distancia_mm}
    ancho = fr.ancho_px * p.distancia_mm / f["frontal"]
    alto = fr.alto_px * p.distancia_mm / f["frontal"]
    fondos = []
    for n, v in vistas.items():
        if n in ("izquierda", "derecha"):     # comparten la altura (Y)
            d[n] = f[n] * alto / v.alto_px
            fondos.append(v.ancho_px * d[n] / f[n])
        elif n == "superior":                 # comparte el ancho (X)
            d[n] = f[n] * ancho / v.ancho_px
            fondos.append(v.alto_px * d[n] / f[n])
        elif n == "trasera":                  # comparte el ancho y el alto; no ve el fondo
            d[n] = f[n] * (ancho / v.ancho_px + alto / v.alto_px) / 2
    fondo = float(np.median(fondos)) if fondos else min(ancho, alto)

    resultado = {}
    for n, v in vistas.items():
        uc, vc = (v.caja[0] + v.caja[2]) / 2, (v.caja[1] + v.caja[3]) / 2
        t = d[n] * (inversa_K(K[n]) @ np.array([uc, vc, 1.0]))   # centro del objeto, ejes de la cámara
        resultado[n] = Camara(K[n], ROTACIONES[n], t)
    return resultado, np.array([ancho, alto, fondo])


def tallar(vistas: dict[str, Vista], cams: dict[str, Camara], medidas: np.ndarray):
    """Rejilla de voxeles y ocupación suave: el mínimo, entre las vistas, del
    valor de la silueta (0 fuera, 1 dentro) donde se proyecta cada voxel.
    Usar siluetas suavizadas e interpolación bilineal evita que la superficie
    quede escalonada."""
    lado = float(medidas.max()) * MARGEN
    paso = lado / LADO_VOXELES
    mitad = medidas * MARGEN / 2
    ejes = [np.arange(-m, m + paso, paso) for m in mitad]
    X, Y, Z = np.meshgrid(*ejes, indexing="ij")
    centros = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    ocupacion = np.ones(len(centros), dtype=np.float32)
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for n, v in vistas.items():
        # La silueta se ensancha un poco: tolera que las fotos a pulso no calcen exacto
        suave = cv2.GaussianBlur(cv2.dilate(v.mascara, nucleo).astype(np.float32), (0, 0), 1.5)
        uv, z = cams[n].proyectar(centros)
        # Lo que cae fuera de la foto (o detrás de la cámara) no se descarta con esta vista
        valor = map_coordinates(suave, [uv[:, 1], uv[:, 0]], order=1, mode="constant", cval=1.0)
        ocupacion = np.minimum(ocupacion, np.where(z > 0, valor, 1.0))
    return ocupacion.reshape(X.shape), np.array([e[0] for e in ejes]), paso


def colorear(V: np.ndarray, normales: np.ndarray, vistas: dict[str, Vista], cams: dict[str, Camara]) -> np.ndarray:
    """Color de cada vértice: el de la foto cuya cámara lo ve más de frente."""
    mejor = np.full(len(V), -np.inf)
    color = np.zeros((len(V), 3), dtype=np.uint8)
    for n, v in vistas.items():
        hacia = cams[n].centro - V
        hacia /= np.linalg.norm(hacia, axis=1, keepdims=True)
        puntaje = np.einsum("ij,ij->i", normales, hacia)
        uv, _ = cams[n].proyectar(V)
        alto, ancho = v.mascara.shape
        u = np.clip(np.round(uv[:, 0]).astype(int), 0, ancho - 1)
        w = np.clip(np.round(uv[:, 1]).astype(int), 0, alto - 1)
        puntaje = np.where(v.mascara[w, u] > 0, puntaje, puntaje - 2)   # fuera de la silueta: último recurso
        gana = puntaje > mejor
        mejor[gana] = puntaje[gana]
        color[gana] = v.foto[w[gana], u[gana], ::-1]   # BGR → RGB
    return color


def reconstruir(vistas: dict[str, Vista], p: Parametros | None = None) -> Resultado:
    p = p or Parametros()
    if "frontal" not in vistas or len(vistas) < 2:
        raise SinObjeto("hacen falta la foto frontal y al menos otra vista")
    tiempos: dict[str, float] = {}
    t0 = time.perf_counter()

    cams, medidas = camaras(vistas, p)
    ocupacion, origen, paso = tallar(vistas, cams, medidas)
    tiempos["talla"] = round((time.perf_counter() - t0) * 1000)
    if (ocupacion > 0.5).sum() < 50:
        raise SinObjeto("las siluetas no coinciden entre sí; repite las fotos con el objeto centrado")

    t1 = time.perf_counter()
    # Se suaviza la ocupación para que la superficie no quede escalonada; el
    # borde de ceros garantiza una superficie cerrada
    campo = gaussian_filter(np.pad(ocupacion, 2), sigma=0.8)
    verts, caras, normales, _ = marching_cubes(campo, 0.5, spacing=(paso, paso, paso))
    P = verts + origen - 2 * paso                       # índices → mm del mundo
    normales = normales / np.maximum(np.linalg.norm(normales, axis=1, keepdims=True), 1e-9)
    # Con 'descent' las normales apuntan hacia donde baja la ocupación: afuera
    if volumen(P, caras) < 0:                           # las caras también deben mirar hacia afuera
        caras = caras[:, ::-1]
    colores = colorear(P, normales, vistas, cams)
    tiempos["malla"] = round((time.perf_counter() - t1) * 1000)

    centro = (P.min(axis=0) + P.max(axis=0)) / 2
    V = ((P - centro) @ R_OBJETO.T).astype(np.float32)
    F = caras.astype(np.uint32)
    malla = Malla(V=V, F=F, uv=None, colores=colores, n_frente=len(V), caras={"todas": len(F)}, centro=centro)
    fr = vistas["frontal"]
    tam = V.max(axis=0) - V.min(axis=0)
    return Resultado(
        foto=fr.foto, mascara=fr.mascara, alfa=fr.alfa, K=cams["frontal"].K, malla=malla,
        metodo=f"envolvente visual · {len(vistas)} vistas · {fr.metodo}",
        tamano_mm=(float(tam[0]), float(tam[1]), float(tam[2])),
        volumen_mm3=volumen(V, F), tiempos=tiempos,
    )


__all__ = ["VISTAS", "Vista", "preparar_vista", "reconstruir", "camaras"]
