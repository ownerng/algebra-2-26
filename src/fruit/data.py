"""
Datos de la via D: Fruits-360 completo, aumentos y fondos aleatorios.

El problema que resuelve este modulo no es de arquitectura sino de dominio.
Fruits-360 son fotos de estudio (fruta girando sobre fondo blanco, luz fija);
una red que solo vio eso saca 99.9 % en el test del dataset y se desploma ante
una webcam. La defensa es entrenar con el dominio ya roto a proposito:

  * RandomBackground reemplaza el fondo blanco (el conectado al borde, no los
    blancos del interior de la fruta) por color liso, degradado, ruido o
    textura. Es el aumento que mas pesa.
  * Variacion de luz, balance de blancos, desenfoque, ruido y compresion JPEG
    imitan una camara barata.
  * Rotacion de 360 grados y espejos: una fruta no tiene arriba.

Estructura esperada (la deja `python -m src.data.download --dataset fruits360_full`):

    <raiz>/Training/<Variedad>/*.jpg
    <raiz>/Test/<Variedad>/*.jpg
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EXTENSIONES = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------
# Indice de archivos
# --------------------------------------------------------------------------

def buscar_raiz(raiz: Path | str | None = None) -> Path:
    """Carpeta que contiene Training/ y Test/ de Fruits-360."""
    candidatos = [Path(raiz)] if raiz else []
    candidatos += [config.RAW_DIR / "fruits360_full"]
    for base in candidatos:
        for c in (base, *base.glob("*")):
            if (c / "Training").is_dir():
                return c
    raise FileNotFoundError(
        "No encuentro Fruits-360 completo (carpeta con Training/ y Test/).\n"
        "Ejecuta:  python -m src.data.download --dataset fruits360_full\n"
        "o pasa --data con la carpeta correcta.")


def indexar(directorio: Path) -> tuple[list[Path], list[str]]:
    """Rutas de imagen y nombre de variedad (carpeta padre), ordenadas."""
    rutas = sorted(p for p in directorio.rglob("*")
                   if p.suffix.lower() in EXTENSIONES)
    return rutas, [p.parent.name for p in rutas]


def clases_de(directorio: Path) -> list[str]:
    nombres = sorted({d.name for d in directorio.iterdir() if d.is_dir()})
    if not nombres:
        raise FileNotFoundError("sin subcarpetas de clase en %s" % directorio)
    return nombres


def cargar_bgr(ruta: Path) -> np.ndarray:
    img = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
    if img is None:
        raise OSError("imagen ilegible: %s" % ruta)
    return img


# --------------------------------------------------------------------------
# Fondo aleatorio
# --------------------------------------------------------------------------

def mascara_fondo_blanco(bgr: np.ndarray, umbral: int = 232) -> np.ndarray:
    """Fondo = pixeles casi blancos CONECTADOS al borde de la imagen.

    Un umbral a secas borraria tambien el brillo especular de una manzana o el
    interior de un coco; exigir conexion con el borde deja la fruta intacta.
    Devuelve bool (H, W), True donde hay fondo.
    """
    blanco = (bgr.min(axis=2) >= umbral).astype(np.uint8)
    n, etiquetas = cv2.connectedComponents(blanco, connectivity=4)
    if n <= 1:
        return np.zeros(blanco.shape, bool)
    borde = np.unique(np.concatenate([
        etiquetas[0], etiquetas[-1], etiquetas[:, 0], etiquetas[:, -1]]))
    borde = borde[borde != 0]
    return np.isin(etiquetas, borde)


def _fondo_sintetico(alto: int, ancho: int, rng: random.Random) -> np.ndarray:
    """Fondo BGR uint8: liso, degradado, ruido de baja frecuencia o rayas."""
    tipo = rng.choice(("liso", "degradado", "ruido", "rayas", "mesa"))
    c1 = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
    c2 = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
    if tipo == "liso":
        base = np.tile(c1, (alto, ancho, 1))
    elif tipo == "degradado":
        t = np.linspace(0, 1, alto if rng.random() < 0.5 else ancho, dtype=np.float32)
        eje = (t[:, None, None] if t.size == alto else t[None, :, None])
        base = c1 * (1 - eje) + c2 * eje
        base = np.broadcast_to(base, (alto, ancho, 3))
    elif tipo == "ruido":
        pequeno = np.random.default_rng(rng.randint(0, 2**31)).random(
            (6, 6, 3), dtype=np.float32)
        base = cv2.resize(pequeno, (ancho, alto), interpolation=cv2.INTER_CUBIC)
        base = c1 + (c2 - c1) * base
    elif tipo == "rayas":
        periodo = rng.randint(6, 40)
        x = (np.arange(ancho) // periodo) % 2
        base = np.where(x[None, :, None] == 0, c1, c2)
        base = np.broadcast_to(base, (alto, ancho, 3))
    else:  # "mesa": tono madera/papel con grano fino
        calido = np.array([rng.randint(60, 120), rng.randint(110, 170),
                           rng.randint(150, 220)], np.float32)
        grano = np.random.default_rng(rng.randint(0, 2**31)).normal(
            0, 9, (alto, ancho, 1)).astype(np.float32)
        base = calido + grano
    return np.clip(base, 0, 255).astype(np.uint8)


class RandomBackground:
    """Sustituye el fondo blanco por uno aleatorio con probabilidad `p`.

    `fondos_reales`: carpeta opcional con fotos de fondos (mesa, pared, mano).
    Si existe se mezclan con los sinteticos; es la mejora mas barata cuando se
    conoce el entorno donde se va a usar la camara.
    """

    def __init__(self, p: float = 0.8, fondos_reales: Path | None = None) -> None:
        self.p = p
        self.reales: list[Path] = []
        if fondos_reales and Path(fondos_reales).is_dir():
            self.reales = [q for q in Path(fondos_reales).rglob("*")
                           if q.suffix.lower() in EXTENSIONES]
            log.info("fondos reales para aumento: %d", len(self.reales))

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return bgr
        mascara = mascara_fondo_blanco(bgr)
        if not mascara.any():
            return bgr
        alto, ancho = bgr.shape[:2]
        if self.reales and random.random() < 0.5:
            try:
                fondo = cv2.resize(cargar_bgr(random.choice(self.reales)),
                                   (ancho, alto), interpolation=cv2.INTER_AREA)
            except OSError:
                fondo = _fondo_sintetico(alto, ancho, random)
        else:
            fondo = _fondo_sintetico(alto, ancho, random)
        # Borde suave: sin el, la red aprende la costura recortada en vez de la fruta.
        alfa = cv2.GaussianBlur((~mascara).astype(np.float32), (0, 0), 0.8)[..., None]
        salida = bgr.astype(np.float32) * alfa + fondo.astype(np.float32) * (1 - alfa)
        return salida.astype(np.uint8)


# --------------------------------------------------------------------------
# Degradaciones tipo camara
# --------------------------------------------------------------------------

def degradar_camara(bgr: np.ndarray, fuerza: float = 1.0) -> np.ndarray:
    """Luz, balance de blancos, desenfoque, ruido y JPEG, al azar y por separado."""
    img = bgr.astype(np.float32)
    if random.random() < 0.8:                       # exposicion y contraste
        img = (img - 128) * random.uniform(1 - 0.35 * fuerza, 1 + 0.35 * fuerza) + 128
        img = img * random.uniform(1 - 0.4 * fuerza, 1 + 0.4 * fuerza)
    if random.random() < 0.6:                       # dominante de color (luz calida/fria)
        img = img * np.array([random.uniform(1 - 0.18 * fuerza, 1 + 0.18 * fuerza)
                              for _ in range(3)], np.float32)
    if random.random() < 0.3:                       # gamma
        g = random.uniform(0.7, 1.4)
        img = 255.0 * (np.clip(img, 0, 255) / 255.0) ** g
    img = np.clip(img, 0, 255).astype(np.uint8)

    if random.random() < 0.4:                       # desenfoque de movimiento / foco
        k = random.choice((3, 5, 7))
        if random.random() < 0.5:
            img = cv2.GaussianBlur(img, (k, k), 0)
        else:
            nucleo = np.zeros((k, k), np.float32)
            nucleo[k // 2, :] = 1.0 / k
            M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5),
                                        random.uniform(0, 180), 1.0)
            nucleo = cv2.warpAffine(nucleo, M, (k, k))
            img = cv2.filter2D(img, -1, nucleo / max(nucleo.sum(), 1e-6))
    if random.random() < 0.4:                       # ruido del sensor
        ruido = np.random.normal(0, random.uniform(2, 9) * fuerza, img.shape)
        img = np.clip(img.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
    if random.random() < 0.4:                       # compresion JPEG
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY,
                                             random.randint(25, 80)])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def _geometria(bgr: np.ndarray, lado: int) -> np.ndarray:
    """Rotacion libre, escala y traslacion; el relleno es el color de la esquina."""
    alto, ancho = bgr.shape[:2]
    angulo = random.uniform(0, 360)
    escala = random.uniform(0.75, 1.15)
    M = cv2.getRotationMatrix2D((ancho / 2, alto / 2), angulo, escala)
    M[:, 2] += (random.uniform(-0.08, 0.08) * ancho,
                random.uniform(-0.08, 0.08) * alto)
    relleno = tuple(int(v) for v in bgr[0, 0])
    out = cv2.warpAffine(bgr, M, (ancho, alto), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=relleno)
    if random.random() < 0.5:
        out = out[:, ::-1]
    return cv2.resize(np.ascontiguousarray(out), (lado, lado),
                      interpolation=cv2.INTER_CUBIC)


def a_tensor(bgr: np.ndarray, lado: int):
    """BGR uint8 -> tensor float normalizado a ImageNet, (3, lado, lado)."""
    import torch
    if bgr.shape[0] != lado or bgr.shape[1] != lado:
        bgr = cv2.resize(bgr, (lado, lado), interpolation=cv2.INTER_CUBIC)
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    rgb = (rgb - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class FrutasDataset:
    """Dataset de torch (sin heredar para no importar torch al importar el modulo).

    modo: 'train' (aumento fuerte), 'eval' (sin aumento) o 'shift' (eval con
    dominio roto de forma DETERMINISTA: mide robustez, no suerte).
    """

    def __init__(self, rutas: list[Path], etiquetas: list[int], modo: str,
                 lado: int = config.FRUIT_IMG,
                 fondos_reales: Path | None = None) -> None:
        assert modo in ("train", "eval", "shift")
        self.rutas, self.etiquetas, self.modo, self.lado = rutas, etiquetas, modo, lado
        self.fondo = RandomBackground(0.85 if modo == "train" else 1.0, fondos_reales)

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, i: int):
        bgr = cargar_bgr(self.rutas[i])
        if self.modo == "train":
            bgr = self.fondo(bgr)
            bgr = _geometria(bgr, self.lado)
            bgr = degradar_camara(bgr)
        elif self.modo == "shift":
            # Semilla por imagen: la misma imagen se degrada igual en cada corrida.
            estado = (random.getstate(), np.random.get_state())
            random.seed(i + config.SEED)
            np.random.seed((i + config.SEED) % (2**32 - 1))
            bgr = self.fondo(bgr)
            bgr = degradar_camara(bgr, fuerza=1.0)
            random.setstate(estado[0])
            np.random.set_state(estado[1])
        return a_tensor(bgr, self.lado), self.etiquetas[i]


def partir_validacion(etiquetas: list[int], fraccion: float, semilla: int = config.SEED
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Indices (train, val) estratificados por clase."""
    rng = np.random.default_rng(semilla)
    y = np.asarray(etiquetas)
    val: list[int] = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        val += idx[:max(1, int(round(len(idx) * fraccion)))].tolist()
    val_mask = np.zeros(len(y), bool)
    val_mask[val] = True
    return np.flatnonzero(~val_mask), np.flatnonzero(val_mask)
