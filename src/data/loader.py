"""
Construccion del cache de trabajo y particiones estratificadas.

El cache guarda el resultado de segmentar y recortar una sola vez:

    crops  (N, CROP_SIZE, CROP_SIZE, 3) uint8   objeto sobre fondo neutro
    masks  (N, CROP_SIZE, CROP_SIZE)    uint8   0 / 255
    y      (N,)                         int64   indice de clase
    clases (C,)                         str

Las tres vias parten de ese mismo tensor, de modo que la comparacion entre
representaciones no arrastra diferencias de preprocesado.

    python -m src.data.loader --dataset fruits360
    python -m src.data.loader --dataset coil100 --rebuild
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from src.data import download  # noqa: E402
from src.vision.crop import crop_to_mask  # noqa: E402
from src.vision.segment import segment  # noqa: E402


def cache_path(dataset: str) -> Path:
    return config.CACHE_DIR / ("%s_%d.npz" % (dataset, config.CROP_SIZE))


# --------------------------------------------------------------------------
# Recoleccion de rutas por clase
# --------------------------------------------------------------------------

def _rutas_fruits360() -> dict[str, list[Path]]:
    rutas: dict[str, list[Path]] = {}
    for canonica in config.FRUITS_CLASSES:
        archivos: list[Path] = []
        for carpeta in download.carpetas_de(canonica):
            archivos += sorted(carpeta.glob("*.jpg"))
            archivos += sorted(carpeta.glob("*.png"))
        if archivos:
            rutas[canonica] = archivos
    return rutas


def _rutas_coil100() -> dict[str, list[Path]]:
    rutas: dict[str, list[Path]] = {}
    if not download.COIL_DIR.exists():
        return rutas
    for idx in config.COIL_OBJECTS:
        archivos = sorted(download.COIL_DIR.rglob("obj%d__*.png" % idx))
        if archivos:
            rutas["obj%02d" % idx] = archivos
    return rutas


def rutas_por_clase(dataset: str) -> dict[str, list[Path]]:
    if dataset == "fruits360":
        return _rutas_fruits360()
    if dataset == "coil100":
        return _rutas_coil100()
    raise ValueError("dataset desconocido: %s" % dataset)


# --------------------------------------------------------------------------
# Construccion del cache
# --------------------------------------------------------------------------

def build_cache(dataset: str, por_clase: int = config.IMAGES_PER_CLASS) -> Path:
    """Segmenta, recorta y guarda el tensor de trabajo.

    Balancea a `por_clase` imagenes por clase muestreando sin reemplazo con la
    semilla global; si una clase tiene menos, se toma completa (COIL-100 solo
    aporta 72 vistas por objeto).
    """
    rutas = rutas_por_clase(dataset)
    if not rutas:
        raise RuntimeError(
            "No hay imagenes de %s en %s. Ejecuta primero:\n"
            "    python -m src.data.download --dataset %s"
            % (dataset, config.RAW_DIR, dataset)
        )

    rng = config.set_seed(config.SEED)
    clases = sorted(rutas)
    crops, masks, etiquetas = [], [], []
    descartadas = 0

    for indice, clase in enumerate(clases):
        archivos = rutas[clase]
        if len(archivos) > por_clase:
            elegidos = rng.choice(len(archivos), size=por_clase, replace=False)
            archivos = [archivos[i] for i in sorted(elegidos)]

        for ruta in tqdm(archivos, desc="%-24s" % clase, unit="img", leave=False):
            imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
            if imagen is None:
                descartadas += 1
                continue
            mascara, contorno = segment(imagen)
            if contorno is None:
                descartadas += 1
                continue
            recorte, recorte_mask = crop_to_mask(imagen, mascara)
            crops.append(recorte)
            masks.append(recorte_mask)
            etiquetas.append(indice)

    if not crops:
        raise RuntimeError("La segmentacion descarto todas las imagenes de %s" % dataset)

    destino = cache_path(dataset)
    np.savez_compressed(
        destino,
        crops=np.asarray(crops, np.uint8),
        masks=np.asarray(masks, np.uint8),
        y=np.asarray(etiquetas, np.int64),
        clases=np.asarray(clases),
    )
    print("%s: %d imagenes, %d clases, %d descartadas -> %s"
          % (dataset, len(crops), len(clases), descartadas, destino))
    return destino


def load_dataset(dataset: str = config.DEFAULT_DATASET, rebuild: bool = False) -> dict:
    """Carga el cache, construyendolo si hace falta."""
    destino = cache_path(dataset)
    if rebuild or not destino.exists():
        build_cache(dataset)
    datos = np.load(destino, allow_pickle=False)
    return {
        "crops": datos["crops"],
        "masks": datos["masks"],
        "y": datos["y"],
        "clases": [str(c) for c in datos["clases"]],
    }


# --------------------------------------------------------------------------
# Particiones
# --------------------------------------------------------------------------

def stratified_split(
    y: np.ndarray,
    test_fraction: float = config.TEST_FRACTION,
    seed: int = config.SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split estratificado con semilla fija.

    Cada clase aporta la misma proporcion a test, de modo que la exactitud no
    se pueda inflar con un test desbalanceado.
    """
    rng = np.random.default_rng(seed)
    train, test = [], []
    for clase in np.unique(y):
        idx = np.flatnonzero(y == clase)
        rng.shuffle(idx)
        corte = int(round(len(idx) * test_fraction))
        test.append(idx[:corte])
        train.append(idx[corte:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def stratified_kfold(
    y: np.ndarray,
    folds: int = config.CV_FOLDS,
    seed: int = config.SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Validacion cruzada estratificada de k pliegues, semilla fija.

    Devuelve [(idx_train, idx_test)] * folds.
    """
    rng = np.random.default_rng(seed)
    asignacion = np.empty(len(y), dtype=np.int64)
    for clase in np.unique(y):
        idx = np.flatnonzero(y == clase)
        rng.shuffle(idx)
        # reparto ciclico: los pliegues quedan igual de balanceados aunque el
        # numero de muestras de la clase no sea multiplo de `folds`
        asignacion[idx] = np.arange(len(idx)) % folds

    particiones = []
    for f in range(folds):
        test = np.flatnonzero(asignacion == f)
        train = np.flatnonzero(asignacion != f)
        particiones.append((train, test))
    return particiones


def main() -> None:
    ap = argparse.ArgumentParser(description="Construye el cache de trabajo.")
    ap.add_argument("--dataset", choices=config.DATASETS,
                    default=config.DEFAULT_DATASET)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    datos = load_dataset(args.dataset, rebuild=args.rebuild)
    conteo = np.bincount(datos["y"])
    for nombre, n in zip(datos["clases"], conteo):
        print("  %-24s %4d" % (nombre, n))
    print("total: %d imagenes" % len(datos["y"]))


if __name__ == "__main__":
    main()
