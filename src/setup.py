"""
Preparacion de cero: un solo comando entre clonar el repositorio y usar la app.

    python -m src.setup

Hace, en orden y saltandose lo que ya este hecho:

    1. descarga el dataset (Fruits-360 y/o COIL-100)
    2. construye el cache de trabajo (segmenta, recorta, balancea)
    3. entrena las tres vias y guarda results/scoal_model.pkl

El mismo codigo lo usa el boton PREPARAR DATOS de la aplicacion, por eso el
progreso sale por un callback y no por print directo: la interfaz le pasa una
funcion que escribe en la barra de estado en vez de en la terminal.

Los datasets no estan versionados (pesan ~1 GB), asi que este paso es
obligatorio en una copia recien clonada salvo que el modelo entrenado venga
incluido en el repositorio, que es el caso por defecto.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.data import download, loader  # noqa: E402
from src.pipeline import MODEL_PATH, Scoal  # noqa: E402

Progreso = Callable[[str], None]


# --------------------------------------------------------------------------
# Diagnostico: que hay y que falta
# --------------------------------------------------------------------------

def estado(dataset: str = config.DEFAULT_DATASET) -> dict:
    """Que piezas existen ya. La interfaz decide con esto que ofrecer.

    Devuelve {'datos': bool, 'cache': bool, 'modelo': bool, 'faltan': [...]}.
    """
    hay_datos = bool(loader.rutas_por_clase(dataset))
    hay_cache = loader.cache_path(dataset).exists()
    hay_modelo = MODEL_PATH.exists()

    faltan = []
    if not hay_datos:
        faltan.append("datos")
    if not hay_cache:
        faltan.append("cache")
    if not hay_modelo:
        faltan.append("modelo")

    return {"dataset": dataset, "datos": hay_datos, "cache": hay_cache,
            "modelo": hay_modelo, "faltan": faltan, "listo": not faltan}


def resumen(dataset: str = config.DEFAULT_DATASET) -> str:
    """Una linea legible con lo que falta, para la barra de estado."""
    est = estado(dataset)
    if est["listo"]:
        return "%s: datos, cache y modelo listos" % dataset
    return "%s: falta %s" % (dataset, ", ".join(est["faltan"]))


# --------------------------------------------------------------------------
# Preparacion
# --------------------------------------------------------------------------

def descargar(dataset: str, progreso: Progreso = print) -> None:
    """Descarga el dataset si todavia no hay imagenes suyas en disco."""
    if loader.rutas_por_clase(dataset):
        progreso("datos de %s ya presentes" % dataset)
        return

    progreso("descargando %s (puede tardar varios minutos)" % dataset)
    if dataset == "fruits360":
        download.descargar_fruits360()
    elif dataset == "coil100":
        download.descargar_coil100()
    else:
        raise ValueError("dataset desconocido: %s" % dataset)

    if not loader.rutas_por_clase(dataset):
        raise RuntimeError(
            "la descarga de %s termino pero no quedaron imagenes utilizables "
            "en %s" % (dataset, config.RAW_DIR))


def construir_cache(dataset: str, progreso: Progreso = print) -> Path:
    """Segmenta y cachea el dataset si el .npz todavia no existe."""
    destino = loader.cache_path(dataset)
    if destino.exists():
        progreso("cache de %s ya construido" % dataset)
        return destino

    progreso("construyendo cache de %s (segmentando imagenes)" % dataset)
    return loader.build_cache(dataset)


def entrenar(dataset: str, progreso: Progreso = print) -> Scoal:
    """Entrena las tres vias y guarda el modelo que usa la aplicacion."""
    progreso("cargando %s" % dataset)
    datos = loader.load_dataset(dataset)

    progreso("entrenando sobre %d imagenes" % len(datos["y"]))
    modelo = Scoal(datos["clases"]).fit(datos, dataset)
    if not modelo.vias:
        raise RuntimeError("ninguna via pudo entrenarse")

    modelo.save(MODEL_PATH)
    progreso("modelo guardado: vias %s" % ", ".join(sorted(modelo.vias)))
    return modelo


def preparar(dataset: str = config.DEFAULT_DATASET,
             progreso: Progreso = print,
             forzar_entrenamiento: bool = False) -> Scoal | None:
    """Deja el proyecto listo para usar. Idempotente: repetirlo no rehace nada.

    Devuelve el modelo entrenado, o None si ya habia uno en disco y no se pidio
    forzar el reentrenamiento.
    """
    descargar(dataset, progreso)
    construir_cache(dataset, progreso)

    if MODEL_PATH.exists() and not forzar_entrenamiento:
        progreso("modelo ya entrenado en %s" % MODEL_PATH)
        return None

    return entrenar(dataset, progreso)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Deja SCOAL listo para usar: datos, cache y modelo.")
    ap.add_argument("--dataset", choices=config.DATASETS,
                    default=config.DEFAULT_DATASET)
    ap.add_argument("--retrain", action="store_true",
                    help="reentrenar aunque ya exista un modelo guardado")
    ap.add_argument("--check", action="store_true",
                    help="solo reportar que falta, sin descargar nada")
    args = ap.parse_args()

    if args.check:
        print(resumen(args.dataset))
        return

    preparar(args.dataset, forzar_entrenamiento=args.retrain)
    print("listo. Ahora:  python -m src.ui.main_window")


if __name__ == "__main__":
    main()
