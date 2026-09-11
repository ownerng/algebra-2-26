"""
SCOAL v2.2 - Configuracion global.

Un unico lugar para rutas, semilla, hiperparametros y seleccion de clases.
Ningun modulo define constantes propias: todo se importa desde aqui.
"""

from __future__ import annotations

import logging
import logging.handlers
import random
import sys
import threading
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Reproducibilidad
# --------------------------------------------------------------------------

SEED = 42


def set_seed(seed: int = SEED) -> np.random.Generator:
    """Fija la semilla global y devuelve un generador dedicado.

    Todo muestreo aleatorio del proyecto (splits, barajado, inicializaciones)
    debe pasar por el generador devuelto aqui para que los experimentos sean
    reproducibles bit a bit.
    """
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


RNG = set_seed(SEED)

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # descargas sin procesar (zip + extraido)
CACHE_DIR = DATA_DIR / "cache"      # tensores .npz listos para entrenar
RESULTS_DIR = ROOT / "results"      # CSV de ablacion, modelos serializados
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figuras"  # figuras generadas DESDE los CSV
TABLES_DIR = DOCS_DIR / "tablas"    # .tex generados DESDE los CSV

for _d in (RAW_DIR, CACHE_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------
#
# Todo punto de entrada (app, setup, doctor, tests) importa config, asi que es
# el unico lugar donde se configuran. Cada modulo solo hace
# logging.getLogger(__name__) y escribe; aqui se decide a donde va.

LOG_PATH = ROOT / "logs.txt"


def _configurar_logs() -> None:
    raiz = logging.getLogger()
    if any(getattr(h, "scoal", False) for h in raiz.handlers):
        return                                  # config importado dos veces
    if "pytest" in sys.modules:
        # Los tests fabrican zips rotos y fallos a proposito: en logs.txt se
        # leerian como incidentes reales. pytest ya captura los logs aparte.
        return

    # Rotacion a 5 MB x 3 copias: la trazabilidad no puede llenar el disco.
    archivo = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    archivo.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"))
    archivo.scoal = True
    raiz.addHandler(archivo)

    # En la terminal solo avisos y errores: el progreso normal ya lo imprime
    # cada comando y duplicarlo ensucia la salida.
    if sys.stderr is not None:
        consola = logging.StreamHandler()
        consola.setLevel(logging.WARNING)
        consola.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        raiz.addHandler(consola)

    raiz.setLevel(logging.INFO)
    for ruidoso in ("matplotlib", "PIL", "urllib3"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)

    # Lo que nadie atrapa tambien queda escrito, con su traceback.
    def no_atrapada(tipo, valor, tb):
        logging.getLogger("scoal").critical("excepcion no atrapada",
                                            exc_info=(tipo, valor, tb))
        sys.__excepthook__(tipo, valor, tb)

    def no_atrapada_en_hilo(args):
        logging.getLogger("scoal").critical(
            "excepcion no atrapada en el hilo %s",
            args.thread.name if args.thread else "?",
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback))

    sys.excepthook = no_atrapada
    threading.excepthook = no_atrapada_en_hilo


_configurar_logs()

# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

# Fruits-360: 10 clases con solapamiento deliberado (varias son rojas y
# redondas, varias son citricos). Forzar al modelo a usar mas de una feature.
#
# La clave es el nombre canonico usado en todo el proyecto; el valor son los
# prefijos de carpeta aceptados en el dataset, que varian entre releases
# ("Apple Golden 1/2/3", "Potato Red" vs "Potato Red Washed", etc.).
FRUITS_CLASSES: dict[str, tuple[str, ...]] = {
    "manzana_golden":         ("Apple Golden",),
    "manzana_granny_smith":   ("Apple Granny Smith", "Granny Smith"),
    "manzana_red_delicious":  ("Apple Red Delicious",),
    "naranja":                ("Orange",),
    "mandarina":              ("Mandarine", "Mandarin"),
    "limon":                  ("Lemon",),
    "lima":                   ("Limes", "Lime"),
    "tomate_cherry":          ("Tomato Cherry Red", "Cherry Tomato"),
    "papa_roja":              ("Potato Red",),
    "zucchini":               ("Zucchini",),
}

# COIL-100: 20 de los 100 objetos. Los objetos van numerados obj1..obj100 y
# cada uno aporta 72 vistas (una cada 5 grados sobre plato giratorio).
COIL_OBJECTS: tuple[int, ...] = tuple(range(1, 21))

DATASETS = ("fruits360", "coil100")
DEFAULT_DATASET = "fruits360"

IMAGES_PER_CLASS = 300   # balanceo; COIL-100 solo tiene 72 vistas por objeto
TEST_FRACTION = 0.25     # split estratificado con semilla fija

# --------------------------------------------------------------------------
# Preprocesamiento
# --------------------------------------------------------------------------

CROP_SIZE = 128     # recorte cuadrado cacheado (BGR + mascara)
EIGEN_SIZE = 64     # via B: 64x64 gris -> vector de 4096 componentes
CNN_INPUT = 224     # via C: entrada de MobileNetV3
CNN_EMBED_DIM = 1280  # penultima capa de MobileNetV3-Small
CNN_PCA_DIM = 60      # PCA propia sobre los embeddings

# Segmentacion (espacio YCbCr; ver justificacion en src/vision/segment.py)
CHROMA_THRESHOLD = 18.0   # distancia croma minima al fondo
MIN_CONTOUR_AREA = 200    # px^2; descarta ruido

# Disparador de estabilidad (solo webcam)
STABILITY_DIFF_THRESHOLD = 4.0   # diferencia media absoluta entre frames
STABILITY_FRAMES = 5             # frames consecutivos estables para disparar

# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------

USE_WEIGHT = False   # activar solo con mediciones reales de masa (HX711)

LAMBDA_DEFAULT = 1e-2   # regularizacion de Tikhonov
K_DEFAULT = 50          # componentes PCA por defecto (vias B y C)
KNN_K = 5               # baseline k-NN

# --------------------------------------------------------------------------
# Rechazo fuera de dominio (la clase "desconocido")
# --------------------------------------------------------------------------
#
# El conjunto de clases es cerrado y argmax(x W) siempre devuelve una: sin
# estos umbrales la aplicacion clasifica un celular o una cara como fruta.
# Ver la justificacion completa en src/model/reject.py.

REJECT_ENABLED = True

# Percentil del propio entrenamiento que fija los dos umbrales. Con 99 se
# rechaza el 1% mas novedoso y el 1% menos confiado del train, asi que la tasa
# de falso rechazo esperada sobre datos del dominio es ~2%. No hay ningun
# numero magico: los dos umbrales se leen de los datos al entrenar.
REJECT_PERCENTILE = 99.0

# --------------------------------------------------------------------------
# Ablacion
# --------------------------------------------------------------------------

ABLATION_K = (5, 10, 20, 30, 50, 80, 120)
ABLATION_LAMBDA = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
CV_FOLDS = 5

VIAS = ("A", "B", "C")
VIA_NOMBRES = {"A": "Fisicas", "B": "Eigen", "C": "CNN"}
