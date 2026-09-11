"""
Doctor de SCOAL: que le falta a esta copia para funcionar.

    python -m src.doctor               # todo, incluida la camara
    python -m src.doctor --sin-camara  # sin abrir la camara

No arregla nada: cada chequeo dice OK, AVISO o FALTA y, si falta algo, el
comando o el paso concreto para resolverlo. Sale con codigo 1 si hay algun
FALTA, asi sirve tambien dentro de un script.

Hasta confirmar que las dependencias estan instaladas solo usa la biblioteca
estandar: tiene que poder diagnosticar un entorno donde ni numpy importa. A
partir de ahi importa config, que deja todo tambien en logs.txt.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import platform
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PYTHON_MINIMO = (3, 11)

# (distribucion de pip, modulo que la app importa, obligatoria)
PAQUETES = (
    ("numpy", "numpy", True),
    ("opencv-python", "cv2", True),
    ("matplotlib", "matplotlib", True),
    ("requests", "requests", True),
    ("tqdm", "tqdm", True),
    ("torch", "torch", True),
    ("torchvision", "torchvision", True),
    ("PySide6", "PySide6.QtMultimedia", True),
    ("scikit-learn", "sklearn", False),   # solo tests
    ("pytest", "pytest", False),          # solo tests
)

OK, AVISO, FALTA = "OK", "AVISO", "FALTA"
resultados: list[tuple[str, str, str]] = []


def reportar(estado: str, que: str, detalle: str = "") -> None:
    resultados.append((estado, que, detalle))
    print("[%-5s] %s%s" % (estado, que, "\n        " + detalle if detalle else ""))


def _version(texto: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", texto)[:3])


def _minimos() -> dict[str, str]:
    """Versiones minimas leidas de requirements.txt, para no duplicarlas aqui."""
    minimos = {}
    for linea in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z0-9_.-]+)\s*>=\s*([0-9.]+)", linea)
        if m:
            minimos[m.group(1).lower()] = m.group(2)
    return minimos


def _venv_sin_usar() -> Path | None:
    """Python del .venv del proyecto, si existe y no es el que esta corriendo."""
    venv = ROOT / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if sys.prefix != sys.base_prefix or not venv.exists():
        return None
    return venv


def _hay_red(url: str) -> bool:
    try:
        urllib.request.urlopen(urllib.request.Request(url, method="HEAD"),
                               timeout=8).close()
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# Chequeos que solo necesitan la biblioteca estandar
# --------------------------------------------------------------------------

def chequear_python() -> bool:
    actual = sys.version_info[:3]
    if actual < PYTHON_MINIMO:
        reportar(FALTA, "Python %d.%d.%d" % actual,
                 "se necesita %d.%d o superior" % PYTHON_MINIMO)
        return False
    en_venv = sys.prefix != sys.base_prefix
    venv = _venv_sin_usar()
    if en_venv:
        detalle = ""
    elif venv:
        detalle = ("el proyecto ya tiene .venv pero lo corriste con otro Python. Usa:  "
                   "%s -m src.doctor   (o activalo: .venv\\Scripts\\activate)"
                   % venv.relative_to(ROOT))
    else:
        detalle = "no es un entorno virtual: python -m venv .venv, activarlo e instalar (README)"
    reportar(OK if en_venv else AVISO,
             "Python %d.%d.%d (%s)" % (*actual, sys.executable), detalle)
    return True


def chequear_paquetes() -> bool:
    """Instalado, version minima e import real: los tres fallan por separado."""
    minimos = _minimos()
    completo = True
    # Con un .venv sin activar, pip instalaria todo otra vez en el Python del
    # sistema: el arreglo real es usar el .venv.
    arreglo = ("usa el Python del .venv (ver arriba), no pip en este Python"
               if _venv_sin_usar() else "pip install -r requirements.txt")
    for dist, modulo, obligatorio in PAQUETES:
        falta = FALTA if obligatorio else AVISO
        nota = "" if obligatorio else "  (solo para los tests)"
        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            reportar(falta, dist, "no instalado: %s%s" % (arreglo, nota))
            completo = completo and not obligatorio
            continue

        minimo = minimos.get(dist.lower())
        if minimo and _version(version) < _version(minimo):
            reportar(falta, "%s %s" % (dist, version),
                     "se necesita >= %s: %s" % (minimo, arreglo))
            completo = completo and not obligatorio
            continue

        try:
            importlib.import_module(modulo)
        except Exception as exc:
            pista = (" (en Windows suele faltar el Microsoft Visual C++ Redistributable)"
                     if dist == "torch" else "")
            reportar(falta, "%s %s" % (dist, version),
                     "instalado pero no importa: %s%s" % (exc, pista))
            completo = completo and not obligatorio
            continue
        reportar(OK, "%s %s" % (dist, version))
    return completo


# --------------------------------------------------------------------------
# Chequeos del proyecto (ya con dependencias)
# --------------------------------------------------------------------------

def chequear_logs() -> None:
    import config
    try:
        with open(config.LOG_PATH, "a", encoding="utf-8"):
            pass
    except OSError as exc:
        reportar(FALTA, "logs", "no se puede escribir %s: %s" % (config.LOG_PATH, exc))
        return
    reportar(OK, "logs", str(config.LOG_PATH))


def chequear_modelo() -> None:
    from src.pipeline import MODEL_PATH, Scoal

    if not MODEL_PATH.exists():
        reportar(FALTA, "modelo entrenado",
                 "no existe %s: python -m src.setup (descarga datos y entrena)" % MODEL_PATH)
        return
    try:
        modelo = Scoal.load(MODEL_PATH)
    except Exception as exc:
        reportar(FALTA, "modelo entrenado",
                 "%s no se puede leer (%s): git checkout -- results/scoal_model.pkl "
                 "o python -m src.setup --retrain" % (MODEL_PATH.name, exc))
        return
    reportar(OK, "modelo entrenado", "%d clases, vias %s"
             % (len(modelo.clases), ", ".join(sorted(modelo.vias))))


def chequear_cnn() -> None:
    import numpy as np

    from src.features import cnn

    pesos = cnn.pesos_disponibles()
    if pesos is None:
        if _hay_red(cnn.WEIGHTS_URL):
            reportar(AVISO, "pesos de MobileNetV3 (via C)",
                     "no estan en disco; se descargan solos (10 MB) la primera vez "
                     "que la via C clasifica o entrena")
        else:
            reportar(FALTA, "pesos de MobileNetV3 (via C)",
                     "no estan y no hay acceso a download.pytorch.org: descargalos de "
                     "%s y guardalos como %s" % (cnn.WEIGHTS_URL, cnn.WEIGHTS_LOCAL))
        return
    try:
        z = cnn.extract_embeddings(np.zeros((1, 128, 128, 3), np.uint8))
    except Exception as exc:
        reportar(FALTA, "via C (CNN)",
                 "los pesos estan en %s pero el modelo no corre: %s" % (pesos, exc))
        return
    reportar(OK, "via C (CNN)", "pesos en %s, embedding de %d componentes"
             % (pesos, z.shape[1]))


def chequear_datos() -> None:
    """Datasets y cache solo hacen falta para reentrenar: son AVISO, no FALTA."""
    import config
    from src import setup
    from src.data import download

    for zip_ in sorted(config.RAW_DIR.glob("*.zip")):
        if not download.zip_valido(zip_):
            reportar(AVISO, "zip %s" % zip_.name,
                     "corrupto o incompleto (%d bytes): PREPARAR DATOS lo borra y "
                     "lo descarga de nuevo" % zip_.stat().st_size)
    for parcial in sorted(config.RAW_DIR.glob("*.part")):
        reportar(AVISO, "descarga %s" % parcial.name,
                 "en curso o interrumpida (%d MB): si no hay otra preparacion "
                 "corriendo, PREPARAR DATOS la rehace" % (parcial.stat().st_size >> 20))

    urls = {"fruits360": download.FRUITS360_URLS[0], "coil100": download.COIL100_URLS[0]}
    for dataset in config.DATASETS:
        faltan = [f for f in setup.estado(dataset)["faltan"] if f != "modelo"]
        if not faltan:
            reportar(OK, "dataset %s" % dataset, "imagenes y cache listos")
            continue
        red = "" if _hay_red(urls[dataset]) else "  (ahora mismo NO hay acceso a %s)" % urls[dataset]
        reportar(AVISO, "dataset %s" % dataset,
                 "falta %s; solo hace falta para reentrenar: PREPARAR DATOS o "
                 "python -m src.setup --dataset %s%s" % (", ".join(faltan), dataset, red))

    libre = shutil.disk_usage(ROOT).free / 1e9
    if libre < 2:
        reportar(AVISO, "espacio en disco",
                 "quedan %.1f GB; descargar y cachear un dataset puede ocupar 1-2 GB" % libre)


def chequear_camara() -> None:
    import cv2
    from PySide6.QtMultimedia import QMediaDevices
    from PySide6.QtWidgets import QApplication

    from src.ui.camera import abrir_camara, frame_vacio, indice_camara

    if sys.platform == "win32":
        backends = [cv2.videoio_registry.getBackendName(b)
                    for b in cv2.videoio_registry.getCameraBackends()]
        if "MSMF" not in backends:
            reportar(FALTA, "OpenCV sin backend MSMF",
                     "backends de camara: %s; reinstala opencv-python" % backends)
            return

    app = QApplication.instance() or QApplication([])   # QMediaDevices lo pide
    nombres = [d.description() for d in QMediaDevices.videoInputs()]
    if not nombres:
        reportar(FALTA, "camaras",
                 "el sistema no reporta ninguna: conecta una webcam o instala el "
                 "cliente de DroidCam (droidcam.app)")
        return

    indice = indice_camara()
    elegida = nombres[indice] if indice < len(nombres) else "?"
    es_droidcam = "droidcam" in elegida.lower()
    reportar(OK if es_droidcam else AVISO,
             "camara elegida: %d (%s)" % (indice, elegida),
             "" if es_droidcam else "no aparece DroidCam entre %s; la app usara esta" % nombres)

    captura = abrir_camara(indice)
    if not captura.isOpened():
        captura.release()
        reportar(FALTA, "abrir camara",
                 "no abre; si SCOAL u otra aplicacion la esta usando, cierrala y reintenta")
        return
    # read() falla al instante si la camara esta ocupada: sin cortar tras unos
    # fallos seguidos, el bucle de 3 s escupe miles de avisos de OpenCV.
    frames, fallos, ultimo, inicio = 0, 0, None, time.time()
    while time.time() - inicio < 3 and fallos < 30:
        ok, frame = captura.read()
        if ok:
            frames, fallos, ultimo = frames + 1, 0, frame
        else:
            fallos += 1
    captura.release()
    del app

    if ultimo is None:
        reportar(FALTA, "video de la camara",
                 "abre pero no entrega frames: casi siempre es que otra aplicacion la "
                 "esta usando (SCOAL con CAMARA encendida, Zoom, OBS). Cierrala y reintenta")
    elif frame_vacio(ultimo):
        reportar(FALTA, "video de la camara",
                 "llega imagen vacia (un solo color): abre DroidCam en el telefono y "
                 "conectalo en el cliente; si el cliente ya muestra imagen, "
                 "Archivo > Salir y vuelve a abrirlo")
    else:
        reportar(OK, "video de la camara", "%d frames en 3 s, %dx%d"
                 % (frames, ultimo.shape[1], ultimo.shape[0]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Dice que le falta a SCOAL para funcionar.")
    ap.add_argument("--sin-camara", action="store_true", help="no abrir la camara")
    args = ap.parse_args()

    print("SCOAL doctor | %s | %s\n" % (platform.platform(), ROOT))
    if chequear_python() and chequear_paquetes():
        import logging

        import config  # noqa: F401  (hay numpy: config configura logs.txt)

        log = logging.getLogger("scoal.doctor")
        chequeos = [chequear_logs, chequear_modelo, chequear_cnn, chequear_datos]
        if not args.sin_camara:
            chequeos.append(chequear_camara)
        for chequeo in chequeos:
            try:
                chequeo()
            except Exception as exc:
                log.exception("doctor: %s fallo", chequeo.__name__)
                reportar(FALTA, chequeo.__name__,
                         "fallo inesperado: %r (traceback en logs.txt)" % exc)
        # INFO y no WARNING/ERROR: la consola solo muestra WARNING+ y el reporte
        # ya se imprimio arriba; el estado va dentro del mensaje.
        for estado, que, detalle in resultados:
            log.info("doctor %s: %s %s", estado, que, detalle)
    else:
        print("\nFaltan dependencias basicas; el resto de chequeos necesita "
              "instalarlas primero.")

    faltas = sum(e == FALTA for e, _, _ in resultados)
    avisos = sum(e == AVISO for e, _, _ in resultados)
    print("\n%d FALTA, %d AVISO. %s" % (
        faltas, avisos, "Listo para usar." if not faltas else "Resuelve los FALTA de arriba."))
    sys.exit(1 if faltas else 0)


if __name__ == "__main__":
    main()
