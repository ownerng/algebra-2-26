"""
Descarga y extraccion automatizada de COIL-100 y Fruits-360.

Un solo comando:

    python -m src.data.download                # ambos datasets
    python -m src.data.download --dataset coil100
    python -m src.data.download --check        # solo verificar lo ya presente

Se descarga el zip completo (el indice central de un zip vive al final del
archivo, no se puede extraer en streaming) pero se extraen unicamente las
clases seleccionadas en config.py.

La descarga se escribe en `<zip>.part` y solo pasa a llamarse `<zip>` cuando
termino y es un zip valido. Antes se escribia directo en el nombre final: un
corte a mitad (cerrar la aplicacion, perder la red) dejaba un zip truncado que
la corrida siguiente daba por bueno y fallaba con "File is not a zip file".
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402

log = logging.getLogger(__name__)

# Espejos en orden de preferencia. El primero que responda gana.
COIL100_URLS = (
    "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip",
    "https://cave.cs.columbia.edu/old/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip",
)

FRUITS360_URLS = (
    "https://github.com/fruits-360/fruits-360-100x100/archive/refs/heads/main.zip",
    "https://github.com/Horea94/Fruit-Images-Dataset/archive/refs/heads/master.zip",
)

COIL_DIR = config.RAW_DIR / "coil100"
FRUITS_DIR = config.RAW_DIR / "fruits360"


# --------------------------------------------------------------------------
# Descarga
# --------------------------------------------------------------------------

def _download(urls: tuple[str, ...], destino: Path) -> Path:
    """Descarga la primera URL que responda, reanudando con cabecera Range."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    ultimo_error: Exception | None = None

    for url in urls:
        try:
            ya = destino.stat().st_size if destino.exists() else 0
            headers = {"Range": "bytes=%d-" % ya} if ya else {}
            log.info("descargando %s -> %s (ya en disco: %d bytes)", url, destino, ya)
            resp = requests.get(url, stream=True, timeout=60, headers=headers)
            log.info("respuesta HTTP %d, content-length %s", resp.status_code,
                     resp.headers.get("content-length", "desconocido"))

            if resp.status_code == 416:          # el archivo ya esta completo
                resp.close()
                return destino
            resp.raise_for_status()

            # 206: el servidor retoma donde quedo. 200: ignoro el Range (GitHub
            # lo hace) y manda todo desde cero, asi que se reescribe.
            modo = "ab" if resp.status_code == 206 else "wb"
            if modo == "wb":
                ya = 0
            total = int(resp.headers.get("content-length", 0)) + ya

            escritos = ya
            with open(destino, modo) as fh, tqdm(
                total=total or None, initial=ya, unit="B", unit_scale=True,
                desc=destino.name,
            ) as barra:
                for bloque in resp.iter_content(chunk_size=1 << 20):
                    fh.write(bloque)
                    barra.update(len(bloque))
                    # Una linea cada 50 MB: sin esto una descarga lenta y una
                    # colgada se ven igual en el log.
                    if (escritos + len(bloque)) >> 26 != escritos >> 26:
                        log.info("  %s: %d MB", destino.name, (escritos + len(bloque)) >> 20)
                    escritos += len(bloque)
            log.info("descarga terminada: %d bytes en %s", escritos, destino)
            return destino

        except Exception as exc:                 # espejo caido: siguiente
            ultimo_error = exc
            log.warning("fallo %s: %s", url, exc)

    raise RuntimeError(
        "Ninguna URL respondio. Ultimo error: %s\n"
        "Descarga manual y deja el zip en %s" % (ultimo_error, destino)
    )


def zip_valido(ruta: Path) -> bool:
    """Un zip cortado conserva la cabecera PK del inicio pero pierde el indice
    central del final; is_zipfile busca justamente ese indice."""
    return ruta.exists() and zipfile.is_zipfile(ruta)


def asegurar_zip(urls: tuple[str, ...], destino: Path) -> Path:
    """Deja en `destino` un zip completo: reusa uno valido o lo (re)descarga."""
    if zip_valido(destino):
        log.info("zip ya descargado y valido: %s (%d bytes)",
                 destino, destino.stat().st_size)
        return destino
    if destino.exists():
        log.warning("zip corrupto o incompleto (%d bytes), se borra y se "
                    "descarga de nuevo: %s", destino.stat().st_size, destino)
        destino.unlink()

    parcial = destino.with_name(destino.name + ".part")
    _download(urls, parcial)
    if not zipfile.is_zipfile(parcial):
        tamano = parcial.stat().st_size
        parcial.unlink()
        raise RuntimeError(
            "la descarga de %s no produjo un zip valido (%d bytes); "
            "vuelve a intentar" % (destino.name, tamano))
    parcial.replace(destino)
    log.info("zip completo y valido: %s (%d bytes)", destino, destino.stat().st_size)
    return destino


def _extraer(zip_path: Path, destino: Path, quiero) -> int:
    """Extrae solo los miembros aceptados por `quiero(nombre) -> bool`.

    Devuelve cuantos archivos se escribieron. Se descarta el primer directorio
    del zip (la raiz que agregan GitHub y Columbia) y se conserva el resto de
    la ruta, que es donde vive el nombre de la clase.
    """
    destino.mkdir(parents=True, exist_ok=True)
    escritos = 0

    with zipfile.ZipFile(zip_path) as zf:
        miembros = [m for m in zf.namelist() if not m.endswith("/") and quiero(m)]
        log.info("extrayendo %d archivos seleccionados de %s -> %s",
                 len(miembros), zip_path.name, destino)
        for miembro in tqdm(miembros, desc="extraer %s" % destino.name, unit="img"):
            partes = Path(miembro).parts[1:]
            if not partes:
                continue
            salida = destino.joinpath(*partes)
            if salida.exists():
                continue
            salida.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(miembro) as origen, open(salida, "wb") as fh:
                shutil.copyfileobj(origen, fh)
            escritos += 1

    return escritos


# --------------------------------------------------------------------------
# COIL-100
# --------------------------------------------------------------------------

def _es_objeto_coil(nombre: str) -> bool:
    """obj7__155.png entra si el objeto 7 esta en config.COIL_OBJECTS."""
    tallo = Path(nombre).name
    if not tallo.startswith("obj") or "__" not in tallo:
        return False
    try:
        idx = int(tallo[3:].split("__", 1)[0])
    except ValueError:
        return False
    return idx in config.COIL_OBJECTS


def descargar_coil100() -> Path:
    zip_path = asegurar_zip(COIL100_URLS, config.RAW_DIR / "coil-100.zip")
    n = _extraer(zip_path, COIL_DIR, _es_objeto_coil)
    log.info("COIL-100: %d imagenes nuevas en %s", n, COIL_DIR)
    return COIL_DIR


# --------------------------------------------------------------------------
# Fruits-360
# --------------------------------------------------------------------------

def clase_de_ruta(nombre: str) -> str | None:
    """Clase canonica de una ruta del zip, o None si no esta seleccionada.

    Las rutas son  <raiz>/Training/<Clase>/<archivo>.jpg  y el nombre de
    <Clase> cambia entre releases ("Apple Golden 1", "Potato Red Washed"),
    por eso se compara por prefijo contra los alias de config.FRUITS_CLASSES.
    """
    partes = Path(nombre).parts
    if len(partes) < 3:
        return None
    carpeta = partes[-2]
    for canonica, alias in config.FRUITS_CLASSES.items():
        if any(carpeta.startswith(a) for a in alias):
            return canonica
    return None


def descargar_fruits360() -> Path:
    zip_path = asegurar_zip(FRUITS360_URLS, config.RAW_DIR / "fruits-360.zip")

    def quiero(nombre: str) -> bool:
        if not nombre.lower().endswith((".jpg", ".jpeg", ".png")):
            return False
        return clase_de_ruta(nombre) is not None

    n = _extraer(zip_path, FRUITS_DIR, quiero)
    log.info("Fruits-360: %d imagenes nuevas en %s", n, FRUITS_DIR)

    faltan = [c for c in config.FRUITS_CLASSES if not carpetas_de(c)]
    if faltan:
        log.warning("clases sin carpeta (revisar alias en config.py): %s",
                    ", ".join(faltan))
    return FRUITS_DIR


def carpetas_de(canonica: str) -> list[Path]:
    """Carpetas ya extraidas que corresponden a una clase canonica."""
    if not FRUITS_DIR.exists():
        return []
    alias = config.FRUITS_CLASSES[canonica]
    return [
        d for d in FRUITS_DIR.rglob("*")
        if d.is_dir() and any(d.name.startswith(a) for a in alias)
    ]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def verificar() -> None:
    coil = sum(1 for _ in COIL_DIR.rglob("*.png")) if COIL_DIR.exists() else 0
    fruits = sum(1 for _ in FRUITS_DIR.rglob("*.jpg")) if FRUITS_DIR.exists() else 0
    print("COIL-100   : %6d imagenes  (%s)" % (coil, COIL_DIR))
    print("Fruits-360 : %6d imagenes  (%s)" % (fruits, FRUITS_DIR))


def main() -> None:
    ap = argparse.ArgumentParser(description="Descarga los datasets de SCOAL.")
    ap.add_argument("--dataset", choices=("both", "coil100", "fruits360"),
                    default="both")
    ap.add_argument("--check", action="store_true",
                    help="solo reportar lo ya descargado")
    args = ap.parse_args()

    if args.check:
        verificar()
        return

    if args.dataset in ("both", "fruits360"):
        descargar_fruits360()
    if args.dataset in ("both", "coil100"):
        descargar_coil100()
    verificar()


if __name__ == "__main__":
    main()
