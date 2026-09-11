"""
Contrato de la descarga: un zip cortado nunca se toma por bueno.

Reproduce el fallo real: la descarga de Fruits-360 se corto a mitad, el zip
quedo con cabecera PK pero sin indice central, y PREPARAR DATOS lo daba por
descargado y fallaba al extraer con "File is not a zip file".
"""

import zipfile

import pytest

from src.data import download


def _zip(ruta):
    with zipfile.ZipFile(ruta, "w") as z:
        z.writestr("raiz/Training/Apple Golden 1/0.jpg", b"x" * 1000)
    return ruta


def test_zip_truncado_no_es_valido(tmp_path):
    bueno = _zip(tmp_path / "bueno.zip")
    cortado = tmp_path / "cortado.zip"
    cortado.write_bytes(bueno.read_bytes()[:-30])   # sin el indice central

    assert download.zip_valido(bueno)
    assert not download.zip_valido(cortado)
    assert not download.zip_valido(tmp_path / "no_existe.zip")


def test_zip_truncado_se_borra_y_se_descarga_de_nuevo(tmp_path, monkeypatch):
    destino = tmp_path / "fruits-360.zip"
    destino.write_bytes(b"PK\x03\x04" + b"\0" * 100)

    descargas = []
    monkeypatch.setattr(download, "_download",
                        lambda urls, parcial: descargas.append(_zip(parcial)))

    assert download.asegurar_zip(("http://espejo",), destino) == destino
    assert len(descargas) == 1
    assert download.zip_valido(destino)
    assert not destino.with_name(destino.name + ".part").exists()

    # Con el zip ya valido no se vuelve a descargar.
    download.asegurar_zip(("http://espejo",), destino)
    assert len(descargas) == 1


def test_descarga_que_no_es_zip_falla_sin_dejar_basura(tmp_path, monkeypatch):
    destino = tmp_path / "coil-100.zip"
    monkeypatch.setattr(download, "_download",
                        lambda urls, parcial: parcial.write_bytes(b"<html>"))

    with pytest.raises(RuntimeError, match="no produjo un zip valido"):
        download.asegurar_zip(("http://espejo",), destino)
    assert not destino.exists()
    assert not destino.with_name(destino.name + ".part").exists()
