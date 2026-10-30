"""iPhone por cable USB, sin instalar nada: usbmuxd (el servicio de Linux que
habla con los iPhone) abre una conexión TCP a un puerto del celular a través
del cable. DroidCam escucha en el puerto 4747 también por ahí.

Protocolo de usbmuxd: cada mensaje es una cabecera de 16 bytes (largo, versión
1, tipo 8 = plist, etiqueta) seguida de un plist XML.
"""

import plistlib
import socket
import struct

SOCKET_USBMUXD = "/var/run/usbmuxd"
_BASE = {"ClientVersionString": "escaner-3d", "ProgName": "escaner-3d"}


def _pedir(s: socket.socket, mensaje: dict, etiqueta: int) -> dict:
    datos = plistlib.dumps({**_BASE, **mensaje})
    s.sendall(struct.pack("<IIII", 16 + len(datos), 1, 8, etiqueta) + datos)
    cabecera = _leer(s, 16)
    largo = struct.unpack("<I", cabecera[:4])[0]
    return plistlib.loads(_leer(s, largo - 16))


def _leer(s: socket.socket, n: int) -> bytes:
    datos = b""
    while len(datos) < n:
        trozo = s.recv(n - len(datos))
        if not trozo:
            raise ConnectionError("usbmuxd cerró la conexión")
        datos += trozo
    return datos


def iphones() -> list[int]:
    """IDs de los iPhone conectados por cable (vacío si usbmuxd no está)."""
    try:
        with socket.socket(socket.AF_UNIX) as s:
            s.settimeout(2)
            s.connect(SOCKET_USBMUXD)
            lista = _pedir(s, {"MessageType": "ListDevices"}, 1).get("DeviceList", [])
    except OSError:
        return []
    return [d["DeviceID"] for d in lista if d.get("Properties", {}).get("ConnectionType") == "USB"]


def conectar(id_dispositivo: int, puerto: int) -> socket.socket:
    """Socket conectado al `puerto` del iPhone; desde aquí es un TCP normal."""
    s = socket.socket(socket.AF_UNIX)
    s.settimeout(5)
    s.connect(SOCKET_USBMUXD)
    r = _pedir(s, {"MessageType": "Connect", "DeviceID": id_dispositivo, "PortNumber": socket.htons(puerto)}, 2)
    if r.get("Number") != 0:
        s.close()
        raise ConnectionError("DroidCam no responde por el cable; ábrelo en el iPhone y déjalo en pantalla")
    return s
