"""Cámara del celular (DroidCam), por WiFi o por cable USB.

  • WiFi: DroidCam sirve un video MJPEG en http://IP:4747/video.
  • iPhone por cable: el mismo MJPEG, pedido por el cable a través de usbmuxd
    (no hace falta instalar nada).
  • USB: una cámara del computador (/dev/videoN), por ejemplo la que crea el
    cliente de DroidCam para Linux o la webcam; se lee con OpenCV.

DroidCam solo acepta un cliente a la vez, así que el servidor abre una única
conexión en un hilo, guarda el último cuadro (JPEG) y lo reparte: a la página
(vista previa) y al escáner (foto).
"""

import glob
import os
import threading
import time

import cv2
import httpx
import numpy as np

from . import usbmux

PUERTO_DROIDCAM = 4747


def url_de(ip: str) -> str:
    """'192.168.1.5' → http://192.168.1.5:4747/video. También acepta 'ip:puerto' o una URL completa."""
    ip = ip.strip()
    if ip.startswith(("http://", "https://")):
        return ip
    if ":" not in ip:
        ip = f"{ip}:{PUERTO_DROIDCAM}"
    return f"http://{ip}/video"


def camaras_locales() -> list[dict]:
    """iPhones por cable y cámaras del computador (/dev/video*), con su nombre."""
    lista = [{"ruta": f"ios:{i}", "nombre": "iPhone por cable (DroidCam)", "droidcam": True} for i in usbmux.iphones()]
    for ruta in sorted(glob.glob("/dev/video*"), key=lambda r: int(r.removeprefix("/dev/video") or 0)):
        nombre_sys = f"/sys/class/video4linux/{os.path.basename(ruta)}/name"
        try:
            nombre = open(nombre_sys).read().strip()
        except OSError:
            nombre = os.path.basename(ruta)
        # Cada webcam suele crear dos dispositivos; el segundo solo da metadatos
        indice = f"/sys/class/video4linux/{os.path.basename(ruta)}/index"
        try:
            if int(open(indice).read().strip()) != 0:
                continue
        except (OSError, ValueError):
            pass
        lista.append({"ruta": ruta, "nombre": nombre, "droidcam": "droidcam" in nombre.lower()})
    return lista


class CamaraRed:
    def __init__(self) -> None:
        self.url: str | None = None
        self.jpeg: bytes | None = None
        self.t_cuadro = 0.0
        self.error = ""
        self._hilo: threading.Thread | None = None
        self._parar = threading.Event()
        self._nuevo = threading.Condition()

    @property
    def conectada(self) -> bool:
        return self.jpeg is not None and time.monotonic() - self.t_cuadro < 3

    def conectar(self, ip: str, espera: float = 8.0) -> bool:
        """Abre el video por WiFi (IP) o USB ('/dev/videoN') y espera el primer cuadro."""
        self.desconectar()
        ip = ip.strip()
        usb = ip.startswith(("/dev/video", "ios:"))
        self.url, self.error, self.jpeg = (ip if usb else url_de(ip)), "", None
        self._parar = threading.Event()
        lector = self._leer_usb if ip.startswith("/dev/video") else self._leer_ios if ip.startswith("ios:") else self._leer
        self._hilo = threading.Thread(target=lector, args=(self.url, self._parar), daemon=True)
        self._hilo.start()
        limite = time.monotonic() + espera
        while time.monotonic() < limite and not self.conectada:
            if self.error and not self._hilo.is_alive():
                break
            time.sleep(0.1)
        if not self.conectada and not self.error:
            self.error = ("no llegó imagen; abre DroidCam en el celular y déjalo en pantalla" if usb
                          else "no llegó imagen; revisa la IP y que DroidCam esté abierto en el celular")
        return self.conectada

    def desconectar(self) -> None:
        self._parar.set()
        self.jpeg = None

    def _repartir(self, trozos, parar: threading.Event) -> None:
        """Separa el MJPEG en cuadros: cada uno es un JPEG entre los marcadores FFD8 y FFD9."""
        bufer = b""
        for trozo in trozos:
            if parar.is_set():
                return
            bufer += trozo
            fin = bufer.rfind(b"\xff\xd9")
            if fin < 0:
                continue
            inicio = bufer.rfind(b"\xff\xd8", 0, fin)
            if inicio >= 0:
                with self._nuevo:
                    self.jpeg, self.t_cuadro, self.error = bufer[inicio:fin + 2], time.monotonic(), ""
                    self._nuevo.notify_all()
            bufer = bufer[fin + 2:]

    def _reintentar(self, abrir, parar: threading.Event) -> None:
        """Lee hasta que se pida parar; si se corta la conexión, la vuelve a abrir."""
        while not parar.is_set():
            try:
                abrir()
            except Exception as e:
                self.error = str(e) or type(e).__name__
                if self.jpeg is None:   # nunca llegó imagen: no se insiste
                    return
                time.sleep(1)

    def _leer(self, url: str, parar: threading.Event) -> None:
        def abrir():
            with httpx.stream("GET", url, timeout=httpx.Timeout(5.0, read=5.0)) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"la cámara respondió {r.status_code}")
                self._repartir(r.iter_bytes(), parar)
        self._reintentar(abrir, parar)

    def _leer_ios(self, ruta: str, parar: threading.Event) -> None:
        def abrir():
            with usbmux.conectar(int(ruta.removeprefix("ios:")), PUERTO_DROIDCAM) as s:
                s.sendall(b"GET /video HTTP/1.1\r\nHost: localhost\r\n\r\n")
                def trozos():
                    while True:
                        trozo = s.recv(65536)
                        if not trozo:
                            raise ConnectionError("el iPhone cerró el video")
                        yield trozo
                self._repartir(trozos(), parar)
        self._reintentar(abrir, parar)

    def _leer_usb(self, ruta: str, parar: threading.Event) -> None:
        """Lee una cámara del computador y guarda cada cuadro como JPEG, igual que por WiFi."""
        cap = cv2.VideoCapture(ruta, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                self.error = f"no se pudo abrir {ruta} (¿otro programa la está usando?)"
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            fallos = 0
            while not parar.is_set():
                ok, cuadro = cap.read()
                if not ok:
                    fallos += 1
                    if fallos > 50:
                        self.error = f"{ruta} dejó de enviar imagen"
                        return
                    time.sleep(0.05)
                    continue
                fallos = 0
                ok, jpeg = cv2.imencode(".jpg", cuadro, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    with self._nuevo:
                        self.jpeg, self.t_cuadro, self.error = jpeg.tobytes(), time.monotonic(), ""
                        self._nuevo.notify_all()
        finally:
            cap.release()

    def esperar_cuadro(self, ultimo: bytes | None, espera: float = 1.0) -> bytes | None:
        with self._nuevo:
            if self.jpeg is ultimo:
                self._nuevo.wait(espera)
            return self.jpeg

    def foto(self) -> np.ndarray:
        if not self.conectada:
            raise RuntimeError("la cámara no está conectada")
        bgr = cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("el cuadro de la cámara está dañado")
        return bgr
