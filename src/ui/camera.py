"""
Captura de camara en un hilo aparte.

Regla de Qt que este archivo respeta: los widgets solo se tocan desde el hilo
de la interfaz. El hilo de camara nunca toca un widget; se comunica por
Signal, que Qt entrega en la cola del hilo receptor.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from src.pipeline import medir
from src.vision.stability import StabilityTrigger

log = logging.getLogger(__name__)

# Ancho de trabajo: segmentacion y mediciones en px asumen esta escala.
ANCHO_TRABAJO = 640


class CameraThread(QThread):
    """Lee frames, segmenta y avisa cuando la escena se estabiliza.

    Senales:
      frameReady(frame, medicion) : cada frame capturado, ya segmentado y medido
      triggered(medicion)         : la escena quedo quieta con un objeto
      failed(mensaje)             : la camara no abrio o dejo de responder

    La medicion se calcula en este hilo porque es lo que permite dibujar el
    overlay a velocidad de camara; la clasificacion la hace la ventana, solo
    cuando llega `triggered`.
    """

    frameReady = Signal(object, object)
    triggered = Signal(object)
    failed = Signal(str)

    def __init__(self, indice: int = 0, parent=None) -> None:
        super().__init__(parent)
        self.indice = indice
        self._corriendo = False
        self._disparador = StabilityTrigger()

    def run(self) -> None:
        captura = abrir_camara(self.indice)
        if not captura.isOpened():
            log.error("no se pudo abrir la camara %d (backend MSMF)", self.indice)
            self.failed.emit("no se pudo abrir la camara %d" % self.indice)
            return

        log.info("camara %d abierta: %.0fx%.0f", self.indice,
                 captura.get(cv2.CAP_PROP_FRAME_WIDTH),
                 captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._corriendo = True
        self._disparador.reset()
        frames = disparos = 0
        estado_previo = None

        try:
            while self._corriendo:
                ok, frame = captura.read()
                if not ok:
                    log.error("la camara %d dejo de entregar frames (tras %d)",
                              self.indice, frames)
                    self.failed.emit("la camara dejo de entregar frames")
                    break
                frames += 1
                if frame.shape[1] > ANCHO_TRABAJO:
                    alto = frame.shape[0] * ANCHO_TRABAJO // frame.shape[1]
                    frame = cv2.resize(frame, (ANCHO_TRABAJO, alto),
                                       interpolation=cv2.INTER_AREA)

                medicion = medir(frame)
                # Un contorno que llega al borde no es un objeto sobre fondo
                # uniforme sino la escena entera (pared, cara, mano). El modelo
                # solo vio frutas sobre blanco: clasificar eso llena la banda
                # de basura, asi que cuenta como "sin objeto".
                fondo_malo = medicion["ok"] and toca_borde(medicion["mask"])
                if fondo_malo:
                    medicion = {**medicion, "ok": False, "contour": None,
                                "crop": None, "vector_a": None, "mediciones": {}}
                disparo = self._disparador.update(frame, medicion["ok"])

                # El estado del disparador viaja dentro de la medicion para que
                # la ventana pueda mostrarlo en cada frame: sin esta lectura el
                # modo camara parece congelado cuando en realidad esta
                # esperando que la escena se quede quieta.
                texto, quietos = self._disparador.progreso(medicion["ok"])
                estado = "con objeto" if medicion["ok"] else "sin objeto"
                if fondo_malo:
                    texto = estado = "fondo no uniforme: usa hoja blanca"
                if frame_vacio(frame):
                    texto = estado = "imagen vacia: conecta el telefono en DroidCam"
                medicion["estabilidad"] = {
                    "texto": texto,
                    "quietos": quietos,
                    "frames": self._disparador.frames,
                    "diferencia": self._disparador.diferencia,
                }

                # Al log solo van los cambios de estado: por frame serian 30
                # lineas por segundo.
                if estado != estado_previo:
                    log.info("camara: %s", estado)
                    estado_previo = estado

                self.frameReady.emit(frame, medicion)
                if disparo:
                    disparos += 1
                    log.info("disparador: escena quieta %d frames con objeto, "
                             "se clasifica", self._disparador.frames)
                    self.triggered.emit(medicion)
        finally:
            captura.release()
            log.info("camara %d cerrada: %d frames, %d disparos",
                     self.indice, frames, disparos)

    def stop(self) -> None:
        self._corriendo = False
        self.wait(2000)

    @property
    def diferencia(self) -> float:
        """Ultima diferencia entre frames, para mostrarla como diagnostico."""
        return self._disparador.diferencia


def abrir_camara(indice: int) -> cv2.VideoCapture:
    """Abre la camara tal como la usa la app. La comparten el hilo y el doctor.

    MSMF y no DSHOW: en algunos equipos DSHOW de OpenCV no abre ninguna camara
    ("raised unknown C++ exception", visto con DroidCam + OBS Virtual Camera).
    MSMF abre en ~0.3 s y su orden de dispositivos coincide con el de Qt.

    16:9 y no 640x480: DroidCam encaja el video del telefono en el tamano
    pedido y a 4:3 le agrega franjas negras, que la segmentacion (mediana del
    marco) toma como fondo y convierte toda la imagen en "objeto".
    """
    captura = cv2.VideoCapture(indice, cv2.CAP_MSMF)
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return captura


def frame_vacio(frame: np.ndarray) -> bool:
    """Frame de un solo color = camara virtual sin fuente.

    DroidCam entrega verde liso sin telefono conectado, o con su salida
    trabada (se arregla con Archivo > Salir y reabrir el cliente).
    """
    return bool(frame[::16, ::16].reshape(-1, 3).std(0).max() < 1)


def camaras_disponibles(maximo: int = 4) -> list[int]:
    """Indices de camara que responden. Se consulta una sola vez al arrancar."""
    encontradas = []
    for i in range(maximo):
        cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
        if cap.isOpened():
            encontradas.append(i)
        cap.release()
    return encontradas


def toca_borde(mascara: np.ndarray) -> bool:
    """True si la mascara llega a cualquier borde del cuadro."""
    return bool(mascara[0].any() or mascara[-1].any()
                or mascara[:, 0].any() or mascara[:, -1].any())


def indice_camara(nombre: str = "droidcam") -> int:
    """Indice MSMF de la primera camara cuyo nombre contiene `nombre`.

    OpenCV no expone nombres; Qt si, y en Windows enumera con Media Foundation
    en el mismo orden que CAP_MSMF, asi que su posicion es el indice valido.
    Requiere una QApplication viva. Sin coincidencia cae a la camara 0.
    """
    from PySide6.QtMultimedia import QMediaDevices

    nombres = [d.description() for d in QMediaDevices.videoInputs()]
    for i, n in enumerate(nombres):
        if nombre.lower() in n.lower():
            log.info("camaras %s: se usa la %d (%s)", nombres, i, n)
            return i
    log.warning("camaras %s: ninguna contiene '%s', se usa la 0", nombres, nombre)
    return 0


def demo() -> None:
    """Comprueba el disparador sin camara: escena quieta con objeto dispara."""
    fondo = np.zeros((120, 160, 3), np.uint8)
    con_objeto = fondo.copy()
    cv2.circle(con_objeto, (80, 60), 25, (40, 40, 220), -1)

    disparador = StabilityTrigger(umbral=1.0, frames=3)
    disparos = [disparador.update(con_objeto, True) for _ in range(6)]
    assert any(disparos), "nunca disparo con la escena quieta"
    assert sum(disparos) == 1, "disparo mas de una vez sin que la escena cambie"

    # Escena en movimiento: no debe disparar.
    disparador.reset()
    movidos = []
    for i in range(6):
        frame = np.roll(con_objeto, i * 20, axis=1)
        movidos.append(disparador.update(frame, True))
    assert not any(movidos)

    # Escena quieta pero vacia: tampoco.
    disparador.reset()
    assert not any(disparador.update(fondo, False) for _ in range(6))

    # Objeto centrado se acepta; la escena entera (mascara hasta el borde) no.
    centrado = np.zeros((120, 160), np.uint8)
    cv2.circle(centrado, (80, 60), 25, 255, -1)
    assert not toca_borde(centrado)
    assert toca_borde(np.full((120, 160), 255, np.uint8))

    # Verde liso de DroidCam sin telefono: vacio. Una escena real: no.
    verde = np.zeros((120, 160, 3), np.uint8)
    verde[..., 1] = 135
    assert frame_vacio(verde) and not frame_vacio(con_objeto)

    print("camera demo ok")


if __name__ == "__main__":
    demo()
