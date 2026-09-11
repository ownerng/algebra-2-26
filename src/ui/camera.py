"""
Captura de camara en un hilo aparte.

Regla de Qt que este archivo respeta: los widgets solo se tocan desde el hilo
de la interfaz. El hilo de camara nunca toca un widget; se comunica por
Signal, que Qt entrega en la cola del hilo receptor.
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from src.pipeline import medir
from src.vision.stability import StabilityTrigger


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
        # CAP_DSHOW evita el arranque lento del backend MSMF en Windows.
        captura = cv2.VideoCapture(self.indice, cv2.CAP_DSHOW)
        if not captura.isOpened():
            self.failed.emit("no se pudo abrir la camara %d" % self.indice)
            return

        captura.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._corriendo = True
        self._disparador.reset()

        try:
            while self._corriendo:
                ok, frame = captura.read()
                if not ok:
                    self.failed.emit("la camara dejo de entregar frames")
                    break

                medicion = medir(frame)
                disparo = self._disparador.update(frame, medicion["ok"])

                # El estado del disparador viaja dentro de la medicion para que
                # la ventana pueda mostrarlo en cada frame: sin esta lectura el
                # modo camara parece congelado cuando en realidad esta
                # esperando que la escena se quede quieta.
                texto, quietos = self._disparador.progreso(medicion["ok"])
                medicion["estabilidad"] = {
                    "texto": texto,
                    "quietos": quietos,
                    "frames": self._disparador.frames,
                    "diferencia": self._disparador.diferencia,
                }

                self.frameReady.emit(frame, medicion)
                if disparo:
                    self.triggered.emit(medicion)
        finally:
            captura.release()

    def stop(self) -> None:
        self._corriendo = False
        self.wait(2000)

    @property
    def diferencia(self) -> float:
        """Ultima diferencia entre frames, para mostrarla como diagnostico."""
        return self._disparador.diferencia


def camaras_disponibles(maximo: int = 4) -> list[int]:
    """Indices de camara que responden. Se consulta una sola vez al arrancar."""
    encontradas = []
    for i in range(maximo):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            encontradas.append(i)
        cap.release()
    return encontradas


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

    print("camera demo ok")


if __name__ == "__main__":
    demo()
