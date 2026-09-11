"""
Disparador de estabilidad para el modo webcam.

Clasificar cada frame es ruidoso y caro. Se clasifica solo cuando la escena
lleva N frames quieta Y hay un contorno valido, que es el equivalente software
de la fotocelda de una banda transportadora.
"""

from __future__ import annotations

import cv2
import numpy as np

import config


class StabilityTrigger:
    """Dispara cuando la escena se queda quieta con un objeto presente.

    Criterio:  mean(|gris_t - gris_{t-1}|) < umbral  durante `frames`
    consecutivos, y ademas la mascara del frame actual no esta vacia.

    Una vez disparado no vuelve a disparar hasta que la escena se mueva otra
    vez, para no reclasificar el mismo objeto quieto 30 veces por segundo.
    """

    def __init__(
        self,
        umbral: float = config.STABILITY_DIFF_THRESHOLD,
        frames: int = config.STABILITY_FRAMES,
    ) -> None:
        self.umbral = umbral
        self.frames = frames
        self._previo: np.ndarray | None = None
        self._quietos = 0
        self._ya_disparo = False

    @property
    def diferencia(self) -> float:
        """Ultima diferencia media medida (para mostrarla en la interfaz)."""
        return self._ultima_diff

    _ultima_diff = float("inf")

    @property
    def quietos(self) -> int:
        """Frames quietos acumulados. La interfaz lo muestra como 3/5.

        Sin esto el disparador es invisible: el usuario ve el feed, no pasa
        nada, y no tiene forma de saber si la escena todavia se mueve, si no
        hay objeto o si ya disparo.
        """
        return self._quietos

    @property
    def armado(self) -> bool:
        """False mientras no se mueva la escena tras un disparo."""
        return not self._ya_disparo

    def progreso(self, hay_objeto: bool) -> tuple[str, int]:
        """Estado legible del disparador: (texto, frames quietos)."""
        if not hay_objeto:
            return "sin objeto", 0
        if not self.armado:
            return "clasificado", self.frames
        if self._quietos == 0:
            return "escena en movimiento", 0
        return "estabilizando", min(self._quietos, self.frames)

    def update(self, frame_bgr: np.ndarray, hay_objeto: bool) -> bool:
        """Alimenta un frame. Devuelve True solo en el frame del disparo."""
        gris = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gris = cv2.resize(gris, (160, 120), interpolation=cv2.INTER_AREA)

        if self._previo is None:
            self._previo = gris
            return False

        self._ultima_diff = float(np.mean(np.abs(gris - self._previo)))
        self._previo = gris

        if self._ultima_diff >= self.umbral:
            self._quietos = 0
            self._ya_disparo = False       # hubo movimiento: rearmar
            return False

        self._quietos += 1
        if self._quietos >= self.frames and hay_objeto and not self._ya_disparo:
            self._ya_disparo = True
            return True
        return False

    def reset(self) -> None:
        self._previo = None
        self._quietos = 0
        self._ya_disparo = False
