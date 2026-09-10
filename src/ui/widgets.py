"""
Widgets base de la aplicacion.

Reglas de diseno que este archivo hace cumplir (seccion 9 del PRD):
  - los numeros van en fuente mono y alineados a la derecha
  - las etiquetas van en mayusculas, TEXT_MID, pequenas, con letter-spacing
  - las separaciones son bordes de 1px, nunca espacio en blanco solo
  - ningun color se escribe a mano: todo sale de theme.py
"""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy,
                               QVBoxLayout, QWidget)

from src.ui import theme


class Panel(QFrame):
    """Contenedor con borde de 1px y, opcionalmente, una etiqueta de seccion."""

    def __init__(self, titulo: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(theme.SPACE, theme.SPACE,
                                        theme.SPACE, theme.SPACE)
        self._layout.setSpacing(theme.SPACE // 2)
        if titulo:
            etiqueta = QLabel(titulo.upper())
            etiqueta.setObjectName("SectionLabel")
            self._layout.addWidget(etiqueta)

    def add(self, widget: QWidget, stretch: int = 0) -> QWidget:
        self._layout.addWidget(widget, stretch)
        return widget

    def add_layout(self, layout) -> None:
        self._layout.addLayout(layout)

    def add_stretch(self) -> None:
        self._layout.addStretch(1)


def separador() -> QFrame:
    """Linea de 1px. La separacion visual se hace con borde, no con aire."""
    linea = QFrame()
    linea.setObjectName("Separator")
    linea.setFrameShape(QFrame.HLine)
    return linea


class ValueRow(QWidget):
    """Fila de medicion: etiqueta a la izquierda, numero mono a la derecha."""

    def __init__(self, etiqueta: str, valor: str = "--",
                 parent: QWidget | None = None):
        super().__init__(parent)
        fila = QHBoxLayout(self)
        fila.setContentsMargins(0, 0, 0, 0)

        self._etiqueta = QLabel(etiqueta.upper())
        self._etiqueta.setObjectName("SectionLabel")

        self._valor = QLabel(valor)
        self._valor.setObjectName("ValueMuted")
        self._valor.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        fila.addWidget(self._etiqueta)
        fila.addStretch(1)
        fila.addWidget(self._valor)

    def set_value(self, texto: str, activo: bool = True) -> None:
        self._valor.setText(texto)
        self._valor.setObjectName("Value" if activo else "ValueMuted")
        # Qt no reevalua el QSS al cambiar objectName: hay que forzarlo.
        self._valor.style().unpolish(self._valor)
        self._valor.style().polish(self._valor)


class PredictionRow(QWidget):
    """Prediccion de una via: marca de color, nombre, clase y confianza.

    Las tres filas comparten formato a proposito: cuando las vias discrepan,
    la discrepancia se lee de un vistazo, que es el punto del experimento.
    """

    def __init__(self, via: str, nombre: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.via = via
        fila = QHBoxLayout(self)
        fila.setContentsMargins(0, 2, 0, 2)
        fila.setSpacing(theme.SPACE)

        self._marca = QLabel()
        self._marca.setFixedSize(3, 16)
        self._marca.setStyleSheet("background: %s;" % theme.VIA_COLORS[via])

        self._via = QLabel(via)
        self._via.setObjectName("Value")
        self._via.setFixedWidth(12)
        self._via.setStyleSheet("color: %s;" % theme.VIA_COLORS[via])

        self._nombre = QLabel(nombre.upper())
        self._nombre.setObjectName("SectionLabel")
        self._nombre.setFixedWidth(72)

        self._clase = QLabel("--")
        self._clase.setObjectName("Value")

        self._confianza = QLabel("--")
        self._confianza.setObjectName("ValueMuted")
        self._confianza.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._confianza.setFixedWidth(48)

        for w in (self._marca, self._via, self._nombre, self._clase):
            fila.addWidget(w)
        fila.addStretch(1)
        fila.addWidget(self._confianza)

    def set_prediction(self, clase: str | None, confianza: float | None) -> None:
        if clase is None:
            self._clase.setText("--")
            self._confianza.setText("--")
            return
        self._clase.setText(clase)
        self._confianza.setText("%.2f" % confianza)

    def set_disponible(self, disponible: bool) -> None:
        """Una via sin modelo entrenado se muestra apagada, no se oculta."""
        self.setEnabled(disponible)


class FeedView(QLabel):
    """Imagen de la camara con el overlay de segmentacion dibujado encima.

    El overlay se pinta sobre el pixmap y no en paintEvent para no rehacer el
    dibujo en cada repintado de Qt: el frame solo cambia cuando llega uno nuevo.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setText("SIN SENAL")
        self.setObjectName("Status")

    def set_frame(self, frame_bgr: np.ndarray, resultado: dict | None = None) -> None:
        alto, ancho = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagen = QImage(rgb.data, ancho, alto, 3 * ancho, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(imagen.copy())

        if resultado and resultado.get("ok"):
            self._dibujar_overlay(pixmap, resultado)

        self.setPixmap(pixmap.scaled(self.size(), Qt.KeepAspectRatio,
                                     Qt.SmoothTransformation))

    def _dibujar_overlay(self, pixmap: QPixmap, resultado: dict) -> None:
        contorno = resultado["contour"]
        medidas = resultado.get("mediciones", {})

        pintor = QPainter(pixmap)
        pintor.setRenderHint(QPainter.Antialiasing)

        # Contorno.
        pintor.setPen(QPen(QColor(theme.ACCENT), 2))
        puntos = contorno.reshape(-1, 2)
        for i in range(len(puntos)):
            x0, y0 = puntos[i]
            x1, y1 = puntos[(i + 1) % len(puntos)]
            pintor.drawLine(int(x0), int(y0), int(x1), int(y1))

        # Caja envolvente.
        x, y = puntos[:, 0].min(), puntos[:, 1].min()
        w, h = puntos[:, 0].max() - x, puntos[:, 1].max() - y
        pintor.setPen(QPen(QColor(theme.BORDER), 1, Qt.DashLine))
        pintor.drawRect(int(x), int(y), int(w), int(h))

        # Ejes principales: los autovectores de la matriz de inercia, dibujados
        # con la longitud que dan sus autovalores.
        cx, cy = float(x + w / 2), float(y + h / 2)
        angulo = float(medidas.get("angulo_rad", 0.0))
        mayor = float(medidas.get("eje_mayor_px", 0.0)) / 2.0
        menor = float(medidas.get("eje_menor_px", 0.0)) / 2.0
        pintor.setPen(QPen(QColor(theme.TEXT_HI), 1))
        for largo, giro in ((mayor, 0.0), (menor, np.pi / 2)):
            dx = largo * np.cos(angulo + giro)
            dy = largo * np.sin(angulo + giro)
            pintor.drawLine(int(cx - dx), int(cy - dy), int(cx + dx), int(cy + dy))

        pintor.end()
