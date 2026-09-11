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
from PySide6.QtGui import (QColor, QFontMetrics, QImage, QPainter, QPen,
                           QPixmap)
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


class EtiquetaElidida(QLabel):
    """QLabel que recorta con puntos suspensivos en vez de desbordar el panel.

    Hace falta porque los nombres de clase son largos ("manzana_granny_smith")
    y el panel derecho ya no tiene ancho fijo: sin esto, o el texto empuja el
    panel, o se corta a la mitad sin aviso.
    """

    def __init__(self, texto: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._completo = texto
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumWidth(48)
        self.setText(texto)

    def setText(self, texto: str) -> None:
        self._completo = texto
        self._repintar()

    def text(self) -> str:
        return self._completo

    def resizeEvent(self, evento) -> None:
        super().resizeEvent(evento)
        self._repintar()

    def _repintar(self) -> None:
        metrica = QFontMetrics(self.font())
        QLabel.setText(self, metrica.elidedText(self._completo, Qt.ElideRight,
                                                max(self.width(), 48)))


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
        self._nombre.setMinimumWidth(56)

        self._clase = EtiquetaElidida("--")
        self._clase.setObjectName("Value")

        self._confianza = QLabel("--")
        self._confianza.setObjectName("ValueMuted")
        self._confianza.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._confianza.setFixedWidth(40)

        for w in (self._marca, self._via, self._nombre):
            fila.addWidget(w)
        fila.addWidget(self._clase, 1)
        fila.addWidget(self._confianza)

    def set_prediction(self, clase: str | None, confianza: float | None,
                       conocido: bool = True, novedad: float | None = None) -> None:
        """Muestra la prediccion de la via.

        Con `conocido=False` la muestra cayo fuera del dominio entrenado: se
        anuncia DESCONOCIDO y la clase que el argmax habria elegido queda como
        pista entre parentesis, en tono apagado. Ocultarla seria peor: en la
        sustentacion interesa ver exactamente que estaba a punto de decir el
        modelo y por que se rechazo.
        """
        if clase is None:
            self._clase.setText("--")
            self._clase.setStyleSheet("")
            self._clase.setToolTip("")
            self._confianza.setText("--")
            return

        if conocido:
            self._clase.setText(clase)
            self._clase.setStyleSheet("color: %s;" % theme.TEXT_HI)
            self._clase.setToolTip("")
        else:
            self._clase.setText("DESCONOCIDO  (%s)" % clase)
            self._clase.setStyleSheet("color: %s;" % theme.ERR)
            self._clase.setToolTip(
                "Fuera del dominio entrenado.\n"
                "El modelo habria dicho '%s' con confianza %.2f." % (clase, confianza)
                + ("\nNovedad: %.2f veces el umbral." % novedad
                   if novedad is not None else ""))

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
        # Minimo pequeno a proposito: con 640x480 la ventana no cabia en un
        # portatil de 13 pulgadas con escalado del sistema. El feed se escala
        # al tamano disponible en set_frame, no necesita reservarlo.
        self.setMinimumSize(280, 210)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setText("SIN SENAL")
        self.setObjectName("Status")

    _original: QPixmap | None = None

    def set_frame(self, frame_bgr: np.ndarray, resultado: dict | None = None) -> None:
        alto, ancho = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagen = QImage(rgb.data, ancho, alto, 3 * ancho, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(imagen.copy())

        if resultado and resultado.get("ok"):
            self._dibujar_overlay(pixmap, resultado)

        # Se guarda el original, no solo el escalado: al redimensionar la
        # ventana hay que reescalar desde la resolucion completa, y con una
        # imagen fija (sin camara) no llega otro frame que lo corrija.
        self._original = pixmap
        self._escalar()

    def resizeEvent(self, evento) -> None:
        super().resizeEvent(evento)
        self._escalar()

    def _escalar(self) -> None:
        if self._original is None or self._original.isNull():
            return
        self.setPixmap(self._original.scaled(self.size(), Qt.KeepAspectRatio,
                                             Qt.SmoothTransformation))

    def setText(self, texto: str) -> None:
        """Mostrar texto descarta la imagen: si no, reaparece al redimensionar."""
        self._original = None
        super().setText(texto)

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


class CaptureStatus(QWidget):
    """Estado del disparador de estabilidad, visible en cada frame.

    La clasificacion no ocurre en cada frame: ocurre cuando la escena lleva N
    frames quieta y hay un objeto segmentado. Sin este indicador el modo camara
    parece roto, porque el usuario ve el feed moverse y las predicciones nunca
    cambian. Aqui se lee, literalmente, por que todavia no dispara.
    """

    ESTADOS = {
        "sin objeto":           (theme.TEXT_LO,  "no hay contorno valido en la escena"),
        "escena en movimiento": (theme.TEXT_MID, "la escena aun se mueve"),
        "estabilizando":        (theme.ACCENT,   "quieta, contando frames"),
        "clasificado":          (theme.VIA_A,    "ya clasificado; mover la escena para repetir"),
        "camara apagada":       (theme.TEXT_LO,  "encender la camara o usar una imagen del dataset"),
    }

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        fila = QHBoxLayout(self)
        fila.setContentsMargins(0, 0, 0, 0)
        fila.setSpacing(theme.SPACE)

        self._punto = QLabel()
        self._punto.setFixedSize(8, 8)

        self._texto = QLabel("camara apagada")
        self._texto.setObjectName("Status")

        self._conteo = QLabel("")
        self._conteo.setObjectName("ValueMuted")
        self._conteo.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        fila.addWidget(self._punto)
        fila.addWidget(self._texto)
        fila.addStretch(1)
        fila.addWidget(self._conteo)
        self.set_estado("camara apagada")

    def set_estado(self, texto: str, quietos: int = 0, frames: int = 0,
                   diferencia: float | None = None) -> None:
        color, ayuda = self.ESTADOS.get(texto, (theme.TEXT_MID, ""))
        self._punto.setStyleSheet(
            "background: %s; border-radius: 4px;" % color)
        self._texto.setText(texto)
        self._texto.setStyleSheet("color: %s;" % color)
        self._conteo.setText("%d/%d" % (quietos, frames) if frames else "")

        if diferencia is not None and diferencia != float("inf"):
            ayuda = "%s\nmovimiento medio entre frames: %.2f" % (ayuda, diferencia)
        self.setToolTip(ayuda)
