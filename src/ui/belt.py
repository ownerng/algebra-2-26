"""
Banda transportadora animada y contadores por clase.

La animacion no es decoracion: representa el recorrido de un objeto real desde
la camara hasta su contenedor, y el contador de cada contenedor es el resumen
de la sesion de clasificacion.
"""

from __future__ import annotations

from PySide6.QtCore import (QEasingCurve, QPointF, QPropertyAnimation,
                            QRectF, QSequentialAnimationGroup, Qt)
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (QGraphicsObject, QGraphicsScene, QGraphicsView,
                               QGridLayout, QVBoxLayout, QWidget)

from src.ui import theme
from src.ui.widgets import ValueRow

ALTO_VISTA = 76
RADIO_OBJETO = 14

# Contenedor de descarte: las muestras que el rechazo marca como fuera de
# dominio tienen que ir a algun lado, y ese conteo es un dato del experimento
# (cuanto de lo que ve la camara no se parece al dataset).
RECHAZO = "desconocido"

# Ancho minimo por contenedor al repartirlos en filas.
ANCHO_CONTENEDOR = 150


class ObjetoEnBanda(QGraphicsObject):
    """Circulo que viaja por la banda.

    Hereda de QGraphicsObject y no de QGraphicsEllipseItem porque
    QPropertyAnimation necesita un QObject con la propiedad `pos`, que
    QGraphicsObject ya expone.
    """

    def __init__(self, color: str) -> None:
        super().__init__()
        self._color = QColor(color)

    def boundingRect(self) -> QRectF:
        return QRectF(-RADIO_OBJETO, -RADIO_OBJETO, 2 * RADIO_OBJETO, 2 * RADIO_OBJETO)

    def paint(self, pintor, opcion, widget=None) -> None:
        pintor.setRenderHint(QPainter.Antialiasing)
        pintor.setBrush(QBrush(self._color))
        pintor.setPen(QPen(QColor(theme.BG_BASE), 1))
        pintor.drawEllipse(self.boundingRect())


class Belt(QWidget):
    """Vista de la banda mas la fila de contenedores con su conteo."""

    def __init__(self, clases: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._orden = list(clases) + [RECHAZO]
        self._conteo = {c: 0 for c in self._orden}
        self._animaciones: list[QSequentialAnimationGroup] = []
        self._columnas = 0

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        raiz.setSpacing(theme.SPACE)

        self._escena = QGraphicsScene(self)
        # El fondo de la escena es independiente del QSS de la vista.
        self._escena.setBackgroundBrush(QBrush(QColor(theme.BG_PANEL)))
        self._linea = self._escena.addLine(
            0, ALTO_VISTA / 2 + RADIO_OBJETO, 4000, ALTO_VISTA / 2 + RADIO_OBJETO,
            QPen(QColor(theme.BORDER), 1))
        self._vista = QGraphicsView(self._escena)
        self._vista.setFixedHeight(ALTO_VISTA)
        self._vista.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._vista.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._vista.setRenderHint(QPainter.Antialiasing)
        raiz.addWidget(self._vista)

        # Los contenedores van en una rejilla que se repliega: con 10 clases y
        # una pantalla ancha queda una sola fila, con 20 clases o en un
        # portatil de 13 pulgadas se reparte en varias en vez de aplastarse.
        self._rejilla = QGridLayout()
        self._rejilla.setHorizontalSpacing(theme.SPACE * 2)
        self._rejilla.setVerticalSpacing(theme.SPACE // 2)
        self._filas: dict[str, ValueRow] = {}
        for clase in self._orden:
            self._filas[clase] = ValueRow(clase.replace("_", " "), "0")
        raiz.addLayout(self._rejilla)
        self._reacomodar(1)

    def _reacomodar(self, columnas: int) -> None:
        """Reparte los contenedores en `columnas` columnas."""
        if columnas == self._columnas:
            return
        self._columnas = columnas

        while self._rejilla.count():
            item = self._rejilla.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)

        for i, clase in enumerate(self._orden):
            self._rejilla.addWidget(self._filas[clase], i // columnas, i % columnas)
        for c in range(columnas):
            self._rejilla.setColumnStretch(c, 1)

    def resizeEvent(self, evento) -> None:
        """La banda ocupa siempre el ancho de la vista, sin barras de scroll."""
        super().resizeEvent(evento)
        ancho = max(self._vista.viewport().width(), 320)
        self._escena.setSceneRect(0, 0, ancho, ALTO_VISTA)
        self._linea.setLine(0, ALTO_VISTA / 2 + RADIO_OBJETO,
                            ancho, ALTO_VISTA / 2 + RADIO_OBJETO)
        self._reacomodar(max(1, min(len(self._orden),
                                    self.width() // ANCHO_CONTENEDOR)))

    # -- animacion ------------------------------------------------------
    def despachar(self, clase: str, via: str = "C", conocido: bool = True) -> None:
        """Lanza un objeto por la banda y lo suma a su contenedor.

        El color es el de la via que produjo la clasificacion mostrada, para
        que se vea de que representacion salio la decision. Una muestra
        rechazada viaja en rojo y cae en el contenedor de descarte, igual que
        en una banda real: el objeto no desaparece, se aparta.
        """
        if not conocido:
            clase = RECHAZO

        ancho = max(self._vista.viewport().width(), 320)
        color = theme.ERR if not conocido else theme.VIA_COLORS.get(via, theme.ACCENT)
        objeto = ObjetoEnBanda(color)
        objeto.setPos(QPointF(RADIO_OBJETO * 2, ALTO_VISTA / 2 - 8))
        self._escena.addItem(objeto)

        avanzar = QPropertyAnimation(objeto, b"pos")
        avanzar.setDuration(900)
        avanzar.setStartValue(objeto.pos())
        avanzar.setEndValue(QPointF(ancho - RADIO_OBJETO * 3, ALTO_VISTA / 2 - 8))
        avanzar.setEasingCurve(QEasingCurve.InOutQuad)

        caer = QPropertyAnimation(objeto, b"pos")
        caer.setDuration(320)
        caer.setStartValue(QPointF(ancho - RADIO_OBJETO * 3, ALTO_VISTA / 2 - 8))
        caer.setEndValue(QPointF(ancho - RADIO_OBJETO * 3, ALTO_VISTA + 40))
        caer.setEasingCurve(QEasingCurve.InQuad)

        grupo = QSequentialAnimationGroup(self)
        grupo.addAnimation(avanzar)
        grupo.addAnimation(caer)
        grupo.finished.connect(lambda: self._retirar(objeto, grupo))
        # Guardar la referencia: si el grupo se recolecta, la animacion muere.
        self._animaciones.append(grupo)
        grupo.start()

        self._conteo[clase] = self._conteo.get(clase, 0) + 1
        if clase in self._filas:
            self._filas[clase].set_value(str(self._conteo[clase]))

    def _retirar(self, objeto: ObjetoEnBanda, grupo) -> None:
        self._escena.removeItem(objeto)
        if grupo in self._animaciones:
            self._animaciones.remove(grupo)

    # -- estado ---------------------------------------------------------
    def conteo(self) -> dict[str, int]:
        return dict(self._conteo)

    def reiniciar(self) -> None:
        for clase in self._conteo:
            self._conteo[clase] = 0
            self._filas[clase].set_value("0", activo=False)
