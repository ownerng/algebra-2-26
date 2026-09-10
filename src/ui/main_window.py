"""
Ventana principal de SCOAL.

    python -m src.ui.main_window

Estructura (seccion 9.3 del PRD):

    barra superior: dataset, via de la banda, camara, entrenar, analisis
    centro izquierda: feed con overlay de segmentacion y ejes principales
    centro derecha: predicciones de las tres vias y mediciones
    inferior: banda animada con el conteo por contenedor
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QHBoxLayout,
                               QLabel, QMainWindow, QPushButton, QVBoxLayout,
                               QWidget)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from src.data.loader import load_dataset  # noqa: E402
from src.pipeline import MODEL_PATH, Scoal  # noqa: E402
from src.ui import theme  # noqa: E402
from src.ui.analysis import AnalysisWindow  # noqa: E402
from src.ui.belt import Belt  # noqa: E402
from src.ui.camera import CameraThread  # noqa: E402
from src.ui.widgets import (FeedView, Panel, PredictionRow, ValueRow,  # noqa: E402
                            separador)

# Mediciones mostradas en el panel derecho: (etiqueta, clave, formato).
MEDICIONES = (
    ("Area", "area", "%.0f px2"),
    ("Perimetro", "perimetro", "%.0f px"),
    ("Aspecto", "aspecto", "%.2f"),
    ("Circularidad", "circularidad", "%.2f"),
    ("Eje mayor", "eje_mayor_px", "%.1f px"),
    ("Eje menor", "eje_menor_px", "%.1f px"),
)


class TrainWorker(QThread):
    """Entrena las tres vias sin bloquear la interfaz."""

    progreso = Signal(str)
    listo = Signal(object)
    fallo = Signal(str)

    def __init__(self, dataset: str, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset

    def run(self) -> None:
        try:
            self.progreso.emit("cargando %s" % self.dataset)
            datos = load_dataset(self.dataset)

            self.progreso.emit("extrayendo caracteristicas (%d imagenes)"
                               % len(datos["y"]))
            modelo = Scoal(datos["clases"])
            modelo.fit(datos, self.dataset)

            if not modelo.vias:
                self.fallo.emit("ninguna via pudo entrenarse")
                return

            modelo.save(MODEL_PATH)
            self.progreso.emit("entrenado: vias %s" % ", ".join(sorted(modelo.vias)))
            self.listo.emit(modelo)
        except Exception as exc:                     # la UI no debe morir
            self.fallo.emit(str(exc))


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SCOAL")
        self.resize(1400, 880)
        self.setStyleSheet(theme.qss())

        self.modelo: Scoal | None = None
        self.camara: CameraThread | None = None
        self.entrenador: TrainWorker | None = None
        self._ultima_medicion: dict | None = None

        raiz = QWidget()
        self.setCentralWidget(raiz)
        columna = QVBoxLayout(raiz)
        columna.setContentsMargins(theme.SPACE, theme.SPACE, theme.SPACE, theme.SPACE)
        columna.setSpacing(theme.SPACE)

        columna.addWidget(self._barra_superior())
        columna.addWidget(self._centro(), 1)
        columna.addWidget(self._panel_banda())

        self._cargar_modelo_guardado()

    # -- construccion ---------------------------------------------------
    def _barra_superior(self) -> QWidget:
        barra = QFrame()
        barra.setObjectName("Panel")
        fila = QHBoxLayout(barra)
        fila.setContentsMargins(theme.SPACE, theme.SPACE // 2,
                                theme.SPACE, theme.SPACE // 2)
        fila.setSpacing(theme.SPACE * 2)

        titulo = QLabel("SCOAL")
        titulo.setObjectName("Title")

        self.combo_dataset = QComboBox()
        self.combo_dataset.addItems(config.DATASETS)
        self.combo_dataset.setCurrentText(config.DEFAULT_DATASET)

        self.combo_via = QComboBox()
        for via in config.VIAS:
            self.combo_via.addItem("via %s - %s" % (via, config.VIA_NOMBRES[via]), via)
        self.combo_via.setCurrentIndex(len(config.VIAS) - 1)

        self.boton_camara = QPushButton("CAMARA")
        self.boton_camara.setObjectName("Toggle")
        self.boton_camara.setCheckable(True)
        self.boton_camara.toggled.connect(self._alternar_camara)

        self.boton_entrenar = QPushButton("ENTRENAR")
        self.boton_entrenar.setObjectName("Primary")
        self.boton_entrenar.clicked.connect(self._entrenar)

        self.boton_analisis = QPushButton("ANALISIS")
        self.boton_analisis.clicked.connect(self._abrir_analisis)

        self.estado = QLabel("sin modelo")
        self.estado.setObjectName("Status")

        fila.addWidget(titulo)
        fila.addWidget(self.combo_dataset)
        fila.addWidget(self.combo_via)
        fila.addWidget(self.boton_camara)
        fila.addStretch(1)
        fila.addWidget(self.estado)
        fila.addWidget(self.boton_entrenar)
        fila.addWidget(self.boton_analisis)
        return barra

    def _centro(self) -> QWidget:
        contenedor = QWidget()
        fila = QHBoxLayout(contenedor)
        fila.setContentsMargins(0, 0, 0, 0)
        fila.setSpacing(theme.SPACE)

        panel_feed = Panel("Feed")
        self.feed = FeedView()
        panel_feed.add(self.feed, 1)

        derecha = QWidget()
        columna = QVBoxLayout(derecha)
        columna.setContentsMargins(0, 0, 0, 0)
        columna.setSpacing(theme.SPACE)
        columna.addWidget(self._panel_predicciones())
        columna.addWidget(self._panel_mediciones())
        columna.addStretch(1)
        derecha.setFixedWidth(380)

        fila.addWidget(panel_feed, 1)
        fila.addWidget(derecha)
        return contenedor

    def _panel_predicciones(self) -> Panel:
        panel = Panel("Predicciones")
        self.filas_prediccion: dict[str, PredictionRow] = {}
        for i, via in enumerate(config.VIAS):
            if i:
                panel.add(separador())
            fila = PredictionRow(via, config.VIA_NOMBRES[via])
            fila.set_disponible(False)
            self.filas_prediccion[via] = fila
            panel.add(fila)
        return panel

    def _panel_mediciones(self) -> Panel:
        panel = Panel("Mediciones")
        self.filas_medicion: dict[str, ValueRow] = {}
        for etiqueta, clave, _fmt in MEDICIONES:
            fila = ValueRow(etiqueta)
            self.filas_medicion[clave] = fila
            panel.add(fila)

        panel.add(separador())
        self.fila_rgb = ValueRow("RGB")
        panel.add(self.fila_rgb)

        # El peso queda cableado y visible pero inactivo: los datasets
        # publicos no traen masa (config.USE_WEIGHT).
        self.fila_peso = ValueRow("Peso")
        panel.add(self.fila_peso)
        return panel

    def _panel_banda(self) -> Panel:
        panel = Panel("Banda")
        self.banda = Belt(self._clases())
        panel.add(self.banda)
        return panel

    # -- modelo ---------------------------------------------------------
    def _clases(self) -> list[str]:
        if self.modelo is not None:
            return self.modelo.clases
        return list(config.FRUITS_CLASSES)

    def _cargar_modelo_guardado(self) -> None:
        if not MODEL_PATH.exists():
            self.estado.setText("sin modelo, pulsar ENTRENAR")
            return
        try:
            self.modelo = Scoal.load(MODEL_PATH)
            self._modelo_listo(self.modelo, "modelo cargado de disco")
        except Exception as exc:
            self.estado.setText("modelo ilegible: %s" % exc)

    def _entrenar(self) -> None:
        if self.entrenador is not None and self.entrenador.isRunning():
            return
        self.boton_entrenar.setEnabled(False)
        self.entrenador = TrainWorker(self.combo_dataset.currentText(), self)
        self.entrenador.progreso.connect(self.estado.setText)
        self.entrenador.listo.connect(lambda m: self._modelo_listo(m, "entrenado"))
        self.entrenador.fallo.connect(self._entrenamiento_fallo)
        self.entrenador.finished.connect(
            lambda: self.boton_entrenar.setEnabled(True))
        self.entrenador.start()

    def _modelo_listo(self, modelo: Scoal, mensaje: str) -> None:
        self.modelo = modelo
        for via, fila in self.filas_prediccion.items():
            fila.set_disponible(via in modelo.vias)
        faltan = [v for v in config.VIAS if v not in modelo.vias]
        texto = "%s: %d clases, vias %s" % (mensaje, len(modelo.clases),
                                            ", ".join(sorted(modelo.vias)))
        if faltan:
            texto += "  (sin %s)" % ", ".join(faltan)
        self.estado.setText(texto)

        # Reconstruir la banda: los contenedores son las clases del modelo.
        nueva = Belt(modelo.clases)
        panel = self.banda.parentWidget()
        panel.layout().replaceWidget(self.banda, nueva)
        self.banda.deleteLater()
        self.banda = nueva

    def _entrenamiento_fallo(self, mensaje: str) -> None:
        self.estado.setText("error: %s" % mensaje)
        self.estado.setStyleSheet("color: %s;" % theme.ERR)

    # -- camara ---------------------------------------------------------
    def _alternar_camara(self, encendida: bool) -> None:
        if encendida:
            self.camara = CameraThread(0, self)
            self.camara.frameReady.connect(self._nuevo_frame)
            self.camara.triggered.connect(self._clasificar)
            self.camara.failed.connect(self._camara_fallo)
            self.camara.start()
            self.estado.setText("camara activa")
        elif self.camara is not None:
            self.camara.stop()
            self.camara = None
            self.feed.setText("SIN SENAL")
            self.estado.setText("camara detenida")

    def _camara_fallo(self, mensaje: str) -> None:
        self.estado.setText("camara: %s" % mensaje)
        self.boton_camara.setChecked(False)

    def _nuevo_frame(self, frame: np.ndarray, medicion: dict) -> None:
        self._ultima_medicion = medicion
        self.feed.set_frame(frame, medicion)
        self._mostrar_mediciones(medicion)

    def _mostrar_mediciones(self, medicion: dict) -> None:
        medidas = medicion.get("mediciones", {})
        for _etiqueta, clave, fmt in MEDICIONES:
            fila = self.filas_medicion[clave]
            if clave in medidas:
                fila.set_value(fmt % medidas[clave])
            else:
                fila.set_value("--", activo=False)

        if medidas:
            self.fila_rgb.set_value("%3.0f / %3.0f / %3.0f"
                                    % (medidas["R_med"], medidas["G_med"],
                                       medidas["B_med"]))
        else:
            self.fila_rgb.set_value("--", activo=False)

        self.fila_peso.set_value(
            "%.1f g" % medidas["peso_g"] if config.USE_WEIGHT and "peso_g" in medidas
            else "--", activo=config.USE_WEIGHT)

    def _clasificar(self, medicion: dict) -> None:
        """Disparo de estabilidad: clasificar por las tres vias."""
        if self.modelo is None or not medicion.get("ok"):
            return

        predicciones = self.modelo.classify(medicion)
        for via, fila in self.filas_prediccion.items():
            if via in predicciones:
                clase, confianza = predicciones[via]
                fila.set_prediction(clase, confianza)
            else:
                fila.set_prediction(None, None)

        # La banda despacha segun la via seleccionada; si esa via no esta
        # entrenada se usa la primera disponible.
        via_banda = self.combo_via.currentData()
        if via_banda not in predicciones and predicciones:
            via_banda = sorted(predicciones)[0]
        if via_banda in predicciones:
            self.banda.despachar(predicciones[via_banda][0], via_banda)

    # -- ciclo de vida --------------------------------------------------
    def _abrir_analisis(self) -> None:
        AnalysisWindow(self).show()

    def closeEvent(self, evento) -> None:
        if self.camara is not None:
            self.camara.stop()
        if self.entrenador is not None and self.entrenador.isRunning():
            self.entrenador.wait(3000)
        super().closeEvent(evento)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(theme.qss())
    ventana = MainWindow()
    ventana.show()

    # Sin camara, mostrar una imagen del dataset para que la ventana no
    # arranque vacia: es un dato real, no un marcador de posicion.
    _mostrar_muestra(ventana)

    sys.exit(app.exec())


def _mostrar_muestra(ventana: MainWindow) -> None:
    from src.data.loader import cache_path
    ruta = cache_path(config.DEFAULT_DATASET)
    if not ruta.exists():
        return
    with np.load(ruta) as z:
        crop = z["crops"][0]
    grande = cv2.resize(crop, (480, 480), interpolation=cv2.INTER_NEAREST)
    from src.pipeline import medir
    ventana._nuevo_frame(grande, medir(grande))


if __name__ == "__main__":
    main()
