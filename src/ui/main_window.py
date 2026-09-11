"""
Ventana principal de SCOAL.

    python -m src.ui.main_window

Estructura (seccion 9.3 del PRD):

    barra superior: dataset, via de la banda, camara, capturar, entrenar, analisis
    barra de estado: que esta haciendo el modelo y que espera el disparador
    panel de preparacion: visible solo si faltan datos, cache o modelo
    centro izquierda: feed con overlay de segmentacion y ejes principales
    centro derecha: predicciones de las tres vias y mediciones
    inferior: banda animada con el conteo por contenedor

Tres decisiones de interfaz que este archivo sostiene:

  1. **Nada invisible.** La clasificacion no ocurre en cada frame sino cuando
     el disparador de estabilidad lo permite; ese estado se muestra siempre,
     porque si no la aplicacion parece congelada.
  2. **Arranque guiado.** Una copia recien clonada no tiene datasets ni cache.
     En vez de fallar con un mensaje tecnico, la ventana dice que falta y
     ofrece prepararlo.
  3. **Tamanos flexibles.** Ningun ancho fijo que impida usar la aplicacion en
     un portatil de 13 pulgadas con escalado del sistema.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QApplication, QComboBox, QFrame, QHBoxLayout,
                               QLabel, QMainWindow, QPushButton, QSizePolicy,
                               QVBoxLayout, QWidget)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402
from src import setup  # noqa: E402
from src.data.loader import cache_path, load_dataset  # noqa: E402
from src.pipeline import MODEL_PATH, Scoal, medir  # noqa: E402
from src.ui import theme  # noqa: E402
from src.ui.analysis import AnalysisWindow  # noqa: E402
from src.ui.belt import Belt  # noqa: E402
from src.ui.camera import CameraThread  # noqa: E402
from src.ui.widgets import (CaptureStatus, FeedView, Panel,  # noqa: E402
                            PredictionRow, ValueRow, separador)

# Mediciones mostradas en el panel derecho: (etiqueta, clave, formato).
MEDICIONES = (
    ("Area", "area", "%.0f px2"),
    ("Perimetro", "perimetro", "%.0f px"),
    ("Aspecto", "aspecto", "%.2f"),
    ("Circularidad", "circularidad", "%.2f"),
    ("Eje mayor", "eje_mayor_px", "%.1f px"),
    ("Eje menor", "eje_menor_px", "%.1f px"),
)

# Geometria de la ventana. El maximo es una preferencia, el minimo un
# compromiso: por debajo de esto los contenedores de la banda se apilan.
ANCHO_PREFERIDO, ALTO_PREFERIDO = 1360, 860
ANCHO_MINIMO, ALTO_MINIMO = 900, 560


class SetupWorker(QThread):
    """Descarga, cachea y entrena sin bloquear la interfaz.

    Corre exactamente el mismo codigo que `python -m src.setup`, para que la
    ruta del boton y la ruta de la terminal no puedan divergir.
    """

    progreso = Signal(str)
    listo = Signal(object)
    fallo = Signal(str)

    def __init__(self, dataset: str, parent=None) -> None:
        super().__init__(parent)
        self.dataset = dataset

    def run(self) -> None:
        try:
            modelo = setup.preparar(self.dataset, self.progreso.emit)
            self.listo.emit(modelo)
        except Exception as exc:                     # la UI no debe morir
            self.fallo.emit(str(exc))


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
        except Exception as exc:
            self.fallo.emit(str(exc))


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SCOAL")
        self.setStyleSheet(theme.qss())
        self._ajustar_geometria()

        self.modelo: Scoal | None = None
        self.camara: CameraThread | None = None
        self.entrenador: TrainWorker | None = None
        self.preparador: SetupWorker | None = None
        self._ultima_medicion: dict | None = None

        raiz = QWidget()
        self.setCentralWidget(raiz)
        columna = QVBoxLayout(raiz)
        columna.setContentsMargins(theme.SPACE, theme.SPACE, theme.SPACE, theme.SPACE)
        columna.setSpacing(theme.SPACE)

        columna.addWidget(self._barra_superior())
        columna.addWidget(self._barra_estado())
        columna.addWidget(self._panel_preparacion())
        columna.addWidget(self._centro(), 1)
        columna.addWidget(self._panel_banda())

        self._cargar_modelo_guardado()
        self._revisar_preparacion()
        self._mostrar_muestra()

    def _ajustar_geometria(self) -> None:
        """Tamano inicial derivado de la pantalla, no una constante.

        Con `resize(1400, 880)` fijo la ventana no cabia en un portatil de 13
        pulgadas: el area util con escalado del 150% ronda 1280x720 logicos.
        """
        self.setMinimumSize(ANCHO_MINIMO, ALTO_MINIMO)
        pantalla = QApplication.primaryScreen()
        if pantalla is None:
            self.resize(ANCHO_PREFERIDO, ALTO_PREFERIDO)
            return
        util = pantalla.availableGeometry()
        self.resize(min(ANCHO_PREFERIDO, max(ANCHO_MINIMO, util.width() - 80)),
                    min(ALTO_PREFERIDO, max(ALTO_MINIMO, util.height() - 80)))

    # -- construccion ---------------------------------------------------
    def _barra_superior(self) -> QWidget:
        barra = QFrame()
        barra.setObjectName("Panel")
        fila = QHBoxLayout(barra)
        fila.setContentsMargins(theme.SPACE, theme.SPACE // 2,
                                theme.SPACE, theme.SPACE // 2)
        fila.setSpacing(theme.SPACE)

        titulo = QLabel("SCOAL")
        titulo.setObjectName("Title")

        self.combo_dataset = QComboBox()
        self.combo_dataset.addItems(config.DATASETS)
        self.combo_dataset.setCurrentText(config.DEFAULT_DATASET)
        self.combo_dataset.currentTextChanged.connect(self._cambio_dataset)

        self.combo_via = QComboBox()
        for via in config.VIAS:
            self.combo_via.addItem("via %s - %s" % (via, config.VIA_NOMBRES[via]), via)
        self.combo_via.setCurrentIndex(len(config.VIAS) - 1)

        self.boton_camara = QPushButton("CAMARA")
        self.boton_camara.setObjectName("Toggle")
        self.boton_camara.setCheckable(True)
        self.boton_camara.toggled.connect(self._alternar_camara)

        # Sin este boton la unica forma de clasificar es que la escena se quede
        # quieta cinco frames seguidos, que sostenido a pulso no pasa nunca.
        self.boton_capturar = QPushButton("CAPTURAR")
        self.boton_capturar.setToolTip(
            "Clasifica el frame actual sin esperar al disparador de estabilidad.")
        self.boton_capturar.clicked.connect(self._capturar_ahora)

        self.boton_entrenar = QPushButton("ENTRENAR")
        self.boton_entrenar.setObjectName("Primary")
        self.boton_entrenar.setToolTip(
            "Reentrena las tres vias con el dataset seleccionado y guarda\n"
            "results/scoal_model.pkl. Necesita el cache ya construido.")
        self.boton_entrenar.clicked.connect(self._entrenar)

        self.boton_analisis = QPushButton("ANALISIS")
        self.boton_analisis.setToolTip(
            "Abre las figuras del informe (curvas de ablacion, confusion,\n"
            "dispersion PCA) leidas de los CSV de results/. No recalcula nada.")
        self.boton_analisis.clicked.connect(self._abrir_analisis)

        for w in (titulo, self.combo_dataset, self.combo_via):
            fila.addWidget(w)
        fila.addStretch(1)
        for w in (self.boton_camara, self.boton_capturar,
                  self.boton_entrenar, self.boton_analisis):
            fila.addWidget(w)
        return barra

    def _barra_estado(self) -> QWidget:
        """Fila propia para el estado: en la barra superior se quedaba sin sitio."""
        barra = QFrame()
        barra.setObjectName("Panel")
        fila = QHBoxLayout(barra)
        fila.setContentsMargins(theme.SPACE, theme.SPACE // 2,
                                theme.SPACE, theme.SPACE // 2)
        fila.setSpacing(theme.SPACE * 2)

        self.estado = QLabel("iniciando")
        self.estado.setObjectName("Status")
        self.estado.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        self.captura = CaptureStatus()
        self.captura.setFixedWidth(260)

        fila.addWidget(self.estado, 1)
        fila.addWidget(self.captura)
        return barra

    def _panel_preparacion(self) -> QWidget:
        """Aviso de arranque: que falta para poder usar la aplicacion.

        Es lo primero que ve quien clona el repositorio. Los datasets pesan
        cerca de 1 GB y no estan versionados, asi que sin este panel la
        aplicacion arranca vacia y el unico camino es leer el README.
        """
        self.panel_prep = Panel("Preparacion")
        self.texto_prep = QLabel()
        self.texto_prep.setObjectName("Status")
        self.texto_prep.setWordWrap(True)

        self.boton_preparar = QPushButton("PREPARAR DATOS")
        self.boton_preparar.setObjectName("Primary")
        self.boton_preparar.clicked.connect(self._preparar)

        fila = QHBoxLayout()
        fila.setSpacing(theme.SPACE * 2)
        fila.addWidget(self.texto_prep, 1)
        fila.addWidget(self.boton_preparar, 0, Qt.AlignTop)
        self.panel_prep.add_layout(fila)
        return self.panel_prep

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
        # Rango en vez de ancho fijo: se encoge en pantallas pequenas y no se
        # estira sin limite en pantallas grandes.
        derecha.setMinimumWidth(280)
        derecha.setMaximumWidth(400)

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

        panel.add(separador())
        self.fila_novedad = ValueRow("Novedad")
        self.fila_novedad.setToolTip(
            "Distancia de la muestra al dominio entrenado, normalizada al\n"
            "umbral: por encima de 1.00 se rechaza como desconocido.\n"
            "Corresponde a la via seleccionada en la barra superior.")
        panel.add(self.fila_novedad)
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

    # -- preparacion ----------------------------------------------------
    def _revisar_preparacion(self) -> None:
        """Muestra u oculta el panel de arranque segun lo que haya en disco."""
        dataset = self.combo_dataset.currentText()
        est = setup.estado(dataset)

        self.panel_prep.setVisible(not est["listo"])
        self.boton_entrenar.setEnabled(est["cache"])
        if est["listo"]:
            return

        detalle = {
            "datos": "las imagenes de %s (~1 GB, se descargan una sola vez)" % dataset,
            "cache": "el cache segmentado de %s" % dataset,
            "modelo": "el modelo entrenado",
        }
        faltantes = "\n".join("  - falta %s" % detalle[f] for f in est["faltan"])
        self.texto_prep.setText(
            "Esta copia todavia no puede clasificar:\n%s\n\n"
            "Pulsa PREPARAR DATOS, o desde la terminal:  "
            "python -m src.setup --dataset %s" % (faltantes, dataset))
        self.estado.setText(setup.resumen(dataset))

    def _preparar(self) -> None:
        if self.preparador is not None and self.preparador.isRunning():
            return
        self.boton_preparar.setEnabled(False)
        self.boton_preparar.setText("PREPARANDO...")

        self.preparador = SetupWorker(self.combo_dataset.currentText(), self)
        self.preparador.progreso.connect(self.estado.setText)
        self.preparador.listo.connect(self._preparacion_lista)
        self.preparador.fallo.connect(self._trabajo_fallo)
        self.preparador.finished.connect(self._preparacion_termino)
        self.preparador.start()

    def _preparacion_lista(self, modelo: Scoal | None) -> None:
        if modelo is not None:
            self._modelo_listo(modelo, "preparado")
        else:
            self._cargar_modelo_guardado()
        self._mostrar_muestra()

    def _preparacion_termino(self) -> None:
        self.boton_preparar.setEnabled(True)
        self.boton_preparar.setText("PREPARAR DATOS")
        self._revisar_preparacion()

    def _cambio_dataset(self, _texto: str) -> None:
        self._revisar_preparacion()

    # -- modelo ---------------------------------------------------------
    def _clases(self) -> list[str]:
        if self.modelo is not None:
            return self.modelo.clases
        return list(config.FRUITS_CLASSES)

    def _cargar_modelo_guardado(self) -> None:
        if not MODEL_PATH.exists():
            self.estado.setText("sin modelo entrenado")
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
        self.entrenador.fallo.connect(self._trabajo_fallo)
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
        self.estado.setStyleSheet("")

        # Reconstruir la banda: los contenedores son las clases del modelo.
        nueva = Belt(modelo.clases)
        panel = self.banda.parentWidget()
        vieja = self.banda
        panel.layout().replaceWidget(vieja, nueva)
        # replaceWidget solo la saca del layout: sigue siendo hija del panel y
        # se sigue pintando sobre la nueva hasta que deleteLater se procese.
        # Desparentarla ahora la quita de la pantalla en el acto.
        vieja.setParent(None)
        vieja.deleteLater()
        self.banda = nueva
        nueva.show()

    def _trabajo_fallo(self, mensaje: str) -> None:
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
            self.estado.setText("camara activa: deja el objeto quieto para clasificar")
        elif self.camara is not None:
            self.camara.stop()
            self.camara = None
            self.captura.set_estado("camara apagada")
            self.estado.setText("camara detenida")
            self._mostrar_muestra()

    def _camara_fallo(self, mensaje: str) -> None:
        self._trabajo_fallo("camara: %s" % mensaje)
        self.boton_camara.setChecked(False)

    def _nuevo_frame(self, frame: np.ndarray, medicion: dict) -> None:
        self._ultima_medicion = medicion
        self.feed.set_frame(frame, medicion)
        self._mostrar_mediciones(medicion)

        estabilidad = medicion.get("estabilidad")
        if estabilidad:
            self.captura.set_estado(estabilidad["texto"], estabilidad["quietos"],
                                    estabilidad["frames"], estabilidad["diferencia"])

    def _capturar_ahora(self) -> None:
        """Clasifica el frame actual saltandose el disparador de estabilidad."""
        if self._ultima_medicion is None:
            self.estado.setText("no hay frame que capturar")
            return
        if not self._ultima_medicion.get("ok"):
            self.estado.setText("no hay objeto segmentado en el frame actual")
            return
        self._clasificar(self._ultima_medicion)

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
        """Clasificar por las tres vias y despachar en la banda."""
        if self.modelo is None:
            self.estado.setText("no hay modelo cargado: pulsa ENTRENAR")
            return
        if not medicion.get("ok"):
            return

        predicciones = self.modelo.classify(medicion)
        for via, fila in self.filas_prediccion.items():
            p = predicciones.get(via)
            if p is None:
                fila.set_prediction(None, None)
            else:
                fila.set_prediction(p["clase"], p["confianza"],
                                    p["conocido"], p["novedad"])

        # La banda despacha segun la via seleccionada; si esa via no esta
        # entrenada se usa la primera disponible.
        via_banda = self.combo_via.currentData()
        if via_banda not in predicciones and predicciones:
            via_banda = sorted(predicciones)[0]
        if via_banda not in predicciones:
            return

        elegida = predicciones[via_banda]
        self.fila_novedad.set_value("%.2f" % elegida["novedad"],
                                    activo=not elegida["conocido"])
        self.banda.despachar(elegida["clase"], via_banda, elegida["conocido"])

        if not elegida["conocido"]:
            self.estado.setText(
                "rechazado: fuera del dominio entrenado "
                "(novedad %.2f, confianza %.2f)"
                % (elegida["novedad"], elegida["confianza"]))
        else:
            self.estado.setText("clasificado por via %s: %s"
                                % (via_banda, elegida["clase"]))

    # -- ciclo de vida --------------------------------------------------
    def _mostrar_muestra(self) -> None:
        """Sin camara, mostrar una imagen real del cache en vez de un vacio.

        Es un dato del dataset, no un marcador de posicion: permite probar
        segmentacion, mediciones y CAPTURAR sin webcam.
        """
        if self.camara is not None:
            return
        ruta = cache_path(self.combo_dataset.currentText())
        if not ruta.exists():
            self.feed.setText("SIN DATOS")
            return
        with np.load(ruta) as z:
            crop = z["crops"][0]
        grande = cv2.resize(crop, (480, 480), interpolation=cv2.INTER_NEAREST)
        self._nuevo_frame(grande, medir(grande))
        self.captura.set_estado("camara apagada")

    def _abrir_analisis(self) -> None:
        AnalysisWindow(self).show()

    def closeEvent(self, evento) -> None:
        if self.camara is not None:
            self.camara.stop()
        for hilo in (self.entrenador, self.preparador):
            if hilo is not None and hilo.isRunning():
                hilo.wait(3000)
        super().closeEvent(evento)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(theme.qss())
    ventana = MainWindow()
    ventana.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
