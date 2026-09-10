"""
Ventana de analisis: las mismas figuras del informe, embebidas en la aplicacion.

No recalcula nada. Lee los CSV que dejo el estudio de ablacion y usa los
mismos constructores de figuras que generan los PNG de docs/figuras, de modo
que lo que se ve en pantalla y lo que se imprime en el informe no pueden
divergir.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (QDialog, QLabel, QTabWidget, QVBoxLayout,
                               QWidget)

import config
from src.experiments import ablation
from src.ui import theme


class Lienzo(FigureCanvasQTAgg):
    """Un FigureCanvasQTAgg que se dibuja con un constructor de ablation.py."""

    def __init__(self, constructor, tam: tuple[float, float]) -> None:
        figura = Figure(figsize=tam)
        super().__init__(figura)
        self.dibujado = constructor(figura)
        if self.dibujado:
            figura.tight_layout()


class AnalysisWindow(QDialog):
    """Pestanas con las curvas de ablacion y las figuras cualitativas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("SCOAL - Analisis")
        self.resize(1000, 720)
        self.setStyleSheet(theme.qss())

        import matplotlib.pyplot as plt
        theme.apply_matplotlib(plt)

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(theme.SPACE, theme.SPACE, theme.SPACE, theme.SPACE)

        pestanas = QTabWidget()
        raiz.addWidget(pestanas)

        vacias = 0
        for _nombre, titulo, constructor, tam in ablation.FIGURAS:
            lienzo = Lienzo(constructor, tam)
            if not lienzo.dibujado:
                vacias += 1
                continue
            pestanas.addTab(lienzo, titulo)

        if pestanas.count() == 0:
            aviso = QLabel(
                "No hay resultados en %s.\n"
                "Ejecutar:  python -m src.experiments.ablation"
                % config.RESULTS_DIR)
            aviso.setObjectName("Status")
            raiz.addWidget(aviso)
        elif vacias:
            aviso = QLabel("%d figuras sin datos todavia" % vacias)
            aviso.setObjectName("Status")
            raiz.addWidget(aviso)
