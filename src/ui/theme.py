"""
Tokens de diseno y generacion del QSS.

Fuente unica de verdad del aspecto de la aplicacion: ningun widget escribe un
color a mano. Tambien alimenta los rcParams de matplotlib para que las figuras
embebidas y las del informe usen exactamente la misma paleta.

Direccion visual: consola de instrumentacion tecnica. Oscura, plana, densa en
datos, tipografia fuerte, un solo color de acento, cero decoracion.
"""

from __future__ import annotations

BG_BASE = "#14161A"
BG_PANEL = "#1C1F25"
BG_ELEVATED = "#242830"
BORDER = "#2F343D"
TEXT_HI = "#E8EAEE"
TEXT_MID = "#98A0AB"
TEXT_LO = "#5E6570"
ACCENT = "#E8734A"      # UNICO acento de interfaz
VIA_A = "#5FB37A"       # verde  - features fisicas
VIA_B = "#D9A441"       # ambar  - eigen-objetos
VIA_C = "#5B9DD9"       # azul   - embeddings CNN
ERR = "#D9534F"

FONT_UI = "Inter, DejaVu Sans, sans-serif"
FONT_MONO = "JetBrains Mono, DejaVu Sans Mono, monospace"
SPACE = 8               # todo el spacing es multiplo de 8
RADIUS = 4              # maximo absoluto

# Los tres colores de via son codificacion de datos, no decoracion: son la
# unica excepcion a la regla de un solo acento y se usan igual en la interfaz,
# en las graficas y en las leyendas.
VIA_COLORS = {"A": VIA_A, "B": VIA_B, "C": VIA_C}


def qss() -> str:
    """Hoja de estilo completa de la aplicacion, generada desde los tokens."""
    return """
QWidget {
    background: %(bg)s;
    color: %(text_hi)s;
    font-family: %(font_ui)s;
    font-size: 12px;
}

QFrame#Panel {
    background: %(panel)s;
    border: 1px solid %(border)s;
    border-radius: %(radius)dpx;
}

QFrame#Separator {
    background: %(border)s;
    max-height: 1px;
    border: none;
}

QLabel#Title {
    color: %(text_hi)s;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1px;
}

QLabel#SectionLabel {
    color: %(text_mid)s;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

QLabel#Value {
    font-family: %(font_mono)s;
    font-size: 12px;
    color: %(text_hi)s;
}

QLabel#ValueMuted {
    font-family: %(font_mono)s;
    font-size: 12px;
    color: %(text_lo)s;
}

QLabel#Status {
    font-family: %(font_mono)s;
    font-size: 11px;
    color: %(text_mid)s;
}

QPushButton {
    background: %(elevated)s;
    color: %(text_hi)s;
    border: 1px solid %(border)s;
    border-radius: %(radius)dpx;
    padding: %(pad)dpx %(pad2)dpx;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
}
QPushButton:hover   { border-color: %(accent)s; }
QPushButton:pressed { background: %(panel)s; }
QPushButton:disabled { color: %(text_lo)s; border-color: %(border)s; }
QPushButton#Primary {
    background: %(accent)s;
    border-color: %(accent)s;
    color: %(bg)s;
}

QPushButton#Toggle:checked {
    border-color: %(accent)s;
    color: %(accent)s;
}

QComboBox {
    background: %(elevated)s;
    border: 1px solid %(border)s;
    border-radius: %(radius)dpx;
    padding: %(pad)dpx %(pad)dpx;
    min-width: 140px;
    font-size: 11px;
}
QComboBox:hover { border-color: %(accent)s; }
QComboBox::drop-down { border: none; width: 16px; }
QComboBox QAbstractItemView {
    background: %(elevated)s;
    border: 1px solid %(border)s;
    selection-background-color: %(accent)s;
    selection-color: %(bg)s;
    outline: none;
}

QTabWidget::pane { border: 1px solid %(border)s; }
QTabBar::tab {
    background: %(panel)s;
    color: %(text_mid)s;
    border: 1px solid %(border)s;
    padding: %(pad)dpx %(pad2)dpx;
    font-size: 11px;
    letter-spacing: 0.5px;
}
QTabBar::tab:selected { color: %(text_hi)s; border-bottom-color: %(accent)s; }

QGraphicsView { background: %(panel)s; border: 1px solid %(border)s; }

QScrollBar:vertical, QScrollBar:horizontal {
    background: %(bg)s; width: 10px; height: 10px; border: none;
}
QScrollBar::handle { background: %(border)s; border-radius: 2px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
""" % {
        "bg": BG_BASE, "panel": BG_PANEL, "elevated": BG_ELEVATED,
        "border": BORDER, "text_hi": TEXT_HI, "text_mid": TEXT_MID,
        "text_lo": TEXT_LO, "accent": ACCENT,
        "font_ui": FONT_UI, "font_mono": FONT_MONO,
        "radius": RADIUS, "pad": SPACE // 2, "pad2": SPACE + SPACE // 2,
    }


def apply_matplotlib(plt) -> None:
    """Alinea matplotlib a la paleta. Mismas figuras en la app y en el informe."""
    plt.rcParams.update({
        "figure.facecolor": BG_PANEL,
        "axes.facecolor": BG_PANEL,
        "savefig.facecolor": BG_PANEL,
        "axes.edgecolor": BORDER,
        "axes.labelcolor": TEXT_MID,
        "axes.titlecolor": TEXT_HI,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": BORDER,
        "grid.linewidth": 0.6,
        "text.color": TEXT_HI,
        "xtick.color": TEXT_MID,
        "ytick.color": TEXT_MID,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "figure.dpi": 130,
    })
