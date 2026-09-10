# PRD — SCOAL v2.2
## Sistema de Clasificación de Objetos por Álgebra Lineal

**Asignatura:** Álgebra Lineal
**Institución:** Universidad Cooperativa de Colombia
**Autor:** Jhonnatan
**Versión:** 2.2

---

## 1. Objetivo del producto

Sistema que clasifica objetos por tamaño, color, forma e identidad, con el motor
de clasificación implementado en NumPy puro, comparando empíricamente tres
representaciones de características bajo un mismo clasificador de mínimos cuadrados.

## 2. Pregunta de investigación

> ¿Qué tan lejos llega un clasificador lineal de mínimos cuadrados, y cuánto de su
> desempeño depende de la representación de entrada en lugar del clasificador mismo?

## 3. Alcance

### Dentro
- Datasets públicos: COIL-100 y Fruits-360, con descarga automatizada
- Tres extractores de características (vías A, B, C)
- Clasificador único de mínimos cuadrados con regularización de Tikhonov, NumPy puro
- Estudio de ablación: accuracy vs componentes PCA, vs lambda, vs vía
- Aplicación de escritorio en PySide6 con banda animada y panel de análisis
- Modo webcam en vivo con las tres predicciones simultáneas
- Andamiaje del informe académico en LaTeX, normas APA 7ª edición

### Fuera
- Captura manual de dataset
- Hardware físico (motores, servos, celdas de carga)
- Peso como característica activa (cableado pero desactivado, ver §8)
- Entrenamiento de redes neuronales (solo extractor congelado)
- Múltiples objetos por frame
- **Operaciones de git, creación de repositorio remoto o publicación.**
  El agente genera estructura local, `.gitignore` y `README.md`. Nada más.
  No ejecutar `git init`, `git remote`, `gh repo create` ni equivalentes.

---

## 4. Datasets

| Dataset | Uso | Volumen | Notas |
|---|---|---|---|
| COIL-100 | Demostración canónica de eigen-objetos | 7.200 img, 100 objetos, ~125 MB | Plato giratorio motorizado, fondo negro, 72 vistas por objeto a intervalos de 5° |
| Fruits-360 | Caso de uso tipo banda transportadora | 106.671 img, 160 clases, 100×100 px | Objetos rotando sobre motor a 3 rpm frente a cámara fija, fondo blanco |

### Subconjunto de trabajo

No usar las 160 clases. Seleccionar **10 clases de Fruits-360** con solapamiento
deliberado, para forzar al modelo a usar más de una característica:

```
manzana_golden, manzana_granny_smith, manzana_red_delicious,
naranja, mandarina, limon, lima,
tomate_cherry, papa_roja, zucchini
```

De COIL-100: 20 objetos. Balancear a ~300 imágenes por clase.

`src/data/download.py` descarga y descomprime automáticamente. Un solo comando.

---

## 5. Arquitectura del pipeline

```
Imagen (dataset o webcam)
    |
    +--[0] (solo webcam) Disparador de estabilidad
    |      diff(frame_t, frame_t-1) < umbral por N frames
    |      + existe contorno valido -> dispara
    |
    +--[1] Segmentacion
    |      RGB->YCbCr (matriz 3x3) -> umbral croma -> mascara -> contorno
    |
    +--[2] Recorte con mascara, fondo neutro, normalizacion
    |
    +------------- TRES VIAS -------------
    |
    +--[A] Features fisicas (13-D) ------- NumPy puro
    |      area, perimetro, aspecto, circularidad,
    |      R_med G_med B_med, Hu[7], lambda1 lambda2 (autovalores inercia)
    |
    +--[B] Eigen-objetos (k-D) ----------- NumPy puro
    |      64x64 gris -> 4096-D -> centrado -> SVD
    |      (Turk-Pentland si m << n) -> k componentes
    |
    +--[C] Embeddings CNN (60-D) --------- modelo prestado
    |      MobileNetV3-Small preentrenado, CONGELADO
    |      penultima capa -> 1280-D -> PCA PROPIA -> 60-D
    |
    +--[3] Para cada via, por separado:
           estandarizar -> W = (X^T X + lambda*I)^-1 X^T Y -> argmax(W^T x)
```

**Nota técnica obligatoria en el informe:** la conversión RGB→HSV no es una
transformación lineal (usa mínimo, máximo y divisiones). Por eso se usa YCbCr,
que sí es una matriz 3×3 exacta. Documentar esta decisión.

---

## 6. Baselines de comparación

| Baseline | Qué mide |
|---|---|
| Clasificación aleatoria | Piso absoluto |
| k-NN (norma euclidiana, NumPy) por vía | ¿El clasificador lineal aporta sobre memorizar? |
| MobileNetV3 completo con su cabeza original | Techo del modelo preentrenado |
| Mínimos cuadrados propio sobre vía C | Cuánto se pierde al reemplazar la cabeza del CNN |

---

## 7. Estudio de ablación

Tres barridos. Cada uno produce CSV y figura.

1. **Accuracy vs k** (componentes PCA): k ∈ {5, 10, 20, 30, 50, 80, 120}, vías B y C
2. **Accuracy vs lambda** (Tikhonov): lambda ∈ {0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10},
   con número de condición de XᵀX reportado en cada punto
3. **Accuracy vs vía**, validación cruzada 5-fold, media ± desviación estándar

Las figuras se generan **desde el CSV**, no re-entrenando. Reproducibles.

---

## 8. Especificaciones del motor

```python
# src/model/lstsq.py
def fit_least_squares(X: np.ndarray, Y: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """
    Resuelve  min_W ||X W - Y||_F^2 + lambda * ||W||_F^2
    Solucion cerrada:  W = (X^T X + lambda I)^-1 X^T Y

    X: (m, n) matriz de diseno, incluye columna de unos
    Y: (m, k) etiquetas one-hot
    Retorna W: (n, k)

    Usar np.linalg.solve, NUNCA np.linalg.inv (estabilidad numerica).
    PROHIBIDO sklearn.
    """

def predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Retorna (m,) indices de clase via argmax(X @ W, axis=1)."""


# src/features/eigen.py
def fit_eigenobjects(images: np.ndarray, k: int) -> dict:
    """
    images: (m, d) imagenes aplanadas
    Centrar restando la media. Sea A la matriz centrada (m, d).

    Si m < d, usar truco de Turk-Pentland (1991):
      calcular autovectores u_i de A A^T  (m x m, manejable)
      mapear:  v_i = A^T u_i / ||A^T u_i||
      donde v_i son los autovectores de A^T A (d x d, inmanejable)

    Retorna {'mean': (d,), 'components': (d, k), 'explained_var': (k,)}
    Implementar con np.linalg.svd o np.linalg.eigh.
    """


# src/features/physical.py
def inertia_axes(mask: np.ndarray) -> tuple[float, float, float]:
    """
    Construye la matriz de inercia 2x2 desde los momentos centrales
    de segundo orden:
      M = [[mu20, mu11], [mu11, mu02]]
    Sus autovalores dan las longitudes de los ejes principales,
    sus autovectores dan la orientacion.
    Retorna (eje_mayor, eje_menor, angulo_rad) via np.linalg.eigh.
    """

def extract_physical(mask, image_bgr, peso_g: float | None = None) -> np.ndarray:
    """
    Si USE_WEIGHT es False o peso_g es None, la componente de peso NO entra
    al vector. El orden de las features debe ser IDENTICO en train e inferencia.
    Existe un test que lo verifica.
    """


# src/features/cnn.py
def extract_embeddings(images: np.ndarray, model) -> np.ndarray:
    """
    MobileNetV3-Small preentrenado, CONGELADO (requires_grad=False, model.eval()).
    Retorna (m, 1280) de la penultima capa.

    Este es el UNICO punto del proyecto donde entra torch, y solo como
    extractor de caracteristicas. La PCA y el clasificador que siguen
    son implementacion propia en NumPy.
    """
```

### Peso (desactivado)

```python
# config.py
USE_WEIGHT = False   # activar solo si se dispone de mediciones reales de masa
```

Los datasets públicos no traen peso. La vía A corre sin esa componente.
La arquitectura lo soporta para trabajo futuro con celda de carga HX711.

---

## 9. Aplicación PySide6 — especificación de diseño

**Sección normativa. El agente no improvisa estética.**

### 9.1 Dirección visual

Consola de instrumentación técnica. Oscura, plana, densa en datos, tipografía
fuerte, un color de acento. Cero decoración.

### 9.2 Tokens — `src/ui/theme.py`, genera el QSS

```python
BG_BASE     = "#14161A"
BG_PANEL    = "#1C1F25"
BG_ELEVATED = "#242830"
BORDER      = "#2F343D"
TEXT_HI     = "#E8EAEE"
TEXT_MID    = "#98A0AB"
TEXT_LO     = "#5E6570"
ACCENT      = "#E8734A"      # UNICO acento de interfaz
VIA_A       = "#5FB37A"      # verde  - features fisicas
VIA_B       = "#D9A441"      # ambar  - eigen-objetos
VIA_C       = "#5B9DD9"      # azul   - embeddings CNN
ERR         = "#D9534F"

FONT_UI     = "Inter, DejaVu Sans, sans-serif"
FONT_MONO   = "JetBrains Mono, DejaVu Sans Mono, monospace"
SPACE       = 8              # todo el spacing es multiplo de 8
RADIUS      = 4              # maximo absoluto
```

Los tres colores de vía son la única excepción a la regla de un acento: son
codificación de datos, no decoración. Se usan consistentemente en predicciones,
gráficas y leyendas.

### 9.3 Layout — ventana principal 1400×880

```
+---------------------------------------------------------------+
| SCOAL  [Dataset v] [Via v] [o Camara]     [Entrenar][Analisis] |
+--------------------------+------------------------------------+
|                          |  PREDICCIONES                      |
|   FEED / IMAGEN          |  +------------------------------+  |
|   overlay:               |  | # A  Fisicas    manzana  0.71|  |
|    - contorno            |  | # B  Eigen      naranja  0.55|  |
|    - bbox                |  | # C  CNN        manzana  0.96|  |
|    - ejes principales    |  +------------------------------+  |
|      (autovectores)      |                                    |
|                          |  MEDICIONES                        |
|                          |  Area          3421 px2            |
|                          |  Aspecto          1.04             |
|                          |  Circularidad     0.91             |
|                          |  RGB        178 / 42 / 38          |
|                          |  L1 / L2     2.31 / 2.22           |
|                          |  Peso             --               |
+--------------------------+------------------------------------+
|  BANDA   >>>   [#] --------------------->                     |
|  v manzana_g 12  v granny 8  v red_del 11  v naranja 9  ...    |
+---------------------------------------------------------------+
```

- **Feed:** `QLabel` con `QPixmap`, overlay dibujado con `QPainter`. Los ejes
  principales (autovectores de la matriz de inercia) se dibujan como dos líneas
  cruzadas sobre el objeto.
- **Predicciones:** tres filas, cada una con su color de vía, clase y confianza.
  Cuando las vías discrepan, la discrepancia es visible y es el punto.
- **Banda:** `QGraphicsScene` con `QPropertyAnimation`. El objeto se desliza y
  cae en su contenedor.
- **Ventana de análisis:** matriz de confusión por vía, scatter PCA 2D coloreado
  por clase, grid de los 16 primeros eigen-objetos, curvas de ablación.
  Matplotlib embebido con `FigureCanvasQTAgg`, `rcParams` alineados a la paleta.

### 9.4 Prohibiciones explícitas

**El diseño no debe ser AI slop generic aesthetics.** Está prohibido:

- Emojis en cualquier parte de la interfaz
- Azul Bootstrap (#007bff, #4A90E2) o cualquier paleta de framework web
- Gradientes de cualquier tipo
- `border-radius` mayor a 4px
- Sombras (`box-shadow`), aunque Qt las soporte
- Títulos gigantes centrados tipo landing page
- Texto explicativo de relleno dentro de la UI
- Iconos decorativos que no representen una acción
- Spinners o barras de progreso que no midan trabajo real
- Fuentes por defecto de Qt sin declarar
- Widgets sin QSS mezclados con widgets estilizados
- Animaciones de entrada o salida de paneles
- Tooltips con texto obvio

**Reglas positivas:**

- Números siempre en fuente mono, alineados a la derecha
- Labels en mayúsculas, color `TEXT_MID`, tamaño pequeño, `letter-spacing: 0.5px`
- Separaciones con bordes de 1px en `BORDER`, nunca con espacio en blanco solo
- Todo el QSS vive en un archivo generado desde `theme.py`
- Si un widget no muestra un dato ni dispara una acción, no va

---

## 10. Informe académico en LaTeX — Normas APA 7ª edición

El agente genera el andamiaje completo en `docs/`, compilable, con las secciones
de contenido marcadas con `% TODO` para que el autor las redacte.
**Los resultados numéricos se inyectan desde los CSV de ablación, nunca se
escriben a mano.**

### 10.1 Estructura obligatoria

```
Portada
Contraportada
Resumen
Abstract
Palabras clave
Keywords
Introduccion
Objetivo general
Objetivos especificos (3)
Marco teorico
Metodologia
Resultados
Discusion
Conclusiones
Trabajo futuro
Referencias
```

### 10.2 Requisitos de formato APA 7

- Clase `apa7` con opción `stu` (documento de estudiante)
- Una sola columna, interlineado doble
- Márgenes de 1 pulgada en los cuatro lados
- Sangría de primera línea de 0.5 pulgadas
- Times New Roman 12 pt (o alternativa APA aprobada)
- Citación autor-fecha mediante `biblatex` estilo `apa`
- **Compilador de bibliografía: `biber`, NO `bibtex`.** `biblatex-apa` lo exige
- Referencias con sangría francesa (la clase lo maneja)
- Encabezados en los cinco niveles APA 7, mapeados a `\section`,
  `\subsection`, `\subsubsection`, `\paragraph`, `\subparagraph`
- Tablas y figuras con numeración y nota APA (`\begin{table}` + `\note{}`)

### 10.3 Archivos a generar

```
docs/
├── informe.tex
├── secciones/
│   ├── portada.tex
│   ├── contraportada.tex
│   ├── resumen.tex
│   ├── abstract.tex
│   ├── introduccion.tex
│   ├── objetivos.tex
│   ├── marco_teorico.tex
│   ├── metodologia.tex
│   ├── resultados.tex
│   ├── discusion.tex
│   ├── conclusiones.tex
│   └── trabajo_futuro.tex
├── figuras/              # generadas por src/experiments/, no editadas a mano
├── tablas/               # .tex generados desde CSV
├── referencias.bib
└── Makefile              # pdflatex -> biber -> pdflatex x2
```

### 10.4 Preámbulo de `informe.tex`

```latex
\documentclass[stu,12pt,floatsintext]{apa7}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[spanish,es-noquoting]{babel}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{csquotes}
\usepackage{url}

\usepackage[style=apa,backend=biber]{biblatex}
\DeclareLanguageMapping{spanish}{spanish-apa}
\addbibresource{referencias.bib}

% --- Metadatos de portada (clase apa7, opcion stu) ---
\title{Clasificacion de objetos mediante algebra lineal:
       estudio comparativo de tres representaciones de caracteristicas
       bajo un clasificador de minimos cuadrados}
\shorttitle{Clasificacion de objetos mediante algebra lineal}
\author{Jhonnatan}
\affiliation{Facultad de Ingenieria, Universidad Cooperativa de Colombia}
\course{Algebra Lineal}
\professor{% TODO: nombre del docente}
\duedate{% TODO: fecha de entrega}

\begin{document}

\maketitle                              % portada APA 7 (estudiante)
\input{secciones/contraportada}

\input{secciones/resumen}
\input{secciones/abstract}
\input{secciones/introduccion}
\input{secciones/objetivos}
\input{secciones/marco_teorico}
\input{secciones/metodologia}
\input{secciones/resultados}
\input{secciones/discusion}
\input{secciones/conclusiones}
\input{secciones/trabajo_futuro}

\printbibliography[title={Referencias}]

\end{document}
```

**Nota:** `\maketitle` con la opción `stu` genera la portada APA de estudiante
completa (título, autor, afiliación, curso, docente, fecha). La contraportada es
una página adicional requerida por la institución, no por APA; se genera aparte.

### 10.5 `secciones/contraportada.tex`

```latex
\newpage
\thispagestyle{empty}
\begin{center}

{\large\textbf{Clasificacion de objetos mediante algebra lineal:
estudio comparativo de tres representaciones de caracteristicas
bajo un clasificador de minimos cuadrados}}

\vspace{2cm}

Jhonnatan

\vspace{2cm}

Trabajo presentado como requisito para la asignatura de Algebra Lineal

\vspace{2cm}

Docente\\
% TODO: nombre completo del docente

\vspace{2cm}

Universidad Cooperativa de Colombia\\
Facultad de Ingenieria\\
Ingenieria de Sistemas\\
Villavicencio, Meta\\
% TODO: ano

\end{center}
\newpage
```

### 10.6 `secciones/resumen.tex` y `abstract.tex`

```latex
% --- resumen.tex ---
\begin{abstract}
% TODO: 150-250 palabras en espanol. Debe cubrir:
% problema, metodo, resultado principal cuantificado, conclusion.
\end{abstract}

\keywords{algebra lineal, clasificacion de objetos,
analisis de componentes principales, minimos cuadrados,
descomposicion en valores singulares, vision por computador}
```

```latex
% --- abstract.tex ---
\newpage
\section*{Abstract}
% TODO: version en ingles del resumen.

\noindent\textit{Keywords:} linear algebra, object classification,
principal component analysis, least squares, singular value
decomposition, computer vision.
\newpage
```

### 10.7 `secciones/objetivos.tex`

Redactados. El autor debe revisarlos antes de entregar.

```latex
\section{Objetivos}

\subsection{Objetivo general}

Desarrollar un sistema de clasificacion de objetos por tamano, color, forma e
identidad, implementado mediante operaciones de algebra lineal, que permita
evaluar comparativamente el aporte de tres representaciones de caracteristicas
distintas bajo un mismo clasificador de minimos cuadrados regularizado.

\subsection{Objetivos especificos}

\begin{enumerate}
\item Implementar tres extractores de caracteristicas ---descriptores
geometricos y cromaticos, proyeccion sobre eigen-objetos mediante
descomposicion en valores singulares, y representaciones de una red
convolucional preentrenada--- empleando unicamente operaciones matriciales
programadas sobre NumPy.

\item Construir un clasificador multiclase de minimos cuadrados con
regularizacion de Tikhonov, verificando su equivalencia numerica con una
implementacion de referencia y caracterizando el efecto del parametro de
regularizacion sobre el numero de condicion de la matriz normal.

\item Evaluar el desempeno comparativo de las tres representaciones mediante
un estudio de ablacion con validacion cruzada, cuantificando la contribucion
de la representacion frente a la del clasificador en la exactitud final del
sistema.
\end{enumerate}
```

### 10.8 `docs/referencias.bib` — semilla mínima

```bibtex
@article{turk1991,
  author  = {Turk, Matthew and Pentland, Alex},
  title   = {Eigenfaces for recognition},
  journal = {Journal of Cognitive Neuroscience},
  volume  = {3},
  number  = {1},
  pages   = {71--86},
  year    = {1991}
}

@book{boyd2018,
  author    = {Boyd, Stephen and Vandenberghe, Lieven},
  title     = {Introduction to applied linear algebra:
               Vectors, matrices, and least squares},
  publisher = {Cambridge University Press},
  year      = {2018}
}

@article{muresan2018,
  author  = {Mure{\c{s}}an, Horea and Oltean, Mihai},
  title   = {Fruit recognition from images using deep learning},
  journal = {Acta Universitatis Sapientiae, Informatica},
  volume  = {10},
  number  = {1},
  pages   = {26--42},
  year    = {2018}
}

@techreport{nene1996,
  author      = {Nene, Sameer A. and Nayar, Shree K. and Murase, Hiroshi},
  title       = {Columbia Object Image Library (COIL-100)},
  institution = {Columbia University},
  number      = {CUCS-006-96},
  year        = {1996}
}

@inproceedings{howard2019,
  author    = {Howard, Andrew and Sandler, Mark and Chu, Grace and others},
  title     = {Searching for {MobileNetV3}},
  booktitle = {Proceedings of the IEEE/CVF International Conference
               on Computer Vision},
  pages     = {1314--1324},
  year      = {2019}
}
```

### 10.9 Makefile

```makefile
informe:
	cd docs && pdflatex informe.tex
	cd docs && biber informe
	cd docs && pdflatex informe.tex
	cd docs && pdflatex informe.tex

clean:
	cd docs && rm -f *.aux *.log *.bbl *.bcf *.blg *.out *.run.xml
```

### 10.10 Tabla de mapeo de conceptos (va en el marco teórico)

El agente genera esta tabla en `docs/tablas/mapeo_conceptos.tex` con formato APA
(`\begin{table}`, `\caption`, `\note`).

| Concepto | Dónde se aplica |
|---|---|
| Matrices y tensores | Imagen como I ∈ R^(H×W×3) |
| Producto matriz-vector | Conversión RGB→YCbCr y RGB→gris |
| Autovalores y autovectores | Matriz de inercia 2×2 → ejes principales, orientación |
| Matriz de covarianza | Estandarización y PCA |
| Descomposición en valores singulares | Eigen-objetos, reducción 4096→k |
| Relación entre AᵀA y AAᵀ | Truco de Turk-Pentland para m ≪ n |
| Espacios y subespacios vectoriales | Espacio de características, subespacio propio |
| Proyección ortogonal | Representación en el subespacio de eigen-objetos |
| Mínimos cuadrados | Entrenamiento del clasificador |
| Pseudoinversa de Moore-Penrose | Solución cerrada W = X⁺Y |
| Rango e independencia lineal | Detección de características redundantes |
| Número de condición | Diagnóstico de XᵀX mal condicionada |
| Regularización de Tikhonov | Estabilización mediante +λI |
| Norma euclidiana | Baseline k-NN, distancia en espacio propio |
| Hiperplanos separadores | Frontera de decisión del clasificador lineal |

---

## 11. Estructura del proyecto

```
scoal/
├── config.py
├── src/
│   ├── data/
│   │   ├── download.py         # COIL-100 + Fruits-360 automatico
│   │   └── loader.py           # split estratificado, semilla fija
│   ├── vision/
│   │   ├── segment.py
│   │   ├── stability.py
│   │   └── crop.py
│   ├── features/
│   │   ├── physical.py         # via A
│   │   ├── eigen.py            # via B
│   │   └── cnn.py              # via C
│   ├── model/
│   │   ├── lstsq.py
│   │   ├── knn.py
│   │   ├── pca.py
│   │   └── metrics.py
│   ├── experiments/
│   │   └── ablation.py         # 3 barridos -> CSV + figuras
│   ├── pipeline.py
│   └── ui/
│       ├── theme.py
│       ├── widgets.py
│       ├── main_window.py
│       ├── camera.py           # QThread + Signal
│       ├── belt.py             # QGraphicsScene
│       └── analysis.py
├── docs/                       # ver seccion 10
├── tests/
│   ├── test_vs_sklearn.py
│   └── test_feature_order.py
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 12. Restricciones para el agente codificador

1. **PROHIBIDO** `sklearn` como motor. PCA, SVD, mínimos cuadrados, pseudoinversa,
   k-NN y métricas: **NumPy puro**.
2. `sklearn` únicamente en `tests/test_vs_sklearn.py`, para verificación numérica.
3. `torch` únicamente en `src/features/cnn.py`, con el modelo **congelado**
   (`requires_grad=False`, `model.eval()`), usado solo como extractor.
   Ninguna capa se entrena.
4. `opencv` solo para I/O, contornos y morfología. Ninguna decisión de
   clasificación vive ahí.
5. Cada función matemática lleva docstring **con la fórmula en notación
   matemática**. Alimenta el marco teórico del informe.
6. `np.linalg.solve` en vez de `np.linalg.inv`, con comentario del porqué.
7. Semilla aleatoria fija global en `config.py`.
8. **Threading:** cámara en `QThread`, comunicación por `Signal`. Nunca tocar
   widgets desde un hilo secundario.
9. La UI sigue §9 al pie de la letra. Tokens desde `theme.py`, nunca hardcodeados.
10. Los experimentos escriben CSV; las figuras se generan desde ese CSV.
11. **No ejecutar comandos de git.** Generar `.gitignore` y `README.md`
    como archivos, nada más.
12. El informe LaTeX debe **compilar sin errores** con las secciones en `% TODO`.
    Verificar corriendo `make informe`.

---

## 13. Criterios de aceptación

| # | Criterio | Umbral |
|---|---|---|
| 1 | Accuracy vía C (embeddings + clasificador propio), 10 clases | ≥ 95% |
| 2 | Accuracy vía B (eigen-objetos) en Fruits-360 | ≥ 85% |
| 3 | Accuracy vía A (características físicas) | ≥ 70% |
| 4 | Brecha entre vía C y MobileNetV3 completo | reportada, < 5 puntos |
| 5 | Coincidencia con sklearn en mínimos cuadrados | error < 1e-8 |
| 6 | Varianza explicada con k óptimo | ≥ 90% |
| 7 | Efecto de lambda | curva con óptimo visible + número de condición |
| 8 | Ablación completa | 3 gráficas generadas desde CSV |
| 9 | Modo webcam | tres predicciones simultáneas en < 2 s |
| 10 | Discrepancia entre vías en vivo | documentada como hallazgo, no como bug |
| 11 | UI cumple §9 | revisión manual contra la lista de prohibiciones |
| 12 | `make informe` compila | PDF generado sin errores |

---

## 14. Riesgos

| Riesgo | Mitigación |
|---|---|
| Vías A y B fallan con webcam en vivo | **Es el hallazgo esperado.** Se presenta como evidencia de domain shift, no se esconde |
| El agente entrena el CNN en vez de congelarlo | Verificar `requires_grad=False` y `model.eval()`; test que falle si no |
| Fruits-360 demasiado fácil (fondo blanco perfecto) | Por eso las 10 clases con solapamiento; COIL-100 aporta reflectancia compleja |
| Datasets pesados | Fruits-360 100×100 y COIL-100 (125 MB); subconjunto de 10 clases |
| El agente mete sklearn a escondidas | `grep -rn "sklearn" src/` debe salir vacío |
| PySide6 curva de aprendizaje | Empezar `main_window.py` con datos mock antes de conectar el motor |
| Overfitting con clases desbalanceadas | Balancear a ~300 img/clase, CV 5-fold, media ± desviación |
| `biblatex-apa` compilado con bibtex | Usar `biber`. El Makefile ya lo fija |
| Falta la clase `apa7` en la distro LaTeX | `tlmgr install apa7 biblatex-apa`; documentar en README |

---

## 15. Entregables

- Código fuente completo con README ejecutable desde cero
- Informe en LaTeX, normas APA 7, compilable, con secciones marcadas para redacción
- Figuras generadas: scatter PCA 2D, grid de eigen-objetos, tres matrices de
  confusión, curvas de ablación
- Aplicación funcional con demo en vivo por webcam

---

## 16. Trabajo futuro

- Peso real con celda de carga HX711 + ESP32 (`USE_WEIGHT` ya cableado)
- Fine-tuning del extractor sobre dominio propio para cerrar el domain shift
- Banda transportadora física
- Comparación contra SVM con kernel

---

## 17. Orden de ejecución sugerido

No entregar este documento completo de una sola vez. Dividir en cuatro órdenes,
validando cada bloque antes de continuar:

1. `config.py` + estructura de directorios + `src/data/` + `.gitignore` + `README.md`
2. `src/vision/` + `src/features/` + `src/model/` + `tests/`
3. `src/experiments/ablation.py` + `docs/` (andamiaje LaTeX APA)
4. `src/ui/`

**Al pedir cualquier trabajo de interfaz, incluir la sección 9.4 literal, incluso
en ajustes menores.**
