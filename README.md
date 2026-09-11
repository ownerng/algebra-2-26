# SCOAL v2.3

Sistema de Clasificación de Objetos por Álgebra Lineal.
Universidad Cooperativa de Colombia — Álgebra Lineal.

Clasifica objetos por tamaño, color, forma e identidad. El motor matemático
(PCA, SVD, mínimos cuadrados, pseudoinversa, k-NN, métricas) está implementado
en **NumPy puro**. El estudio compara tres representaciones de características
bajo **un mismo** clasificador de mínimos cuadrados regularizado.

**Pregunta de investigación:** ¿qué tan lejos llega un clasificador lineal de
mínimos cuadrados, y cuánto de su desempeño depende de la representación de
entrada en lugar del clasificador mismo?

---

## Arranque rápido

```bash
git clone https://github.com/ownerng/algebra-2-26.git
cd algebra-2-26

python -m venv .venv
.venv\Scripts\activate          # Windows;  en Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python -m src.ui.main_window     # la aplicación
```

**Si algo no funciona, primero el doctor.** Revisa Python, cada paquete
(instalado, versión mínima y que realmente importe), el modelo, los pesos de
la CNN, los datasets y sus zips, y la cámara (incluido si DroidCam entrega
imagen vacía). Cada problema viene con el comando o paso para resolverlo:

```bash
python -m src.doctor             # todo
python -m src.doctor --sin-camara
```

**Logs.** Todo lo que hace la aplicación queda en `logs.txt`, en la raíz del
proyecto: cada botón, descarga, cache, entrenamiento, cambio de estado de la
cámara, cada clasificación con sus tres vías y cada error con su traceback.
Rota a 5 MB (se conservan 3 copias) y no se versiona.

**El modelo entrenado viene en el repositorio** (`results/scoal_model.pkl`,
1.7 MB), así que la aplicación clasifica desde el primer arranque sin descargar
nada. Requiere Python 3.11 o superior.

Lo único que **no** viene son los datasets: pesan cerca de 1 GB. Hacen falta
solo si querés reentrenar, cambiar de dataset o reproducir el estudio de
ablación. Un único comando los deja listos:

```bash
python -m src.setup              # descarga + cache + entrenamiento
python -m src.setup --check      # solo dice qué falta, sin descargar
```

La aplicación detecta sola lo que falte y ofrece un botón **PREPARAR DATOS**
que ejecuta exactamente ese mismo código, con el progreso en la barra de
estado. No hace falta leer este archivo para arrancar.

## La aplicación

```
barra superior   dataset | vía de la banda | CAMARA | CAPTURAR | ENTRENAR | ANALISIS
barra de estado  qué hace el modelo  ·  qué espera el disparador de estabilidad
feed             imagen con el contorno, la caja y los ejes de inercia dibujados
predicciones     las tres vías a la vez, con su confianza y la novedad
banda            el objeto viaja y cae en el contenedor de su clase
```

| Control | Qué hace |
|---|---|
| **CAMARA** | Enciende la webcam. La clasificación **no** ocurre en cada frame: el disparador de estabilidad espera a que la escena lleve 5 frames quieta y haya un objeto segmentado. La barra de estado muestra en qué va (`sin objeto`, `escena en movimiento`, `estabilizando 3/5`, `clasificado`). |
| **CAPTURAR** | Clasifica el frame actual sin esperar al disparador. Es lo que se usa sosteniendo un objeto a pulso, que nunca queda del todo quieto. |
| **ENTRENAR** | Reentrena las tres vías con el dataset seleccionado y guarda el modelo. Necesita el cache construido; si falta, el botón queda apagado y el panel de preparación dice qué hacer. |
| **ANALISIS** | Abre las figuras del informe (curvas de ablación, matrices de confusión, dispersión PCA) leídas de los CSV de `results/`. No recalcula nada, así que lo que se ve en pantalla y lo que se imprime en el informe no pueden divergir. |

Sin cámara la ventana no arranca vacía: muestra una imagen real del cache, de
modo que segmentación, mediciones y CAPTURAR se pueden probar igual.

## Qué clasifica, y qué no

El conjunto de clases es **cerrado**: 10 frutas de Fruits-360 o 20 objetos de
COIL-100, según el dataset seleccionado. No es un detector de objetos genérico.

Como `argmax(x W)` siempre devuelve una clase, un clasificador cerrado sin más
responde "manzana" ante un celular, una cara o una pared. Por eso el modelo
trae un **criterio de rechazo** y la interfaz puede decir `DESCONOCIDO`:

- **Distancia al subespacio** (vías B y C): la muestra se proyecta sobre los
  componentes principales y se reconstruye; si el residuo relativo es grande,
  no vive en el subespacio que generan las clases. Es la *distance from face
  space* de Turk y Pentland (1991).
- **Norma estandarizada** (vía A, que no reduce dimensión): tras estandarizar,
  una muestra típica tiene norma ≈ 1; las features fuera del rango visto la
  disparan.
- **Margen de la clase ganadora**: confianza por debajo del percentil bajo del
  entrenamiento.

Los tres umbrales se leen del propio conjunto de entrenamiento como percentiles
(`config.REJECT_PERCENTILE = 99`), no se escriben a mano. La justificación
completa está en `src/model/reject.py`; el contrato está fijado en
`tests/test_reject.py`.

### Tasas de rechazo medidas

Modelo entrenado sobre Fruits-360, evaluado sobre 120 imágenes de cada origen:

| Origen | Vía A | Vía B | Vía C |
|---|---|---|---|
| Fruits-360 (lo que entrenó) — *falso rechazo* | 5.0 % | 2.5 % | 6.7 % |
| COIL-100 (objetos que nunca vio) | 29.2 % | 37.5 % | **100 %** |
| Ruido aleatorio | 100 % | 100 % | 100 % |

La vía C separa dominios casi perfectamente; las vías A y B no. Tiene sentido y
es parte del hallazgo: área, color y circularidad de un patito de goma caen
dentro del rango de una fruta, y su imagen en gris se reconstruye
razonablemente con eigen-objetos de frutas. Los embeddings de la CNN, no.

**Consecuencia práctica:** para el modo webcam conviene dejar la banda en la
vía C. Y aun así, la segmentación asume un objeto sobre fondo aproximadamente
uniforme (estima el color de fondo con la mediana del marco de la imagen), así
que una escena con pared, muebles y sombras produce contornos sin sentido. Una
hoja blanca detrás del objeto cambia el resultado por completo.

### Cámara real: por qué la vía C dice DESCONOCIDO

Medido con DroidCam apuntando a una foto de manzana roja en pantalla: la vía C
**acierta la clase** (`manzana_red_delicious`, 12 de 12) pero su novedad queda
entre 1.48 y 1.60, por encima del límite 1.00, y la muestra se rechaza. La
barra de estado lo dice así: *"DESCONOCIDO: vía C cree que es …, pero la imagen
no se parece a las fotos de entrenamiento"*.

No se arregla subiendo el límite. Novedad de la vía C sobre 500 imágenes de
Fruits-360 y 400 de COIL-100 (solo el criterio de novedad):

| Límite | Fruits-360 rechazadas | COIL-100 rechazados |
|---|---|---|
| 1.00 (actual) | 1 % | 100 % |
| 1.25 | 0 % | 98.5 % |
| 1.50 | 0 % | 86.5 % |
| 1.75 | 0 % | 26 % |
| 2.00 | 0 % | 0 % |

Una fruta vista por la cámara queda a la misma distancia del subespacio que un
objeto ajeno. El dominio aprendido son fotos de estudio (fruta girando sobre
blanco, luz controlada); la cámara cambia luz, sombras, enfoque y compresión.
Es un **desplazamiento de dominio**, no un fallo del clasificador: aceptar esa
manzana obligaría a aceptar 3 de cada 4 objetos que no son fruta. El arreglo de
fondo sería entrenar también con capturas de la propia cámara.

Con el mismo criterio, la vía A deja pasar el 98 % de los objetos de COIL-100 y
la vía B el 53 %: por eso con la banda en A o B caen objetos cualquiera en los
contenedores, y la aplicación lo advierte al elegirlas.

## De cero al informe

```bash
python -m src.setup --dataset fruits360             # datos + cache + modelo
python -m src.experiments.ablation --dataset fruits360   # CSV, figuras y tablas
python -m src.ui.main_window                        # aplicación
make informe                                        # o docs/compilar.ps1 en Windows
```

Para compilar el informe hace falta una distribución LaTeX con las clases
`apa7` y `biblatex-apa`, y el compilador de bibliografía **biber** (no bibtex).
Con MiKTeX los paquetes se instalan solos al compilar; con TeX Live:
`tlmgr install apa7 biblatex-apa biber`.

Verificación numérica del motor contra sklearn (sklearn solo se usa aquí):

```bash
python -m pytest tests -q
```

Cada módulo matemático trae además una comprobación mínima ejecutable:

```bash
python -m src.model.lstsq        # y .pca, .knn, .metrics, .reject
python -m src.features.eigen     # y .physical, .cnn
python -m src.pipeline
```

## Arquitectura

```
Imagen (dataset o webcam)
  |
  +-- [0] (webcam) disparador de estabilidad: escena quieta N frames + contorno
  +-- [1] segmentación: RGB->YCbCr (matriz 3x3) -> umbral -> máscara -> contorno
  +-- [2] recorte con máscara, fondo neutro, normalización a 128x128
  |
  +-- [A] features físicas (16-D)   área, perímetro, aspecto, circularidad,
  |                                 RGB medio, Hu[7], ejes de inercia
  +-- [B] eigen-objetos (k-D)       64x64 gris -> 4096-D -> SVD (Turk-Pentland)
  +-- [C] embeddings CNN (k-D)      MobileNetV3-Small congelado -> PCA propia
  |
  +-- [3] por vía: estandarizar -> W = (X^T X + λI)^-1 X^T Y -> argmax(X W)
  +-- [4] rechazo: distancia al dominio + margen -> clase o "desconocido"
```

**Por qué YCbCr y no HSV:** RGB→HSV no es una transformación lineal (usa
mínimo, máximo y divisiones). RGB→YCbCr sí es una matriz 3×3 exacta, así que
la etapa de segmentación también queda dentro del álgebra lineal. Está
documentado en `src/vision/segment.py` y en el marco teórico del informe.

## Restricciones del proyecto

| Regla | Dónde se verifica |
|---|---|
| sklearn no es motor, solo referencia | `grep -rn "sklearn" src/` debe salir vacío |
| torch solo como extractor congelado | `src/features/cnn.py`, `demo()` falla si `requires_grad` o `training` |
| `np.linalg.solve`, nunca `np.linalg.inv` | `src/model/lstsq.py` |
| Orden de features estable train/inferencia | `tests/test_feature_order.py` |
| Umbrales de rechazo ajustados, no escritos a mano | `tests/test_reject.py` |
| Figuras generadas desde CSV, no reentrenando | `src/experiments/ablation.py` |
| Semilla fija global | `config.py` |

## Estado actual de los resultados

Ejecutado sobre Fruits-360, 10 clases con solapamiento, 300 imágenes por clase,
partición estratificada y validación cruzada de 5 pliegues:

| Vía | Representación | Exactitud (media ± DE) |
|---|---|---|
| A | Físicas (16-D) | 0.841 ± 0.015 |
| B | Eigen-objetos (k=50) | 0.900 ± 0.011 |
| C | Embeddings CNN (k=50) | 0.998 ± 0.003 |

Baselines sobre la misma partición:

| Baseline | Vía A | Vía B | Vía C |
|---|---|---|---|
| Aleatorio (1/10) | 0.100 | 0.100 | 0.100 |
| Mínimos cuadrados (el clasificador del estudio) | 0.821 | 0.872 | 0.997 |
| k-NN (k=5) sobre la misma representación | 0.927 | 0.976 | 0.997 |
| MobileNetV3 con su cabeza original de ImageNet | — | — | 0.612 |

Dos observaciones que el informe debe discutir, no esconder:

1. **k-NN supera a mínimos cuadrados en las vías A y B, pero empata en la C.**
   Esa es la pregunta de investigación respondida con números: cuando la
   representación separa mal, el clasificador lineal pierde terreno frente a
   uno que memoriza; cuando la representación es buena, las fronteras planas
   bastan y ambos coinciden en 0.9973. El desempeño dependía de la
   representación, no del clasificador. Nótese además que la cabeza original
   de ImageNet solo llega a 0.612: el mérito está en los embeddings más el
   clasificador propio, no en la CNN preentrenada tal cual.
2. **En las vías B y C la curva de λ es plana y cond(XᵀX+λI) ≈ 1.** No es un
   error: tras proyectar sobre componentes principales y estandarizar, las
   columnas quedan ortogonales entre sí, la matriz normal queda diagonal y no
   hay nada que regularizar. El efecto de λ se ve en la vía A, cuyas features
   están correlacionadas (cond pasa de 1.06e5 a 1.62e3 al subir λ).

## Estructura

```
config.py                  semilla, rutas, clases, hiperparámetros, rechazo
src/setup.py               preparación de cero en un comando
src/data/                  descarga y cache
src/vision/                segmentación, recorte, disparador de estabilidad
src/features/              vía A (físicas), vía B (eigen), vía C (CNN)
src/model/                 mínimos cuadrados, PCA, k-NN, métricas, rechazo
src/experiments/           barridos de ablación, figuras y tablas
src/ui/                    aplicación PySide6
docs/                      informe LaTeX en normas APA 7
tests/                     verificación contra sklearn, contrato de features y de rechazo
results/                   CSV de los experimentos y el modelo entrenado (versionado)
```

## Notas

- El **peso** está cableado pero desactivado (`config.USE_WEIGHT = False`):
  los datasets públicos no traen masa. Al activarlo, la componente entra
  siempre al final del vector; hay un test que lo fija.
- La **discrepancia entre vías** en el modo webcam es un hallazgo esperado
  (desplazamiento de dominio entre dataset y cámara), no un defecto.
- Los **pesos de MobileNetV3-Small** los descarga torchvision sola la primera
  vez que se entrena la vía C. Si la red los bloquea, se pueden bajar a mano de
  <https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth> y
  guardarlos en `data/raw/torch/hub/checkpoints/`.
- El proyecto no crea repositorios remotos ni toca ramas por su cuenta.
