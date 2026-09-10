# SCOAL v2.2

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

## Instalación

Requiere Python 3.11 o superior.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Para compilar el informe hace falta además una distribución LaTeX con las
clases `apa7` y `biblatex-apa`, y el compilador de bibliografía **biber**
(no bibtex). Con MiKTeX los paquetes se instalan solos al compilar; con TeX
Live: `tlmgr install apa7 biblatex-apa biber`.

## Uso, de cero a informe

```bash
# 1. Descargar los datasets (COIL-100 + Fruits-360, solo las clases usadas)
python -m src.data.download

# 2. Construir el cache de trabajo (segmenta, recorta y balancea)
python -m src.data.loader --dataset fruits360

# 3. Estudio de ablación: CSV en results/<dataset>/, figuras en docs/figuras/,
#    tablas .tex en docs/tablas/
python -m src.experiments.ablation --dataset fruits360

# 4. Entrenar el modelo que usa la aplicación
python -m src.pipeline --train --dataset fruits360

# 5. Aplicación de escritorio
python -m src.ui.main_window

# 6. Informe
make informe                    # o, en Windows sin make:  docs/compilar.ps1
```

Verificación numérica del motor contra sklearn (sklearn solo se usa aquí):

```bash
python -m pytest tests -q
```

Cada módulo matemático trae además una comprobación mínima ejecutable:

```bash
python -m src.model.lstsq        # y .pca, .knn, .metrics
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
| Figuras generadas desde CSV, no reentrenando | `src/experiments/ablation.py` |
| Semilla fija global | `config.py` |

El proyecto **no** ejecuta comandos de git ni crea repositorios remotos.

## Estado actual de los resultados

Ejecutado sobre Fruits-360, 10 clases con solapamiento, 300 imágenes por clase,
partición estratificada y validación cruzada de 5 pliegues:

| Vía | Representación | Exactitud (media ± DE) |
|---|---|---|
| A | Físicas (16-D) | 0.841 ± 0.015 |
| B | Eigen-objetos (k=50) | 0.900 ± 0.011 |
| C | Embeddings CNN | pendiente, ver abajo |

Baselines sobre la misma partición: aleatorio 0.100, k-NN sobre vía A 0.927,
k-NN sobre vía B 0.976.

Dos observaciones que el informe debe discutir, no esconder:

1. **k-NN supera a mínimos cuadrados en ambas vías.** La representación ya
   separa las clases; el clasificador lineal pierde parte de esa separación al
   imponer fronteras planas. Es exactamente la pregunta de investigación
   respondida con números.
2. **En las vías B y C la curva de λ es plana y cond(XᵀX+λI) ≈ 1.** No es un
   error: tras proyectar sobre componentes principales y estandarizar, las
   columnas quedan ortogonales entre sí, la matriz normal queda diagonal y no
   hay nada que regularizar. El efecto de λ se ve en la vía A, cuyas features
   están correlacionadas (cond pasa de 1.06e5 a 1.62e3 al subir λ).

### Vía C: pesos de MobileNetV3

La vía C necesita los pesos preentrenados de MobileNetV3-Small. En la red donde
se preparó este proyecto `download.pytorch.org` está bloqueado, así que la vía
quedó pendiente de ejecución; el código está completo y se activa solo cuando
los pesos existen. Dos formas de conseguirlos:

- Dejar que torchvision los descargue: basta con volver a correr el pipeline
  desde una red sin ese bloqueo.
- Descargar a mano
  <https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth>
  y guardarlo como `data/raw/torch/mobilenet_v3_small.pth`.

Después, `python -m src.experiments.ablation` regenera todo incluyendo la vía C
y el baseline de la cabeza original de ImageNet.

## Estructura

```
config.py                  semilla, rutas, clases, hiperparámetros
src/data/                  descarga y cache
src/vision/                segmentación, recorte, disparador de estabilidad
src/features/              vía A (físicas), vía B (eigen), vía C (CNN)
src/model/                 mínimos cuadrados, PCA, k-NN, métricas
src/experiments/           barridos de ablación, figuras y tablas
src/ui/                    aplicación PySide6
docs/                      informe LaTeX en normas APA 7
tests/                     verificación contra sklearn y contrato de features
results/                   CSV de los experimentos y modelo entrenado
```

## Notas

- El **peso** está cableado pero desactivado (`config.USE_WEIGHT = False`):
  los datasets públicos no traen masa. Al activarlo, la componente entra
  siempre al final del vector; hay un test que lo fija.
- La **discrepancia entre vías** en el modo webcam es un hallazgo esperado
  (desplazamiento de dominio entre dataset y cámara), no un defecto.
- Los datasets pesan ~1 GB descargados; `data/` está en `.gitignore`.
