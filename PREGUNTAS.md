# Preguntas pendientes sobre el PRD

Todo el sistema está construido y corriendo. Estas son las ambigüedades que
encontré: en cada una tomé una decisión para no bloquear el trabajo, y aquí
queda escrito qué decidí y qué habría que cambiar si la respuesta es otra.

---

## 1. Vía A: ¿13 o 16 componentes?

El PRD dice "features físicas (13-D)" pero la lista tiene 16:
área, perímetro, aspecto, circularidad (4) + R,G,B medios (3) + Hu[7] (7) +
λ1, λ2 (2) = **16**.

**Decidí:** implementar las 16 de la lista. El número no está escrito en
ningún lado del código: sale de `physical.FEATURE_NAMES`, y hay un test que
verifica que el orden declarado es el orden producido.

**Si querías 13:** decime cuáles tres se caen (¿los Hu de orden alto?).

---

## 2. Vía C: la penúltima capa de MobileNetV3-Small no da 1280

El PRD pide "MobileNetV3-Small → penúltima capa → 1280-D". MobileNetV3-Small
termina en `Linear(576→1024) → Hardswish → Dropout → Linear(1024→1000)`: su
penúltima capa entrega **1024**. El 1280 es el ancho de MobileNetV3-**Large**.

**Decidí:** usar Small y medir la dimensión del modelo cargado
(`cnn.embed_dim()`), sin fijar el número. La PCA propia reduce a 60 igual.

**Si querías 1280 literal:** hay que cambiar a `mobilenet_v3_large`. Es un
cambio de una línea en `src/features/cnn.py`, pero el modelo pesa ~5× más.

---

## 3. Baseline "MobileNetV3 completo con su cabeza original"

Esa cabeza predice las 1000 clases de ImageNet, no las 10 tuyas. No existe una
comparación directa sin decidir cómo mapear.

**Decidí:** asignarle a cada clase de ImageNet la clase del proyecto más
frecuente entre las muestras de entrenamiento que la activan (voto
mayoritario, tabla de conteo, cero entrenamiento). Queda documentado en la
docstring de `_baseline_mobilenet` y hay que decirlo en la metodología.

**Alternativa:** reportarlo solo cualitativamente ("ImageNet dice *Granny
Smith* para las tres manzanas") sin número. Menos comparable.

---

## 4. Vía C no se pudo ejecutar todavía: red bloqueada

`download.pytorch.org` no responde desde esta máquina (probado también fuera
del entorno: la conexión expira). PyPI, GitHub y Columbia sí responden, así
que los datasets se descargaron completos y las vías A y B corrieron enteras.

**Estado:** el código de la vía C está completo y probado en su parte
verificable; se activa solo cuando existan los pesos. Dos caminos:

- correr el pipeline desde otra red (torchvision los baja solo), o
- bajar a mano `mobilenet_v3_small-047dcff4.pth` y dejarlo en
  `data/raw/torch/mobilenet_v3_small.pth`.

Después, `python -m src.experiments.ablation` regenera todo con la vía C
incluida, sin tocar código.

**Criterios de aceptación 1 y 4 quedan pendientes de eso.**

---

## 5. Nombres de las clases de Fruits-360

El PRD las nombra en español; el dataset usa carpetas en inglés que además
cambian entre releases. Mapeé por prefijo en `config.FRUITS_CLASSES`:

| PRD | Carpeta usada |
|---|---|
| manzana_golden | Apple Golden 1/2/3 |
| manzana_granny_smith | Apple Granny Smith |
| manzana_red_delicious | Apple Red Delicious |
| naranja | Orange |
| mandarina | Mandarine |
| limon | Lemon |
| lima | Limes |
| tomate_cherry | Tomato Cherry Red |
| papa_roja | Potato Red (+ Potato Red Washed) |
| zucchini | Zucchini (+ Zucchini dark) |

Las 10 quedaron con 300 imágenes cada una, 0 descartadas por la segmentación.
**Confirmá que `papa_roja` es "Potato Red" y no otra papa.**

---

## 6. COIL-100 no tiene nombres de clase

Los 20 objetos quedaron como `obj01 … obj20`. Si querés nombres legibles en el
informe hay que escribirlos a mano (el dataset no los trae).

---

## 7. Raíz del proyecto

El árbol del PRD dice `scoal/`. Puse todo en la raíz de `algebra-2026-2`, sin
carpeta intermedia. Si lo querés anidado es un `git mv` cuando decidas.

---

## 8. Dos hallazgos que quiero que veas antes de escribir la discusión

1. **k-NN le gana a mínimos cuadrados en las dos vías** (0.927 vs 0.821 en A;
   0.976 vs 0.872 en B). No es un bug: es literalmente la pregunta de
   investigación respondida. La representación ya separa las clases y el
   clasificador lineal pierde parte de esa separación al imponer fronteras
   planas.

2. **En las vías B y C la curva de λ sale plana, con cond(XᵀX+λI) ≈ 1.**
   Tampoco es un bug: después de proyectar sobre componentes principales y
   estandarizar, las columnas quedan ortogonales, la matriz normal queda
   diagonal y no hay nada mal condicionado que regularizar. El criterio 7
   ("curva con óptimo visible + número de condición") sí se ve en la vía A,
   donde las features están correlacionadas: cond baja de 1.06e5 a 1.62e3.

   Si querés que el óptimo se vea también en B, hay que quitar la
   estandarización posterior a la proyección, o barrer λ sobre las 4096
   componentes crudas en vez de las k proyectadas. Decime si vale la pena.

---

## 9. Qué instalé en tu máquina

No había Python ni LaTeX. Para poder ejecutar y verificar instalé, todo por
scoop y a nivel de usuario (sin admin, desinstalable con `scoop uninstall`):

- `python` 3.14.7
- `latex` (MiKTeX 25.12) — necesario para el criterio 12
- el entorno virtual `.venv` dentro del proyecto, con las dependencias de
  `requirements.txt` (numpy 2.5, opencv 5.0, torch 2.14, PySide6 6.11,
  matplotlib, scikit-learn solo para tests)

Si preferís otra versión de Python o instalar LaTeX de otra forma, decime y lo
rehago.

---

## 10. Lo único que no pude verificar

**Criterio 9 (webcam, tres predicciones en menos de 2 s).** No tengo cámara
acá. El código está armado y el disparador de estabilidad tiene su prueba
propia sin hardware (`python -m src.ui.camera`), pero el camino completo
cámara → predicción hay que probarlo vos: abrí la app y activá CAMARA.
