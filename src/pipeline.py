"""
Orquestacion: representaciones crudas, entrenamiento y prediccion.

Separacion deliberada en dos etapas:

  1. Representaciones CRUDAS (no dependen de la particion): features fisicas,
     vectores de gris de 4096 componentes y embeddings de la CNN. Se calculan
     una vez por dataset y se cachean, porque la CNN es lo caro del proyecto.

  2. Representaciones AJUSTADAS (dependen de la particion): eigen-objetos,
     PCA y estandarizacion se ajustan SOLO con el conjunto de entrenamiento de
     cada pliegue y se aplican a test. Ajustarlas con todo el dataset filtraria
     informacion del test y la exactitud reportada seria optimista.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from src.features import eigen, physical  # noqa: E402
from src.model import lstsq, reject  # noqa: E402
from src.model.pca import PCA, Standardizer  # noqa: E402
from src.vision.crop import crop_to_mask, to_gray_vector  # noqa: E402
from src.vision.segment import segment  # noqa: E402


# --------------------------------------------------------------------------
# Etapa 1: representaciones crudas (cacheadas por dataset)
# --------------------------------------------------------------------------

def raw_path(dataset: str) -> Path:
    return config.CACHE_DIR / ("raw_%s.npz" % dataset)


def compute_raw(dataset: str, datos: dict, vias=config.VIAS,
                rebuild: bool = False) -> dict:
    """Devuelve {'A': (m, nA), 'B': (m, 4096), 'C': (m, d)} cacheado en disco.

    La via C se omite silenciosamente si no hay pesos de la CNN disponibles:
    el resto del estudio no depende de ella y debe poder correr igual.
    """
    destino = raw_path(dataset)
    cache: dict[str, np.ndarray] = {}
    if destino.exists() and not rebuild:
        with np.load(destino) as z:
            cache = {k: z[k] for k in z.files}

    crops, masks = datos["crops"], datos["masks"]
    cambio = False

    if "A" in vias and "A" not in cache:
        cache["A"] = physical.extract_batch(masks, crops)
        cambio = True

    if "B" in vias and "B" not in cache:
        cache["B"] = np.stack([to_gray_vector(c) for c in crops])
        cambio = True

    if "C" in vias and "C" not in cache:
        from src.features import cnn
        try:
            cache["C"] = cnn.extract_embeddings(crops)
            # Logits de la cabeza original: solo alimentan el baseline
            # "MobileNetV3 completo", no la via C.
            cache["C_logits"] = cnn.imagenet_logits(crops)
            cambio = True
        except RuntimeError as exc:
            print("via C no disponible: %s" % exc)

    if cambio:
        np.savez_compressed(destino, **cache)
    return cache


# --------------------------------------------------------------------------
# Etapa 2: representacion ajustada por particion
# --------------------------------------------------------------------------

def fit_representation(via: str, raw: np.ndarray, train_idx: np.ndarray,
                       k: int = config.K_DEFAULT) -> dict:
    """Ajusta la reduccion y la estandarizacion de una via usando solo train.

    Via A: no hay reduccion, las 16 componentes son interpretables y pocas.
    Via B: eigen-objetos con el truco de Turk-Pentland (m << 4096).
    Via C: PCA propia sobre los embeddings.
    """
    Xtr = raw[train_idx]
    modelo: dict = {"via": via, "k": k}

    if via == "A":
        proyectado = Xtr
    elif via == "B":
        modelo["eigen"] = eigen.fit_eigenobjects(Xtr, k)
        proyectado = eigen.project(Xtr, modelo["eigen"])
    elif via == "C":
        modelo["pca"] = PCA(k).fit(Xtr)
        proyectado = modelo["pca"].transform(Xtr)
    else:
        raise ValueError("via desconocida: %s" % via)

    modelo["scaler"] = Standardizer().fit(proyectado)
    return modelo


def project_representation(modelo: dict, raw: np.ndarray) -> np.ndarray:
    """Reduccion de dimension de una via, sin estandarizar ni sesgo."""
    if modelo["via"] == "A":
        return np.asarray(raw, dtype=np.float64)
    if modelo["via"] == "B":
        return eigen.project(raw, modelo["eigen"])
    return modelo["pca"].transform(raw)


def apply_representation(modelo: dict, raw: np.ndarray) -> np.ndarray:
    """Aplica una representacion ya ajustada. Devuelve la matriz de diseno."""
    proyectado = project_representation(modelo, raw)
    return lstsq.add_bias(modelo["scaler"].transform(proyectado))


def train_via(via: str, raw: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
              k: int = config.K_DEFAULT,
              lam: float = config.LAMBDA_DEFAULT,
              n_clases: int | None = None) -> dict:
    """Ajusta representacion + clasificador de una via sobre train_idx."""
    modelo = fit_representation(via, raw, train_idx, k)
    X = apply_representation(modelo, raw[train_idx])
    Y = lstsq.one_hot(y[train_idx], n_clases)
    modelo["W"] = lstsq.fit_least_squares(X, Y, lam)
    modelo["lam"] = lam
    fit_rejection(modelo, raw[train_idx])
    return modelo


def predict_via(modelo: dict, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (clases predichas, confianzas softmax por fila)."""
    X = apply_representation(modelo, raw)
    S = lstsq.scores(X, modelo["W"])
    return np.argmax(S, axis=1), lstsq.confidence(S)


# --------------------------------------------------------------------------
# Rechazo fuera de dominio
# --------------------------------------------------------------------------

def distancia_dominio(modelo: dict, raw: np.ndarray) -> np.ndarray:
    """Cuanto se aleja cada muestra del dominio que la via aprendio, (m,).

    Vias B y C: residuo relativo de reconstruccion sobre el subespacio de
    componentes principales, que es donde vive el dataset. Via A: norma del
    vector estandarizado, porque no reduce dimension y no hay subespacio del
    que salirse. Ver src/model/reject.py.
    """
    via = modelo["via"]
    if via == "A":
        return reject.norma_estandarizada(
            modelo["scaler"].transform(project_representation(modelo, raw)))

    Z = project_representation(modelo, raw)
    if via == "B":
        rec = eigen.reconstruct(Z, modelo["eigen"])
        media = modelo["eigen"]["mean"]
    else:
        rec = modelo["pca"].inverse_transform(Z)
        media = modelo["pca"].mean_
    raw2d = np.atleast_2d(np.asarray(raw, dtype=np.float64))
    return reject.distancia_al_subespacio(raw2d, rec, media)


def fit_rejection(modelo: dict, raw_train: np.ndarray) -> dict:
    """Fija los dos umbrales de rechazo de una via desde su entrenamiento.

    Ninguno se elige a mano. El de novedad es el percentil alto de la distancia
    al dominio; el de confianza, el percentil bajo de la confianza ganadora.
    Fijar este ultimo a ojo no funciona: con k clases la confianza de minimos
    cuadrados sobre un acierto ronda unas pocas decimas y cambia con k, con la
    via y con lambda.
    """
    modelo["umbral_novedad"] = reject.umbral(distancia_dominio(modelo, raw_train))

    _, conf = predict_via(modelo, raw_train)
    modelo["umbral_confianza"] = reject.umbral(
        conf.max(axis=1), 100.0 - config.REJECT_PERCENTILE)
    return modelo


def novedad_via(modelo: dict, raw: np.ndarray) -> np.ndarray:
    """Novedad normalizada al umbral de la via: 1.0 es la frontera, (m,)."""
    return reject.novedad(distancia_dominio(modelo, raw),
                          modelo.get("umbral_novedad", float("inf")))


# --------------------------------------------------------------------------
# Modelo completo (las tres vias) para la aplicacion
# --------------------------------------------------------------------------

MODEL_PATH = config.RESULTS_DIR / "scoal_model.pkl"


def medir(image_bgr: np.ndarray, peso_g: float | None = None) -> dict:
    """Segmenta una imagen y extrae sus mediciones fisicas, sin clasificar.

    Devuelve
      {'ok': bool, 'mask':, 'contour':, 'crop':, 'vector_a':,
       'mediciones': {nombre: valor, ...}}

    Si no hay contorno valido devuelve ok=False: es preferible no decir nada a
    medir el fondo. Esta funcion corre en el hilo de camara, por eso no toca
    ningun widget ni el modelo.
    """
    mascara, contorno = segment(image_bgr)
    if contorno is None:
        return {"ok": False, "mask": mascara, "contour": None, "crop": None,
                "vector_a": None, "mediciones": {}}

    recorte, recorte_mask = crop_to_mask(image_bgr, mascara)
    vector_a = physical.extract_physical(recorte_mask, recorte, peso_g)
    medidas = dict(zip(physical.feature_names(), vector_a))

    # Los ejes se recalculan sobre la mascara en coordenadas de la imagen
    # original, no del recorte, porque el overlay se dibuja sobre el frame.
    eje_mayor, eje_menor, angulo = physical.inertia_axes(mascara)
    medidas["angulo_rad"] = angulo
    medidas["eje_mayor_px"] = eje_mayor
    medidas["eje_menor_px"] = eje_menor

    return {"ok": True, "mask": mascara, "contour": contorno, "crop": recorte,
            "vector_a": vector_a, "mediciones": medidas}


class Scoal:
    """Modelo entrenado de las tres vias, listo para inferencia en vivo."""

    def __init__(self, clases: list[str]) -> None:
        self.clases = list(clases)
        self.vias: dict[str, dict] = {}

    # -- entrenamiento --------------------------------------------------
    def fit(self, datos: dict, dataset: str, vias=config.VIAS,
            k: int = config.K_DEFAULT, lam: float = config.LAMBDA_DEFAULT,
            train_idx: np.ndarray | None = None) -> "Scoal":
        raw = compute_raw(dataset, datos, vias)
        y = datos["y"]
        if train_idx is None:
            train_idx = np.arange(len(y))
        for via in vias:
            if via not in raw:
                continue
            self.vias[via] = train_via(via, raw[via], y, train_idx, k, lam,
                                       n_clases=len(self.clases))
        return self

    # -- inferencia sobre una imagen ------------------------------------
    def classify(self, medicion: dict) -> dict:
        """Clasifica una medicion ya hecha por `medir`, por las tres vias.

        Recibe el resultado de `medir` (que es lo que produce el hilo de
        camara) y devuelve, por via:

            {'A': {'clase':, 'confianza':, 'novedad':, 'conocido':}, ...}

        `conocido` es False cuando la muestra cae fuera del dominio entrenado;
        en ese caso `clase` sigue siendo la que el argmax eligio, porque para
        el informe interesa ver que habria dicho el modelo sin el rechazo, pero
        la interfaz debe mostrar "desconocido".

        Separar medir de clasificar permite dibujar el overlay en vivo a
        velocidad de camara y clasificar solo cuando el disparador de
        estabilidad lo pide.
        """
        if not medicion.get("ok"):
            return {}

        crudo = {
            "A": medicion["vector_a"][None, :],
            "B": to_gray_vector(medicion["crop"])[None, :],
        }
        if "C" in self.vias:
            from src.features import cnn
            crudo["C"] = cnn.extract_embeddings(medicion["crop"][None, ...])

        predicciones = {}
        for via, modelo in self.vias.items():
            if via not in crudo:
                continue
            idx, conf = predict_via(modelo, crudo[via])
            confianza = float(conf[0, idx[0]])
            nov = float(novedad_via(modelo, crudo[via])[0])
            conocido = bool(reject.es_conocido(
                [nov], [confianza], modelo.get("umbral_confianza", 0.0))[0])
            predicciones[via] = {
                "clase": self.clases[int(idx[0])],
                "confianza": confianza,
                "novedad": nov,
                "conocido": conocido or not config.REJECT_ENABLED,
            }
        return predicciones

    def predict_image(self, image_bgr: np.ndarray,
                      peso_g: float | None = None) -> dict:
        """Segmenta, mide y clasifica una imagen completa."""
        medicion = medir(image_bgr, peso_g)
        medicion["predicciones"] = self.classify(medicion)
        return medicion

    # -- persistencia ---------------------------------------------------
    def save(self, ruta: Path = MODEL_PATH) -> Path:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, "wb") as fh:
            pickle.dump(self, fh)
        return ruta

    @staticmethod
    def load(ruta: Path = MODEL_PATH) -> "Scoal":
        with open(ruta, "rb") as fh:
            return pickle.load(fh)


def demo() -> None:
    """Smoke test sin datasets: dos clases sinteticas separables."""
    rng = np.random.default_rng(0)
    crops, masks, y = [], [], []
    import cv2

    for etiqueta in (0, 1):
        for _ in range(20):
            img = np.full((128, 128, 3), 128, np.uint8)
            m = np.zeros((128, 128), np.uint8)
            r = 40 if etiqueta == 0 else 20
            color = (40, 40, 220) if etiqueta == 0 else (60, 200, 60)
            cx, cy = rng.integers(50, 78, 2)
            cv2.circle(img, (int(cx), int(cy)), r, color, -1)
            cv2.circle(m, (int(cx), int(cy)), r, 255, -1)
            crops.append(img)
            masks.append(m)
            y.append(etiqueta)

    datos = {"crops": np.array(crops), "masks": np.array(masks),
             "y": np.array(y), "clases": ["grande_rojo", "pequeno_verde"]}
    raw = {"A": physical.extract_batch(datos["masks"], datos["crops"]),
           "B": np.stack([to_gray_vector(c) for c in datos["crops"]])}

    idx = np.arange(len(y))
    for via in ("A", "B"):
        modelo = train_via(via, raw[via], datos["y"], idx, k=10, lam=1e-2,
                           n_clases=2)
        pred, conf = predict_via(modelo, raw[via])
        assert (pred == datos["y"]).mean() > 0.95, via
        assert np.allclose(conf.sum(axis=1), 1.0)

    # Sin objeto en la escena no debe haber prediccion.
    vacio = Scoal(datos["clases"]).predict_image(np.full((64, 64, 3), 128, np.uint8))
    assert vacio["ok"] is False and not vacio["predicciones"]

    # Rechazo: lo que se parece al entrenamiento se acepta, lo ajeno no.
    modelo_b = train_via("B", raw["B"], datos["y"], idx, k=10, lam=1e-2,
                         n_clases=2)
    assert np.isfinite(modelo_b["umbral_novedad"])
    nov_dentro = novedad_via(modelo_b, raw["B"])
    assert (nov_dentro <= 1.0).mean() >= 0.95, "rechaza demasiado entrenamiento"

    ajeno = np.stack([to_gray_vector(
        rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)) for _ in range(5)])
    assert (novedad_via(modelo_b, ajeno) > 1.0).all(), "ruido debe ser novedoso"

    print("pipeline demo ok")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Entrena el modelo de las tres vias.")
    ap.add_argument("--train", action="store_true",
                    help="entrenar y guardar en results/scoal_model.pkl")
    ap.add_argument("--dataset", choices=config.DATASETS,
                    default=config.DEFAULT_DATASET)
    args = ap.parse_args()

    if not args.train:
        demo()
        return

    from src.data.loader import load_dataset

    datos = load_dataset(args.dataset)
    modelo = Scoal(datos["clases"]).fit(datos, args.dataset)
    ruta = modelo.save()
    print("modelo entrenado (vias %s) -> %s"
          % (", ".join(sorted(modelo.vias)), ruta))


if __name__ == "__main__":
    # Reimportar el modulo bajo su nombre real antes de entrenar: ejecutado
    # como  python -m src.pipeline  este archivo es __main__, y pickle
    # guardaria la clase como __main__.Scoal, que la aplicacion no sabe leer.
    from src.pipeline import main as _main

    _main()
