"""
Rechazo de muestras fuera de dominio: la clase "desconocido".

Por que hace falta
------------------
El clasificador de minimos cuadrados decide con  argmax(x W), y el argmax de
un vector siempre existe. Da igual que la imagen sea una manzana, un celular o
una cara: el modelo esta obligado a devolver una de las clases entrenadas, la
menos lejana. Eso no es un error del ajuste, es una propiedad del conjunto
cerrado de clases, y sin un criterio de rechazo la aplicacion miente con
aplomo.

Dos criterios, los dos de algebra lineal, ninguno heuristico
------------------------------------------------------------
1. **Distancia al subespacio** (vias B y C). Turk y Pentland (1991) la llaman
   *distance from face space*: una muestra se proyecta sobre los k vectores
   propios y se reconstruye; si la reconstruccion se aleja del original, la
   muestra no vive en el subespacio que generan las clases entrenadas.

       d(x) = ||x - x_rec||_2 / ||x - mu||_2

   El denominador vuelve la medida adimensional y comparable entre muestras
   de energia distinta.

2. **Norma estandarizada** (via A, que no reduce dimension y por lo tanto no
   tiene subespacio del que salirse). Tras estandarizar, cada componente de
   entrenamiento tiene media 0 y desviacion 1, de modo que

       n(z) = ||z||_2 / sqrt(p)

   vale ~1 para una muestra tipica y crece cuando las features se salen del
   rango visto. Es la distancia de Mahalanobis con covarianza diagonal, que es
   la que el estandarizador ya estima.

El umbral no se elige a mano: se lee del propio conjunto de entrenamiento como
un percentil alto de su distribucion. Con el percentil 99, por construccion, se
rechaza el 1% de las muestras de entrenamiento, y ese numero es el que se
reporta como tasa de falso rechazo esperada.
"""

from __future__ import annotations

import numpy as np

import config

EPS = 1e-12


def distancia_al_subespacio(X: np.ndarray, X_rec: np.ndarray,
                            media: np.ndarray) -> np.ndarray:
    """Residuo relativo de reconstruccion, (m,).

        d(x) = ||x - x_rec||_2 / ||x - mu||_2

    Vale 0 si la muestra queda exactamente dentro del subespacio y tiende a 1
    cuando la proyeccion no captura nada de ella.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    X_rec = np.atleast_2d(np.asarray(X_rec, dtype=np.float64))
    numerador = np.linalg.norm(X - X_rec, axis=1)
    denominador = np.linalg.norm(X - np.asarray(media, dtype=np.float64), axis=1)
    return numerador / np.maximum(denominador, EPS)


def norma_estandarizada(Z: np.ndarray) -> np.ndarray:
    """Norma euclidea por fila dividida por sqrt(p), (m,).

    Z ya debe venir estandarizado y SIN la columna de unos: el termino
    independiente no es una medicion y sumaria una constante a la norma.
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    return np.linalg.norm(Z, axis=1) / np.sqrt(max(Z.shape[1], 1))


def umbral(valores: np.ndarray,
           percentil: float = config.REJECT_PERCENTILE) -> float:
    """Umbral de rechazo leido del entrenamiento como un percentil alto."""
    valores = np.asarray(valores, dtype=np.float64).ravel()
    if valores.size == 0:
        return float("inf")
    return float(np.percentile(valores, percentil))


def novedad(valores: np.ndarray, umbral_: float) -> np.ndarray:
    """Distancia normalizada al umbral: 1.0 es exactamente el limite.

    Se expone asi, y no como el valor crudo, porque un unico numero con el 1
    como frontera se puede mostrar en la interfaz sin explicar que mide cada
    via de forma distinta.
    """
    if not np.isfinite(umbral_) or umbral_ <= 0:
        return np.zeros_like(np.atleast_1d(np.asarray(valores, dtype=np.float64)))
    return np.atleast_1d(np.asarray(valores, dtype=np.float64)) / umbral_


def es_conocido(novedad_: np.ndarray, confianza: np.ndarray,
                umbral_confianza: float) -> np.ndarray:
    """True si la muestra pertenece al dominio entrenado, (m,).

    Se exigen las dos condiciones a la vez. Son independientes y detectan
    cosas distintas: la novedad mide si la muestra se parece a lo visto, la
    confianza mide si alguna clase la reclama con margen. Un objeto parecido
    al dataset pero ambiguo entre dos clases pasa la primera y falla la
    segunda; un objeto ajeno al dataset falla la primera aunque alguna clase
    lo reclame con fuerza.

    `umbral_confianza` tambien sale del entrenamiento (percentil bajo de la
    confianza ganadora sobre train) y NO es una constante: con 10 clases la
    confianza tipica de minimos cuadrados sobre un acierto ronda 0.3, asi que
    cualquier umbral fijado a ojo rechaza el dataset entero o no rechaza nada.
    """
    novedad_ = np.atleast_1d(np.asarray(novedad_, dtype=np.float64))
    confianza = np.atleast_1d(np.asarray(confianza, dtype=np.float64))
    return (novedad_ <= 1.0) & (confianza >= umbral_confianza)


def demo() -> None:
    """Muestras del dominio se aceptan; muestras ajenas se rechazan."""
    rng = np.random.default_rng(0)

    # Subespacio: un plano de R^10 generado por dos direcciones ortonormales.
    base = np.linalg.qr(rng.normal(size=(10, 2)))[0]
    media = np.zeros(10)
    coef = rng.normal(size=(200, 2))
    dentro = coef @ base.T
    rec_dentro = (dentro @ base) @ base.T
    d_dentro = distancia_al_subespacio(dentro, rec_dentro, media)
    assert d_dentro.max() < 1e-9, "una muestra del subespacio no tiene residuo"

    # Muestra ajena: componente fuerte fuera del plano.
    fuera = dentro[:20] + 5.0 * np.linalg.qr(rng.normal(size=(10, 3)))[0][:, 2]
    rec_fuera = (fuera @ base) @ base.T
    d_fuera = distancia_al_subespacio(fuera, rec_fuera, media)
    assert d_fuera.min() > 0.5, "una muestra ajena debe dejar residuo grande"

    # El umbral sale del entrenamiento y rechaza el percentil pactado.
    valores = rng.gamma(2.0, 1.0, size=5000)
    u = umbral(valores, 99.0)
    rechazados = (valores > u).mean()
    assert abs(rechazados - 0.01) < 0.005, rechazados

    # Norma estandarizada: ~1 en el dominio, grande fuera.
    Z = rng.normal(size=(500, 16))
    assert abs(norma_estandarizada(Z).mean() - 1.0) < 0.1
    assert norma_estandarizada(Z + 6.0).min() > 3.0

    # Decision conjunta.
    assert es_conocido([0.5], [0.9], 0.2)[0]
    assert not es_conocido([3.0], [0.9], 0.2)[0], "novedad alta debe rechazar"
    assert not es_conocido([0.5], [0.05], 0.2)[0], "confianza baja debe rechazar"

    # El umbral de confianza sale del train, no de una constante: con 10 clases
    # una confianza de 0.28 es un acierto normal y no debe rechazarse.
    conf_train = rng.beta(3.0, 7.0, size=5000)          # media ~0.3
    u_conf = umbral(conf_train, 100.0 - 99.0)
    assert u_conf < 0.28, "el umbral no puede rechazar la confianza tipica"
    assert (conf_train < u_conf).mean() < 0.02

    print("reject demo ok")


if __name__ == "__main__":
    demo()
