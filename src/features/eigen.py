"""
Via B - eigen-objetos.

Cada imagen de 64x64 en gris es un punto de R^4096. El conjunto de imagenes de
un objeto no llena ese espacio: vive en un subespacio de dimension mucho menor.
Se busca una base ortonormal de ese subespacio (los eigen-objetos) y se
representa cada imagen por sus coordenadas en esa base.

Es el metodo de Turk y Pentland (1991) para eigenfaces, aplicado a objetos.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def fit_eigenobjects(images: np.ndarray, k: int) -> dict:
    """Calcula los k eigen-objetos de un conjunto de imagenes aplanadas.

    Parametros
    ----------
    images : (m, d) imagenes aplanadas, una por fila
    k : numero de componentes a retener

    Devuelve
    --------
    {'mean': (d,), 'components': (d, k), 'explained_var': (k,),
     'explained_var_ratio': (k,)}

    Metodo
    ------
    Sea A la matriz centrada (m, d), A = X - mu. La covarianza es
    C = A^T A / (m-1), de tamano d x d. Con d = 4096 esa matriz tiene 16.7
    millones de entradas y diagonalizarla es innecesario cuando m << d, porque
    su rango es a lo sumo m-1: solo m-1 autovalores son distintos de cero.

    Truco de Turk-Pentland: se diagonaliza la matriz pequena A A^T (m x m). Si

        A A^T u = mu_i u        (u autovector de la matriz pequena)

    multiplicando por A^T por la izquierda:

        A^T A (A^T u) = mu_i (A^T u)

    o sea que  v = A^T u / ||A^T u||  es autovector de A^T A con el mismo
    autovalor. Se obtienen los autovectores de la matriz grande sin construirla.

    Cuando m >= d el truco no ahorra nada y se usa la SVD de A directamente,
    que ademas es mas estable que formar A^T A (elevar al cuadrado los datos
    eleva al cuadrado el numero de condicion).
    """
    X = np.asarray(images, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("images debe ser (m, d)")
    m, d = X.shape
    k = int(min(k, m - 1 if m <= d else d))
    if k < 1:
        raise ValueError("k debe ser >= 1 y caber en el rango de los datos")

    mu = X.mean(axis=0)
    A = X - mu

    if m < d:
        # --- Turk-Pentland: diagonalizar A A^T, de tamano m x m ---
        G = A @ A.T                              # matriz de Gram, simetrica
        valores, U = np.linalg.eigh(G)           # ascendente, base ortonormal
        orden = np.argsort(valores)[::-1][:k]
        valores = np.clip(valores[orden], 0.0, None)
        V = A.T @ U[:, orden]                    # mapear al espacio grande
        normas = np.linalg.norm(V, axis=0)
        normas[normas < EPS] = 1.0
        V /= normas                              # base ortonormal en R^d
        total = np.clip(np.linalg.eigvalsh(G), 0.0, None).sum()
    else:
        _, s, Vt = np.linalg.svd(A, full_matrices=False)
        valores = s[:k] ** 2
        V = Vt[:k].T
        total = (s ** 2).sum()

    var = valores / max(m - 1, 1)
    ratio = valores / (total if total > EPS else 1.0)

    return {
        "mean": mu,
        "components": V,
        "explained_var": var,
        "explained_var_ratio": ratio,
    }


def project(images: np.ndarray, modelo: dict) -> np.ndarray:
    """Proyeccion ortogonal sobre el subespacio propio.

        z = V_k^T (x - mu)

    Devuelve (m, k): las coordenadas de cada imagen en la base de eigen-objetos.
    """
    X = np.asarray(images, dtype=np.float64)
    return (X - modelo["mean"]) @ modelo["components"]


def reconstruct(Z: np.ndarray, modelo: dict) -> np.ndarray:
    """Reconstruccion desde el subespacio:  x ~ V_k z + mu.

    Util para la figura del informe: mostrar como se degrada la reconstruccion
    al bajar k hace visible cuanta informacion retiene el subespacio.
    """
    return np.asarray(Z, dtype=np.float64) @ modelo["components"].T + modelo["mean"]


def demo() -> None:
    rng = np.random.default_rng(0)

    # Caso m < d: 30 imagenes de 400 pixeles generadas desde 3 patrones.
    base = rng.normal(size=(3, 400))
    X = rng.normal(size=(30, 3)) @ base
    modelo = fit_eigenobjects(X, k=3)

    assert modelo["components"].shape == (400, 3)
    # Base ortonormal.
    G = modelo["components"].T @ modelo["components"]
    assert np.allclose(G, np.eye(3), atol=1e-8)
    # Tres patrones explican toda la varianza.
    assert modelo["explained_var_ratio"].sum() > 0.999
    # Reconstruccion exacta dentro del subespacio que genero los datos.
    Z = project(X, modelo)
    assert np.abs(reconstruct(Z, modelo) - X).max() < 1e-8

    # Caso m > d: mismo resultado por la via de la SVD directa.
    Y = rng.normal(size=(200, 5)) @ rng.normal(size=(5, 5))
    m2 = fit_eigenobjects(Y, k=5)
    assert m2["components"].shape == (5, 5)
    assert np.allclose(m2["components"].T @ m2["components"], np.eye(5), atol=1e-8)

    # k mayor que el rango disponible se recorta en vez de fallar.
    assert fit_eigenobjects(X, k=999)["components"].shape[1] == 29

    print("eigen demo ok")


if __name__ == "__main__":
    demo()
