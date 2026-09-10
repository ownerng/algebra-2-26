"""
Clasificador multiclase de minimos cuadrados con regularizacion de Tikhonov.

NumPy puro. Prohibido sklearn como motor (solo aparece en
tests/test_vs_sklearn.py para verificar que este codigo da lo mismo).
"""

from __future__ import annotations

import numpy as np


def one_hot(y: np.ndarray, n_clases: int | None = None) -> np.ndarray:
    """Etiquetas enteras -> matriz indicadora Y in {0,1}^(m x k).

    Y[i, j] = 1 si la muestra i pertenece a la clase j, 0 en otro caso.
    """
    y = np.asarray(y, dtype=np.int64).ravel()
    k = int(n_clases if n_clases is not None else y.max() + 1)
    Y = np.zeros((y.size, k), dtype=np.float64)
    Y[np.arange(y.size), y] = 1.0
    return Y


def add_bias(X: np.ndarray) -> np.ndarray:
    """Agrega la columna de unos: X -> [1, X].

    El termino independiente entra como una columna mas de la matriz de
    diseno, no como un parametro aparte, para que la solucion cerrada siga
    siendo un unico sistema lineal.
    """
    X = np.asarray(X, dtype=np.float64)
    return np.hstack([np.ones((X.shape[0], 1)), X])


def fit_least_squares(X: np.ndarray, Y: np.ndarray, lam: float = 0.0) -> np.ndarray:
    """Resuelve el problema de minimos cuadrados regularizado.

        min_W  ||X W - Y||_F^2 + lambda ||W||_F^2

    Anulando el gradiente respecto de W:

        2 X^T (X W - Y) + 2 lambda W = 0
        (X^T X + lambda I) W = X^T Y
        W = (X^T X + lambda I)^{-1} X^T Y

    Con lambda = 0 esto es W = X^+ Y, la pseudoinversa de Moore-Penrose.

    Parametros
    ----------
    X : (m, n) matriz de diseno, ya debe incluir la columna de unos
    Y : (m, k) etiquetas one-hot
    lam : lambda >= 0

    Devuelve
    --------
    W : (n, k)

    Nota numerica: se usa np.linalg.solve y NUNCA np.linalg.inv. Invertir y
    despues multiplicar amplifica el error por el numero de condicion de la
    matriz una segunda vez, ademas de costar mas operaciones; resolver el
    sistema factoriza una sola vez. Si aun asi la matriz normal resulta
    singular (lambda = 0 con columnas linealmente dependientes) se cae a
    np.linalg.lstsq, que usa SVD y devuelve la solucion de norma minima.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X e Y deben tener el mismo numero de filas")
    if lam < 0:
        raise ValueError("lambda debe ser >= 0")

    n = X.shape[1]
    A = X.T @ X + lam * np.eye(n)
    b = X.T @ Y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(X, Y, rcond=None)[0]


def scores(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Puntajes crudos por clase: S = X W, de forma (m, k)."""
    return np.asarray(X, dtype=np.float64) @ W


def predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Indices de clase predichos, (m,), via argmax(X W, axis=1)."""
    return np.argmax(scores(X, W), axis=1)


def confidence(S: np.ndarray) -> np.ndarray:
    """Puntajes -> pseudo-probabilidades por fila.

        p_j = max(s_j, 0) / sum_i max(s_i, 0)

    Por que sirve el puntaje crudo y no hace falta una softmax: si la matriz
    de diseno incluye la columna de unos, el vector 1 esta en el espacio
    columna de X, y como las filas de Y (one-hot) suman uno, la solucion de
    minimos cuadrados cumple  X W 1 = 1  exactamente cuando lambda = 0, y de
    forma aproximada cuando lambda > 0. Es decir, los puntajes de cada fila ya
    suman ~1 por construccion; solo falta recortar los negativos, que existen
    porque nada obliga a la regresion a quedarse dentro de [0, 1].

    Sigue sin ser una probabilidad calibrada y no debe reportarse como tal en
    el informe: es una lectura del margen entre clases, util en la interfaz.
    """
    S = np.atleast_2d(np.asarray(S, dtype=np.float64))
    positivos = np.clip(S, 0.0, None)
    total = positivos.sum(axis=1, keepdims=True)
    # Fila entera negativa (ninguna clase reclama la muestra): reparto uniforme
    # antes que dividir por cero.
    uniforme = np.full_like(positivos, 1.0 / positivos.shape[1])
    return np.where(total > 0, positivos / np.where(total > 0, total, 1.0), uniforme)


def condition_number(X: np.ndarray, lam: float = 0.0) -> float:
    """Numero de condicion de (X^T X + lambda I) en norma 2.

        cond_2(A) = sigma_max(A) / sigma_min(A)

    Es el diagnostico que acompana el barrido de lambda: mide cuanto amplifica
    el sistema los errores de los datos. La regularizacion desplaza todos los
    autovalores en +lambda, asi que cond baja monotonamente con lambda.
    """
    X = np.asarray(X, dtype=np.float64)
    A = X.T @ X + lam * np.eye(X.shape[1])
    sv = np.linalg.svd(A, compute_uv=False)
    if sv[-1] <= 0:
        return float("inf")
    return float(sv[0] / sv[-1])


def demo() -> None:
    """Comprobacion minima: datos linealmente separables se clasifican bien."""
    rng = np.random.default_rng(0)
    centros = np.array([[0.0, 0.0], [6.0, 0.0], [0.0, 6.0]])
    X = np.vstack([c + rng.normal(0, 0.5, (60, 2)) for c in centros])
    y = np.repeat(np.arange(3), 60)

    Xb = add_bias(X)
    W = fit_least_squares(Xb, one_hot(y, 3), lam=1e-3)
    assert W.shape == (3, 3)
    assert (predict(Xb, W) == y).mean() > 0.98

    p = confidence(scores(Xb, W))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert p.min() >= 0.0
    # La clase predicha es la de mayor confianza, y con datos separables la
    # confianza debe ser alta, no repartida.
    assert (np.argmax(p, axis=1) == predict(Xb, W)).all()
    assert p.max(axis=1).mean() > 0.8

    # Los puntajes de cada fila suman uno por construccion cuando lambda = 0.
    W0 = fit_least_squares(Xb, one_hot(y, 3), lam=0.0)
    assert np.abs(scores(Xb, W0).sum(axis=1) - 1.0).max() < 1e-9

    # La regularizacion nunca empeora el condicionamiento.
    assert condition_number(Xb, 0.0) >= condition_number(Xb, 1.0)

    # Con una columna duplicada (rango deficiente) y lambda = 0 la matriz
    # normal es singular; la caida a lstsq debe seguir devolviendo algo usable.
    Xdup = np.hstack([Xb, Xb[:, 1:2]])
    Wdup = fit_least_squares(Xdup, one_hot(y, 3), lam=0.0)
    assert np.isfinite(Wdup).all()

    print("lstsq demo ok")


if __name__ == "__main__":
    demo()
