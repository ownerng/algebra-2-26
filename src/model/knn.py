"""
Baseline k-NN con norma euclidiana, NumPy puro.

Responde a la pregunta: el clasificador lineal, aporta algo sobre memorizar
el conjunto de entrenamiento? Si k-NN empata o gana, la representacion ya
separa las clases y el clasificador no esta haciendo el trabajo.
"""

from __future__ import annotations

import numpy as np


def pairwise_sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matriz (m_a, m_b) de distancias euclidianas al cuadrado.

    Identidad usada:  ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2

    Se calcula asi para reducirlo a un unico producto de matrices A B^T, mucho
    mas rapido que el bucle directo. El maximo con cero corrige los negativos
    minusculos que introduce la cancelacion en punto flotante.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    a2 = np.einsum("ij,ij->i", A, A)[:, None]
    b2 = np.einsum("ij,ij->i", B, B)[None, :]
    return np.maximum(a2 - 2.0 * (A @ B.T) + b2, 0.0)


class KNN:
    """k vecinos mas cercanos por voto mayoritario, sin ponderar."""

    def __init__(self, k: int = 5) -> None:
        self.k = int(k)
        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None
        self.n_clases_ = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        self.X_ = np.asarray(X, dtype=np.float64)
        self.y_ = np.asarray(y, dtype=np.int64).ravel()
        self.n_clases_ = int(self.y_.max()) + 1
        return self

    def predict(self, X: np.ndarray, bloque: int = 512) -> np.ndarray:
        if self.X_ is None:
            raise RuntimeError("KNN sin ajustar")
        X = np.asarray(X, dtype=np.float64)
        k = min(self.k, len(self.X_))
        salida = np.empty(len(X), dtype=np.int64)

        # Por bloques: la matriz de distancias completa es m_test x m_train y
        # con 100k muestras no cabe en memoria.
        for ini in range(0, len(X), bloque):
            fin = min(ini + bloque, len(X))
            d = pairwise_sqdist(X[ini:fin], self.X_)
            vecinos = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
            votos = self.y_[vecinos]
            conteo = np.apply_along_axis(
                np.bincount, 1, votos, minlength=self.n_clases_)
            salida[ini:fin] = np.argmax(conteo, axis=1)
        return salida


def demo() -> None:
    rng = np.random.default_rng(0)
    centros = np.array([[0.0, 0.0], [5.0, 5.0]])
    X = np.vstack([c + rng.normal(0, 0.4, (50, 2)) for c in centros])
    y = np.repeat(np.arange(2), 50)

    modelo = KNN(3).fit(X, y)
    assert (modelo.predict(X) == y).mean() > 0.98

    d = pairwise_sqdist(X[:3], X[:3])
    assert np.allclose(np.diag(d), 0.0)
    assert (d >= 0).all()
    assert np.allclose(d, d.T)

    print("knn demo ok")


if __name__ == "__main__":
    demo()
