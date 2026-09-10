"""
Estandarizacion y analisis de componentes principales, NumPy puro.

PCA aqui es la de proposito general (sobre vectores de features, via C y
diagnosticos). La version para imagenes con el truco de Turk-Pentland vive en
src/features/eigen.py, porque alli m << n y conviene la matriz pequena.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


class Standardizer:
    """Centrado y escalado por columna:  z = (x - mu) / sigma.

    Los estadisticos se ajustan SOLO con el conjunto de entrenamiento y se
    aplican tal cual a test. Ajustarlos con todo el dataset filtraria
    informacion del test al entrenamiento.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        sigma = X.std(axis=0)
        # Una columna constante tiene sigma = 0: se deja en 1 para no dividir
        # por cero. Esa columna aporta cero informacion, no cero validez.
        self.scale_ = np.where(sigma < EPS, 1.0, sigma)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Standardizer sin ajustar")
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PCA:
    """Analisis de componentes principales por SVD de la matriz centrada.

    Sea A la matriz de datos centrada (m x n) y A = U S V^T su SVD. Las
    columnas de V son los autovectores de la matriz de covarianza

        C = A^T A / (m - 1)

    y los autovalores de C valen  s_i^2 / (m - 1). Se usa la SVD de A en vez
    de formar C explicitamente: elevar al cuadrado los datos eleva al cuadrado
    tambien el numero de condicion.
    """

    def __init__(self, n_components: int) -> None:
        self.n_components = int(n_components)
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None   # (n, k)
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        X = np.asarray(X, dtype=np.float64)
        m = X.shape[0]
        self.mean_ = X.mean(axis=0)
        A = X - self.mean_

        k = min(self.n_components, min(A.shape))
        _, s, Vt = np.linalg.svd(A, full_matrices=False)

        var = s ** 2 / max(m - 1, 1)
        total = var.sum()
        self.components_ = Vt[:k].T
        self.explained_variance_ = var[:k]
        self.explained_variance_ratio_ = var[:k] / (total if total > EPS else 1.0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Proyeccion ortogonal sobre el subespacio:  Z = (X - mu) V_k."""
        if self.components_ is None:
            raise RuntimeError("PCA sin ajustar")
        return (np.asarray(X, dtype=np.float64) - self.mean_) @ self.components_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Vuelta al espacio original:  X ~ Z V_k^T + mu (reconstruccion)."""
        return np.asarray(Z, dtype=np.float64) @ self.components_.T + self.mean_

    @property
    def cumulative_variance(self) -> float:
        """Fraccion de varianza explicada por las k componentes retenidas."""
        return float(self.explained_variance_ratio_.sum())


def demo() -> None:
    rng = np.random.default_rng(0)
    # Datos en un plano de R^5: dos componentes deben explicar casi todo.
    base = rng.normal(size=(2, 5))
    X = rng.normal(size=(200, 2)) @ base + rng.normal(0, 1e-3, (200, 5))

    p = PCA(2).fit(X)
    assert p.cumulative_variance > 0.99
    Z = p.transform(X)
    assert Z.shape == (200, 2)
    # Reconstruccion casi exacta porque los datos viven en ese plano.
    assert np.abs(p.inverse_transform(Z) - X).max() < 1e-2
    # Base ortonormal.
    assert np.allclose(p.components_.T @ p.components_, np.eye(2), atol=1e-8)

    s = Standardizer().fit(X)
    Z2 = s.transform(X)
    assert np.allclose(Z2.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(Z2.std(axis=0), 1, atol=1e-8)

    cte = np.hstack([X, np.ones((200, 1))])       # columna constante
    assert np.isfinite(Standardizer().fit_transform(cte)).all()

    print("pca demo ok")


if __name__ == "__main__":
    demo()
