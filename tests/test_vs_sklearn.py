"""
Verificacion numerica del motor propio contra sklearn.

sklearn NO es parte del sistema: aparece unicamente en este archivo y solo
como implementacion de referencia. Si `grep -rn "sklearn" src/` devuelve algo,
el proyecto esta incumpliendo su propia restriccion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import lstsq  # noqa: E402
from src.model.knn import KNN  # noqa: E402
from src.model.metrics import accuracy, confusion_matrix  # noqa: E402
from src.model.pca import PCA, Standardizer  # noqa: E402

TOL = 1e-8


@pytest.fixture
def datos():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 12))
    y = rng.integers(0, 4, size=300)
    return X, y


def test_minimos_cuadrados_coincide_con_ridge(datos):
    """W propia vs sklearn.linear_model.Ridge, error < 1e-8."""
    from sklearn.linear_model import Ridge

    X, y = datos
    Xb = lstsq.add_bias(X)
    Y = lstsq.one_hot(y, 4)

    for lam in (1e-3, 1e-2, 1.0, 10.0):
        W = lstsq.fit_least_squares(Xb, Y, lam)
        # fit_intercept=False porque la columna de unos ya esta en Xb, y ahi
        # el termino independiente si queda regularizado en ambas partes.
        ref = Ridge(alpha=lam, fit_intercept=False, solver="cholesky").fit(Xb, Y)
        assert np.abs(W - ref.coef_.T).max() < TOL, lam


def test_pseudoinversa_sin_regularizacion(datos):
    """Con lambda = 0 la solucion debe ser la de la pseudoinversa."""
    X, y = datos
    Xb = lstsq.add_bias(X)
    Y = lstsq.one_hot(y, 4)
    assert np.abs(lstsq.fit_least_squares(Xb, Y, 0.0) - np.linalg.pinv(Xb) @ Y).max() < 1e-9


def test_pca_coincide_con_sklearn(datos):
    """Componentes, varianza explicada y proyeccion coinciden salvo signo."""
    from sklearn.decomposition import PCA as SkPCA

    X, _ = datos
    propia = PCA(5).fit(X)
    ref = SkPCA(n_components=5, svd_solver="full").fit(X)

    assert np.abs(propia.explained_variance_ - ref.explained_variance_).max() < 1e-9
    assert np.abs(propia.explained_variance_ratio_
                  - ref.explained_variance_ratio_).max() < 1e-9

    # El signo de un autovector es arbitrario: se compara en valor absoluto.
    assert np.abs(np.abs(propia.components_.T) - np.abs(ref.components_)).max() < 1e-8
    assert np.abs(np.abs(propia.transform(X)) - np.abs(ref.transform(X))).max() < 1e-8


def test_standardizer_coincide_con_sklearn(datos):
    from sklearn.preprocessing import StandardScaler

    X, _ = datos
    propia = Standardizer().fit_transform(X)
    ref = StandardScaler().fit_transform(X)
    assert np.abs(propia - ref).max() < 1e-10


def test_knn_coincide_con_sklearn():
    """Mismo voto que sklearn sobre datos sin empates."""
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(1)
    centros = rng.normal(0, 6, (4, 6))
    X = np.vstack([c + rng.normal(0, 0.6, (60, 6)) for c in centros])
    y = np.repeat(np.arange(4), 60)

    propia = KNN(5).fit(X, y).predict(X)
    ref = KNeighborsClassifier(5, algorithm="brute").fit(X, y).predict(X)
    assert (propia == ref).mean() == 1.0


def test_metricas_coinciden_con_sklearn(datos):
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix as sk_cm

    _, y = datos
    rng = np.random.default_rng(2)
    pred = np.where(rng.random(len(y)) < 0.7, y, rng.integers(0, 4, len(y)))

    assert abs(accuracy(y, pred) - accuracy_score(y, pred)) < 1e-12
    assert (confusion_matrix(y, pred, 4) == sk_cm(y, pred, labels=range(4))).all()


def test_eigenobjetos_coinciden_con_pca_de_sklearn():
    """El truco de Turk-Pentland (m << d) da la misma base que la PCA de sklearn."""
    from sklearn.decomposition import PCA as SkPCA

    from src.features.eigen import fit_eigenobjects, project

    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 500))          # m << d, se activa el truco

    propia = fit_eigenobjects(X, k=6)
    ref = SkPCA(n_components=6, svd_solver="full").fit(X)

    assert np.abs(np.abs(propia["components"].T)
                  - np.abs(ref.components_)).max() < 1e-7
    assert np.abs(np.abs(project(X, propia)) - np.abs(ref.transform(X))).max() < 1e-7
