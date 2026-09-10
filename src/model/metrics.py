"""
Metricas de evaluacion, NumPy puro.
"""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraccion de aciertos."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_clases: int | None = None) -> np.ndarray:
    """Matriz de confusion C, con C[i, j] = casos de clase real i predichos j.

    Se construye con np.add.at sobre los pares (real, predicho), que es el
    conteo bidimensional sin bucle en Python.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    k = int(n_clases if n_clases is not None
            else max(y_true.max(), y_pred.max()) + 1)
    C = np.zeros((k, k), dtype=np.int64)
    np.add.at(C, (y_true, y_pred), 1)
    return C


def per_class_accuracy(C: np.ndarray) -> np.ndarray:
    """Recall por clase: diagonal dividida por el total de cada fila."""
    filas = C.sum(axis=1)
    return np.divide(np.diag(C), filas, out=np.zeros(len(C), float),
                     where=filas > 0)


def macro_f1(C: np.ndarray) -> float:
    """F1 promediada sin ponderar por soporte.

        precision_i = C_ii / sum_j C_ji ,  recall_i = C_ii / sum_j C_ij
        F1_i = 2 P R / (P + R)

    Con clases balanceadas a ~300 imagenes es casi la exactitud, pero delata
    una clase que el modelo nunca predice.
    """
    tp = np.diag(C).astype(float)
    pred = C.sum(axis=0).astype(float)
    real = C.sum(axis=1).astype(float)
    p = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    r = np.divide(tp, real, out=np.zeros_like(tp), where=real > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return float(f.mean())


def demo() -> None:
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 1, 1, 1, 2, 2])
    C = confusion_matrix(y, p, 3)

    assert C.sum() == len(y)
    assert C[0, 0] == 1 and C[0, 1] == 1
    assert abs(accuracy(y, p) - 5 / 6) < 1e-12
    assert np.allclose(per_class_accuracy(C), [0.5, 1.0, 1.0])
    assert 0.0 <= macro_f1(C) <= 1.0

    # Una clase nunca predicha no debe dividir por cero.
    C2 = confusion_matrix(np.array([0, 1]), np.array([0, 0]), 3)
    assert np.isfinite(macro_f1(C2))

    print("metrics demo ok")


if __name__ == "__main__":
    demo()
