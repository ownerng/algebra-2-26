"""Modelo de cámara estenopeica y retroproyección.

Un punto P = (X, Y, Z) del espacio (ejes de la cámara: X a la derecha, Y hacia
abajo, Z hacia adelante) se ve en el píxel (u, v) con

    λ·[u, v, 1]ᵀ = K·P,      K = [[f, 0, cx], [0, f, cy], [0, 0, 1]],   λ = Z.

Conociendo la profundidad Z de un píxel, el punto se recupera con la inversa:

    P = Z · K⁻¹ · [u, v, 1]ᵀ.
"""

import numpy as np


def matriz_K(ancho: int, alto: int, fov_horizontal: float) -> np.ndarray:
    """Matriz intrínseca: la focal en píxeles sale del ángulo de visión horizontal."""
    f = (ancho / 2) / np.tan(np.radians(fov_horizontal) / 2)
    return np.array([[f, 0, ancho / 2], [0, f, alto / 2], [0, 0, 1]], dtype=np.float64)


def inversa_K(K: np.ndarray) -> np.ndarray:
    """K es triangular superior, así que su inversa se escribe directamente:

    K⁻¹ = [[1/fx, 0, −cx/fx], [0, 1/fy, −cy/fy], [0, 0, 1]].
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([[1 / fx, 0, -cx / fx], [0, 1 / fy, -cy / fy], [0, 0, 1]], dtype=np.float64)


def retroproyectar(K_inv: np.ndarray, u: np.ndarray, v: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Lleva N píxeles a R³ en una sola multiplicación de matrices.

    Los píxeles se apilan en coordenadas homogéneas como columnas de una matriz
    3 × N; K⁻¹ los convierte en rayos (con componente z = 1) y cada rayo se
    escala por su profundidad. Devuelve una matriz N × 3.
    """
    pixeles = np.vstack([u, v, np.ones_like(u)]).astype(np.float64)  # 3 × N
    rayos = K_inv @ pixeles                                           # 3 × N
    return (rayos * Z).T                                              # N × 3


def proyectar(K: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Operación inversa (para comprobar): N × 3 puntos → N × 2 píxeles."""
    q = (K @ P.T).T
    return q[:, :2] / q[:, 2:3]
