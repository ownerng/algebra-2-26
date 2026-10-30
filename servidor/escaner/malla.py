"""Malla cerrada: V ∈ R^(N×3) (vértices) y F ∈ Z^(M×3) (caras, índices de V).

  • Frente: una rejilla sobre los píxeles del objeto; cada nodo se retroproyecta
    con P = Z_frente·K⁻¹·[u, v, 1]ᵀ y cada celda se parte en dos triángulos.
  • Atrás (extrusión): paralela al eje óptico, para no salirse de la silueta:
    P = Z₀·K⁻¹·[u, v, 1]ᵀ + (Z_atras − Z₀)·e₃. Sus triángulos van con el orden
    invertido para mirar hacia afuera.
  • Paredes: las aristas que usa un solo triángulo del frente forman el
    contorno; cada una se une con su copia trasera en un cuadrilátero.

Así toda arista queda compartida por exactamente dos caras: un sólido cerrado.

Al final se pasa a los ejes del objeto (x a la derecha, y hacia arriba, z hacia
la cámara) con  p = R·(P − c),  R = diag(1, −1, −1)  (giro de 180° en x).
"""

from dataclasses import dataclass

import numpy as np

from .geometria import inversa_K, retroproyectar

LADO_MAX = 180  # nodos de la rejilla a lo largo del lado mayor del objeto
R_OBJETO = np.diag([1.0, -1.0, -1.0])


@dataclass
class Malla:
    V: np.ndarray        # N × 3, mm, ejes del objeto
    F: np.ndarray        # M × 3, índices
    uv: np.ndarray | None        # N × 2, coordenadas de textura en la foto (una sola foto)
    colores: np.ndarray | None   # N × 3 (RGB 0–255), color por vértice (varias fotos)
    n_frente: int        # los primeros n_frente vértices son el frente; los siguientes n_frente, atrás
    caras: dict[str, int]  # cuántas caras de cada parte (frente, atrás, paredes), en ese orden en F
    centro: np.ndarray   # c, en ejes de la cámara


def construir(mascara: np.ndarray, Z_frente: np.ndarray, Z_atras: np.ndarray, K: np.ndarray, Z0: float) -> Malla:
    alto, ancho = mascara.shape
    vs_obj, us_obj = np.nonzero(mascara)
    u0, u1, v0, v1 = us_obj.min(), us_obj.max(), vs_obj.min(), vs_obj.max()
    paso = max(1, int(np.ceil(max(u1 - u0 + 1, v1 - v0 + 1) / LADO_MAX)))
    U, Vg = np.meshgrid(np.arange(u0, u1 + 1, paso), np.arange(v0, v1 + 1, paso))
    dentro = mascara[Vg, U] > 0

    # Índice de vértice de cada nodo (−1 fuera del objeto)
    indice = np.full(dentro.shape, -1, dtype=np.int64)
    n = int(dentro.sum())
    indice[dentro] = np.arange(n)
    u, v = U[dentro], Vg[dentro]

    K_inv = inversa_K(K)
    frente = retroproyectar(K_inv, u, v, Z_frente[v, u])
    atras = retroproyectar(K_inv, u, v, np.full(n, Z0))
    atras[:, 2] = Z_atras[v, u]

    # Celdas: a b / d e. La v de la foto crece hacia abajo, así que (a, d, b)
    # gira en sentido antihorario visto desde la cámara.
    a, b = indice[:-1, :-1], indice[:-1, 1:]
    d, e = indice[1:, :-1], indice[1:, 1:]
    A, B, D, E = a >= 0, b >= 0, d >= 0, e >= 0
    tri = lambda sel, *vs: np.stack([x[sel] for x in vs], axis=1)  # noqa: E731
    cuatro = A & B & D & E
    F_frente = np.concatenate([
        tri(cuatro, a, d, b), tri(cuatro, b, d, e),
        tri(A & B & D & ~E, a, d, b), tri(~A & B & D & E, b, d, e),
        tri(A & ~B & D & E, a, d, e), tri(A & B & ~D & E, a, e, b),
    ])
    F_atras = F_frente[:, ::-1] + n

    # Contorno: aristas (dirigidas) que aparecen en un solo triángulo
    aristas = np.concatenate([F_frente[:, [0, 1]], F_frente[:, [1, 2]], F_frente[:, [2, 0]]])
    clave = np.minimum(aristas[:, 0], aristas[:, 1]) * n + np.maximum(aristas[:, 0], aristas[:, 1])
    _, inv, veces = np.unique(clave, return_inverse=True, return_counts=True)
    borde = aristas[veces[inv] == 1]
    p, q = borde[:, 0], borde[:, 1]
    F_paredes = np.concatenate([np.stack([q, p, p + n], 1), np.stack([q, p + n, q + n], 1)])

    P = np.vstack([frente, atras])
    centro = (P.min(axis=0) + P.max(axis=0)) / 2
    V = (P - centro) @ R_OBJETO.T

    uv = np.stack([(u + 0.5) / ancho, 1 - (v + 0.5) / alto], axis=1)
    return Malla(
        V=V.astype(np.float32),
        F=np.vstack([F_frente, F_atras, F_paredes]).astype(np.uint32),
        uv=np.vstack([uv, uv]).astype(np.float32),
        colores=None,
        n_frente=n,
        caras={"frente": len(F_frente), "atras": len(F_atras), "paredes": len(F_paredes)},
        centro=centro,
    )


def volumen(V: np.ndarray, F: np.ndarray) -> float:
    """Volumen de un sólido cerrado (teorema de la divergencia):
    V = Σ det[p₀ p₁ p₂] / 6 sobre todos los triángulos."""
    t = V[F.astype(np.int64)].astype(np.float64)
    return float(np.linalg.det(t).sum() / 6)


def a_obj(m: Malla) -> str:
    """Formato OBJ (Blender, Visor 3D de Windows, etc.)."""
    lineas = ["# Escáner 3D de una foto: V (vértices, mm) y F (caras)"]
    if m.colores is not None:   # color por vértice: "v x y z r g b" (Blender, MeshLab)
        rgb = m.colores / 255
        lineas += [f"v {x:.2f} {y:.2f} {z:.2f} {r:.3f} {g:.3f} {b:.3f}" for (x, y, z), (r, g, b) in zip(m.V, rgb)]
        lineas += [f"f {a} {b} {c}" for a, b, c in (m.F.astype(np.int64) + 1)]
        return "\n".join(lineas) + "\n"
    lineas += [f"v {x:.2f} {y:.2f} {z:.2f}" for x, y, z in m.V]
    lineas += [f"vt {s:.4f} {t:.4f}" for s, t in m.uv]
    lineas += [f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in (m.F.astype(np.int64) + 1)]
    return "\n".join(lineas) + "\n"
