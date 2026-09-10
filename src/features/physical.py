"""
Via A - features fisicas: geometria, color y ejes de inercia.

Todo el vector se calcula con NumPy; OpenCV solo aporta el contorno y el
perimetro (medicion, no decision de clasificacion).

El ORDEN de las componentes es parte del contrato del modelo: se entrena y se
infiere con el mismo orden o los pesos aprendidos dejan de significar lo
mismo. FEATURE_NAMES es la unica fuente de verdad de ese orden y
tests/test_feature_order.py lo verifica.
"""

from __future__ import annotations

import cv2
import numpy as np

import config

EPS = 1e-12

# Orden canonico del vector de la via A.
FEATURE_NAMES: tuple[str, ...] = (
    "area",
    "perimetro",
    "aspecto",
    "circularidad",
    "R_med",
    "G_med",
    "B_med",
    "hu1", "hu2", "hu3", "hu4", "hu5", "hu6", "hu7",
    "eje_mayor",
    "eje_menor",
)

WEIGHT_NAME = "peso_g"      # se agrega al final solo si USE_WEIGHT


def feature_names(use_weight: bool | None = None) -> tuple[str, ...]:
    """Nombres del vector en el orden en que se produce."""
    if use_weight is None:
        use_weight = config.USE_WEIGHT
    return FEATURE_NAMES + ((WEIGHT_NAME,) if use_weight else ())


def inertia_axes(mask: np.ndarray) -> tuple[float, float, float]:
    """Ejes principales del objeto a partir de la matriz de inercia.

    Con los momentos centrales de segundo orden normalizados por el area,

        M = (1/mu00) [[mu20, mu11],
                      [mu11, mu02]]

    M es simetrica y definida positiva, asi que np.linalg.eigh entrega
    autovalores reales lambda1 >= lambda2 >= 0 y autovectores ortogonales.
    Los autovectores son las direcciones principales del objeto; los
    autovalores son las varianzas a lo largo de esas direcciones.

    Se devuelven longitudes en pixeles usando la elipse equivalente,
    L = 4 sqrt(lambda), que es la convencion habitual (para un disco de radio
    r da L = 2r en ambos ejes).

    Devuelve (eje_mayor, eje_menor, angulo_rad), con el angulo del eje mayor
    medido desde el eje x en (-pi/2, pi/2].
    """
    m = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    if m["m00"] <= EPS:
        return 0.0, 0.0, 0.0

    M = np.array([[m["mu20"], m["mu11"]],
                  [m["mu11"], m["mu02"]]], dtype=np.float64) / m["m00"]

    valores, vectores = np.linalg.eigh(M)      # eigh: ascendente
    valores = np.clip(valores, 0.0, None)
    mayor, menor = valores[1], valores[0]
    v = vectores[:, 1]                          # autovector del mayor
    angulo = float(np.arctan2(v[1], v[0]))
    if angulo <= -np.pi / 2:
        angulo += np.pi
    elif angulo > np.pi / 2:
        angulo -= np.pi

    return float(4.0 * np.sqrt(mayor)), float(4.0 * np.sqrt(menor)), angulo


def hu_moments(mask: np.ndarray) -> np.ndarray:
    """Los 7 momentos invariantes de Hu, en escala logaritmica.

        h'_i = -sign(h_i) log10(|h_i|)

    Los momentos crudos abarcan varios ordenes de magnitud (10^-1 a 10^-20) y
    sin comprimir dominarian la estandarizacion posterior. Son invariantes a
    traslacion, escala y rotacion, que es justo lo que se necesita con objetos
    que llegan girados sobre la banda.
    """
    m = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    hu = cv2.HuMoments(m).ravel()
    return -np.sign(hu) * np.log10(np.abs(hu) + EPS)


def color_medio(mask: np.ndarray, image_bgr: np.ndarray) -> tuple[float, float, float]:
    """Color medio del objeto (solo pixeles de la mascara), en orden R, G, B."""
    sel = mask > 0
    if not sel.any():
        return 0.0, 0.0, 0.0
    medio = image_bgr[sel].astype(np.float64).mean(axis=0)   # B, G, R
    return float(medio[2]), float(medio[1]), float(medio[0])


def extract_physical(
    mask: np.ndarray,
    image_bgr: np.ndarray,
    peso_g: float | None = None,
    use_weight: bool | None = None,
) -> np.ndarray:
    """Vector de features fisicas en el orden de FEATURE_NAMES.

    area          : numero de pixeles de la mascara
    perimetro     : longitud del contorno externo
    aspecto       : ancho / alto de la caja envolvente
    circularidad  : 4 pi A / P^2, vale 1 para un circulo perfecto
    R,G,B medios  : promedio del color dentro de la mascara
    hu1..hu7      : momentos de Hu en escala logaritmica
    eje_mayor/menor: elipse equivalente de la matriz de inercia

    Si use_weight (por defecto config.USE_WEIGHT) es False o peso_g es None,
    la componente de peso NO entra al vector. Los datasets publicos no traen
    masa, asi que la via A corre sin ella; la arquitectura la soporta para el
    trabajo futuro con celda de carga.
    """
    if use_weight is None:
        use_weight = config.USE_WEIGHT

    binaria = (mask > 0).astype(np.uint8)
    area = float(binaria.sum())

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        mayor = max(contornos, key=cv2.contourArea)
        perimetro = float(cv2.arcLength(mayor, True))
        _, _, w, h = cv2.boundingRect(mayor)
    else:
        perimetro, w, h = 0.0, 1, 1

    aspecto = float(w) / float(max(h, 1))
    circularidad = float(4.0 * np.pi * area / (perimetro ** 2)) if perimetro > EPS else 0.0

    r, g, b = color_medio(mask, image_bgr)
    hu = hu_moments(mask)
    eje_mayor, eje_menor, _ = inertia_axes(mask)

    vector = np.array(
        [area, perimetro, aspecto, circularidad, r, g, b,
         *hu, eje_mayor, eje_menor],
        dtype=np.float64,
    )

    if use_weight:
        if peso_g is None:
            raise ValueError(
                "USE_WEIGHT esta activo pero no se recibio peso_g: el vector "
                "quedaria mas corto en inferencia que en entrenamiento"
            )
        vector = np.append(vector, float(peso_g))

    assert vector.size == len(feature_names(use_weight))
    return vector


def extract_batch(masks: np.ndarray, crops: np.ndarray) -> np.ndarray:
    """Aplica extract_physical a un lote. Devuelve (m, n_features)."""
    return np.stack([extract_physical(mk, im) for mk, im in zip(masks, crops)])


def demo() -> None:
    """Un circulo y una barra deben separarse por circularidad y ejes."""
    circulo = np.zeros((128, 128), np.uint8)
    cv2.circle(circulo, (64, 64), 30, 255, -1)
    barra = np.zeros((128, 128), np.uint8)
    cv2.rectangle(barra, (20, 58), (108, 70), 255, -1)
    imagen = np.full((128, 128, 3), 128, np.uint8)
    imagen[..., 2] = 200                       # rojizo

    v_c = extract_physical(circulo, imagen)
    v_b = extract_physical(barra, imagen)

    nombres = feature_names(False)
    assert v_c.size == len(nombres) == 16
    assert 0.9 < v_c[nombres.index("circularidad")] <= 1.05
    assert v_b[nombres.index("circularidad")] < v_c[nombres.index("circularidad")]

    # Circulo: ejes casi iguales. Barra: eje mayor mucho mas largo.
    assert abs(v_c[nombres.index("eje_mayor")] - v_c[nombres.index("eje_menor")]) < 2.0
    assert v_b[nombres.index("eje_mayor")] > 4 * v_b[nombres.index("eje_menor")]

    # El eje mayor de una barra horizontal apunta en x.
    assert abs(inertia_axes(barra)[2]) < 0.05

    # Color medio dentro de la mascara, en orden R, G, B.
    assert abs(v_c[nombres.index("R_med")] - 200) < 1.0

    # Mascara vacia no debe explotar.
    assert np.isfinite(extract_physical(np.zeros((32, 32), np.uint8),
                                        np.zeros((32, 32, 3), np.uint8))).all()

    print("physical demo ok")


if __name__ == "__main__":
    demo()
