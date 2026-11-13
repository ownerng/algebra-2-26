"""Procesamiento alto: el objeto completo con IA generativa (imagen → 3D).

Una foto no trae la parte de atrás del objeto; el modo bajo la extruye o la
talla con varias vistas. Aquí la inventa Hunyuan3D v3.1 Pro (Tencent), que se
ejecuta en fal.ai: recibe la foto del objeto (y, si la hay, la trasera) y
devuelve una malla con textura en GLB.

La clave de fal.ai va en la variable de entorno FAL_KEY del servidor (nunca en
el navegador). La API es una cola: se envía el pedido, se pregunta su estado
hasta que termina y se descarga el resultado.
"""

import base64
import json
import os
import ssl
import time
import urllib.request

import certifi
import cv2
import numpy as np

MODELO = "fal-ai/hunyuan-3d/v3.1/pro/image-to-3d"
SEGMENTAR = "fal-ai/sam-3/image"   # SAM 3: una máscara por cada objeto que coincide con un texto ("sofa")
COLA = "https://queue.fal.run/"
ESPERA_MAX = 600   # segundos
# Los certificados de Windows no validan el CDN de fal.ai (fal.media); los de certifi, sí
SSL = ssl.create_default_context(cafile=certifi.where())


class SinClave(Exception):
    pass


def _pedir(url: str, datos: dict | None = None) -> dict:
    clave = os.environ.get("FAL_KEY", "").strip()
    if not clave:
        raise SinClave("falta la clave de fal.ai: inicia el servidor con la variable FAL_KEY")
    cuerpo = None if datos is None else json.dumps(datos).encode()
    pedido = urllib.request.Request(url, cuerpo, {"Authorization": f"Key {clave}", "Content-Type": "application/json"},
                                    method="GET" if datos is None else "POST")
    with urllib.request.urlopen(pedido, timeout=60, context=SSL) as r:
        return json.loads(r.read())


def imagen(foto: np.ndarray, alfa: np.ndarray, lado_max: int = 1536) -> str:
    """El objeto sobre fondo blanco, centrado en un cuadrado con margen (como las
    fotos de producto con que se entrenan estos modelos), en PNG base64."""
    vs, us = np.nonzero(alfa > 127)
    v0, v1, u0, u1 = vs.min(), vs.max() + 1, us.min(), us.max() + 1
    a = (alfa[v0:v1, u0:u1, None] / 255.0)
    objeto = foto[v0:v1, u0:u1] * a + 255 * (1 - a)
    h, w = objeto.shape[:2]
    lado = int(max(h, w) * 1.15)
    lienzo = np.full((lado, lado, 3), 255, np.uint8)
    y, x = (lado - h) // 2, (lado - w) // 2
    lienzo[y:y + h, x:x + w] = objeto.astype(np.uint8)
    if lado > lado_max:
        lienzo = cv2.resize(lienzo, (lado_max, lado_max), interpolation=cv2.INTER_AREA)
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", lienzo)[1].tobytes()).decode()


def ejecutar(modelo: str, entrada: dict) -> dict:
    """Envía un pedido a la cola de fal.ai, espera a que termine y devuelve el resultado."""
    cola = _pedir(COLA + modelo, entrada)
    print(f"[generativo] pedido {cola.get('request_id')} a {modelo}")   # para recuperarlo si algo falla después
    limite = time.monotonic() + ESPERA_MAX
    while _pedir(cola["status_url"]).get("status") != "COMPLETED":
        if time.monotonic() > limite:
            raise TimeoutError("fal.ai tardó demasiado en generar el modelo")
        time.sleep(2)
    return _pedir(cola["response_url"])


def descargar(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120, context=SSL) as r:
        return r.read()


def generar(frente: str, trasera: str | None = None, caras: int | None = None) -> bytes:
    """GLB del objeto completo. `frente` y `trasera` son imágenes (data URI);
    `caras` limita la malla (por defecto 500 000)."""
    entrada = {"input_image_url": frente}
    if trasera:
        entrada["back_image_url"] = trasera
    if caras:
        entrada["face_count"] = caras
    return descargar(ejecutar(MODELO, entrada)["model_glb"]["url"])


def instancias(foto: np.ndarray, texto: str, umbral: float = 0.8) -> list[np.ndarray]:
    """Máscaras (del tamaño de la foto) de cada objeto que SAM 3 encuentra con
    ese texto, con confianza ≥ umbral (los reflejos en un espejo quedan abajo)."""
    alto, ancho = foto.shape[:2]
    jpg = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", foto, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()).decode()
    r = ejecutar(SEGMENTAR, {"image_url": jpg, "prompt": texto, "apply_mask": False, "return_multiple_masks": True,
                             "max_masks": 8, "include_scores": True})
    mascaras = []
    for m, puntaje in zip(r.get("masks", []), r.get("scores") or [1.0] * len(r.get("masks", []))):
        if puntaje >= umbral:
            img = cv2.imdecode(np.frombuffer(descargar(m["url"]), np.uint8), cv2.IMREAD_GRAYSCALE)
            mascaras.append(cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_NEAREST) > 127)
    return mascaras


def _json(glb: bytes) -> dict:
    largo = int.from_bytes(glb[12:16], "little")
    return json.loads(glb[20:20 + largo])


def _matriz(nodo: dict) -> np.ndarray:
    """Transformación 4 × 4 de un nodo glTF: `matrix` (por columnas) o T·R·S."""
    if "matrix" in nodo:
        return np.array(nodo["matrix"], float).reshape(4, 4).T
    x, y, z, w = nodo.get("rotation", [0, 0, 0, 1])
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3] = R * np.array(nodo.get("scale", [1, 1, 1]))
    M[:3, 3] = nodo.get("translation", [0, 0, 0])
    return M


def caja(glb: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Mínimo y máximo de los vértices tal como se ven en la escena: las 8
    esquinas de la caja de cada malla (min/max de su POSITION) pasan por la
    transformación de su nodo (Hunyuan3D gira 90° en x para que y quede arriba)."""
    g = _json(glb)
    puntos = []
    for nodo in g["nodes"]:
        if "mesh" not in nodo:
            continue
        M = _matriz(nodo)
        for p in g["meshes"][nodo["mesh"]]["primitives"]:
            a = g["accessors"][p["attributes"]["POSITION"]]
            esquinas = np.array([[x, y, z, 1] for x in (a["min"][0], a["max"][0]) for y in (a["min"][1], a["max"][1])
                                 for z in (a["min"][2], a["max"][2])])
            puntos.append((esquinas @ M.T)[:, :3])
    puntos = np.vstack(puntos)
    return puntos.min(axis=0), puntos.max(axis=0)


def contar(glb: bytes) -> tuple[int, int]:
    """Vértices y triángulos de un GLB. Tras la cabecera (12 bytes) viene el
    bloque JSON (8 bytes de cabecera propia) que describe cada malla; sus
    `accessors` dicen cuántos elementos tiene cada arreglo."""
    g = _json(glb)
    N = M = 0
    for malla in g.get("meshes", []):
        for p in malla["primitives"]:
            n = g["accessors"][p["attributes"]["POSITION"]]["count"]
            N += n
            M += (g["accessors"][p["indices"]]["count"] if "indices" in p else n) // 3
    return N, M
