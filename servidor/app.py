"""Servidor del escáner 3D (FastAPI).

  POST /api/camara/conectar   {"ip": "192.168.1.5"}  → abre DroidCam
  GET  /api/camara/usb                               → cámaras del computador (USB)
  GET  /api/camara/estado                            → ¿hay imagen?
  GET  /api/camara/video                             → vista previa MJPEG
  POST /api/escanear          (foto opcional)        → malla V, F en base64
  POST /api/vistas/{nombre}   (foto opcional)        → guarda y segmenta una vista
  DELETE /api/vistas                                 → borra las vistas
  POST /api/reconstruir                              → envolvente visual de las vistas
  POST /api/obj                                      → último modelo en .obj
  GET  /api/tablero.svg                              → tablero de marcadores para imprimir
  GET  /api/calibracion                              → estado de la calibración de la cámara
  POST /api/calibracion/foto  (foto opcional)        → suma una foto del tablero
  POST /api/calibracion/calcular                     → calibra (K y distorsión) y la guarda
  DELETE /api/calibracion                            → borra las fotos y la calibración
  POST /api/habitacion        (foto opcional, calidad: bajo | alto) → maqueta 3D de una habitación
  GET  /api/mueble/{i}                               → GLB de un mueble generado (habitación, alto)
  POST /api/generativo        {modo: una | vistas}   → objeto completo con IA generativa (fal.ai)
  GET  /api/glb                                      → último modelo generativo en .glb

Uso:  servidor/.venv/bin/uvicorn app:app --app-dir servidor --port 8000
"""

import asyncio
import base64
import time
import urllib.error
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel

from escaner import calibracion, generativo, habitacion, profundidad, segmentacion, tablero
from escaner.camara import CamaraRed, camaras_locales
from escaner.malla import a_obj
from escaner.multivista import VISTAS, Vista, es_vista, preparar_vista, reconstruir
from escaner.proceso import MARCO, Parametros, Resultado, SinObjeto, escanear, reducir

camara = CamaraRed()
ultimo: Resultado | None = None
ultimo_glb: bytes | None = None
ultimos_muebles: list[bytes] = []
vistas: dict[str, Vista] = {}


@asynccontextmanager
async def ciclo(_: FastAPI):
    calibracion.cargar()
    await asyncio.to_thread(segmentacion.precargar)
    await asyncio.to_thread(profundidad.precargar)
    yield
    camara.desconectar()


app = FastAPI(title="Escáner 3D", lifespan=ciclo)
app.add_middleware(CORSMiddleware, allow_origin_regex=r"https?://.*", allow_methods=["*"], allow_headers=["*"])


class PedidoConexion(BaseModel):
    ip: str


def estado() -> dict:
    return {"conectada": camara.conectada, "url": camara.url, "error": camara.error, "marco": MARCO}


@app.get("/api/camara/usb")
def listar_usb() -> list[dict]:
    return camaras_locales()


@app.get("/api/camara/estado")
def ver_estado() -> dict:
    return estado()


@app.post("/api/camara/conectar")
async def conectar(p: PedidoConexion) -> dict:
    if not p.ip.strip():
        raise HTTPException(400, "escribe la IP del celular o elige una cámara USB")
    await asyncio.to_thread(camara.conectar, p.ip)
    return estado()


@app.get("/api/camara/video")
async def video() -> StreamingResponse:
    async def cuadros():
        anterior = None
        while camara.jpeg is not None:
            jpeg = await asyncio.to_thread(camara.esperar_cuadro, anterior)
            if jpeg is None:
                break
            if jpeg is anterior:
                continue
            anterior = jpeg
            yield b"--cuadro\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    return StreamingResponse(cuadros(), media_type="multipart/x-mixed-replace; boundary=cuadro")


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def imagen_b64(img: np.ndarray, formato: str) -> str:
    ok, datos = cv2.imencode(formato, img, [cv2.IMWRITE_JPEG_QUALITY, 88] if formato == ".jpg" else [])
    return base64.b64encode(datos.tobytes()).decode()


def recorte_b64(foto: np.ndarray, alfa: np.ndarray) -> str:
    """El objeto sin fondo (PNG con alfa), recortado a su silueta."""
    vs, us = np.nonzero(alfa > 127)
    rgba = np.dstack([foto, alfa])
    if len(us):
        rgba = rgba[vs.min():vs.max() + 1, us.min():us.max() + 1]
    return imagen_b64(rgba, ".png")


def respuesta(r: Resultado) -> dict:
    m = r.malla
    return {
        # V ∈ R^(N×3) (float32) y F ∈ Z^(M×3) (uint32), en orden de filas
        "V": b64(m.V), "F": b64(m.F),
        # Una foto: textura (uv); varias fotos: un color RGB por vértice
        "uv": b64(m.uv) if m.uv is not None else None,
        "colores": b64(m.colores) if m.colores is not None else None,
        "N": int(len(m.V)), "M": int(len(m.F)), "n_frente": m.n_frente, "caras": m.caras,
        "K": r.K.round(3).tolist(),
        "foto": imagen_b64(r.foto, ".jpg"),
        "recorte": recorte_b64(r.foto, r.alfa),
        "metodo": r.metodo,
        "tamano_mm": [round(x, 1) for x in r.tamano_mm],
        "volumen_cm3": round(r.volumen_mm3 / 1000, 1),
        "tiempos_ms": r.tiempos,
        "vista": r.vista,
        "muebles": [{"glb": f"/api/mueble/{i}?t={int(time.time())}", "matriz": M} for i, (_, M) in enumerate(r.muebles or [])],
    }


async def leer_foto(foto: UploadFile | None) -> np.ndarray:
    """La foto subida o, si no hay, el cuadro actual de la cámara."""
    if foto is not None:
        bgr = cv2.imdecode(np.frombuffer(await foto.read(), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(400, "la imagen no se pudo leer")
        return bgr
    try:
        return camara.foto()
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e


@app.post("/api/escanear")
async def escanear_foto(foto: UploadFile | None = File(None), distancia_mm: float = Form(400.0)) -> dict:
    global ultimo
    bgr = await leer_foto(foto)
    try:
        ultimo = await asyncio.to_thread(escanear, bgr, Parametros(distancia_mm=distancia_mm))
    except SinObjeto as e:
        raise HTTPException(422, str(e)) from e
    return respuesta(ultimo)


@app.post("/api/vistas/{nombre}")
async def tomar_vista(nombre: str, foto: UploadFile | None = File(None)) -> dict:
    if not es_vista(nombre):
        raise HTTPException(404, f"vista desconocida; usa {', '.join(VISTAS)} o extra-N")
    bgr = await leer_foto(foto)
    try:
        v = await asyncio.to_thread(preparar_vista, nombre, bgr, MARCO)
    except SinObjeto as e:
        raise HTTPException(422, str(e)) from e
    vistas[nombre] = v
    return {"nombre": nombre, "recorte": recorte_b64(v.foto, v.alfa), "tablero": v.camara is not None, "listas": list(vistas)}


@app.delete("/api/vistas")
def borrar_vistas() -> dict:
    vistas.clear()
    return {"listas": []}


@app.post("/api/reconstruir")
async def reconstruir_vistas(distancia_mm: float = Form(400.0)) -> dict:
    global ultimo
    try:
        ultimo = await asyncio.to_thread(reconstruir, dict(vistas), Parametros(distancia_mm=distancia_mm))
    except SinObjeto as e:
        raise HTTPException(422, str(e)) from e
    return respuesta(ultimo)


@app.post("/api/habitacion")
async def escanear_habitacion(foto: UploadFile | None = File(None), calidad: str = Form("bajo")) -> dict:
    global ultimo, ultimos_muebles
    bgr = await leer_foto(foto)
    try:
        ultimo = await asyncio.to_thread(habitacion.reconstruir, bgr, None, calidad == "alto")
    except SinObjeto as e:
        raise HTTPException(503, str(e)) from e
    ultimos_muebles = [glb for glb, _ in ultimo.muebles or []]
    return respuesta(ultimo)


@app.get("/api/mueble/{i}")
def descargar_mueble(i: int) -> Response:
    if not 0 <= i < len(ultimos_muebles):
        raise HTTPException(404, "no hay ese mueble")
    return Response(ultimos_muebles[i], media_type="model/gltf-binary")


@app.post("/api/generativo")
async def escanear_generativo(modo: str = Form("una"), foto: UploadFile | None = File(None)) -> dict:
    """Procesamiento alto: la foto del objeto (o la frontal y la trasera de las
    vistas) va a Hunyuan3D en fal.ai, que devuelve el objeto completo."""
    global ultimo_glb
    t = time.perf_counter()
    trasera = None
    if modo == "vistas":
        if not vistas:
            raise HTTPException(409, "toma al menos una vista")
        fr = vistas.get("frontal") or next(iter(vistas.values()))
        bgr, alfa = fr.foto, fr.alfa
        if "trasera" in vistas:
            trasera = generativo.imagen(vistas["trasera"].foto, vistas["trasera"].alfa)
    else:
        bgr = await leer_foto(foto)
        mascara, alfa, _ = await asyncio.to_thread(segmentacion.segmentar, reducir(bgr), MARCO)
        if mascara.sum() < 100:
            raise HTTPException(422, "no se distinguió el objeto; ponlo dentro del recuadro")
        # El recorte sale de la foto original, con todo su detalle
        alfa = cv2.resize(alfa, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    try:
        glb = await asyncio.to_thread(generativo.generar, generativo.imagen(bgr, alfa), trasera)
    except generativo.SinClave as e:
        raise HTTPException(503, str(e)) from e
    except urllib.error.HTTPError as e:
        raise HTTPException(502, f"fal.ai respondió {e.code}: {e.read()[:300].decode(errors='replace')}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise HTTPException(502, f"no se pudo hablar con fal.ai: {e}") from e
    ultimo_glb = glb
    N, M = generativo.contar(glb)
    return {
        "V": "", "F": "", "uv": None, "colores": None, "N": N, "M": M, "n_frente": 0, "caras": {}, "K": [],
        "foto": imagen_b64(reducir(bgr), ".jpg"), "recorte": recorte_b64(bgr, alfa),
        "metodo": "Hunyuan3D v3.1 Pro (fal.ai)" + (" · frente y trasera" if trasera else ""),
        "tamano_mm": [0, 0, 0], "volumen_cm3": 0, "tiempos_ms": {"generacion": round((time.perf_counter() - t) * 1000)},
        "vista": None, "glb": f"/api/glb?t={int(time.time())}",
    }


@app.get("/api/glb")
def descargar_glb() -> Response:
    if ultimo_glb is None:
        raise HTTPException(404, "todavía no hay modelo generativo")
    return Response(ultimo_glb, media_type="model/gltf-binary", headers={"Content-Disposition": 'attachment; filename="objeto.glb"'})


@app.get("/api/obj", response_class=PlainTextResponse)
def descargar_obj() -> PlainTextResponse:
    if ultimo is None:
        raise HTTPException(404, "todavía no hay modelo")
    return PlainTextResponse(a_obj(ultimo.malla), headers={"Content-Disposition": 'attachment; filename="objeto.obj"'})


@app.get("/api/tablero.svg")
def tablero_svg() -> Response:
    return Response(tablero.svg(), media_type="image/svg+xml")


@app.get("/api/calibracion")
def ver_calibracion() -> dict:
    return calibracion.estado()


@app.post("/api/calibracion/foto")
async def foto_calibracion(foto: UploadFile | None = File(None)) -> dict:
    # La foto completa, sin reducir: la calibración vale para la resolución de la cámara
    marcadores = await asyncio.to_thread(calibracion.agregar, await leer_foto(foto))
    if not marcadores:
        raise HTTPException(422, f"no se ve el tablero (hacen falta {calibracion.MIN_MARCADORES} marcadores o más)")
    return {**calibracion.estado(), "marcadores": marcadores}


@app.post("/api/calibracion/calcular")
async def calcular_calibracion() -> dict:
    try:
        await asyncio.to_thread(calibracion.calcular)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    return calibracion.estado()


@app.delete("/api/calibracion")
def borrar_calibracion() -> dict:
    calibracion.borrar()
    return calibracion.estado()
