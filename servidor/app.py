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

Uso:  servidor/.venv/bin/uvicorn app:app --app-dir servidor --port 8000
"""

import asyncio
import base64
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from escaner import segmentacion
from escaner.camara import CamaraRed, camaras_locales
from escaner.malla import a_obj
from escaner.multivista import VISTAS, Vista, preparar_vista, reconstruir
from escaner.proceso import MARCO, Parametros, Resultado, SinObjeto, escanear

camara = CamaraRed()
ultimo: Resultado | None = None
vistas: dict[str, Vista] = {}


@asynccontextmanager
async def ciclo(_: FastAPI):
    await asyncio.to_thread(segmentacion.precargar)
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
    if nombre not in VISTAS:
        raise HTTPException(404, f"vista desconocida; usa {', '.join(VISTAS)}")
    bgr = await leer_foto(foto)
    try:
        v = await asyncio.to_thread(preparar_vista, nombre, bgr, MARCO)
    except SinObjeto as e:
        raise HTTPException(422, str(e)) from e
    vistas[nombre] = v
    return {"nombre": nombre, "recorte": recorte_b64(v.foto, v.alfa), "listas": [n for n in VISTAS if n in vistas]}


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


@app.get("/api/obj", response_class=PlainTextResponse)
def descargar_obj() -> PlainTextResponse:
    if ultimo is None:
        raise HTTPException(404, "todavía no hay modelo")
    return PlainTextResponse(a_obj(ultimo.malla), headers={"Content-Disposition": 'attachment; filename="objeto.obj"'})
