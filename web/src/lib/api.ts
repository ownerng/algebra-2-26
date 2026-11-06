// Cliente del servidor en Python (servidor/app.py). La cámara se conecta desde
// el servidor, que también hace la segmentación, la profundidad y la malla.

export const API = `${location.protocol}//${location.hostname}:8000`;
export const VIDEO = `${API}/api/camara/video`;

// Recuadro de encuadre (x0, y0, x1, y1 en fracciones de la imagen); el servidor
// usa el mismo para elegir el objeto (servidor/escaner/proceso.py)
export const MARCO = [0.2, 0.15, 0.8, 0.85] as const;

export interface EstadoCamara { conectada: boolean; url: string | null; error: string }

export interface Modelo {
  V: Float32Array;        // N × 3 (mm): x derecha, y arriba, z hacia la cámara
  F: Uint32Array;         // M × 3, índices de V
  uv: Float32Array | null;      // N × 2, coordenadas en la foto (una sola foto)
  colores: Uint8Array | null;   // N × 3, RGB por vértice (varias vistas)
  N: number; M: number;
  nFrente: number;
  caras: { frente?: number; atras?: number; paredes?: number; todas?: number };
  K: number[][];
  foto: string;           // data URL de la foto procesada
  recorte: string;        // data URL del objeto sin fondo (PNG con alfa)
  metodo: string;
  tamanoMm: [number, number, number];
  volumenCm3: number;
  tiemposMs: Record<string, number>;
  // Habitaciones: la cámara de la foto (mm, ejes del modelo) y su ángulo de visión vertical
  vista: { origen: number[]; mira: number[]; fov_v: number } | null;
  glb: string | null;     // procesamiento alto: URL del modelo generativo (sin V, F)
  muebles: { glb: string; matriz: number[] }[];   // habitación (alto): cada mueble y su matriz 4 × 4 (por columnas)
}

async function pedir<T>(ruta: string, init?: RequestInit): Promise<T> {
  let r: Response;
  try {
    r = await fetch(API + ruta, init);
  } catch {
    throw new Error('el servidor no responde (inicia ./iniciar-web.sh)');
  }
  const datos = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((datos as { detail?: string }).detail ?? `error ${r.status}`);
  return datos as T;
}

export const estadoCamara = () => pedir<EstadoCamara>('/api/camara/estado');

export interface CamaraUsb { ruta: string; nombre: string; droidcam: boolean }

export const camarasUsb = () => pedir<CamaraUsb[]>('/api/camara/usb');

// `ip` es la IP del celular (WiFi) o la ruta de una cámara USB (/dev/videoN)
export const conectarCamara = (ip: string) =>
  pedir<EstadoCamara>('/api/camara/conectar', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ip }) });

const bytes = (b64: string) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer;

function aModelo(d: Record<string, unknown>): Modelo {
  return {
    V: new Float32Array(bytes(d.V as string)),
    F: new Uint32Array(bytes(d.F as string)),
    uv: d.uv ? new Float32Array(bytes(d.uv as string)) : null,
    colores: d.colores ? new Uint8Array(bytes(d.colores as string)) : null,
    N: d.N as number, M: d.M as number,
    nFrente: d.n_frente as number,
    caras: d.caras as Modelo['caras'],
    K: d.K as number[][],
    foto: `data:image/jpeg;base64,${d.foto}`,
    recorte: `data:image/png;base64,${d.recorte}`,
    metodo: d.metodo as string,
    tamanoMm: d.tamano_mm as Modelo['tamanoMm'],
    volumenCm3: d.volumen_cm3 as number,
    tiemposMs: d.tiempos_ms as Record<string, number>,
    vista: (d.vista as Modelo['vista']) ?? null,
    glb: d.glb ? API + (d.glb as string) : null,
    muebles: ((d.muebles as Modelo['muebles']) ?? []).map((m) => ({ ...m, glb: API + m.glb })),
  };
}

// Una sola foto: relieve de la silueta con la foto como textura
export const escanear = async () => aModelo(await pedir('/api/escanear', { method: 'POST' }));

// Habitación: la foto subida o, sin ella, el cuadro actual de la cámara
export async function escanearHabitacion(foto?: Blob, alto = false) {
  const datos = new FormData();
  if (foto) datos.append('foto', foto);
  datos.append('calidad', alto ? 'alto' : 'bajo');
  return aModelo(await pedir('/api/habitacion', { method: 'POST', body: datos }));
}

// Procesamiento alto: el objeto completo con IA generativa (Hunyuan3D en fal.ai) desde la foto actual
export async function generar3D() {
  const datos = new FormData();
  datos.append('modo', 'una');
  return aModelo(await pedir('/api/generativo', { method: 'POST', body: datos }));
}

export const URL_OBJ = `${API}/api/obj`;
export const URL_TABLERO = `${API}/api/tablero.svg`;

// Calibración de la cámara con el tablero (servidor/escaner/calibracion.py)
export interface EstadoCalibracion {
  fotos: number; minimo: number; calibrada: boolean;
  f_px: number | null; error_px: number | null; resolucion: string | null;
}
export const estadoCalibracion = () => pedir<EstadoCalibracion>('/api/calibracion');
export const fotoCalibracion = () => pedir<EstadoCalibracion>('/api/calibracion/foto', { method: 'POST' });
export const calcularCalibracion = () => pedir<EstadoCalibracion>('/api/calibracion/calcular', { method: 'POST' });
export const borrarCalibracion = () => pedir<EstadoCalibracion>('/api/calibracion', { method: 'DELETE' });
