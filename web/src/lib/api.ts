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
  };
}

// Una sola foto: relieve de la silueta con la foto como textura
export const escanear = async () => aModelo(await pedir('/api/escanear', { method: 'POST' }));

// Varias vistas: cada foto se segmenta al tomarla; al final, la envolvente visual
export const VISTAS = ['frontal', 'izquierda', 'trasera', 'derecha', 'superior'] as const;
export type NombreVista = (typeof VISTAS)[number];

export async function tomarVista(nombre: NombreVista): Promise<string> {
  const d = await pedir<{ recorte: string }>(`/api/vistas/${nombre}`, { method: 'POST' });
  return `data:image/png;base64,${d.recorte}`;
}

export const borrarVistas = () => pedir('/api/vistas', { method: 'DELETE' });

export const reconstruir = async () => aModelo(await pedir('/api/reconstruir', { method: 'POST' }));

export const URL_OBJ = `${API}/api/obj`;
