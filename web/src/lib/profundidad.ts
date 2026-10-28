// Profundidad a partir de una sola foto con una red neuronal (Depth Anything V2,
// versión pequeña) que corre en el navegador con transformers.js. Se descarga
// una vez (~25 MB) y queda guardada en el navegador.
//
// La red devuelve una profundidad RELATIVA (qué está más cerca y qué más lejos),
// no milímetros: la escala real la pone el sensor VL53L0X.

const MODELO = 'onnx-community/depth-anything-v2-small';

type Estimador = (img: unknown) => Promise<{ depth: { data: Uint8Array | Uint8ClampedArray; width: number; height: number } }>;
let estimador: Promise<Estimador> | null = null;

export type Progreso = (texto: string) => void;

export function cargarModelo(progreso?: Progreso): Promise<Estimador> {
  if (!estimador) {
    estimador = (async () => {
      const { pipeline, env } = await import('@huggingface/transformers');
      env.allowLocalModels = false;
      const archivos = new Map<string, number>();
      const p = await pipeline('depth-estimation', MODELO, {
        dtype: 'q8',
        progress_callback: (e: { status: string; file?: string; progress?: number }) => {
          if (e.status === 'progress' && e.file) {
            archivos.set(e.file, e.progress ?? 0);
            const media = [...archivos.values()].reduce((a, b) => a + b, 0) / archivos.size;
            progreso?.(`Descargando el modelo de IA (solo la primera vez): ${Math.round(media)} %`);
          }
        },
      });
      return p as unknown as Estimador;
    })();
    estimador.catch(() => { estimador = null; });
  }
  return estimador;
}

// Mapa de profundidad del mismo tamaño que la foto: 0 = lejos, 255 = cerca
export async function estimarProfundidad(foto: ImageData, progreso?: Progreso): Promise<Uint8Array> {
  const est = await cargarModelo(progreso);
  progreso?.('Estimando la profundidad de cada píxel…');
  const { RawImage } = await import('@huggingface/transformers');
  const img = new RawImage(foto.data, foto.width, foto.height, 4);
  const { depth } = await est(img);
  if (depth.width === foto.width && depth.height === foto.height) return Uint8Array.from(depth.data);
  // Si la red devolvió otro tamaño, se reescala al de la foto (vecino más cercano)
  const res = new Uint8Array(foto.width * foto.height);
  for (let v = 0; v < foto.height; v++) for (let u = 0; u < foto.width; u++) {
    const su = Math.min(depth.width - 1, Math.floor((u * depth.width) / foto.width));
    const sv = Math.min(depth.height - 1, Math.floor((v * depth.height) / foto.height));
    res[v * foto.width + u] = depth.data[sv * depth.width + su]!;
  }
  return res;
}
