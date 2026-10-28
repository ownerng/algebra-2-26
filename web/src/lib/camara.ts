// Cámara del celular (DroidCam por WiFi) o webcam. De una sola foto:
//  - una red neuronal estima la profundidad relativa de cada píxel;
//  - el objeto es lo más cercano (umbral de Otsu sobre la profundidad);
//  - el VL53L0X mide la distancia real Z y, con la matriz K de la cámara,
//    cada píxel se convierte en milímetros.
// Si la red no está disponible, se usa la silueta por color y un relieve
// estimado a partir de la distancia al borde.

import { useEffect, useReducer } from 'react';
import type { Matriz } from './matematica.ts';
import { colorFondo, componenteMayor, distanciaAlBorde, distanciaColor, matrizK, umbralOtsu } from './vision.ts';
import { escaneo } from './escaneo.ts';
import { estimarProfundidad } from './profundidad.ts';
import { ajustarPlano, type Plano, type Vec3 } from './algebra3d.ts';

const CLAVE_IP = 'escaner-3d-droidcam-ip';
export const PUENTE_HTTP = 'http://localhost:3001';
export const DROIDCAM = 'droidcam-wifi';
const ANCHO_PROCESO = 640; // la foto se reduce a este ancho para procesarla rápido

export interface ModeloCamara {
  ancho: number; alto: number;           // píxeles de la foto reducida
  rgb: Uint8ClampedArray;
  mascara: Uint8Array;
  altura: Float32Array;                  // mm sobre el plano de la foto
  K: Matriz;
  Z: number;                             // distancia al objeto (mm)
  zDelSensor: boolean;
  conIA: boolean;                        // la profundidad la estimó la red neuronal
  planoFondo: Plano | null;              // d = a·u + b·v + c ajustado al borde de la foto
  profundidad: Uint8Array | null;        // profundidad relativa de la red (0 lejos, 255 cerca)
  zMm: Float32Array;                     // distancia de cada píxel del objeto a la cámara (mm)
  grosorBase: number;                    // mm que se suman a la altura (grosor de la base)
  mmPorPx: number;
  fondo: [number, number, number];
  umbral: number;
  areaPx: number;
  caja: { u0: number; u1: number; v0: number; v1: number; anchoMm: number; altoMm: number };
}

class Camara extends EventTarget {
  dispositivos: MediaDeviceInfo[] = [];
  deviceId = DROIDCAM;
  stream: MediaStream | null = null;
  video: HTMLVideoElement | null = null;
  error = '';
  ipCelular = (() => { try { return localStorage.getItem(CLAVE_IP) ?? ''; } catch { return ''; } })();
  anguloVision = 65;      // grados, ángulo de visión horizontal de la cámara
  distanciaManual = 300;  // mm, si el sensor no está conectado
  grosor = 3;             // mm, base de la figura
  relieve = 0.5;          // 0 = figura plana; 1 = relieve alto
  capturando = false;
  estadoTexto = '';
  ultimaProfundidad: Uint8Array | null = null;
  modelo: ModeloCamara | null = null;
  vista: 'camara' | '3d' = 'camara';
  version = 0;

  // Modelo 3D completo con TRELLIS 2 (fal.ai), a través del puente
  ia = {
    clave: null as boolean | null,   // ¿el puente tiene la clave de fal.ai?
    usar: true,
    estado: '' as '' | 'enviando' | 'en cola' | 'generando' | 'listo' | 'error',
    mensaje: '',
    glb: null as ArrayBuffer | null,
    segundos: 0,
  };
  private tareaIA = 0;

  async revisarIA() {
    try {
      const r = await fetch(`${PUENTE_HTTP}/ia3d/listo`);
      this.ia.clave = !!((await r.json()) as { clave: boolean }).clave;
    } catch { this.ia.clave = null; }
    this.avisar();
  }

  async guardarClaveIA(clave: string): Promise<string> {
    try {
      const r = await fetch(`${PUENTE_HTTP}/ia3d/clave`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ clave }) });
      const d = (await r.json()) as { error?: string };
      await this.revisarIA();
      return d.error ?? '';
    } catch { return 'el puente no está corriendo (inicia ./iniciar-web.sh)'; }
  }

  cambiarIA(usar: boolean) { this.ia.usar = usar; this.avisar(); }

  // Recorta el objeto (con margen) y lo manda a TRELLIS 2; al terminar deja el .glb en ia.glb
  private async generarIA(foto: ImageData, m: ModeloCamara) {
    const tarea = ++this.tareaIA;
    const vigente = () => tarea === this.tareaIA;
    const poner = (estado: Camara['ia']['estado'], mensaje = '') => { if (vigente()) { this.ia.estado = estado; this.ia.mensaje = mensaje; this.avisar(); } };
    this.ia.glb = null;
    const t0 = performance.now();
    try {
      poner('enviando');
      const lado = Math.max(m.caja.u1 - m.caja.u0, m.caja.v1 - m.caja.v0) * 1.3 + 16;
      const cu = (m.caja.u0 + m.caja.u1) / 2, cv = (m.caja.v0 + m.caja.v1) / 2;
      const fuente = document.createElement('canvas');
      fuente.width = foto.width; fuente.height = foto.height;
      fuente.getContext('2d')!.putImageData(foto, 0, 0);
      const c = document.createElement('canvas');
      c.width = c.height = 768;
      const g = c.getContext('2d')!;
      g.fillStyle = '#ffffff'; g.fillRect(0, 0, c.width, c.height);
      g.imageSmoothingQuality = 'high';
      g.drawImage(fuente, cu - lado / 2, cv - lado / 2, lado, lado, 0, 0, c.width, c.height);
      const imagen = c.toDataURL('image/jpeg', 0.92);
      const r = await fetch(`${PUENTE_HTTP}/ia3d/crear`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ imagen }) });
      const d = (await r.json()) as { id?: string; error?: string };
      if (!d.id) throw new Error(d.error ?? 'sin respuesta');
      for (;;) {
        await new Promise((res) => setTimeout(res, 2000));
        if (!vigente()) return;
        this.ia.segundos = Math.round((performance.now() - t0) / 1000);
        const e = (await (await fetch(`${PUENTE_HTTP}/ia3d/estado?id=${d.id}`)).json()) as { estado?: string; posicion?: number; glb?: string; error?: string };
        if (e.error) throw new Error(e.error);
        if (e.estado === 'IN_QUEUE') poner('en cola', e.posicion ? `puesto ${e.posicion} en la fila` : '');
        else if (e.estado === 'IN_PROGRESS') poner('generando');
        else if (e.estado === 'COMPLETED' && e.glb) {
          const b = await fetch(`${PUENTE_HTTP}/ia3d/modelo?url=${encodeURIComponent(e.glb)}`);
          if (!b.ok) throw new Error('no se pudo bajar el modelo');
          const glb = await b.arrayBuffer();
          if (!vigente()) return;
          this.ia.glb = glb;
          poner('listo');
          return;
        }
        if (performance.now() - t0 > 6 * 60_000) throw new Error('tardó más de 6 minutos');
      }
    } catch (e) {
      poner('error', (e as Error).message);
    }
  }

  private avisar() { this.version++; this.dispatchEvent(new Event('cambio')); }

  get usaDroidcam() { return this.deviceId === DROIDCAM; }
  get urlDroidcam() { return `http://${this.ipCelular.trim()}:4747/video`; }
  get vistaDroidcam() { return `${PUENTE_HTTP}/droidcam/video?url=${encodeURIComponent(this.urlDroidcam)}`; }
  get encendida() { return this.usaDroidcam ? !!this.ipCelular.trim() : !!this.stream; }

  async listar() {
    try {
      this.dispositivos = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'videoinput');
    } catch { /* sin permiso todavía */ }
    this.avisar();
  }

  async elegir(deviceId: string) {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.deviceId = deviceId;
    this.error = '';
    if (deviceId !== DROIDCAM) {
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({ video: { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } } });
        await this.listar();
      } catch (e) {
        this.error = 'No se pudo abrir esa cámara: ' + (e as Error).message;
      }
    }
    this.avisar();
  }

  cambiarIp(ip: string) {
    this.ipCelular = ip;
    try { localStorage.setItem(CLAVE_IP, ip); } catch { /* sin almacenamiento */ }
    this.avisar();
  }

  cambiar(c: Partial<Pick<Camara, 'anguloVision' | 'distanciaManual' | 'grosor' | 'relieve' | 'vista'>>) {
    Object.assign(this, c);
    if (this.ultimaFoto && ('anguloVision' in c || 'distanciaManual' in c || 'grosor' in c || 'relieve' in c)) {
      this.procesar(this.ultimaFoto, this.modelo?.zDelSensor ? this.modelo.Z : undefined, this.ultimaProfundidad ?? undefined);
    }
    this.avisar();
  }

  private async capturar(): Promise<ImageData | null> {
    let fuente: CanvasImageSource | null = null, w = 0, h = 0;
    if (this.usaDroidcam) {
      const r = await fetch(`${PUENTE_HTTP}/droidcam/foto?url=${encodeURIComponent(this.urlDroidcam)}`);
      if (!r.ok) throw new Error(await r.text());
      const bmp = await createImageBitmap(await r.blob());
      fuente = bmp; w = bmp.width; h = bmp.height;
    } else if (this.video?.videoWidth) {
      fuente = this.video; w = this.video.videoWidth; h = this.video.videoHeight;
    }
    if (!fuente) return null;
    // Se reduce la foto: el escalamiento es una transformación lineal de la imagen
    const escala = Math.min(1, ANCHO_PROCESO / w);
    const c = document.createElement('canvas');
    c.width = Math.round(w * escala); c.height = Math.round(h * escala);
    const g = c.getContext('2d', { willReadFrequently: true })!;
    g.drawImage(fuente, 0, 0, c.width, c.height);
    return g.getImageData(0, 0, c.width, c.height);
  }

  ultimaFoto: ImageData | null = null;

  async tomarFoto() {
    if (this.capturando) return;
    this.capturando = true;
    this.error = '';
    this.avisar();
    try {
      const foto = await this.capturar();
      if (!foto) throw new Error('la cámara todavía no muestra imagen');
      this.ultimaFoto = foto;
      // Distancia al objeto: la del sensor si está conectado y midiendo
      const vivo = escaneo.vivo && performance.now() - escaneo.vivo.t < 1500 && escaneo.vivo.r > 0 ? escaneo.vivo.r : null;
      try {
        this.ultimaProfundidad = await estimarProfundidad(foto, (t) => { this.estadoTexto = t; this.avisar(); });
      } catch (e) {
        this.ultimaProfundidad = null;
        console.warn('Sin red de profundidad:', e);
      }
      this.estadoTexto = '';
      this.procesar(foto, vivo ?? undefined, this.ultimaProfundidad ?? undefined);
      if (this.modelo) {
        this.vista = '3d';
        Object.assign(escaneo.vista, { theta: -25, phi: 55, zoom: 1 });
        this.tareaIA++;
        this.ia.glb = null; this.ia.estado = ''; this.ia.mensaje = '';
        if (this.ia.usar && this.ia.clave) void this.generarIA(foto, this.modelo);
      }
    } catch (e) {
      this.error = 'No se pudo tomar la foto: ' + (e as Error).message;
    }
    this.capturando = false;
    this.avisar();
  }

  procesar(foto: ImageData, zSensor?: number, prof?: Uint8Array) {
    const { width: ancho, height: alto, data: rgb } = foto;
    const fondo = colorFondo(rgb, ancho, alto);
    // Objeto: con la red, lo que está más cerca que el umbral de Otsu de la
    // profundidad; sin la red, lo que es de otro color que el fondo.
    // La mesa o el piso también tienen profundidad (más cerca abajo de la foto).
    // Se ajusta un plano d = a·u + b·v + c a la profundidad del borde de la foto
    // por mínimos cuadrados y se resta: queda solo lo que sobresale del fondo.
    let planoFondo: Plano | null = null;
    let relieve: Uint8Array | null = null;
    if (prof) {
      const marco = Math.round(0.06 * Math.min(ancho, alto));
      const borde: Vec3[] = [];
      for (let v = 0; v < alto; v += 3) for (let u = 0; u < ancho; u += 3) {
        if (u >= marco && u < ancho - marco && v >= marco && v < alto - marco) continue;
        borde.push([u, v, prof[v * ancho + u]!]);
      }
      planoFondo = ajustarPlano(borde);
      relieve = new Uint8Array(prof.length);
      for (let v = 0; v < alto; v++) for (let u = 0; u < ancho; u++) {
        const k = v * ancho + u;
        const base = planoFondo ? planoFondo.a * u + planoFondo.b * v + planoFondo.c : 0;
        relieve[k] = Math.max(0, Math.min(255, prof[k]! - base));
      }
    }
    const separar = relieve ?? distanciaColor(rgb, fondo);
    const umbral = Math.max(relieve ? 8 : 25, umbralOtsu(separar));
    let mascara: Uint8Array = new Uint8Array(separar.length);
    for (let k = 0; k < separar.length; k++) mascara[k] = separar[k]! > umbral ? 1 : 0;
    mascara = componenteMayor(mascara, ancho, alto);
    let areaPx = 0, u0 = ancho, u1 = 0, v0 = alto, v1 = 0;
    for (let v = 0; v < alto; v++) for (let u = 0; u < ancho; u++) {
      if (!mascara[v * ancho + u]) continue;
      areaPx++; u0 = Math.min(u0, u); u1 = Math.max(u1, u); v0 = Math.min(v0, v); v1 = Math.max(v1, v);
    }
    if (areaPx < 30 || areaPx > 0.9 * ancho * alto) {
      this.modelo = null;
      this.error = 'No se distinguió el objeto. Acerca la cámara para que el objeto ocupe buena parte de la imagen, sin tocar los bordes.';
      return;
    }
    const zDelSensor = zSensor !== undefined;
    const Z = zSensor ?? this.distanciaManual;
    const K = matrizK(ancho, alto, this.anguloVision);
    const mmPorPx = Z / K[0]![0]!;
    const anchoMm = (u1 - u0 + 1) * mmPorPx, altoMm = (v1 - v0 + 1) * mmPorPx;
    const altura = new Float32Array(mascara.length);
    if (relieve) {
      // Relieve de la red (sin el plano del fondo), llevado a mm con una escala
      // proporcional al tamaño real del objeto
      let pMax = 1;
      for (let k = 0; k < relieve.length; k++) if (mascara[k]) pMax = Math.max(pMax, relieve[k]!);
      const escala = (this.relieve * 1.2 * Math.min(anchoMm, altoMm)) / pMax;
      for (let k = 0; k < altura.length; k++) if (mascara[k]) altura[k] = this.grosor + relieve[k]! * escala;
    } else {
      // Sin red: más alto en el centro de la silueta, cero en el borde
      const dist = distanciaAlBorde(mascara, ancho, alto);
      let dMax = 0;
      for (const d of dist) dMax = Math.max(dMax, d);
      const hMax = this.relieve * 0.5 * Math.min(anchoMm, altoMm);
      for (let k = 0; k < altura.length; k++) if (mascara[k]) altura[k] = this.grosor + (dMax ? hMax * Math.sqrt(dist[k]! / dMax) : 0);
    }
    // Se suaviza la altura dentro del objeto (promedio 7×7, dos pasadas
    // separables) para quitar el ruido de la estimación
    const suave = suavizar(altura, mascara, ancho, alto, 3);
    // Distancia a la cámara de cada píxel: el sensor da la del objeto (Z) y lo
    // que sobresale queda más cerca
    let hMed = 0, nMed = 0;
    for (let k = 0; k < suave.length; k++) if (mascara[k]) { hMed += suave[k]!; nMed++; }
    hMed = nMed ? hMed / nMed : 0;
    const zMm = new Float32Array(suave.length);
    for (let k = 0; k < suave.length; k++) zMm[k] = Z + hMed - suave[k]!;
    this.modelo = { ancho, alto, rgb, mascara, altura: suave, zMm, grosorBase: this.grosor, K, Z, zDelSensor, conIA: !!prof, planoFondo, profundidad: prof ?? null, mmPorPx, fondo, umbral, areaPx, caja: { u0, u1, v0, v1, anchoMm, altoMm } };
  }
}

// Promedio de una ventana (2r+1)×(2r+1) solo con los píxeles del objeto
function suavizar(h: Float32Array, m: Uint8Array, w: number, alto: number, r: number): Float32Array {
  const tmp = new Float32Array(h.length), res = new Float32Array(h.length);
  for (let v = 0; v < alto; v++) for (let u = 0; u < w; u++) {
    let s = 0, n = 0;
    for (let d = -r; d <= r; d++) { const x = u + d; if (x >= 0 && x < w && m[v * w + x]) { s += h[v * w + x]!; n++; } }
    tmp[v * w + u] = n ? s / n : 0;
  }
  for (let v = 0; v < alto; v++) for (let u = 0; u < w; u++) {
    if (!m[v * w + u]) continue;
    let s = 0, n = 0;
    for (let d = -r; d <= r; d++) { const y = v + d; if (y >= 0 && y < alto && m[y * w + u]) { s += tmp[y * w + u]!; n++; } }
    res[v * w + u] = n ? s / n : 0;
  }
  return res;
}

export const camara = new Camara();

export function useCamara(ms = 0) {
  const [, refrescar] = useReducer((n: number) => n + 1, 0);
  useEffect(() => {
    let ultimo = 0, pendiente: ReturnType<typeof setTimeout> | null = null;
    const alCambio = () => {
      const ahora = performance.now();
      if (ahora - ultimo >= ms) { ultimo = ahora; refrescar(); return; }
      if (!pendiente) pendiente = setTimeout(() => { pendiente = null; ultimo = performance.now(); refrescar(); }, ms);
    };
    camara.addEventListener('cambio', alCambio);
    escaneo.addEventListener('cambio', alCambio);
    return () => { camara.removeEventListener('cambio', alCambio); escaneo.removeEventListener('cambio', alCambio); if (pendiente) clearTimeout(pendiente); };
  }, [ms]);
  return camara;
}
