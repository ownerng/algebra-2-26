// Cámara (DroidCam u otra webcam) mirando la cartulina desde arriba. Con las
// 4 esquinas de la cuadrícula se calcula la homografía H; con una foto de la
// cartulina vacía y otra con el objeto se obtiene su silueta exacta en mm.

import { useEffect, useReducer } from 'react';
import type { Matriz } from './matematica.ts';
import { aplicarH, homografiaDesde4, inversa3, umbralOtsu, type Homografia, type Punto2 } from './homografia.ts';
import { escaneo } from './escaneo.ts';

const CLAVE_ESQUINAS = 'escaner-3d-esquinas';
const CLAVE_IP = 'escaner-3d-droidcam-ip';
export const PUENTE_HTTP = 'http://localhost:3001';
export const DROIDCAM = 'droidcam-wifi';
export const RESOLUCION_MM = 1; // un píxel de la imagen enderezada = 1 mm

export interface ModeloCamara {
  x0: number; y0: number; ancho: number; alto: number; // en mm
  rgb: Uint8ClampedArray;      // foto enderezada, fila 0 = y más alta
  mascara: Uint8Array;         // 1 donde está el objeto
  homografia: Homografia;
  umbral: number;
  areaMm2: number;
  caja: { xMin: number; xMax: number; yMin: number; yMax: number } | null;
}

function leerEsquinas(): Punto2[] {
  try { return JSON.parse(localStorage.getItem(CLAVE_ESQUINAS) ?? '[]'); } catch { return []; }
}

class Camara extends EventTarget {
  dispositivos: MediaDeviceInfo[] = [];
  deviceId = '';
  stream: MediaStream | null = null;
  video: HTMLVideoElement | null = null;
  imagen: HTMLImageElement | null = null;
  // DroidCam por WiFi: la app del celular transmite en http://IP:4747/video
  ipCelular = (() => { try { return localStorage.getItem(CLAVE_IP) ?? ''; } catch { return ''; } })();
  get usaDroidcam() { return this.deviceId === DROIDCAM; }
  get urlDroidcam() { return `http://${this.ipCelular.trim()}:4747/video`; }
  get vistaDroidcam() { return `${PUENTE_HTTP}/droidcam/video?url=${encodeURIComponent(this.urlDroidcam)}`; }
  error = '';
  esquinas: Punto2[] = leerEsquinas(); // en píxeles del video
  fondo: ImageData | null = null;
  foto: ImageData | null = null;
  grosor = 3;                          // mm, para objetos planos que el sensor no alcanza a ver
  modelo: ModeloCamara | null = null;
  version = 0;

  private avisar() { this.version++; this.dispatchEvent(new Event('cambio')); }

  async listar() {
    try {
      const todos = await navigator.mediaDevices.enumerateDevices();
      this.dispositivos = todos.filter((d) => d.kind === 'videoinput');
      if (!this.deviceId) this.deviceId = (this.dispositivos.find((d) => /droid/i.test(d.label)) ?? this.dispositivos[0])?.deviceId ?? '';
    } catch (e) {
      this.error = 'No se pudieron listar las cámaras: ' + (e as Error).message;
    }
    this.avisar();
  }

  cambiarIp(ip: string) {
    this.ipCelular = ip;
    try { localStorage.setItem(CLAVE_IP, ip); } catch { /* sin almacenamiento */ }
    this.avisar();
  }

  get encendida() { return this.usaDroidcam ? !!this.ipCelular.trim() : !!this.stream; }

  async iniciar(deviceId = this.deviceId) {
    this.detener();
    this.error = '';
    if (deviceId === DROIDCAM) { this.deviceId = deviceId; this.avisar(); return; }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { deviceId: deviceId ? { exact: deviceId } : undefined, width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      this.deviceId = deviceId;
      await this.listar(); // con permiso ya aparecen los nombres
    } catch (e) {
      this.error = 'No se pudo abrir la cámara. ¿DroidCam está conectado? (' + (e as Error).message + ')';
    }
    this.avisar();
  }

  detener() {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }

  private async capturar(): Promise<ImageData | null> {
    if (this.usaDroidcam) {
      try {
        const r = await fetch(`${PUENTE_HTTP}/droidcam/foto?url=${encodeURIComponent(this.urlDroidcam)}`);
        if (!r.ok) throw new Error(await r.text());
        const bmp = await createImageBitmap(await r.blob());
        const c = document.createElement('canvas');
        c.width = bmp.width; c.height = bmp.height;
        const g = c.getContext('2d', { willReadFrequently: true })!;
        g.drawImage(bmp, 0, 0);
        return g.getImageData(0, 0, c.width, c.height);
      } catch (e) {
        this.error = 'No llegó la imagen del celular. Revisa la IP, que DroidCam esté abierto y que el puente esté corriendo. (' + (e as Error).message + ')';
        return null;
      }
    }
    const v = this.video;
    if (!v || !v.videoWidth) return null;
    const c = document.createElement('canvas');
    c.width = v.videoWidth; c.height = v.videoHeight;
    const g = c.getContext('2d', { willReadFrequently: true })!;
    g.drawImage(v, 0, 0);
    return g.getImageData(0, 0, c.width, c.height);
  }

  marcarEsquina(p: Punto2) {
    if (this.esquinas.length >= 4) this.esquinas = [];
    this.esquinas = [...this.esquinas, p];
    try { localStorage.setItem(CLAVE_ESQUINAS, JSON.stringify(this.esquinas)); } catch { /* sin almacenamiento */ }
    if (this.esquinas.length === 4) this.procesar();
    this.avisar();
  }

  borrarEsquinas() {
    this.esquinas = [];
    try { localStorage.removeItem(CLAVE_ESQUINAS); } catch { /* sin almacenamiento */ }
    this.modelo = null;
    this.avisar();
  }

  async tomarFondo() { this.error = ''; this.fondo = await this.capturar(); this.procesar(); this.avisar(); }
  async tomarFoto() { this.error = ''; this.foto = await this.capturar(); this.procesar(); this.avisar(); }

  cambiarGrosor(mm: number) { this.grosor = mm; this.avisar(); }

  // Esquinas de la cuadrícula dibujada, en mm, en el orden en que se marcan:
  // abajo-izquierda, abajo-derecha, arriba-derecha, arriba-izquierda.
  esquinasMm(): Punto2[] {
    const { nx, ny, paso } = escaneo.config;
    const a = -paso / 2, bx = (nx - 0.5) * paso, by = (ny - 0.5) * paso;
    return [[a, a], [bx, a], [bx, by], [a, by]];
  }

  procesar() {
    this.modelo = null;
    if (this.esquinas.length !== 4 || !this.fondo || !this.foto) return;
    if (this.fondo.width !== this.foto.width || this.fondo.height !== this.foto.height) return;
    const homografia = homografiaDesde4(this.esquinas, this.esquinasMm());
    if (!homografia) { this.error = 'Las esquinas no forman un cuadrilátero válido. Márcalas de nuevo.'; return; }
    const Hinv = inversa3(homografia.H)!;
    const [[x0, y0], , [x1, y1]] = this.esquinasMm() as [Punto2, Punto2, Punto2, Punto2];
    const ancho = Math.round((x1 - x0) / RESOLUCION_MM), alto = Math.round((y1 - y0) / RESOLUCION_MM);
    const rgb = new Uint8ClampedArray(ancho * alto * 4);
    const dif = new Uint8Array(ancho * alto);
    const { width: W, height: Hh } = this.foto;
    const F = this.foto.data, B = this.fondo.data;
    // Cada píxel de la imagen enderezada se busca en la foto con H⁻¹
    for (let r = 0; r < alto; r++) {
      const y = y1 - (r + 0.5) * RESOLUCION_MM;
      for (let c = 0; c < ancho; c++) {
        const x = x0 + (c + 0.5) * RESOLUCION_MM;
        const [u, v] = aplicarH(Hinv as Matriz, [x, y]);
        const ui = Math.round(u), vi = Math.round(v);
        const k = r * ancho + c;
        if (ui < 0 || vi < 0 || ui >= W || vi >= Hh) continue;
        const p = (vi * W + ui) * 4;
        rgb[k * 4] = F[p]!; rgb[k * 4 + 1] = F[p + 1]!; rgb[k * 4 + 2] = F[p + 2]!; rgb[k * 4 + 3] = 255;
        dif[k] = Math.min(255, (Math.abs(F[p]! - B[p]!) + Math.abs(F[p + 1]! - B[p + 1]!) + Math.abs(F[p + 2]! - B[p + 2]!)) / 1.5);
      }
    }
    const umbral = Math.max(30, umbralOtsu(dif));
    let mascara = new Uint8Array(ancho * alto);
    for (let k = 0; k < dif.length; k++) mascara[k] = dif[k]! > umbral ? 1 : 0;
    // Limpieza: cada píxel toma el valor de la mayoría de su vecindad 3×3
    const limpia = new Uint8Array(ancho * alto);
    for (let r = 1; r < alto - 1; r++) for (let c = 1; c < ancho - 1; c++) {
      let s = 0;
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) s += mascara[(r + dr) * ancho + c + dc]!;
      limpia[r * ancho + c] = s >= 5 ? 1 : 0;
    }
    mascara = limpia;
    let area = 0, xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (let r = 0; r < alto; r++) for (let c = 0; c < ancho; c++) {
      if (!mascara[r * ancho + c]) continue;
      area++;
      const x = x0 + (c + 0.5) * RESOLUCION_MM, y = y1 - (r + 0.5) * RESOLUCION_MM;
      xMin = Math.min(xMin, x); xMax = Math.max(xMax, x); yMin = Math.min(yMin, y); yMax = Math.max(yMax, y);
    }
    this.modelo = {
      x0, y0, ancho, alto, rgb, mascara, homografia, umbral,
      areaMm2: area * RESOLUCION_MM ** 2,
      caja: area ? { xMin, xMax, yMin, yMax } : null,
    };
  }

  // Altura en (x, y): la del sensor si la midió (más alta que el grosor), si no el grosor
  alturaEn(x: number, y: number): number {
    const { paso } = escaneo.config;
    const { puntos } = escaneo.geometria();
    if (!puntos.length) return this.grosor;
    const i = Math.round(x / paso), j = Math.round(y / paso);
    const p = puntos.find((q) => q.i === i && q.j === j);
    return Math.max(this.grosor, p ? p.h : 0);
  }
}

export const camara = new Camara();

export function useCamara() {
  const [, refrescar] = useReducer((n: number) => n + 1, 0);
  useEffect(() => {
    camara.addEventListener('cambio', refrescar);
    escaneo.addEventListener('cambio', refrescar);
    return () => { camara.removeEventListener('cambio', refrescar); escaneo.removeEventListener('cambio', refrescar); };
  }, []);
  return camara;
}
