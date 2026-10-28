// Cámara del celular (DroidCam por WiFi) o webcam. Con una sola foto se
// obtiene la silueta del objeto; con la distancia que mide el VL53L0X y el
// modelo de cámara estenopeica, su tamaño real; y con la distancia al borde de
// la silueta, un relieve 3D estimado.

import { useEffect, useReducer } from 'react';
import type { Matriz } from './matematica.ts';
import { colorFondo, componenteMayor, distanciaAlBorde, distanciaColor, matrizK, umbralOtsu } from './vision.ts';
import { escaneo } from './escaneo.ts';

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
  modelo: ModeloCamara | null = null;
  vista: 'camara' | '3d' = 'camara';
  version = 0;

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
    if (this.ultimaFoto && ('anguloVision' in c || 'distanciaManual' in c || 'grosor' in c || 'relieve' in c)) this.procesar(this.ultimaFoto, this.modelo?.Z);
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
      this.procesar(foto, vivo ?? undefined);
      if (this.modelo) this.vista = '3d';
    } catch (e) {
      this.error = 'No se pudo tomar la foto: ' + (e as Error).message;
    }
    this.capturando = false;
    this.avisar();
  }

  procesar(foto: ImageData, zSensor?: number) {
    const { width: ancho, height: alto, data: rgb } = foto;
    const fondo = colorFondo(rgb, ancho, alto);
    const dif = distanciaColor(rgb, fondo);
    const umbral = Math.max(25, umbralOtsu(dif));
    let mascara: Uint8Array = new Uint8Array(dif.length);
    for (let k = 0; k < dif.length; k++) mascara[k] = dif[k]! > umbral ? 1 : 0;
    mascara = componenteMayor(mascara, ancho, alto);
    let areaPx = 0, u0 = ancho, u1 = 0, v0 = alto, v1 = 0;
    for (let v = 0; v < alto; v++) for (let u = 0; u < ancho; u++) {
      if (!mascara[v * ancho + u]) continue;
      areaPx++; u0 = Math.min(u0, u); u1 = Math.max(u1, u); v0 = Math.min(v0, v); v1 = Math.max(v1, v);
    }
    if (areaPx < 30 || areaPx > 0.9 * ancho * alto) {
      this.modelo = null;
      this.error = 'No se distinguió el objeto del fondo. Pon el objeto en el centro, sobre un fondo liso de otro color, sin que toque los bordes de la imagen.';
      return;
    }
    const zDelSensor = zSensor !== undefined;
    const Z = zSensor ?? this.distanciaManual;
    const K = matrizK(ancho, alto, this.anguloVision);
    const mmPorPx = Z / K[0]![0]!;
    const dist = distanciaAlBorde(mascara, ancho, alto);
    let dMax = 0;
    for (const d of dist) dMax = Math.max(dMax, d);
    const anchoMm = (u1 - u0 + 1) * mmPorPx, altoMm = (v1 - v0 + 1) * mmPorPx;
    // Relieve estimado: más alto en el centro de la silueta, cero en el borde
    const hMax = this.relieve * 0.5 * Math.min(anchoMm, altoMm);
    const altura = new Float32Array(mascara.length);
    for (let k = 0; k < altura.length; k++) {
      if (mascara[k]) altura[k] = this.grosor + (dMax ? hMax * Math.sqrt(dist[k]! / dMax) : 0);
    }
    this.modelo = { ancho, alto, rgb, mascara, altura, K, Z, zDelSensor, mmPorPx, fondo, umbral, areaPx, caja: { u0, u1, v0, v1, anchoMm, altoMm } };
  }
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
