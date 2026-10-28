// Vista 3D dibujada a mano: cada punto pasa por V = Rx(φ)·Rz(θ) y por la
// proyección Π; cada cara se sombrea con su normal (producto vectorial) y la
// luz (producto punto), y se pinta de atrás hacia adelante.

import { useEffect, useRef } from 'react';
import { matrizVista, sombreado, type Vec3 } from '../lib/algebra3d.ts';
import { multiplicarMatVec } from '../lib/matematica.ts';
import { escaneo } from '../lib/escaneo.ts';
import { camara, type ModeloCamara } from '../lib/camara.ts';

export const LUZ: Vec3 = [-0.35, -0.55, 1];
const MONO = '500 11px "IBM Plex Mono", ui-monospace, monospace';
const CELDAS_MAX = 90; // celdas a lo largo del lado mayor del objeto

interface Cara { pts: Vec3[]; color: [number, number, number]; arriba: boolean }

// Del píxel (u, v) al plano de la foto en mm, con el centro del objeto en el origen
export function aPlano(m: ModeloCamara, u: number, v: number): [number, number] {
  const cu = (m.caja.u0 + m.caja.u1) / 2, cv = (m.caja.v0 + m.caja.v1) / 2;
  return [(u - cu) * m.mmPorPx, -(v - cv) * m.mmPorPx];
}

export function caras(m: ModeloCamara): Cara[] {
  const { u0, u1, v0, v1 } = m.caja;
  const n = Math.max(1, Math.ceil(Math.max(u1 - u0 + 1, v1 - v0 + 1) / CELDAS_MAX));
  const cols = Math.ceil((u1 - u0 + 1) / n), filas = Math.ceil((v1 - v0 + 1) / n);
  const h = new Float32Array(cols * filas);
  const color: [number, number, number][] = [];
  for (let fr = 0; fr < filas; fr++) for (let fc = 0; fc < cols; fc++) {
    let s = 0, r = 0, g = 0, b = 0, alt = 0;
    for (let dv = 0; dv < n; dv++) for (let du = 0; du < n; du++) {
      const u = u0 + fc * n + du, v = v0 + fr * n + dv;
      if (u > u1 || v > v1) continue;
      const k = v * m.ancho + u;
      if (!m.mascara[k]) continue;
      s++; alt += m.altura[k]!; r += m.rgb[k * 4]!; g += m.rgb[k * 4 + 1]!; b += m.rgb[k * 4 + 2]!;
    }
    if (s * 2 >= n * n) { h[fr * cols + fc] = alt / s; color[fr * cols + fc] = [r / s, g / s, b / s]; }
  }
  const alturaDe = (fr: number, fc: number) => (fr < 0 || fc < 0 || fr >= filas || fc >= cols ? 0 : h[fr * cols + fc]!);
  const res: Cara[] = [];
  for (let fr = 0; fr < filas; fr++) for (let fc = 0; fc < cols; fc++) {
    const z = h[fr * cols + fc]!;
    if (!z) continue;
    const [xa, yb] = aPlano(m, u0 + fc * n, v0 + fr * n);
    const [xb, ya] = aPlano(m, u0 + (fc + 1) * n, v0 + (fr + 1) * n);
    const c = color[fr * cols + fc]!;
    res.push({ pts: [[xa, ya, z], [xb, ya, z], [xb, yb, z], [xa, yb, z]], color: c, arriba: true });
    // Paredes donde la vecina es más baja
    const vecinas: [number, Vec3, Vec3][] = [
      [alturaDe(fr + 1, fc), [xa, ya, 0], [xb, ya, 0]],
      [alturaDe(fr - 1, fc), [xa, yb, 0], [xb, yb, 0]],
      [alturaDe(fr, fc - 1), [xa, ya, 0], [xa, yb, 0]],
      [alturaDe(fr, fc + 1), [xb, ya, 0], [xb, yb, 0]],
    ];
    for (const [zv, a, b] of vecinas) {
      if (zv >= z - 0.01) continue;
      res.push({ pts: [[a[0], a[1], zv], [b[0], b[1], zv], [b[0], b[1], z], [a[0], a[1], z]], color: c, arriba: false });
    }
  }
  return res;
}

let cache = { version: -1, caras: [] as Cara[] };

export function Vista3D() {
  const lienzo = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = lienzo.current!;
    const g = canvas.getContext('2d')!;
    let ancho = 0, alto = 0;
    const redimensionar = () => {
      const dpr = window.devicePixelRatio || 1;
      const r = canvas.getBoundingClientRect();
      ancho = r.width; alto = r.height;
      canvas.width = Math.round(ancho * dpr);
      canvas.height = Math.round(alto * dpr);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const obs = new ResizeObserver(redimensionar);
    obs.observe(canvas);
    redimensionar();

    let arrastre: { x: number; y: number } | null = null;
    const abajo = (e: PointerEvent) => { arrastre = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); };
    const mover = (e: PointerEvent) => {
      if (!arrastre) return;
      escaneo.vista.theta += (e.clientX - arrastre.x) * 0.4;
      escaneo.vista.phi = Math.min(89, Math.max(5, escaneo.vista.phi + (e.clientY - arrastre.y) * 0.4));
      arrastre = { x: e.clientX, y: e.clientY };
    };
    const arriba = () => { arrastre = null; };
    const rueda = (e: WheelEvent) => { e.preventDefault(); escaneo.vista.zoom = Math.min(5, Math.max(0.3, escaneo.vista.zoom * Math.exp(-e.deltaY * 0.0015))); };
    canvas.addEventListener('pointerdown', abajo);
    canvas.addEventListener('pointermove', mover);
    canvas.addEventListener('pointerup', arriba);
    canvas.addEventListener('wheel', rueda, { passive: false });

    let cuadro = 0;
    const dibujar = () => {
      cuadro = requestAnimationFrame(dibujar);
      if (!ancho) return;
      const fondo = g.createRadialGradient(ancho / 2, alto / 2, 0, ancho / 2, alto / 2, Math.max(ancho, alto) * 0.75);
      fondo.addColorStop(0, '#1a1309');
      fondo.addColorStop(1, '#0a0704');
      g.fillStyle = fondo;
      g.fillRect(0, 0, ancho, alto);
      const m = camara.modelo;
      if (!m) return;

      if (escaneo.vista.girar && !arrastre) escaneo.vista.theta += 0.15;
      const V = matrizVista(escaneo.vista.theta, escaneo.vista.phi);
      const tam = Math.max(m.caja.anchoMm, m.caja.altoMm);
      const k = (Math.min(ancho, alto) / (tam * 1.7)) * escaneo.vista.zoom;
      const pantalla = (p: Vec3) => {
        const v = multiplicarMatVec(V, p);
        return { x: ancho / 2 + k * v[0]!, y: alto / 2 - k * v[2]!, prof: v[1]! };
      };

      // Piso: una cuadrícula de 1 cm (5 cm para objetos grandes) bajo el objeto
      const paso = tam > 200 ? 50 : 10;
      const lim = Math.ceil((tam * 0.8) / paso) * paso;
      g.lineWidth = 1;
      g.strokeStyle = 'rgba(255,178,56,0.10)';
      g.beginPath();
      for (let s = -lim; s <= lim; s += paso) {
        const a = pantalla([s, -lim, 0]), b = pantalla([s, lim, 0]);
        const c = pantalla([-lim, s, 0]), d = pantalla([lim, s, 0]);
        g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.moveTo(c.x, c.y); g.lineTo(d.x, d.y);
      }
      g.stroke();

      if (cache.version !== camara.version) cache = { version: camara.version, caras: caras(m) };
      const pintadas = cache.caras.map((c) => {
        const pts = c.pts.map(pantalla);
        const { brillo } = sombreado(c.pts[0]!, c.pts[1]!, c.pts[2]!, LUZ);
        const luz = 0.35 + 0.65 * (c.arriba ? Math.max(brillo, 0.55) : brillo);
        return { pts, prof: pts.reduce((s, q) => s + q.prof, 0) / pts.length, color: `rgb(${c.color.map((x) => Math.round(x * luz)).join(',')})` };
      });
      pintadas.sort((a, b) => b.prof - a.prof);
      for (const t of pintadas) {
        g.fillStyle = t.color; g.strokeStyle = t.color; g.lineWidth = 0.6;
        g.beginPath();
        t.pts.forEach((q, n) => (n ? g.lineTo(q.x, q.y) : g.moveTo(q.x, q.y)));
        g.closePath(); g.fill(); g.stroke();
      }

      // Escala sobre el piso
      const a = pantalla([-lim, -lim, 0]), b = pantalla([-lim + paso, -lim, 0]);
      g.strokeStyle = '#ffb238'; g.lineWidth = 3;
      g.beginPath(); g.moveTo(a.x, a.y); g.lineTo(b.x, b.y); g.stroke();
      g.fillStyle = '#ffb238'; g.font = MONO;
      g.fillText(`${paso / 10} cm`, b.x + 6, b.y + 4);
      g.fillStyle = 'rgba(255,178,56,0.7)';
      g.fillText('arrastra para girar · rueda para acercar', 14, alto - 14);
    };
    cuadro = requestAnimationFrame(dibujar);

    return () => {
      cancelAnimationFrame(cuadro);
      obs.disconnect();
      canvas.removeEventListener('pointerdown', abajo);
      canvas.removeEventListener('pointermove', mover);
      canvas.removeEventListener('pointerup', arriba);
      canvas.removeEventListener('wheel', rueda);
    };
  }, []);

  return <canvas ref={lienzo} className="lienzo3d" aria-label="Objeto en 3D" />;
}
