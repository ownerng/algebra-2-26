// Vista 3D dibujada a mano: cada punto pasa por V = Rx(φ)·Rz(θ) y por la
// proyección Π; cada triángulo se sombrea con su normal (producto vectorial)
// y la luz (producto punto). Se pinta de atrás hacia adelante.

import { useEffect, useRef } from 'react';
import { matrizVista, sombreado, type Vec3 } from '../lib/algebra3d.ts';
import { multiplicarMatVec } from '../lib/matematica.ts';
import { escaneo } from '../lib/escaneo.ts';
import { camara } from '../lib/camara.ts';

const CELDA_CAMARA_MM = 2;

interface Prisma { x: number; y: number; h: number; color: [number, number, number]; paredes: [Vec3, Vec3][] }

// Objeto de la cámara: la silueta en celdas de 2 mm, cada una levantada a su
// altura, con el color de la foto. Las paredes van donde la vecina está vacía.
function prismasCamara(): Prisma[] {
  const m = camara.modelo;
  if (!m) return [];
  const n = CELDA_CAMARA_MM;
  const cols = Math.floor(m.ancho / n), filas = Math.floor(m.alto / n);
  const dentro = new Uint8Array(cols * filas);
  const colores: [number, number, number][] = [];
  for (let fr = 0; fr < filas; fr++) for (let fc = 0; fc < cols; fc++) {
    let s = 0, r = 0, g = 0, b = 0;
    for (let dr = 0; dr < n; dr++) for (let dc = 0; dc < n; dc++) {
      const k = (fr * n + dr) * m.ancho + fc * n + dc;
      if (m.mascara[k]) { s++; r += m.rgb[k * 4]!; g += m.rgb[k * 4 + 1]!; b += m.rgb[k * 4 + 2]!; }
    }
    if (s * 2 >= n * n) { dentro[fr * cols + fc] = 1; colores[fr * cols + fc] = [r / s, g / s, b / s]; }
  }
  const lado = (fr: number, fc: number) => fr >= 0 && fc >= 0 && fr < filas && fc < cols && dentro[fr * cols + fc] === 1;
  const y1 = m.y0 + m.alto;
  const res: Prisma[] = [];
  for (let fr = 0; fr < filas; fr++) for (let fc = 0; fc < cols; fc++) {
    if (!dentro[fr * cols + fc]) continue;
    const xa = m.x0 + fc * n, xb = xa + n, yb = y1 - fr * n, ya = yb - n;
    const h = camara.alturaEn((xa + xb) / 2, (ya + yb) / 2);
    const paredes: [Vec3, Vec3][] = [];
    if (!lado(fr + 1, fc)) paredes.push([[xa, ya, 0], [xb, ya, 0]]);
    if (!lado(fr - 1, fc)) paredes.push([[xa, yb, 0], [xb, yb, 0]]);
    if (!lado(fr, fc - 1)) paredes.push([[xa, ya, 0], [xa, yb, 0]]);
    if (!lado(fr, fc + 1)) paredes.push([[xb, ya, 0], [xb, yb, 0]]);
    res.push({ x: xa, y: ya, h, color: colores[fr * cols + fc]!, paredes });
  }
  return res;
}

let cachePrismas = { clave: '', prismas: [] as Prisma[] };

const LUZ: Vec3 = [-0.35, -0.55, 1];
const MONO = '500 11px "IBM Plex Mono", ui-monospace, monospace';

function colorAltura(t: number, brillo: number) {
  // De ámbar oscuro (mesa) a ámbar claro (lo más alto), oscurecido por la sombra
  const base = [58 + t * (255 - 58), 40 + t * (214 - 40), 18 + t * (120 - 18)];
  const k = 0.3 + 0.7 * brillo;
  return `rgb(${base.map((c) => Math.round(c * k)).join(',')})`;
}

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
    const rueda = (e: WheelEvent) => { e.preventDefault(); escaneo.vista.zoom = Math.min(4, Math.max(0.4, escaneo.vista.zoom * Math.exp(-e.deltaY * 0.0015))); };
    canvas.addEventListener('pointerdown', abajo);
    canvas.addEventListener('pointermove', mover);
    canvas.addEventListener('pointerup', arriba);
    canvas.addEventListener('wheel', rueda, { passive: false });

    let cuadro = 0;
    const dibujar = () => {
      cuadro = requestAnimationFrame(dibujar);
      if (!ancho) return;
      const { nx, ny, paso, altura } = escaneo.config;
      if (escaneo.vista.girar && !arrastre) escaneo.vista.theta += 0.12;
      const V = matrizVista(escaneo.vista.theta, escaneo.vista.phi);
      const centro: Vec3 = [((nx - 1) * paso) / 2, ((ny - 1) * paso) / 2, altura * 0.35];
      const k = (Math.min(ancho, alto) / (Math.max(Math.max(nx, ny) * paso, altura) * 1.6)) * escaneo.vista.zoom;
      // Pantalla = Π · V · (p − c): Π se queda con la 1.ª y 3.ª componente
      const pantalla = (p: Vec3) => {
        const v = multiplicarMatVec(V, [p[0] - centro[0], p[1] - centro[1], p[2] - centro[2]]);
        return { x: ancho / 2 + k * v[0]!, y: alto / 2 - k * v[2]!, prof: v[1]! };
      };

      const fondo = g.createRadialGradient(ancho / 2, alto / 2, 0, ancho / 2, alto / 2, Math.max(ancho, alto) * 0.75);
      fondo.addColorStop(0, '#1a1309');
      fondo.addColorStop(1, '#0a0704');
      g.fillStyle = fondo;
      g.fillRect(0, 0, ancho, alto);

      // Mesa: la cuadrícula en z = 0
      g.lineWidth = 1;
      g.strokeStyle = 'rgba(255,178,56,0.13)';
      g.beginPath();
      for (let i = 0; i < nx; i++) {
        const a = pantalla([i * paso, 0, 0]), b = pantalla([i * paso, (ny - 1) * paso, 0]);
        g.moveTo(a.x, a.y); g.lineTo(b.x, b.y);
      }
      for (let j = 0; j < ny; j++) {
        const a = pantalla([0, j * paso, 0]), b = pantalla([(nx - 1) * paso, j * paso, 0]);
        g.moveTo(a.x, a.y); g.lineTo(b.x, b.y);
      }
      g.stroke();

      // Ejes x, y, z desde la casilla (0, 0)
      const o = pantalla([0, 0, 0]);
      const ejes: [Vec3, string, string][] = [[[60, 0, 0], '#ff6f59', 'x'], [[0, 60, 0], '#a6e36b', 'y'], [[0, 0, 60], '#8fd8ff', 'z']];
      g.font = MONO;
      for (const [d, color, nombre] of ejes) {
        const f = pantalla(d);
        g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 2;
        g.beginPath(); g.moveTo(o.x, o.y); g.lineTo(f.x, f.y); g.stroke();
        g.fillText(nombre, f.x + 4, f.y - 4);
      }

      const { puntos } = escaneo.geometria();
      const tris: { pts: { x: number; y: number }[]; prof: number; color: string }[] = [];

      // Con silueta de la cámara: el objeto exacto (prismas de 2 mm con el color de la foto)
      if (camara.modelo) {
        const clave = `${camara.version}-${escaneo.version}`;
        if (cachePrismas.clave !== clave) cachePrismas = { clave, prismas: prismasCamara() };
        const n = CELDA_CAMARA_MM;
        for (const p of cachePrismas.prismas) {
          const sube = (q: Vec3): Vec3 => [q[0], q[1], p.h];
          const caras: Vec3[][] = [
            [[p.x, p.y, p.h], [p.x + n, p.y, p.h], [p.x + n, p.y + n, p.h], [p.x, p.y + n, p.h]],
            ...p.paredes.map(([a, b]) => [a, b, sube(b), sube(a)]),
          ];
          for (const cara of caras) {
            const { brillo } = sombreado(cara[0]!, cara[1]!, cara[2]!, LUZ);
            const k = 0.35 + 0.65 * (cara === caras[0] ? Math.max(brillo, 0.85) : brillo);
            const pts = cara.map(pantalla);
            tris.push({ pts, prof: pts.reduce((s, q) => s + q.prof, 0) / pts.length, color: `rgb(${p.color.map((c) => Math.round(c * k)).join(',')})` });
          }
        }
      }

      // Superficie del sensor: dos triángulos por cada cuadro con sus 4 esquinas medidas
      const mapa = new Map(puntos.map((p) => [`${p.i},${p.j}`, p]));
      const maxH = Math.max(10, ...puntos.map((p) => p.h));
      for (let j = 0; j < ny - 1 && !camara.modelo; j++) {
        for (let i = 0; i < nx - 1; i++) {
          const esquinas = [[i, j], [i + 1, j], [i + 1, j + 1], [i, j + 1]].map(([a, b]) => mapa.get(`${a},${b}`));
          if (esquinas.some((e) => !e)) continue;
          const v = esquinas.map((e) => [e!.x, e!.y, Math.max(0, e!.h)] as Vec3);
          for (const [a, b, c] of [[0, 1, 2], [0, 2, 3]] as const) {
            const { brillo } = sombreado(v[a]!, v[b]!, v[c]!, LUZ);
            const pts = [v[a]!, v[b]!, v[c]!].map(pantalla);
            const t = Math.min(1, ((v[a]![2] + v[b]![2] + v[c]![2]) / 3) / maxH);
            tris.push({ pts, prof: (pts[0]!.prof + pts[1]!.prof + pts[2]!.prof) / 3, color: colorAltura(t, brillo) });
          }
        }
      }
      tris.sort((a, b) => b.prof - a.prof);
      for (const t of tris) {
        g.fillStyle = t.color; g.strokeStyle = t.color; g.lineWidth = 0.6;
        g.beginPath();
        t.pts.forEach((q, n) => (n ? g.lineTo(q.x, q.y) : g.moveTo(q.x, q.y)));
        g.closePath(); g.fill(); g.stroke();
      }

      // Puntos medidos
      for (const p of puntos) {
        const s = pantalla([p.x, p.y, Math.max(0, p.h)]);
        g.fillStyle = p.mesa ? 'rgba(143,216,255,0.55)' : '#ffe2a8';
        g.beginPath(); g.arc(s.x, s.y, 2.2, 0, Math.PI * 2); g.fill();
      }

      // Casilla congelada en el cuaderno
      const res = escaneo.resaltada && mapa.get(`${escaneo.resaltada.i},${escaneo.resaltada.j}`);
      if (res) {
        const s = pantalla([res.x, res.y, Math.max(0, res.h)]);
        g.strokeStyle = '#fffaf0'; g.lineWidth = 2;
        g.beginPath(); g.arc(s.x, s.y, 9, 0, Math.PI * 2); g.stroke();
      }

      // La caja en la casilla actual y su haz hacia abajo
      if (escaneo.cursor < escaneo.total) {
        const [i, j] = escaneo.casillaActual;
        const x = i * paso, y = j * paso;
        const arribaCaja = pantalla([x, y, altura]);
        const vivo = escaneo.vivo && performance.now() - escaneo.vivo.t < 1000 && escaneo.vivo.r > 0 ? escaneo.vivo.r : null;
        const fin = pantalla([x, y, vivo ? Math.max(0, altura - vivo) : 0]);
        g.setLineDash([4, 4]);
        g.strokeStyle = 'rgba(255,178,56,0.8)'; g.lineWidth = 1.5;
        g.beginPath(); g.moveTo(arribaCaja.x, arribaCaja.y); g.lineTo(fin.x, fin.y); g.stroke();
        g.setLineDash([]);
        const m = 14;
        const tapa = [[-m, -m], [m, -m], [m, m], [-m, m]].map(([a, b]) => pantalla([x + a!, y + b!, altura]));
        g.fillStyle = 'rgba(255,178,56,0.18)'; g.strokeStyle = '#ffb238';
        g.beginPath(); tapa.forEach((p, n) => (n ? g.lineTo(p.x, p.y) : g.moveTo(p.x, p.y))); g.closePath(); g.fill(); g.stroke();
        g.fillStyle = '#ffb238';
        g.fillText('caja', tapa[1]!.x + 6, tapa[1]!.y);
      }

      g.fillStyle = 'rgba(255,178,56,0.7)';
      g.font = MONO;
      g.fillText(`θ = ${Math.round(((escaneo.vista.theta % 360) + 360) % 360)}°   φ = ${Math.round(escaneo.vista.phi)}°   arrastra para girar`, 14, alto - 14);
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

  return <canvas ref={lienzo} className="lienzo3d" aria-label="Superficie 3D del objeto escaneado" />;
}
