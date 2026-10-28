// Pantalla de radar: mapa global en milímetros, eje y hacia arriba,
// dibujado como un tubo de fósforo ámbar.

import { useEffect, useRef } from 'react';
import { aRad, type Pose } from '../lib/matematica.ts';
import { COLORES, escaner, type Posicion, type PuntoMapa } from '../lib/escaner.ts';
import { CIRCULOS, SEGMENTOS } from '../lib/simulador.ts';

const ALCANCE = 1200;
const FOSFORO = '255,178,56';
const EJE_X = '#ff6f59';
const EJE_Y = '#a6e36b';
const MONO = '500 11px "IBM Plex Mono", ui-monospace, monospace';

function brillo(color: string, tam = 32) {
  const c = document.createElement('canvas');
  c.width = c.height = tam;
  const g = c.getContext('2d')!;
  const grad = g.createRadialGradient(tam / 2, tam / 2, 0, tam / 2, tam / 2, tam / 2);
  grad.addColorStop(0, '#fffaf0');
  grad.addColorStop(0.2, color);
  grad.addColorStop(0.5, color + '50');
  grad.addColorStop(1, color + '00');
  g.fillStyle = grad;
  g.fillRect(0, 0, tam, tam);
  return c;
}

class PantallaRadar {
  private g: CanvasRenderingContext2D;
  private ancho = 0;
  private alto = 0;
  escala = 0.3;
  origen: [number, number] = [0, 0];
  private sprites: Record<string, HTMLCanvasElement> = {};

  constructor(private canvas: HTMLCanvasElement) {
    this.g = canvas.getContext('2d')!;
  }

  redimensionar() {
    const dpr = window.devicePixelRatio || 1;
    const { width, height } = this.canvas.getBoundingClientRect();
    if (!width || !height) return;
    const primera = !this.ancho;
    this.ancho = width;
    this.alto = height;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.g.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (primera) this.encuadrar();
  }

  encuadrar() {
    if (!this.ancho) return;
    // Esquinas del semicírculo de cada posición, llevadas al sistema global
    const esquinas = [escaner.poses[1], escaner.poses[2]].flatMap((p) => {
      const c = Math.cos(aRad(p.alpha)), s = Math.sin(aRad(p.alpha));
      return [[-ALCANCE, -150], [ALCANCE, -150], [-ALCANCE, ALCANCE], [ALCANCE, ALCANCE]]
        .map(([x, y]) => [p.dx + c * x! - s * y!, p.dy + s * x! + c * y!] as const);
    });
    const xs = esquinas.map((e) => e[0]);
    const ys = esquinas.map((e) => e[1]);
    const [minX, maxX, minY, maxY] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    this.escala = Math.min(this.ancho / (maxX - minX), this.alto / (maxY - minY)) * 0.94;
    this.origen = [this.ancho / 2 - ((minX + maxX) / 2) * this.escala, this.alto / 2 + ((minY + maxY) / 2) * this.escala];
  }

  aPantalla(x: number, y: number): [number, number] {
    return [this.origen[0] + x * this.escala, this.origen[1] - y * this.escala];
  }

  aMundo(px: number, py: number): [number, number] {
    return [(px - this.origen[0]) / this.escala, (this.origen[1] - py) / this.escala];
  }

  zoom(px: number, py: number, f: number) {
    this.origen = [px - (px - this.origen[0]) * f, py - (py - this.origen[1]) * f];
    this.escala *= f;
  }

  private sprite(color: string) {
    return (this.sprites[color] ??= brillo(color));
  }

  dibujar(ahora: number) {
    const g = this.g;
    if (!this.ancho) return;
    // Fondo con viñeta, como el cristal de un tubo
    const fondo = g.createRadialGradient(this.ancho / 2, this.alto / 2, 0, this.ancho / 2, this.alto / 2, Math.max(this.ancho, this.alto) * 0.75);
    fondo.addColorStop(0, '#1a1309');
    fondo.addColorStop(1, '#0a0704');
    g.fillStyle = fondo;
    g.fillRect(0, 0, this.ancho, this.alto);

    this.cuadricula();
    if (escaner.tipoFuente === 'simulacion' && escaner.verHabitacion) this.habitacion();
    const activa = escaner.posicionActiva;
    this.anillos(escaner.poses[activa], COLORES[activa]);
    for (const id of [1, 2] as Posicion[]) this.pose(id, escaner.poses[id], id === activa);
    this.haz(ahora);
    const puntos = escaner.puntos();
    for (const id of [1, 2] as Posicion[]) this.puntos(puntos[id], COLORES[id], ahora);
    this.escalaGrafica();
  }

  private cuadricula() {
    const g = this.g;
    const paso = this.escala * 100 < 8 ? 500 : 100;
    const [x0, y1] = this.aMundo(0, 0);
    const [x1, y0] = this.aMundo(this.ancho, this.alto);
    g.lineWidth = 1;
    for (let x = Math.floor(x0 / paso) * paso; x <= x1; x += paso) {
      const [sx] = this.aPantalla(x, 0);
      g.strokeStyle = `rgba(${FOSFORO},${x % 500 === 0 ? 0.11 : 0.04})`;
      g.beginPath(); g.moveTo(sx, 0); g.lineTo(sx, this.alto); g.stroke();
    }
    for (let y = Math.floor(y0 / paso) * paso; y <= y1; y += paso) {
      const [, sy] = this.aPantalla(0, y);
      g.strokeStyle = `rgba(${FOSFORO},${y % 500 === 0 ? 0.11 : 0.04})`;
      g.beginPath(); g.moveTo(0, sy); g.lineTo(this.ancho, sy); g.stroke();
    }
  }

  // Lo que "debería" ver el sensor: la habitación real del simulador
  private habitacion() {
    const g = this.g;
    g.save();
    g.strokeStyle = 'rgba(255,240,220,0.16)';
    g.lineWidth = 1.2;
    g.setLineDash([4, 5]);
    g.beginPath();
    for (const [a, b] of SEGMENTOS) {
      const [ax, ay] = this.aPantalla(a[0], a[1]);
      const [bx, by] = this.aPantalla(b[0], b[1]);
      g.moveTo(ax, ay); g.lineTo(bx, by);
    }
    for (const { c, r } of CIRCULOS) {
      const [cx, cy] = this.aPantalla(c[0], c[1]);
      g.moveTo(cx + r * this.escala, cy);
      g.arc(cx, cy, r * this.escala, 0, Math.PI * 2);
    }
    g.stroke();
    g.restore();
  }

  private anillos(pose: Pose, color: string) {
    const g = this.g;
    const [cx, cy] = this.aPantalla(pose.dx, pose.dy);
    const a0 = -aRad(pose.alpha);
    g.font = MONO;
    g.lineWidth = 1;
    for (let r = 300; r <= ALCANCE; r += 300) {
      g.strokeStyle = color + '38';
      g.beginPath();
      g.arc(cx, cy, r * this.escala, a0 - Math.PI, a0, false);
      g.stroke();
      const lx = cx + Math.cos(a0) * r * this.escala;
      const ly = cy + Math.sin(a0) * r * this.escala;
      g.fillStyle = color + '90';
      g.fillText(r / 10 + ' cm', lx - 18, ly + 15);
    }
    for (let t = 0; t <= 180; t += 30) {
      const a = a0 - aRad(t);
      g.strokeStyle = color + '1c';
      g.beginPath();
      g.moveTo(cx, cy);
      g.lineTo(cx + Math.cos(a) * ALCANCE * this.escala, cy + Math.sin(a) * ALCANCE * this.escala);
      g.stroke();
      g.fillStyle = color + '80';
      g.fillText(t + '°', cx + Math.cos(a) * (ALCANCE + 70) * this.escala - 10, cy + Math.sin(a) * (ALCANCE + 70) * this.escala + 4);
    }
  }

  private flecha(x0: number, y0: number, x1: number, y1: number, color: string, texto: string) {
    const g = this.g;
    const ang = Math.atan2(y1 - y0, x1 - x0);
    g.strokeStyle = color;
    g.fillStyle = color;
    g.lineWidth = 2;
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    g.beginPath();
    g.moveTo(x1, y1);
    g.lineTo(x1 - 9 * Math.cos(ang - 0.4), y1 - 9 * Math.sin(ang - 0.4));
    g.lineTo(x1 - 9 * Math.cos(ang + 0.4), y1 - 9 * Math.sin(ang + 0.4));
    g.closePath(); g.fill();
    g.fillText(texto, x1 + 8 * Math.cos(ang) - 5, y1 + 12 * Math.sin(ang) + 4);
  }

  private pose(id: Posicion, pose: Pose, activa: boolean) {
    const g = this.g;
    const [cx, cy] = this.aPantalla(pose.dx, pose.dy);
    const L = 220;
    const a = aRad(pose.alpha);
    g.font = MONO;
    if (id === 2) {
      const [ox, oy] = this.aPantalla(0, 0);
      g.setLineDash([5, 5]);
      g.strokeStyle = 'rgba(255,240,220,0.35)';
      g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(ox, oy); g.lineTo(cx, cy); g.stroke();
      g.setLineDash([]);
      g.fillStyle = 'rgba(255,240,220,0.6)';
      g.fillText('t = (dx, dy)', (ox + cx) / 2 + 8, (oy + cy) / 2);
    }
    // Columnas de R(α): los ejes locales vistos desde el sistema global
    const ex = this.aPantalla(pose.dx + L * Math.cos(a), pose.dy + L * Math.sin(a));
    const ey = this.aPantalla(pose.dx - L * Math.sin(a), pose.dy + L * Math.cos(a));
    g.globalAlpha = activa ? 1 : 0.5;
    this.flecha(cx, cy, ex[0], ex[1], EJE_X, 'x' + '₀₁₂'[id]);
    this.flecha(cx, cy, ey[0], ey[1], EJE_Y, 'y' + '₀₁₂'[id]);
    const color = COLORES[id];
    g.shadowColor = color;
    g.shadowBlur = activa ? 16 : 4;
    g.fillStyle = color;
    g.beginPath(); g.arc(cx, cy, activa ? 7 : 5, 0, Math.PI * 2); g.fill();
    g.shadowBlur = 0;
    g.font = '600 13px "Barlow Condensed", sans-serif';
    g.fillText('P' + id, cx - 26, cy + 19);
    g.globalAlpha = 1;
  }

  private haz(ahora: number) {
    const u = escaner.ultima;
    if (!u || ahora - u.t > 1500) return;
    const g = this.g;
    const pose = escaner.poses[u.posicion];
    const color = COLORES[u.posicion];
    const theta = escaner.invertir ? 180 - u.thetaServo : u.thetaServo;
    const dir = escaner.invertir ? -escaner.direccion : escaner.direccion;
    const [cx, cy] = this.aPantalla(pose.dx, pose.dy);
    const R = ALCANCE * this.escala;
    const base = pose.alpha + theta;
    // Estela del barrido
    for (let i = 20; i >= 1; i--) {
      const a = -aRad(base - dir * i);
      g.strokeStyle = color + Math.round((1 - i / 20) * 60).toString(16).padStart(2, '0');
      g.lineWidth = 3;
      g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + Math.cos(a) * R, cy + Math.sin(a) * R); g.stroke();
    }
    const a = -aRad(base);
    const largo = u.r > 0 ? Math.min(u.r + escaner.d, ALCANCE) * this.escala : R;
    g.strokeStyle = color;
    g.lineWidth = 2;
    g.shadowColor = color;
    g.shadowBlur = 10;
    g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + Math.cos(a) * largo, cy + Math.sin(a) * largo); g.stroke();
    g.shadowBlur = 0;
  }

  private puntos(puntos: PuntoMapa[], color: string, ahora: number) {
    if (!puntos.length) return;
    const g = this.g;
    // Contorno: une lecturas vecinas del mismo objeto
    g.strokeStyle = color + '80';
    g.lineWidth = 1.5;
    g.beginPath();
    for (let i = 1; i < puntos.length; i++) {
      const p = puntos[i - 1]!, q = puntos[i]!;
      if (q.theta - p.theta <= 4 && Math.hypot(q.x - p.x, q.y - p.y) < 110) {
        const [ax, ay] = this.aPantalla(p.x, p.y);
        const [bx, by] = this.aPantalla(q.x, q.y);
        g.moveTo(ax, ay); g.lineTo(bx, by);
      }
    }
    g.stroke();
    g.globalCompositeOperation = 'lighter';
    const spr = this.sprite(color);
    const blanco = this.sprite('#ffffff');
    const tam = Math.max(9, Math.min(18, 40 * this.escala));
    for (const p of puntos) {
      const [sx, sy] = this.aPantalla(p.x, p.y);
      const edad = ahora - p.t;
      if (edad >= 0 && edad < 600) {
        const k = 1 + (1 - edad / 600) * 0.8;
        g.globalAlpha = 0.5 * (1 - edad / 600);
        g.drawImage(blanco, sx - (tam * k) / 2, sy - (tam * k) / 2, tam * k, tam * k);
      }
      g.globalAlpha = 0.85;
      g.drawImage(spr, sx - tam / 2, sy - tam / 2, tam, tam);
    }
    g.globalAlpha = 1;
    g.globalCompositeOperation = 'source-over';
  }

  private escalaGrafica() {
    const g = this.g;
    const mm = this.escala * 100 < 30 ? 500 : 100;
    const px = mm * this.escala;
    const x = 18, y = this.alto - 20;
    g.strokeStyle = `rgba(${FOSFORO},0.8)`;
    g.fillStyle = `rgba(${FOSFORO},0.8)`;
    g.lineWidth = 2;
    g.beginPath(); g.moveTo(x, y - 5); g.lineTo(x, y); g.lineTo(x + px, y); g.lineTo(x + px, y - 5); g.stroke();
    g.font = MONO;
    g.fillText(mm / 10 + ' cm', x + px + 8, y + 1);
  }
}

export function Radar({ encuadre }: { encuadre: number }) {
  const lienzo = useRef<HTMLCanvasElement>(null);
  const pantalla = useRef<PantallaRadar | null>(null);

  useEffect(() => {
    const canvas = lienzo.current!;
    const p = new PantallaRadar(canvas);
    pantalla.current = p;
    const obs = new ResizeObserver(() => p.redimensionar());
    obs.observe(canvas);
    p.redimensionar();

    let arrastre: { x: number; y: number; o: [number, number] } | null = null;
    const abajo = (e: PointerEvent) => { arrastre = { x: e.clientX, y: e.clientY, o: [...p.origen] }; canvas.setPointerCapture(e.pointerId); };
    const mover = (e: PointerEvent) => { if (arrastre) p.origen = [arrastre.o[0] + e.clientX - arrastre.x, arrastre.o[1] + e.clientY - arrastre.y]; };
    const arriba = () => { arrastre = null; };
    const rueda = (e: WheelEvent) => {
      e.preventDefault();
      const r = canvas.getBoundingClientRect();
      p.zoom(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * 0.0015));
    };
    canvas.addEventListener('pointerdown', abajo);
    canvas.addEventListener('pointermove', mover);
    canvas.addEventListener('pointerup', arriba);
    canvas.addEventListener('wheel', rueda, { passive: false });

    let cuadro = 0;
    const bucle = () => { p.dibujar(performance.now()); cuadro = requestAnimationFrame(bucle); };
    cuadro = requestAnimationFrame(bucle);
    return () => {
      cancelAnimationFrame(cuadro);
      obs.disconnect();
      canvas.removeEventListener('pointerdown', abajo);
      canvas.removeEventListener('pointermove', mover);
      canvas.removeEventListener('pointerup', arriba);
      canvas.removeEventListener('wheel', rueda);
    };
  }, []);

  useEffect(() => { if (encuadre) pantalla.current?.encuadrar(); }, [encuadre]);

  return <canvas ref={lienzo} className="radar" aria-label="Radar 2D con la nube de puntos" />;
}
