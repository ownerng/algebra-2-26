// Simulador del prototipo: recorre 0–180° como el firmware y mide distancias
// lanzando rayos contra una habitación virtual. Sirve para ensayar la
// sustentación sin el sensor soldado. Emite exactamente las mismas líneas JSON.

import { interpretarLinea, type Fuente, type OpcionesFuente } from './conexion.ts';
import { aRad, type Pose } from './matematica.ts';

const ALCANCE_MM = 1200;
const PASO = 2;
const INTERVALO_MS = 45;

type Punto = [number, number];
export type Segmento = [Punto, Punto];

// Habitación en coordenadas globales (mm). La posición 1 está en el origen mirando hacia +y.
function caja(x: number, y: number, ancho: number, alto: number): Segmento[] {
  const a: Punto = [x, y], b: Punto = [x + ancho, y], c: Punto = [x + ancho, y + alto], d: Punto = [x, y + alto];
  return [[a, b], [b, c], [c, d], [d, a]];
}
function patas(x0: number, y0: number, x1: number, y1: number, lado = 40): Segmento[] {
  return ([[x0, y0], [x1, y0], [x1, y1], [x0, y1]] as Punto[]).flatMap(([x, y]) => caja(x - lado / 2, y - lado / 2, lado, lado));
}

export const SEGMENTOS: Segmento[] = [
  [[-900, -200], [-900, 1250]],          // pared izquierda
  [[-900, 1250], [950, 1250]],           // pared del fondo
  [[950, 1250], [950, 700]],             // pared derecha (con puerta)
  [[950, 350], [950, -200]],
  ...patas(150, 450, 650, 800),          // mesa
  ...patas(-520, 520, -220, 820, 30),    // silla
  ...caja(-560, 820, 380, 40),           // respaldo de la silla
  ...caja(600, 60, 220, 160),            // caja en el piso
];
export const CIRCULOS = [{ c: [-560, 180] as Punto, r: 110 }]; // papelera

// Pose real del aparato en cada posición (la que el estudiante debe medir con cinta).
export const POSES_REALES: Record<1 | 2, Pose> = {
  1: { dx: 0, dy: 0, alpha: 0 },
  2: { dx: 250, dy: 1100, alpha: 180 },
};
export const DESFASE_LENTE_MM = 15;

function rayoSegmento(o: Punto, u: Punto, [a, b]: Segmento) {
  const ex = b[0] - a[0], ey = b[1] - a[1];
  const den = u[0] * ey - u[1] * ex;
  if (Math.abs(den) < 1e-9) return Infinity;
  const ax = a[0] - o[0], ay = a[1] - o[1];
  const t = (ax * ey - ay * ex) / den;
  const s = (ax * u[1] - ay * u[0]) / den;
  return t > 0 && s >= 0 && s <= 1 ? t : Infinity;
}

function rayoCirculo(o: Punto, u: Punto, { c, r }: { c: Punto; r: number }) {
  const fx = o[0] - c[0], fy = o[1] - c[1];
  const b = fx * u[0] + fy * u[1];
  const disc = b * b - (fx * fx + fy * fy - r * r);
  if (disc < 0) return Infinity;
  const t = -b - Math.sqrt(disc);
  return t > 0 ? t : Infinity;
}

// conRuido = false devuelve la distancia exacta (para los experimentos del informe)
export function medirSimulado(pose: Pose, thetaServo: number, conRuido = true): number {
  const ang = aRad(pose.alpha + thetaServo);
  const u: Punto = [Math.cos(ang), Math.sin(ang)];
  const o: Punto = [pose.dx, pose.dy];
  let t = Infinity;
  for (const s of SEGMENTOS) t = Math.min(t, rayoSegmento(o, u, s));
  for (const c of CIRCULOS) t = Math.min(t, rayoCirculo(o, u, c));
  const r = t - DESFASE_LENTE_MM;
  if (!isFinite(r) || r > ALCANCE_MM) return -1;
  if (!conRuido) return r;
  if (Math.random() < 0.03) return -1;
  const ruido = (Math.random() - 0.5) * (6 + r * 0.02);
  return Math.max(0, Math.round(r + ruido));
}

export class Simulador implements Fuente {
  private angulo = 0;
  private direccion = 1;
  private escaneando = true;
  private temporizador: ReturnType<typeof setInterval> | undefined;

  constructor(private opciones: OpcionesFuente & { posicionActiva: () => 1 | 2 }) {}

  private emitir(objeto: object) {
    const dato = interpretarLinea(JSON.stringify(objeto));
    if (dato) this.opciones.alDato(dato);
  }

  async conectar() {
    this.opciones.alEstado('conectado', 'Simulación');
    this.emitir({ estado: 'listo' });
    this.temporizador = setInterval(() => this.paso(), INTERVALO_MS);
  }

  private paso() {
    if (!this.escaneando) return;
    const pose = POSES_REALES[this.opciones.posicionActiva()];
    this.emitir({ r: medirSimulado(pose, this.angulo), theta: this.angulo });
    this.angulo += this.direccion * PASO;
    if (this.angulo >= 180) { this.angulo = 180; this.direccion = -1; }
    if (this.angulo <= 0) { this.angulo = 0; this.direccion = 1; }
  }

  async enviar(comando: string) {
    if (comando === 'S') { this.escaneando = true; this.emitir({ estado: 'escaneando' }); }
    if (comando === 'P') { this.escaneando = false; this.emitir({ estado: 'pausado' }); }
    if (comando === 'C') { this.escaneando = false; this.angulo = 90; this.emitir({ r: -1, theta: 90 }); this.emitir({ estado: 'pausado' }); }
  }

  async desconectar() {
    clearInterval(this.temporizador);
    this.opciones.alEstado('desconectado', 'Sin conexión');
  }
}
