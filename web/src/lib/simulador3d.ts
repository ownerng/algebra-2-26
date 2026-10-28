// Simulador del escáner 3D: una mesa (ligeramente inclinada, como una mesa
// real) con un objeto encima y la caja con el sensor a una altura fija. Emite
// los mismos mensajes JSON que el firmware escaner_3d.

import { interpretarLinea, type Fuente, type OpcionesFuente } from './conexion.ts';

export type ObjetoSimulado = 'taza' | 'caja' | 'libros' | 'vacia';

export const OBJETOS: Record<ObjetoSimulado, string> = {
  taza: 'Taza (cilindro de 8 cm de diámetro y 9,5 cm de alto)',
  caja: 'Caja (7 × 5 cm, 4,5 cm de alto)',
  libros: 'Libros escalonados (2, 4 y 6 cm)',
  vacia: 'Mesa vacía (para ver el ajuste del plano)',
};

// Altura de la superficie (mm) en el punto (x, y) de la mesa, en mm
export function alturaObjeto(objeto: ObjetoSimulado, x: number, y: number): number {
  if (objeto === 'vacia') return 0;
  if (objeto === 'taza') {
    return Math.hypot(x - 90, y - 70) <= 40 ? 95 : 0;
  }
  if (objeto === 'caja') {
    return x >= 55 && x <= 125 && y >= 45 && y <= 95 ? 45 : 0;
  }
  // libros apilados en escalera
  if (y < 30 || y > 110 || x < 30) return 0;
  if (x < 80) return 20;
  if (x < 120) return 40;
  if (x < 160) return 60;
  return 0;
}

export const ALTURA_REAL_MM = 150;          // del lente a la mesa en el origen
export const INCLINACION = { a: 0.03, b: -0.015 };  // la mesa no está perfectamente nivelada
const MEDIO_CONO = (12.5 * Math.PI) / 180;   // el VL53L0X ve un cono de unos 25°

// Distancia que mediría el sensor sobre (x, y): promedio dentro del cono,
// pesando más las superficies cercanas, como pasa con la luz que regresa.
export function distanciaSimulada(objeto: ObjetoSimulado, x: number, y: number, conRuido = true): number {
  const radio = ALTURA_REAL_MM * Math.tan(MEDIO_CONO) * 0.6;
  let suma = 0, pesos = 0;
  for (let k = 0; k < 49; k++) {
    const ang = k * 2.39996, rho = radio * Math.sqrt((k + 0.5) / 49);
    const px = x + rho * Math.cos(ang), py = y + rho * Math.sin(ang);
    const mesa = INCLINACION.a * px + INCLINACION.b * py;
    const d = ALTURA_REAL_MM - mesa - alturaObjeto(objeto, px, py);
    const w = 1 / (d * d);
    suma += w * d;
    pesos += w;
  }
  const r = suma / pesos;
  return conRuido ? r + (Math.random() - 0.5) * 4 : r;
}

export class Simulador3D implements Fuente {
  private temporizador: ReturnType<typeof setInterval> | undefined;

  constructor(private opciones: OpcionesFuente & { posicionCaja: () => [number, number]; objeto: () => ObjetoSimulado }) {}

  private emitir(objeto: object) {
    const dato = interpretarLinea(JSON.stringify(objeto));
    if (dato) this.opciones.alDato(dato);
  }

  async conectar() {
    this.opciones.alEstado('conectado', 'Simulación');
    this.emitir({ estado: 'sensor listo' });
    this.temporizador = setInterval(() => {
      const [x, y] = this.opciones.posicionCaja();
      this.emitir({ r: Math.round(distanciaSimulada(this.opciones.objeto(), x, y)) });
    }, 120);
  }

  async enviar(comando: string) {
    if (comando !== 'M') return;
    const [x, y] = this.opciones.posicionCaja();
    const lecturas = Array.from({ length: 10 }, () => distanciaSimulada(this.opciones.objeto(), x, y));
    const media = lecturas.reduce((a, b) => a + b, 0) / lecturas.length;
    const desv = Math.sqrt(lecturas.reduce((a, b) => a + (b - media) ** 2, 0) / 9);
    setTimeout(() => this.emitir({ medida: +media.toFixed(1), desv: +desv.toFixed(1), n: 10, origen: 'web' }), 250);
  }

  async desconectar() {
    clearInterval(this.temporizador);
    this.opciones.alEstado('desconectado', 'Sin conexión');
  }
}
