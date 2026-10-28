// Estado del escáner fuera de React: las lecturas llegan ~22 veces por segundo
// y los lienzos las dibujan en cada cuadro. React solo se suscribe a lo que
// muestra (con un ritmo limitado para el panel de matrices).

import { useEffect, useReducer } from 'react';
import { ConexionSerial, type Dato, type Fuente, type TipoEstado } from './conexion.ts';
import { procesarLectura, type Pose } from './matematica.ts';
import { Simulador } from './simulador.ts';

export type Posicion = 1 | 2;
export interface Lectura { r: number; thetaServo: number; posicion: Posicion; t: number }
export interface PuntoMapa {
  id: Posicion; thetaServo: number; theta: number; r: number; t: number;
  xl: number; yl: number; x: number; y: number;
}

export const COLORES: Record<Posicion, string> = { 1: '#ffb238', 2: '#8fd8ff' };

class Escaner extends EventTarget {
  fuente: Fuente | null = null;
  tipoFuente: 'ninguna' | 'serial' | 'simulacion' = 'ninguna';
  estadoConexion: TipoEstado = 'desconectado';
  mensaje = 'Sin conexión';
  servo = 'listo';

  posicionActiva: Posicion = 1;
  grabando = true;
  d = 15;
  invertir = false;
  verHabitacion = true;
  poses: Record<Posicion, Pose> = { 1: { dx: 0, dy: 0, alpha: 0 }, 2: { dx: 250, dy: 1100, alpha: 180 } };
  escaneos: Record<Posicion, Map<number, { r: number; t: number }>> = { 1: new Map(), 2: new Map() };
  ultima: Lectura | null = null;
  ultimaValida: Lectura | null = null;
  direccion = 1;
  version = 0;
  private marcas: number[] = [];

  private avisar() {
    this.dispatchEvent(new Event('cambio'));
  }

  // ---------- Datos entrantes ----------
  private alDato = (dato: Dato) => {
    if (dato.error) return this.alEstado('error', dato.error);
    if (dato.estado) {
      this.servo = dato.estado;
      return this.avisar();
    }
    if (typeof dato.theta !== 'number' || typeof dato.r !== 'number') return;
    const ahora = performance.now();
    const anterior = this.ultima;
    if (anterior && dato.theta !== anterior.thetaServo) this.direccion = dato.theta > anterior.thetaServo ? 1 : -1;
    this.ultima = { r: dato.r, thetaServo: dato.theta, posicion: this.posicionActiva, t: ahora };
    if (dato.r > 0) this.ultimaValida = this.ultima;
    this.marcas.push(ahora);
    if (this.grabando) {
      const mapa = this.escaneos[this.posicionActiva];
      if (dato.r > 0) mapa.set(dato.theta, { r: dato.r, t: ahora });
      else mapa.delete(dato.theta);
      this.version++;
    }
    this.avisar();
  };

  private alEstado = (tipo: TipoEstado, texto: string) => {
    this.estadoConexion = tipo;
    this.mensaje = texto;
    if (tipo === 'desconectado') { this.fuente = null; this.tipoFuente = 'ninguna'; }
    this.avisar();
  };

  lecturasPorSegundo() {
    const ahora = performance.now();
    this.marcas = this.marcas.filter((t) => ahora - t < 1000);
    return this.marcas.length;
  }

  // ---------- Conexión ----------
  private conectando = false;

  async conectar(tipo: 'serial' | 'simulacion') {
    // Un doble clic no debe arrancar dos fuentes a la vez
    if (this.conectando) return;
    this.conectando = true;
    try {
      await this.abrir(tipo);
    } finally {
      this.conectando = false;
    }
  }

  private async abrir(tipo: 'serial' | 'simulacion') {
    await this.fuente?.desconectar();
    const opciones = { alDato: this.alDato, alEstado: this.alEstado, posicionActiva: () => this.posicionActiva };
    const fuente = tipo === 'serial' ? new ConexionSerial(opciones) : new Simulador(opciones);
    this.fuente = fuente;
    this.tipoFuente = tipo;
    try {
      await fuente.conectar();
    } catch (error) {
      this.fuente = null;
      this.tipoFuente = 'ninguna';
      if ((error as Error).name === 'NotFoundError') this.alEstado('desconectado', 'No elegiste ningún puerto');
      else this.alEstado('error', 'No se pudo abrir el puerto. ¿Está abierto el monitor serial del Arduino IDE?');
    }
  }

  async desconectar() {
    await this.fuente?.desconectar();
  }

  enviar(comando: 'S' | 'P' | 'C') {
    void this.fuente?.enviar(comando);
  }

  // ---------- Controles ----------
  cambiar(cambios: Partial<Pick<Escaner, 'posicionActiva' | 'grabando' | 'd' | 'invertir' | 'verHabitacion'>>) {
    Object.assign(this, cambios);
    this.version++;
    this.avisar();
  }

  cambiarPose2(pose: Partial<Pose>) {
    this.poses = { ...this.poses, 2: { ...this.poses[2], ...pose } };
    this.version++;
    this.avisar();
  }

  borrar(pos: Posicion) {
    this.escaneos[pos].clear();
    this.version++;
    this.avisar();
  }

  cargarSesion(datos: SesionGuardada) {
    this.d = datos.d;
    this.invertir = datos.invertir;
    this.poses = datos.poses;
    for (const id of [1, 2] as Posicion[]) {
      this.escaneos[id] = new Map(datos.escaneos[id].map(({ theta, r }) => [theta, { r, t: 0 }]));
    }
    this.version++;
    this.avisar();
  }

  sesion(): SesionGuardada {
    const lista = (id: Posicion) => [...this.escaneos[id]].map(([theta, { r }]) => ({ theta, r }));
    return { formato: 'escaner-lidar-v1', d: this.d, invertir: this.invertir, poses: this.poses, escaneos: { 1: lista(1), 2: lista(2) } };
  }

  // ---------- Geometría derivada (se recalcula solo si cambió la versión) ----------
  private cache: { version: number; puntos: Record<Posicion, PuntoMapa[]> } = { version: -1, puntos: { 1: [], 2: [] } };
  puntos(): Record<Posicion, PuntoMapa[]> {
    if (this.cache.version === this.version) return this.cache.puntos;
    const calcular = (id: Posicion) =>
      [...this.escaneos[id].entries()]
        .map(([thetaServo, { r, t }]) => {
          const c = procesarLectura({ r, thetaServo, d: this.d, invertir: this.invertir, pose: this.poses[id] });
          return { id, thetaServo, theta: c.theta, r, t, xl: c.local[0]!, yl: c.local[1]!, x: c.global[0]!, y: c.global[1]! };
        })
        .sort((a, b) => a.theta - b.theta);
    this.cache = { version: this.version, puntos: { 1: calcular(1), 2: calcular(2) } };
    return this.cache.puntos;
  }
}

export interface SesionGuardada {
  formato: 'escaner-lidar-v1';
  d: number;
  invertir: boolean;
  poses: Record<Posicion, Pose>;
  escaneos: Record<Posicion, { theta: number; r: number }[]>;
}

export const escaner = new Escaner();

// Vuelve a dibujar el componente cuando cambia el escáner, como máximo cada `ms`.
export function useEscaner(ms = 0) {
  const [, refrescar] = useReducer((n: number) => n + 1, 0);
  useEffect(() => {
    let pendiente: ReturnType<typeof setTimeout> | null = null;
    let ultimo = 0;
    const alCambio = () => {
      const ahora = performance.now();
      if (ahora - ultimo >= ms) { ultimo = ahora; refrescar(); return; }
      if (!pendiente) pendiente = setTimeout(() => { pendiente = null; ultimo = performance.now(); refrescar(); }, ms - (ahora - ultimo));
    };
    escaner.addEventListener('cambio', alCambio);
    return () => {
      escaner.removeEventListener('cambio', alCambio);
      if (pendiente) clearTimeout(pendiente);
    };
  }, [ms]);
  return escaner;
}
