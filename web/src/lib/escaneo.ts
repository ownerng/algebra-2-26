// Estado del escáner 3D fuera de React. La persona recorre la cuadrícula en
// zigzag; cada medición se guarda en su casilla y se convierte en un punto 3D.

import { useEffect, useReducer } from 'react';
import { ConexionPuente, ConexionSerial, type Dato, type Fuente, type TipoEstado } from './conexion.ts';
import { ajustarPlano, puntoDeCasilla, type Plano, type Vec3 } from './algebra3d.ts';
import { ALTURA_REAL_MM, Simulador3D, type ObjetoSimulado } from './simulador3d.ts';

export interface Medida { r: number; desv: number; n: number; t: number }
export interface Punto3D { i: number; j: number; r: number; x: number; y: number; z: number; h: number; mesa: boolean }
export interface Config { nx: number; ny: number; paso: number; altura: number }

// Escaneo continuo: una marca recorre cada fila a velocidad constante y la
// persona desliza la caja siguiéndola. La posición de cada lectura es
//   x = v·(t − t₀)   (o el largo de la fila menos eso, en las filas de regreso)
//   y = Δ·j
export type FaseBarrido = 'cuenta' | 'fila' | 'fin';
export interface Barrido { fila: number; fase: FaseBarrido; tFase: number }
export const CUENTA_MS = 3000;

export interface EscaneoGuardado {
  formato: 'escaner-3d-v1';
  nombre: string;
  fecha: string;
  config: Config;
  medidas: { i: number; j: number; r: number; desv: number; n: number }[];
}

const CLAVE_GUARDADOS = 'escaner-3d-guardados';
const clave = (i: number, j: number) => `${i},${j}`;

class Escaneo extends EventTarget {
  fuente: Fuente | null = null;
  tipoFuente: 'ninguna' | 'serial' | 'simulacion' = 'ninguna';
  estadoConexion: TipoEstado = 'desconectado';
  mensaje = 'Sin conexión';
  sensorListo = false;

  config: Config = { nx: 10, ny: 8, paso: 20, altura: ALTURA_REAL_MM };
  alturaMedida = false;            // true cuando la altura H se midió sobre la mesa vacía
  objeto: ObjetoSimulado = 'taza';
  corregirMesa = true;

  medidas = new Map<string, Medida>();
  historial: string[] = [];        // orden en que se midieron las casillas (para deshacer)
  cursor = 0;                      // índice en el recorrido en zigzag
  vivo: { r: number; t: number } | null = null;
  ultima: { i: number; j: number } | null = null;
  midiendo = false;
  calibrando = false;
  automatico = false;
  velocidad = 20;                  // mm/s de la marca en el escaneo continuo
  barrido: Barrido | null = null;
  private acumulado = new Map<string, { suma: number; suma2: number; n: number }>();
  private reloj: ReturnType<typeof setInterval> | undefined;
  version = 0;
  // Ángulos de la vista 3D (los mueve el lienzo sin avisar en cada cuadro)
  vista = { theta: -35, phi: 58, zoom: 1, girar: true };
  // Casilla congelada en el cuaderno, marcada en la cuadrícula y en 3D
  resaltada: { i: number; j: number } | null = null;

  resaltar(c: { i: number; j: number } | null) {
    this.resaltada = c;
    this.avisar();
  }

  private avisar() {
    this.dispatchEvent(new Event('cambio'));
  }

  // ---------- Recorrido ----------
  get total() { return this.config.nx * this.config.ny; }

  // Zigzag: las filas pares de izquierda a derecha y las impares al revés,
  // para que la caja siempre se mueva a la casilla vecina.
  casilla(k: number): [number, number] {
    const j = Math.floor(k / this.config.nx);
    const p = k % this.config.nx;
    return [j % 2 === 0 ? p : this.config.nx - 1 - p, j];
  }

  indice(i: number, j: number) {
    return j * this.config.nx + (j % 2 === 0 ? i : this.config.nx - 1 - i);
  }

  get casillaActual(): [number, number] { return this.casilla(Math.min(this.cursor, this.total - 1)); }

  posicionCaja(): [number, number] {
    if (this.barrido) return this.posicionMarca();
    const [i, j] = this.casillaActual;
    return [i * this.config.paso, j * this.config.paso];
  }

  get largoFila() { return (this.config.nx - 1) * this.config.paso; }

  // Recorrido de la marca en la fila actual (0 durante la cuenta regresiva)
  recorrido(): number {
    const b = this.barrido;
    if (!b || b.fase !== 'fila') return 0;
    return Math.min(this.largoFila, (this.velocidad * (performance.now() - b.tFase)) / 1000);
  }

  posicionMarca(): [number, number] {
    const b = this.barrido!;
    const s = this.recorrido();
    const x = b.fila % 2 === 0 ? s : this.largoFila - s;
    return [x, b.fila * this.config.paso];
  }

  // ---------- Escaneo continuo ----------
  escanearContinuo() {
    if (this.barrido) return this.detenerContinuo();
    if (!this.fuente) return;
    this.reiniciar();
    this.barrido = { fila: 0, fase: 'cuenta', tFase: performance.now() };
    clearInterval(this.reloj);
    this.reloj = setInterval(() => this.avanzarBarrido(), 50);
    this.avisar();
  }

  detenerContinuo() {
    clearInterval(this.reloj);
    this.barrido = null;
    this.avisar();
  }

  private avanzarBarrido() {
    const b = this.barrido;
    if (!b) return;
    const ahora = performance.now();
    if (b.fase === 'cuenta' && ahora - b.tFase >= CUENTA_MS) {
      this.barrido = { ...b, fase: 'fila', tFase: ahora };
    } else if (b.fase === 'fila' && this.velocidad * (ahora - b.tFase) / 1000 >= this.largoFila + this.config.paso * 0.4) {
      if (b.fila >= this.config.ny - 1) {
        clearInterval(this.reloj);
        this.barrido = { ...b, fase: 'fin', tFase: ahora };
        this.cursor = this.total;
      } else {
        this.barrido = { fila: b.fila + 1, fase: 'cuenta', tFase: ahora };
      }
    }
    this.avisar();
  }

  // Cada lectura en vivo durante una fila se suma a la casilla más cercana a la marca
  private acumular(r: number) {
    const [x, y] = this.posicionMarca();
    const i = Math.round(x / this.config.paso), j = Math.round(y / this.config.paso);
    if (i < 0 || i >= this.config.nx) return;
    const k = clave(i, j);
    const a = this.acumulado.get(k) ?? { suma: 0, suma2: 0, n: 0 };
    a.suma += r; a.suma2 += r * r; a.n++;
    this.acumulado.set(k, a);
    const media = a.suma / a.n;
    const desv = a.n > 1 ? Math.sqrt(Math.max(0, (a.suma2 - a.n * media * media) / (a.n - 1))) : 0;
    if (!this.medidas.has(k)) this.historial.push(k);
    this.medidas.set(k, { r: media, desv, n: a.n, t: performance.now() });
    this.ultima = { i, j };
    this.version++;
  }

  private siguienteLibre(desde: number) {
    for (let k = desde; k < this.total; k++) if (!this.medidas.has(clave(...this.casilla(k)))) return k;
    return this.total;
  }

  // Tocar una casilla la vuelve la siguiente a medir y, si ya tiene medición,
  // muestra su cálculo en el cuaderno.
  irA(i: number, j: number) {
    this.cursor = this.indice(i, j);
    if (this.medidas.has(clave(i, j))) this.ultima = { i, j };
    this.avisar();
  }

  // ---------- Datos entrantes ----------
  private alDato = (dato: Dato) => {
    if (dato.error) {
      this.midiendo = false;
      if (dato.error.includes('no encontrado')) this.sensorListo = false;
      return this.alEstado('error', dato.error);
    }
    if (dato.estado) {
      if (dato.estado === 'sensor listo') this.sensorListo = true;
      if (this.estadoConexion === 'error') this.alEstado('conectado', this.tipoFuente === 'simulacion' ? 'Simulación' : 'ESP32 conectado');
      return this.avisar();
    }
    if (typeof dato.medida === 'number') return this.alMedir(dato);
    if (typeof dato.r === 'number') {
      this.vivo = { r: dato.r, t: performance.now() };
      this.sensorListo = true;
      if (this.barrido?.fase === 'fila' && dato.r > 0) this.acumular(dato.r);
      this.avisar();
    }
  };

  private alMedir(dato: Dato) {
    this.midiendo = false;
    const r = dato.medida!;
    if (r <= 0) {
      this.alEstado('error', 'El sensor no recibió eco: acerca la caja a la mesa o revisa que no haya nada tapando el lente');
      return;
    }
    if (this.calibrando) {
      this.calibrando = false;
      this.config = { ...this.config, altura: Math.round(r) };
      this.alturaMedida = true;
      this.version++;
      return this.avisar();
    }
    if (this.cursor >= this.total) return this.avisar();
    const [i, j] = this.casillaActual;
    this.medidas.set(clave(i, j), { r, desv: dato.desv ?? 0, n: dato.n ?? 1, t: performance.now() });
    this.historial.push(clave(i, j));
    this.ultima = { i, j };
    this.cursor = this.siguienteLibre(this.cursor + 1);
    this.version++;
    this.guardarAutomatico();
    this.avisar();
    if (this.automatico) setTimeout(() => this.continuarAutomatico(), 60);
  }

  private alEstado = (tipo: TipoEstado, texto: string) => {
    this.estadoConexion = tipo;
    this.mensaje = texto;
    if (tipo === 'desconectado') { this.fuente = null; this.tipoFuente = 'ninguna'; this.automatico = false; }
    this.avisar();
  };

  // ---------- Conexión ----------
  private conectando = false;

  async conectar(tipo: 'serial' | 'simulacion') {
    if (this.conectando) return;
    this.conectando = true;
    try {
      await this.fuente?.desconectar();
      const opciones = {
        alDato: this.alDato, alEstado: this.alEstado,
        posicionCaja: () => this.posicionCaja(), objeto: () => this.objeto,
      };
      if (tipo === 'simulacion') {
        this.fuente = new Simulador3D(opciones);
        this.tipoFuente = 'simulacion';
        await this.fuente.conectar();
        return;
      }
      this.tipoFuente = 'serial';
      try {
        const puente = new ConexionPuente(opciones);
        this.fuente = puente;
        await puente.conectar();
        return;
      } catch {
        this.fuente = null;
      }
      if (!ConexionSerial.disponible()) {
        this.tipoFuente = 'ninguna';
        this.alEstado('error', 'No encontré el puente USB. Ábrelo con ./iniciar-web.sh');
        return;
      }
      const serial = new ConexionSerial(opciones);
      this.fuente = serial;
      try {
        await serial.conectar();
      } catch (error) {
        this.fuente = null;
        this.tipoFuente = 'ninguna';
        if ((error as Error).name === 'NotFoundError') this.alEstado('desconectado', 'No elegiste ningún puerto');
        else this.alEstado('error', 'No se pudo abrir el puerto. ¿Está abierto el monitor serial del Arduino IDE?');
      }
    } finally {
      this.conectando = false;
    }
  }

  async desconectar() {
    this.automatico = false;
    this.detenerContinuo();
    await this.fuente?.desconectar();
  }

  // ---------- Acciones ----------
  medir() {
    if (!this.fuente || this.midiendo || this.cursor >= this.total) return;
    this.midiendo = true;
    this.avisar();
    void this.fuente.enviar('M');
    // Si no llega respuesta (sensor desconectado), se libera el botón
    setTimeout(() => { if (this.midiendo) { this.midiendo = false; this.avisar(); } }, 4000);
  }

  medirMesa() {
    if (!this.fuente || this.midiendo) return;
    this.calibrando = true;
    this.medir();
  }

  deshacer() {
    const ultima = this.historial.pop();
    if (!ultima) return;
    this.medidas.delete(ultima);
    const [i, j] = ultima.split(',').map(Number) as [number, number];
    this.cursor = this.indice(i, j);
    this.ultima = null;
    if (this.historial.length) {
      const [a, b] = this.historial.at(-1)!.split(',').map(Number);
      this.ultima = { i: a!, j: b! };
    }
    this.version++;
    this.avisar();
  }

  reiniciar() {
    this.automatico = false;
    this.acumulado.clear();
    this.medidas.clear();
    this.historial = [];
    this.cursor = 0;
    this.ultima = null;
    this.version++;
    this.avisar();
  }

  cambiarConfig(cambios: Partial<Config>) {
    const antes = this.config;
    this.config = { ...this.config, ...cambios };
    if (cambios.altura !== undefined && cambios.altura !== antes.altura) this.alturaMedida = false;
    if (cambios.nx !== undefined || cambios.ny !== undefined) this.reiniciar();
    this.version++;
    this.avisar();
  }

  cambiar(cambios: Partial<Pick<Escaneo, 'objeto' | 'corregirMesa' | 'velocidad'>>) {
    Object.assign(this, cambios);
    this.version++;
    this.avisar();
  }

  // Solo en simulación: la caja "se mueve sola" de casilla en casilla
  escanearAutomatico() {
    if (this.tipoFuente !== 'simulacion') return;
    this.automatico = !this.automatico;
    this.avisar();
    if (this.automatico) this.continuarAutomatico();
  }

  private continuarAutomatico() {
    if (!this.automatico) return;
    if (this.cursor >= this.total) { this.automatico = false; return this.avisar(); }
    this.medir();
  }

  // ---------- Geometría derivada ----------
  private cache: { version: number; puntos: Punto3D[]; plano: Plano | null } = { version: -1, puntos: [], plano: null };

  geometria() {
    if (this.cache.version === this.version) return this.cache;
    const { paso, altura, nx, ny } = this.config;
    const base = [...this.medidas.entries()].map(([k, m]) => {
      const [i, j] = k.split(',').map(Number) as [number, number];
      const [x, y, z] = puntoDeCasilla(i, j, m.r, paso, altura);
      return { i, j, r: m.r, x, y, z, h: z, mesa: false };
    });
    // La mesa se estima con las casillas del borde; se descartan las que
    // quedan lejos del primer ajuste (pueden ser parte del objeto).
    let plano: Plano | null = null;
    if (this.corregirMesa) {
      let borde = base.filter((p) => p.i === 0 || p.j === 0 || p.i === nx - 1 || p.j === ny - 1);
      if (borde.length >= 6) {
        plano = ajustarPlano(borde.map((p) => [p.x, p.y, p.z] as Vec3));
        if (plano) {
          const pl = plano;
          const cerca = borde.filter((p) => Math.abs(p.z - (pl.a * p.x + pl.b * p.y + pl.c)) < 6);
          if (cerca.length >= 6) { borde = cerca; plano = ajustarPlano(borde.map((p) => [p.x, p.y, p.z] as Vec3)) ?? plano; }
          const usados = new Set(borde.map((p) => clave(p.i, p.j)));
          for (const p of base) {
            p.h = p.z - (plano!.a * p.x + plano!.b * p.y + plano!.c);
            p.mesa = usados.has(clave(p.i, p.j));
          }
        }
      }
    }
    this.cache = { version: this.version, puntos: base, plano };
    return this.cache;
  }

  // ---------- Guardar y exportar ----------
  datos(nombre = 'Escaneo'): EscaneoGuardado {
    return {
      formato: 'escaner-3d-v1', nombre, fecha: new Date().toISOString(), config: this.config,
      medidas: [...this.medidas.entries()].map(([k, m]) => {
        const [i, j] = k.split(',').map(Number) as [number, number];
        return { i, j, r: m.r, desv: m.desv, n: m.n };
      }),
    };
  }

  cargar(d: EscaneoGuardado) {
    this.config = d.config;
    this.medidas = new Map(d.medidas.map((m) => [clave(m.i, m.j), { r: m.r, desv: m.desv, n: m.n, t: 0 }]));
    this.historial = d.medidas.map((m) => clave(m.i, m.j));
    this.cursor = this.siguienteLibre(0);
    this.ultima = d.medidas.length ? { i: d.medidas.at(-1)!.i, j: d.medidas.at(-1)!.j } : null;
    this.alturaMedida = true;
    this.version++;
    this.avisar();
  }

  guardados(): EscaneoGuardado[] {
    try { return JSON.parse(localStorage.getItem(CLAVE_GUARDADOS) ?? '[]'); } catch { return []; }
  }

  guardarComo(nombre: string) {
    const lista = this.guardados().filter((g) => g.nombre !== nombre);
    lista.unshift(this.datos(nombre));
    try { localStorage.setItem(CLAVE_GUARDADOS, JSON.stringify(lista.slice(0, 20))); } catch { /* almacenamiento lleno o bloqueado */ }
    this.avisar();
  }

  borrarGuardado(nombre: string) {
    try { localStorage.setItem(CLAVE_GUARDADOS, JSON.stringify(this.guardados().filter((g) => g.nombre !== nombre))); } catch { /* sin almacenamiento */ }
    this.avisar();
  }

  private guardarAutomatico() {
    try { localStorage.setItem('escaner-3d-actual', JSON.stringify(this.datos('En curso'))); } catch { /* sin almacenamiento */ }
  }

  recuperarEnCurso(): EscaneoGuardado | null {
    try { const d = JSON.parse(localStorage.getItem('escaner-3d-actual') ?? 'null'); return d?.medidas?.length ? d : null; } catch { return null; }
  }

  // Malla en formato OBJ (se abre en cualquier visor 3D)
  obj(): string {
    const { puntos } = this.geometria();
    const indice = new Map<string, number>();
    const lineas = ['# Escáner 3D por cuadrícula — VL53L0X + ESP32', `# ${puntos.length} puntos, unidades en mm`];
    puntos.forEach((p, k) => { indice.set(clave(p.i, p.j), k + 1); lineas.push(`v ${p.x.toFixed(1)} ${p.y.toFixed(1)} ${p.h.toFixed(1)}`); });
    for (let j = 0; j < this.config.ny - 1; j++) {
      for (let i = 0; i < this.config.nx - 1; i++) {
        const a = indice.get(clave(i, j)), b = indice.get(clave(i + 1, j)), c = indice.get(clave(i + 1, j + 1)), d = indice.get(clave(i, j + 1));
        if (a && b && c) lineas.push(`f ${a} ${b} ${c}`);
        if (a && c && d) lineas.push(`f ${a} ${c} ${d}`);
      }
    }
    return lineas.join('\n') + '\n';
  }

  csv(): string {
    const { puntos } = this.geometria();
    return ['i,j,r_mm,x_mm,y_mm,z_mm,altura_objeto_mm', ...puntos.map((p) => [p.i, p.j, p.r, p.x, p.y, p.z, p.h].map((v) => +v.toFixed(2)).join(','))].join('\n') + '\n';
  }
}

export const escaneo = new Escaneo();

export function useEscaneo(ms = 0) {
  const [, refrescar] = useReducer((n: number) => n + 1, 0);
  useEffect(() => {
    let pendiente: ReturnType<typeof setTimeout> | null = null;
    let ultimo = 0;
    const alCambio = () => {
      const ahora = performance.now();
      if (ahora - ultimo >= ms) { ultimo = ahora; refrescar(); return; }
      if (!pendiente) pendiente = setTimeout(() => { pendiente = null; ultimo = performance.now(); refrescar(); }, ms - (ahora - ultimo));
    };
    escaneo.addEventListener('cambio', alCambio);
    return () => {
      escaneo.removeEventListener('cambio', alCambio);
      if (pendiente) clearTimeout(pendiente);
    };
  }, [ms]);
  return escaneo;
}
