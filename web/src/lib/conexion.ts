// Fuentes de datos: el ESP32 por Web Serial o el simulador.
// Ambas entregan objetos ya parseados del tipo {"r":523,"theta":84} o {"estado":"listo"}.

export interface Dato {
  r?: number;
  theta?: number;
  estado?: string;
  error?: string;
}

export type TipoEstado = 'conectado' | 'desconectado' | 'error';

export interface Fuente {
  conectar(): Promise<void>;
  enviar(comando: string): Promise<void>;
  desconectar(): Promise<void>;
}

export interface OpcionesFuente {
  alDato: (dato: Dato) => void;
  alEstado: (tipo: TipoEstado, texto: string) => void;
}

export function interpretarLinea(linea: string): Dato | null {
  const texto = linea.trim();
  if (!texto.startsWith('{')) return null;
  try {
    return JSON.parse(texto);
  } catch {
    return null;
  }
}

// Tipos mínimos de Web Serial (no vienen en lib.dom)
interface PuertoSerial {
  open(opciones: { baudRate: number }): Promise<void>;
  close(): Promise<void>;
  readable: ReadableStream<Uint8Array> | null;
  writable: WritableStream<Uint8Array> | null;
}
const serial = () => (navigator as unknown as { serial?: { requestPort(): Promise<PuertoSerial> } }).serial;

export class ConexionSerial implements Fuente {
  private puerto: PuertoSerial | null = null;
  private lector: ReadableStreamDefaultReader<string> | null = null;
  private cierre: Promise<void> | null = null;
  private activa = false;

  constructor(private opciones: OpcionesFuente) {}

  static disponible() {
    return !!serial();
  }

  async conectar() {
    this.puerto = await serial()!.requestPort();
    await this.puerto.open({ baudRate: 115200 });
    this.activa = true;
    this.opciones.alEstado('conectado', 'ESP32 conectado');
    void this.leer();
  }

  private async leer() {
    const decodificador = new TextDecoderStream();
    this.cierre = this.puerto!.readable!.pipeTo(decodificador.writable as WritableStream<Uint8Array>).catch(() => {});
    this.lector = decodificador.readable.getReader();
    let resto = '';
    try {
      while (this.activa) {
        const { value, done } = await this.lector.read();
        if (done) break;
        resto += value;
        const lineas = resto.split('\n');
        resto = lineas.pop() ?? '';
        for (const linea of lineas) {
          const dato = interpretarLinea(linea);
          if (dato) this.opciones.alDato(dato);
        }
      }
    } catch (error) {
      this.opciones.alEstado('error', 'Se perdió la conexión: ' + (error as Error).message);
    } finally {
      this.lector.releaseLock();
    }
  }

  async enviar(comando: string) {
    if (!this.puerto?.writable) return;
    const escritor = this.puerto.writable.getWriter();
    await escritor.write(new TextEncoder().encode(comando));
    escritor.releaseLock();
  }

  async desconectar() {
    this.activa = false;
    try {
      await this.lector?.cancel();
      await this.cierre;
      await this.puerto?.close();
    } catch {
      // el puerto ya estaba cerrado
    }
    this.opciones.alEstado('desconectado', 'Sin conexión');
  }
}

// Conexión a través del puente USB (web/puente.ts). Funciona en cualquier
// navegador, incluido Firefox.
export const URL_PUENTE = 'ws://localhost:3001';

export class ConexionPuente implements Fuente {
  private ws: WebSocket | null = null;
  private cerrando = false;

  constructor(private opciones: OpcionesFuente) {}

  conectar() {
    return new Promise<void>((resolver, rechazar) => {
      const ws = new WebSocket(URL_PUENTE);
      this.ws = ws;
      const espera = setTimeout(() => { this.cerrando = true; ws.close(); rechazar(new Error('puente')); }, 2000);
      ws.onopen = () => { clearTimeout(espera); resolver(); };
      ws.onerror = () => { clearTimeout(espera); this.cerrando = true; rechazar(new Error('puente')); };
      ws.onmessage = (ev) => {
        const dato = interpretarLinea(String(ev.data)) as (Dato & { puente?: string }) | null;
        if (!dato) return;
        if (dato.puente) {
          const listo = dato.puente.startsWith('conectado');
          this.opciones.alEstado(listo ? 'conectado' : 'error', listo ? 'ESP32 conectado' : `Puente: ${dato.puente}`);
          return;
        }
        this.opciones.alDato(dato);
      };
      ws.onclose = () => {
        if (!this.cerrando) this.opciones.alEstado('desconectado', 'Se cerró el puente USB');
      };
    });
  }

  async enviar(comando: string) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(comando);
  }

  async desconectar() {
    this.cerrando = true;
    this.ws?.close();
    this.opciones.alEstado('desconectado', 'Sin conexión');
  }
}
