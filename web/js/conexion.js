// Fuentes de datos: el ESP32 por Web Serial o el simulador.
// Ambas entregan objetos ya parseados del tipo {"r":523,"theta":84} o {"estado":"listo"}.

export function interpretarLinea(linea) {
  const texto = linea.trim();
  if (!texto.startsWith('{')) return null;
  try {
    return JSON.parse(texto);
  } catch {
    return null;
  }
}

export class ConexionSerial {
  constructor({ alDato, alEstado }) {
    this.alDato = alDato;
    this.alEstado = alEstado;
    this.puerto = null;
    this.lector = null;
    this.activa = false;
  }

  static disponible() {
    return 'serial' in navigator;
  }

  async conectar() {
    this.puerto = await navigator.serial.requestPort();
    await this.puerto.open({ baudRate: 115200 });
    this.activa = true;
    this.alEstado('conectado', 'ESP32 conectado');
    this.leer();
  }

  async leer() {
    const decodificador = new TextDecoderStream();
    this.cierre = this.puerto.readable.pipeTo(decodificador.writable).catch(() => {});
    this.lector = decodificador.readable.getReader();
    let resto = '';
    try {
      while (this.activa) {
        const { value, done } = await this.lector.read();
        if (done) break;
        resto += value;
        const lineas = resto.split('\n');
        resto = lineas.pop();
        for (const linea of lineas) {
          const dato = interpretarLinea(linea);
          if (dato) this.alDato(dato);
        }
      }
    } catch (error) {
      this.alEstado('error', 'Se perdió la conexión: ' + error.message);
    } finally {
      this.lector.releaseLock();
    }
  }

  async enviar(comando) {
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
    this.alEstado('desconectado', 'Sin conexión');
  }
}
