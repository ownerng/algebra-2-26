// Puente USB → WebSocket (y la cámara del celular por WiFi).
// Lee el puerto serie del ESP32 y reenvía cada línea JSON a la página por
// ws://localhost:3001. Así la interfaz funciona en cualquier navegador,
// incluido Firefox, que no tiene Web Serial.
//
// Uso:  bun puente.ts            (busca /dev/ttyUSB* o /dev/ttyACM*)
//       PUERTO=/dev/ttyUSB1 bun puente.ts

import { existsSync, readdirSync } from 'node:fs';
import { open, type FileHandle } from 'node:fs/promises';
import type { ServerWebSocket } from 'bun';

const PUERTO_WS = Number(process.env.PUERTO_WS ?? 3001);

function buscarPuerto(): string | null {
  if (process.env.PUERTO) return existsSync(process.env.PUERTO) ? process.env.PUERTO : null;
  const nombre = readdirSync('/dev').find((n) => /^tty(USB|ACM)\d+$/.test(n));
  return nombre ? `/dev/${nombre}` : null;
}

const clientes = new Set<ServerWebSocket<unknown>>();
let archivo: FileHandle | null = null;
let estado = 'buscando el ESP32';

function avisar(texto: string) {
  estado = texto;
  console.log(`[puente] ${texto}`);
  const linea = JSON.stringify({ puente: texto });
  for (const c of clientes) c.send(linea);
}

async function leerPuerto(ruta: string) {
  // 115200 baudios, modo crudo, sin eco
  const stty = Bun.spawnSync(['stty', '-F', ruta, '115200', 'raw', '-echo', '-echoe', '-echok', 'cs8', '-cstopb', '-parenb']);
  if (stty.exitCode !== 0) throw new Error(stty.stderr.toString().trim() || 'stty falló');
  archivo = await open(ruta, 'r+');
  avisar(`conectado a ${ruta}`);
  let resto = '';
  const decodificador = new TextDecoder();
  const buffer = new Uint8Array(4096);
  while (true) {
    const { bytesRead } = await archivo.read(buffer, 0, buffer.length, null);
    if (bytesRead === 0) {
      // Al desconectar el USB la lectura devuelve 0 bytes para siempre:
      // si el archivo del puerto ya no existe, se suelta para buscar el nuevo.
      if (!existsSync(ruta)) throw new Error('se desconectó el USB');
      await Bun.sleep(10);
      continue;
    }
    resto += decodificador.decode(buffer.subarray(0, bytesRead), { stream: true });
    const lineas = resto.split('\n');
    resto = lineas.pop() ?? '';
    for (const linea of lineas) {
      const limpia = linea.trim();
      if (!limpia.startsWith('{')) continue;
      for (const c of clientes) c.send(limpia);
    }
  }
}

// Reintenta mientras el ESP32 no esté conectado o si se desconecta
async function bucle() {
  while (true) {
    const ruta = buscarPuerto();
    if (!ruta) {
      if (estado !== 'buscando el ESP32') avisar('buscando el ESP32');
      await Bun.sleep(1500);
      continue;
    }
    try {
      await leerPuerto(ruta);
    } catch (error) {
      avisar(`sin conexión (${(error as Error).message})`);
      await archivo?.close().catch(() => {});
      archivo = null;
      await Bun.sleep(1500);
    }
  }
}

// ---------- Cámara del celular (DroidCam por WiFi) ----------
// DroidCam transmite MJPEG en http://IP:4747/video y acepta una sola conexión,
// así que el puente abre una y reparte el último cuadro: /droidcam/video
// (MJPEG para ver en vivo) y /droidcam/foto (un JPEG para capturar).
const CORS = { 'Access-Control-Allow-Origin': '*', 'Cache-Control': 'no-store' };
interface Flujo { ultimo: Uint8Array | null; clientes: Set<ReadableStreamDefaultController<Uint8Array>>; error: string }
const flujos = new Map<string, Flujo>();

function buscar(buf: Uint8Array, a: number, b: number, desde: number) {
  for (let k = desde; k < buf.length - 1; k++) if (buf[k] === a && buf[k + 1] === b) return k;
  return -1;
}

async function seguirCamara(origen: string, f: Flujo) {
  const sep = new TextEncoder().encode('--cuadro\r\nContent-Type: image/jpeg\r\n\r\n');
  while (true) {
    try {
      const r = await fetch(origen);
      if (!r.ok || !r.body) throw new Error(`respuesta ${r.status}`);
      f.error = '';
      const lector = r.body.getReader();
      let buf = new Uint8Array(0);
      while (true) {
        const { value, done } = await lector.read();
        if (done) break;
        const nuevo = new Uint8Array(buf.length + value.length);
        nuevo.set(buf); nuevo.set(value, buf.length); buf = nuevo;
        // Cada JPEG va de FF D8 a FF D9
        let ini = buscar(buf, 0xff, 0xd8, 0);
        let fin = ini >= 0 ? buscar(buf, 0xff, 0xd9, ini + 2) : -1;
        while (ini >= 0 && fin >= 0) {
          const cuadro = buf.slice(ini, fin + 2);
          f.ultimo = cuadro;
          for (const c of f.clientes) {
            try { c.enqueue(sep); c.enqueue(cuadro); c.enqueue(new TextEncoder().encode('\r\n')); } catch { f.clientes.delete(c); }
          }
          buf = buf.slice(fin + 2);
          ini = buscar(buf, 0xff, 0xd8, 0);
          fin = ini >= 0 ? buscar(buf, 0xff, 0xd9, ini + 2) : -1;
        }
        if (buf.length > 4_000_000) buf = new Uint8Array(0);
      }
    } catch (error) {
      f.error = (error as Error).message;
    }
    await Bun.sleep(1500);
  }
}

function flujo(origen: string): Flujo {
  let f = flujos.get(origen);
  if (!f) {
    f = { ultimo: null, clientes: new Set(), error: '' };
    flujos.set(origen, f);
    void seguirCamara(origen, f);
  }
  return f;
}

async function camaraHttp(url: URL): Promise<Response> {
  const origen = url.searchParams.get('url') ?? '';
  if (!/^http:\/\/[\w.-]+(:\d+)?\/[\w./-]*$/.test(origen)) return new Response('URL de cámara inválida', { status: 400, headers: CORS });
  const f = flujo(origen);
  if (url.pathname === '/droidcam/foto') {
    for (let k = 0; k < 50 && !f.ultimo; k++) await Bun.sleep(100);
    if (!f.ultimo) return new Response(`Sin imagen de la cámara${f.error ? ': ' + f.error : ''}`, { status: 504, headers: CORS });
    return new Response(f.ultimo, { headers: { ...CORS, 'Content-Type': 'image/jpeg' } });
  }
  let control: ReadableStreamDefaultController<Uint8Array>;
  const cuerpo = new ReadableStream<Uint8Array>({
    start(c) { control = c; f.clientes.add(c); },
    cancel() { f.clientes.delete(control); },
  });
  return new Response(cuerpo, { headers: { ...CORS, 'Content-Type': 'multipart/x-mixed-replace; boundary=cuadro' } });
}

Bun.serve({
  port: PUERTO_WS,
  hostname: '127.0.0.1',
  idleTimeout: 0,
  fetch(req, server) {
    const url = new URL(req.url);
    if (url.pathname.startsWith('/droidcam/')) return camaraHttp(url);
    if (server.upgrade(req)) return;
    return new Response(`Puente del escáner: ${estado}\n`);
  },
  websocket: {
    open(ws) {
      clientes.add(ws);
      ws.send(JSON.stringify({ puente: estado }));
    },
    close(ws) {
      clientes.delete(ws);
    },
    async message(_ws, mensaje) {
      // Comandos de un carácter para el firmware: M (medir) y S, P, C (versión con servo)
      const comando = String(mensaje).trim().toUpperCase();
      if (archivo && /^[MSPC]$/.test(comando)) await archivo.write(comando);
    },
  },
});

console.log(`[puente] escuchando en ws://localhost:${PUERTO_WS}`);
void bucle();
