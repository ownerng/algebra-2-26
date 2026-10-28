// Puente USB → WebSocket.
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

Bun.serve({
  port: PUERTO_WS,
  hostname: '127.0.0.1',
  fetch(req, server) {
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
      // Comandos de un carácter para el firmware: S, P, C
      const comando = String(mensaje).trim().toUpperCase();
      if (archivo && /^[SPC]$/.test(comando)) await archivo.write(comando);
    },
  },
});

console.log(`[puente] escuchando en ws://localhost:${PUERTO_WS}`);
void bucle();
