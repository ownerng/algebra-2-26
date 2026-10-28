// Captura mediciones reales del prototipo a través del puente USB
// (web/puente.ts debe estar corriendo).
//
//   bun experimentos/capturar.ts escaneo [barridos]
//       Guarda barridos completos en docs/datos/real_escaneo.csv
//   bun experimentos/capturar.ts distancia <referencia_mm> [segundos]
//       Registra las lecturas con el servo entre 88° y 92° y agrega una fila
//       a docs/datos/real_distancias.csv con la referencia medida con cinta.

import { appendFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { procesarLectura } from '../web/src/lib/matematica.ts';

const D_MM = 15; // distancia del eje del servo al lente
const datos = join(dirname(fileURLToPath(import.meta.url)), '..', 'docs', 'datos');
mkdirSync(datos, { recursive: true });

const [modo = 'escaneo', arg1, arg2] = process.argv.slice(2);

function escuchar(alLeer: (d: { r: number; theta: number }) => boolean) {
  return new Promise<void>((resolver, rechazar) => {
    const ws = new WebSocket('ws://localhost:3001');
    ws.onerror = () => rechazar(new Error('No hay puente en ws://localhost:3001: ábrelo con ./iniciar-web.sh'));
    ws.onopen = () => ws.send('S'); // por si el barrido estaba en pausa
    ws.onmessage = (ev) => {
      const d = JSON.parse(String(ev.data));
      if (typeof d.r === 'number' && typeof d.theta === 'number' && alLeer(d)) {
        ws.close();
        resolver();
      }
    };
  });
}

if (modo === 'escaneo') {
  const barridos = Number(arg1 ?? 2);
  const filas: string[] = [];
  let vueltas = 0;
  let anterior: number | null = null;
  console.log(`Capturando ${barridos} barridos…`);
  await escuchar(({ r, theta }) => {
    if (anterior !== null && (theta === 0 || theta === 180) && theta !== anterior) vueltas++;
    anterior = theta;
    if (vueltas === 0) return false; // empieza en un extremo para tener barridos completos
    if (vueltas > barridos) return true;
    if (r > 0) {
      const c = procesarLectura({ r, thetaServo: theta, d: D_MM, invertir: false, pose: { dx: 0, dy: 0, alpha: 0 } });
      filas.push([vueltas, theta, r, c.local[0]!.toFixed(2), c.local[1]!.toFixed(2)].join(','));
    }
    return false;
  });
  writeFileSync(join(datos, 'real_escaneo.csv'), ['barrido,theta,r,x,y', ...filas].join('\n') + '\n');
  console.log(`${filas.length} lecturas con eco guardadas en docs/datos/real_escaneo.csv`);
} else if (modo === 'distancia') {
  const referencia = Number(arg1);
  const segundos = Number(arg2 ?? 20);
  if (!(referencia > 0)) throw new Error('Indica la distancia de referencia en mm, por ejemplo: distancia 200');
  const lecturas: number[] = [];
  const fin = Date.now() + segundos * 1000;
  console.log(`Registrando ${segundos} s con el objeto a ${referencia} mm…`);
  await escuchar(({ r, theta }) => {
    if (theta >= 88 && theta <= 92 && r > 0) lecturas.push(r);
    return Date.now() > fin;
  });
  if (!lecturas.length) throw new Error('No hubo lecturas con eco entre 88° y 92°');
  const media = lecturas.reduce((a, b) => a + b, 0) / lecturas.length;
  const desv = Math.sqrt(lecturas.reduce((a, b) => a + (b - media) ** 2, 0) / Math.max(1, lecturas.length - 1));
  const archivo = join(datos, 'real_distancias.csv');
  if (!existsSync(archivo)) writeFileSync(archivo, 'referencia_mm,n,media_mm,desviacion_mm,minimo_mm,maximo_mm\n');
  appendFileSync(archivo, [referencia, lecturas.length, media.toFixed(1), desv.toFixed(1), Math.min(...lecturas), Math.max(...lecturas)].join(',') + '\n');
  console.log(`n = ${lecturas.length}, media = ${media.toFixed(1)} mm, error = ${(media - referencia).toFixed(1)} mm, desviación = ${desv.toFixed(1)} mm`);
} else {
  throw new Error(`Modo desconocido: ${modo}`);
}
