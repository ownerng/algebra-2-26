// Cifras del informe a partir de las mediciones reales del escáner 3D.
//   docs/datos/real_repetibilidad.json   10 mediciones en la misma posición
//   docs/datos/real_escaneo.json         (opcional) escaneo real exportado desde la página
// Uso:  bun experimentos/generar_reales3d.ts

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ajustarPlano, puntoDeCasilla, type Vec3 } from '../web/src/lib/algebra3d.ts';

const docs = join(dirname(fileURLToPath(import.meta.url)), '..', 'docs');
const coma = (v: number, d = 1) => v.toFixed(d).replace('.', ',');

const rep = JSON.parse(readFileSync(join(docs, 'datos', 'real_repetibilidad.json'), 'utf8'));
const m: number[] = rep.medidas.map((d: { medida: number }) => d.medida);
let cifras = `% Generado por experimentos/generar_reales3d.ts. No editar a mano.
\\newcommand{\\repN}{${m.length}}
\\newcommand{\\repMedia}{${coma(rep.media, 2)}}
\\newcommand{\\repDesv}{${coma(rep.desv_entre, 2)}}
\\newcommand{\\repInterna}{${coma(rep.desv_interna_media, 2)}}
\\newcommand{\\repMin}{${coma(Math.min(...m))}}
\\newcommand{\\repMax}{${coma(Math.max(...m))}}
`;

// Escaneo real (si ya se hizo): resumen y CSV para la figura
const archivo = join(docs, 'datos', 'real_escaneo.json');
if (existsSync(archivo)) {
  const d = JSON.parse(readFileSync(archivo, 'utf8'));
  const { nx, ny, paso, altura } = d.config;
  const pts = d.medidas.map((q: { i: number; j: number; r: number }) => {
    const [x, y, z] = puntoDeCasilla(q.i, q.j, q.r, paso, altura);
    return { ...q, x, y, z };
  });
  const borde = pts.filter((p: { i: number; j: number }) => p.i === 0 || p.j === 0 || p.i === nx - 1 || p.j === ny - 1);
  const plano = ajustarPlano(borde.map((p: { x: number; y: number; z: number }) => [p.x, p.y, p.z] as Vec3))!;
  const h = (p: { x: number; y: number; z: number }) => p.z - (plano.a * p.x + plano.b * p.y + plano.c);
  const mapa = new Map(pts.map((p: { i: number; j: number }) => [`${p.i},${p.j}`, p]));
  const filas = ['x,y,h'];
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const p = mapa.get(`${i},${j}`) as { x: number; y: number; z: number } | undefined;
      filas.push(p ? [p.x, p.y, Math.max(0, h(p)).toFixed(2)].join(',') : [i * paso, j * paso, 'nan'].join(','));
    }
    filas.push('');
  }
  writeFileSync(join(docs, 'datos', 'real_escaneo.csv'), filas.join('\n'));
  const alturas = pts.map(h);
  cifras += `\\newcommand{\\realNombre}{${d.nombre}}
\\newcommand{\\realNX}{${nx}}
\\newcommand{\\realNY}{${ny}}
\\newcommand{\\realPaso}{${paso}}
\\newcommand{\\realH}{${altura}}
\\newcommand{\\realPuntos}{${pts.length}}
\\newcommand{\\realAltoMax}{${coma(Math.max(...alturas))}}
\\newcommand{\\realPlanoA}{${coma(plano.a, 4)}}
\\newcommand{\\realPlanoB}{${coma(plano.b, 4)}}
\\newcommand{\\realHayEscaneo}{1}
`;
} else {
  cifras += `\\newcommand{\\realHayEscaneo}{0}\n`;
}

writeFileSync(join(docs, 'tablas', 'cifras_reales3d.tex'), cifras);
console.log(cifras);
