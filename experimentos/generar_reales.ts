// Genera las tablas y cifras del informe a partir de las mediciones reales
// capturadas con experimentos/capturar.ts. No escribe cifras a mano.
// Uso:  bun experimentos/generar_reales.ts

import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const docs = join(dirname(fileURLToPath(import.meta.url)), '..', 'docs');
const coma = (v: number, dec = 1) => v.toFixed(dec).replace('.', ',');
const leerCSV = (archivo: string) => {
  const [cabecera, ...filas] = readFileSync(archivo, 'utf8').trim().split('\n');
  const campos = cabecera!.split(',');
  return filas.map((f) => Object.fromEntries(f.split(',').map((v, i) => [campos[i], Number(v)])));
};

const escaneo = leerCSV(join(docs, 'datos', 'real_escaneo.csv'));
if (escaneo.length < 10) throw new Error('real_escaneo.csv casi vacío: captura de nuevo con experimentos/capturar.ts');

const barridos = [...new Set(escaneo.map((f) => f.barrido!))].sort();
const rs = escaneo.map((f) => f.r!);
const angulosConEco = new Set(escaneo.map((f) => f.theta!)).size;

// Repetibilidad: misma θ en dos barridos consecutivos
const porBarrido = new Map(barridos.map((b) => [b, new Map(escaneo.filter((f) => f.barrido === b).map((f) => [f.theta!, f.r!]))]));
const difs: number[] = [];
for (let i = 1; i < barridos.length; i++) {
  const a = porBarrido.get(barridos[i - 1]!)!, b = porBarrido.get(barridos[i]!)!;
  for (const [theta, r] of a) if (b.has(theta)) difs.push(Math.abs(r - b.get(theta)!));
}
const media = (v: number[]) => v.reduce((x, y) => x + y, 0) / v.length;
const mediana = (v: number[]) => { const s = [...v].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]!; };

let tabla = `% Generado por experimentos/generar_reales.ts. No editar a mano.
\\begin{table}[htbp]
\\caption{Escaneo con el prototipo real}
\\label{tab:real}
\\begin{tabular}{lr}
\\toprule
Medida & Valor \\\\
\\midrule
Barridos completos capturados & ${barridos.length} \\\\
Lecturas con eco & ${escaneo.length} \\\\
Ángulos distintos con eco (de 91) & ${angulosConEco} \\\\
Distancia mínima medida (mm) & ${Math.min(...rs)} \\\\
Distancia máxima medida (mm) & ${Math.max(...rs)} \\\\
Pares de lecturas comparables entre barridos & ${difs.length} \\\\
Diferencia media entre barridos, mismo ángulo (mm) & ${difs.length ? coma(media(difs)) : '---'} \\\\
Diferencia mediana entre barridos, mismo ángulo (mm) & ${difs.length ? coma(mediana(difs)) : '---'} \\\\
\\bottomrule
\\end{tabular}
\\note{Lecturas del VL53L0X tomadas por el ESP32 y recibidas por la interfaz a través del puente USB. Elaboración propia.}
\\end{table}
`;

let cifras = `% Generado por experimentos/generar_reales.ts. No editar a mano.
\\newcommand{\\realBarridos}{${barridos.length}}
\\newcommand{\\realLecturas}{${escaneo.length}}
\\newcommand{\\realAngulos}{${angulosConEco}}
\\newcommand{\\realMin}{${Math.min(...rs)}}
\\newcommand{\\realMax}{${Math.max(...rs)}}
\\newcommand{\\realRepMedia}{${difs.length ? coma(media(difs)) : '---'}}
\\newcommand{\\realRepMediana}{${difs.length ? coma(mediana(difs)) : '---'}}
\\newcommand{\\realHayDistancias}{0}
`;

// Prueba de distancias contra cinta métrica (solo si se hizo)
const archivoDist = join(docs, 'datos', 'real_distancias.csv');
if (existsSync(archivoDist)) {
  const filas = leerCSV(archivoDist);
  tabla += `
\\begin{table}[htbp]
\\caption{Distancia medida por el sensor frente a la distancia real}
\\label{tab:real-distancias}
\\begin{tabular}{rrrrr}
\\toprule
Real (mm) & Lecturas & Media (mm) & Error (mm) & Desviación (mm) \\\\
\\midrule
${filas.map((f) => `${f.referencia_mm} & ${f.n} & ${coma(f.media_mm!)} & ${coma(f.media_mm! - f.referencia_mm!)} & ${coma(f.desviacion_mm!)} \\\\`).join('\n')}
\\bottomrule
\\end{tabular}
\\note{Objeto plano frente al sensor con el servo entre $88^\\circ$ y $92^\\circ$; distancia real medida con cinta desde el lente. Elaboración propia.}
\\end{table}
`;
  cifras = cifras.replace('\\realHayDistancias}{0}', '\\realHayDistancias}{1}');
}

writeFileSync(join(docs, 'tablas', 'reales.tex'), tabla);
writeFileSync(join(docs, 'tablas', 'cifras_reales.tex'), cifras);
console.log(`barridos ${barridos.length}, lecturas ${escaneo.length}, ángulos ${angulosConEco}, r ${Math.min(...rs)}–${Math.max(...rs)} mm, repetibilidad media ${difs.length ? media(difs).toFixed(1) : '-'} mm (${difs.length} pares)`);
