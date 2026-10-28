// Experimentos del escáner 3D con el simulador. Genera los CSV de las gráficas
// y las tablas del informe; las cifras del texto salen de aquí.
// Uso:  bun experimentos/generar3d.ts

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// Semilla fija: el ruido del simulador es reproducible
let semilla = 20261029;
Math.random = () => {
  semilla = (semilla + 0x6d2b79f5) | 0;
  let t = Math.imul(semilla ^ (semilla >>> 15), 1 | semilla);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const { ajustarPlano, descomposicionCasilla, matrizCasilla, matrizVista, puntoDeCasilla, rotX, rotZ } = await import('../web/src/lib/algebra3d.ts');
const { multiplicarMat } = await import('../web/src/lib/matematica.ts');
const { ALTURA_REAL_MM, INCLINACION, alturaObjeto, distanciaSimulada } = await import('../web/src/lib/simulador3d.ts');
type Vec3 = [number, number, number];

const docs = join(dirname(fileURLToPath(import.meta.url)), '..', 'docs');
mkdirSync(join(docs, 'datos'), { recursive: true });
mkdirSync(join(docs, 'tablas'), { recursive: true });
const coma = (v: number, d = 1) => v.toFixed(d).replace('.', ',');

const NX = 10, NY = 8, PASO = 20, H = ALTURA_REAL_MM;

function escanear(objeto: 'taza' | 'caja' | 'libros' | 'vacia', paso = PASO, nx = NX, ny = NY) {
  // 10 lecturas por casilla, como el firmware
  const pts: { i: number; j: number; x: number; y: number; z: number; real: number }[] = [];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const x = i * paso, y = j * paso;
    const r = Array.from({ length: 10 }, () => distanciaSimulada(objeto, x, y)).reduce((a, b) => a + b, 0) / 10;
    const [px, py, pz] = puntoDeCasilla(i, j, r, paso, H);
    pts.push({ i, j, x: px, y: py, z: pz, real: alturaObjeto(objeto, x, y) });
  }
  return pts;
}

function corregir(pts: ReturnType<typeof escanear>, nx = NX, ny = NY) {
  let borde = pts.filter((p) => p.i === 0 || p.j === 0 || p.i === nx - 1 || p.j === ny - 1);
  let plano = ajustarPlano(borde.map((p) => [p.x, p.y, p.z] as Vec3))!;
  const cerca = borde.filter((p) => Math.abs(p.z - (plano.a * p.x + plano.b * p.y + plano.c)) < 6);
  if (cerca.length >= 6) { borde = cerca; plano = ajustarPlano(borde.map((p) => [p.x, p.y, p.z] as Vec3))!; }
  return { plano, alturas: pts.map((p) => p.z - (plano.a * p.x + plano.b * p.y + plano.c)) };
}

const errores = (est: number[], real: number[]) => {
  const e = est.map((v, k) => v - real[k]!);
  return { rmse: Math.sqrt(e.reduce((a, b) => a + b * b, 0) / e.length), max: Math.max(...e.map(Math.abs)) };
};

// ---------- 1. Escaneo de la taza y la caja ----------
const filasTabla: string[] = [];
const resumen: Record<string, number> = {};
for (const objeto of ['taza', 'caja', 'libros'] as const) {
  const pts = escanear(objeto);
  const real = pts.map((p) => p.real);
  const sinCorregir = errores(pts.map((p) => p.z), real);
  const { plano, alturas } = corregir(pts);
  const corregido = errores(alturas, real);
  const mesa = pts.map((p, k) => (p.real === 0 ? k : -1)).filter((k) => k >= 0);
  const mesaSin = errores(mesa.map((k) => pts[k]!.z), mesa.map(() => 0));
  const mesaCon = errores(mesa.map((k) => alturas[k]!), mesa.map(() => 0));
  const maxMedida = Math.max(...alturas);
  const maxReal = Math.max(...real);
  filasTabla.push(`${objeto[0]!.toUpperCase() + objeto.slice(1)} & ${maxReal} & ${coma(maxMedida)} & ${coma(corregido.rmse)} \\\\`);
  resumen[`${objeto}_mesa_sin`] = mesaSin.rmse;
  resumen[`${objeto}_mesa_con`] = mesaCon.rmse;
  void sinCorregir;
  resumen[`${objeto}_rmse_sin`] = sinCorregir.rmse;
  resumen[`${objeto}_rmse_con`] = corregido.rmse;
  resumen[`${objeto}_max`] = maxMedida;
  resumen[`${objeto}_a`] = plano.a;
  resumen[`${objeto}_b`] = plano.b;
  // CSV para la superficie: x, y, altura medida, altura real (orden por filas para pgfplots)
  const filas = ['x,y,h,real'];
  for (let j = 0; j < NY; j++) {
    for (let i = 0; i < NX; i++) {
      const k = j * NX + i;
      filas.push([pts[k]!.x, pts[k]!.y, Math.max(0, alturas[k]!).toFixed(2), real[k]].join(','));
    }
    filas.push('');
  }
  writeFileSync(join(docs, 'datos', `sim_${objeto}.csv`), filas.join('\n'));
}

// ---------- 2. Mesa vacía: qué corrige el plano por mínimos cuadrados ----------
const vacia = escanear('vacia');
const { plano: planoVacia, alturas: alturasVacia } = corregir(vacia);
const vaciaSin = errores(vacia.map((p) => p.z), vacia.map(() => 0));
const vaciaCon = errores(alturasVacia, vacia.map(() => 0));
writeFileSync(join(docs, 'datos', 'sim_vacia.csv'), ['x,y,sin,con', ...vacia.map((p, k) => [p.x, p.y, p.z.toFixed(2), alturasVacia[k]!.toFixed(2)].join(','))].join('\n') + '\n');

// ---------- 3. Verificaciones numéricas ----------
const { E, F, T } = descomposicionCasilla(PASO, H);
const TFE = multiplicarMat(T, multiplicarMat(F, E));
const A = matrizCasilla(PASO, H);
let difA = 0;
A.forEach((fila, i) => fila.forEach((v, j) => { difA = Math.max(difA, Math.abs(v - TFE[i]![j]!)); }));
const det3 = (R: number[][]) => R[0]![0]! * (R[1]![1]! * R[2]![2]! - R[1]![2]! * R[2]![1]!) - R[0]![1]! * (R[1]![0]! * R[2]![2]! - R[1]![2]! * R[2]![0]!) + R[0]![2]! * (R[1]![0]! * R[2]![1]! - R[1]![1]! * R[2]![0]!);
let maxDet = 0, maxOrto = 0;
for (let t = 0; t < 360; t += 5) for (let f = 5; f < 90; f += 5) {
  const V = matrizVista(t, f);
  maxDet = Math.max(maxDet, Math.abs(det3(V) - 1));
  const Vt = V[0]!.map((_, j) => V.map((fila) => fila[j]!));
  const I = multiplicarMat(Vt, V);
  I.forEach((fila, i) => fila.forEach((v, j) => { maxOrto = Math.max(maxOrto, Math.abs(v - (i === j ? 1 : 0))); }));
}
const planoConocido = ajustarPlano(Array.from({ length: 80 }, (_, k) => [(k % 10) * 20, Math.floor(k / 10) * 20, 0.03 * (k % 10) * 20 - 0.015 * Math.floor(k / 10) * 20 + 2] as Vec3))!;
const difPlano = Math.max(Math.abs(planoConocido.a - 0.03), Math.abs(planoConocido.b + 0.015), Math.abs(planoConocido.c - 2));
void rotX; void rotZ;

const cient = (v: number) => {
  if (v === 0) return '$0$';
  const [m, e] = v.toExponential(1).split('e');
  return `$${m!.replace('.', '{,}')} \\times 10^{${Number(e)}}$`;
};

writeFileSync(join(docs, 'tablas', 'sim3d.tex'), `% Generado por experimentos/generar3d.ts. No editar a mano.
\\begin{table}[htbp]
\\caption{Escaneo simulado de tres objetos: altura máxima y error frente a la forma real}
\\label{tab:sim3d}
\\begin{tabular}{lrrr}
\\toprule
Objeto & Alto real (mm) & Alto medido (mm) & RMSE de la altura (mm) \\\\
\\midrule
${filasTabla.join('\n')}
\\bottomrule
\\end{tabular}
\\note{Cuadrícula de ${NX} × ${NY} casillas de ${PASO} mm, caja a ${H} mm de la mesa, mesa inclinada y ruido de $\\pm 2$ mm. Alto medido: la casilla más alta. RMSE: raíz del error cuadrático medio de la altura en todas las casillas, después de restar el plano de la mesa. Elaboración propia.}
\\end{table}

\\begin{table}[htbp]
\\caption{Escaneo simulado de la mesa vacía: efecto del plano ajustado por mínimos cuadrados}
\\label{tab:mesa}
\\begin{tabular}{lrr}
\\toprule
 & Sin corregir & Corregido \\\\
\\midrule
RMSE de la altura (mm) & ${coma(vaciaSin.rmse, 2)} & ${coma(vaciaCon.rmse, 2)} \\\\
Error máximo (mm) & ${coma(vaciaSin.max, 2)} & ${coma(vaciaCon.max, 2)} \\\\
\\midrule
Pendiente en $x$, $a$ (real ${coma(INCLINACION.a, 3)}) & \\multicolumn{2}{r}{${coma(planoVacia.a, 4)}} \\\\
Pendiente en $y$, $b$ (real ${coma(INCLINACION.b, 3)}) & \\multicolumn{2}{r}{${coma(planoVacia.b, 4)}} \\\\
\\bottomrule
\\end{tabular}
\\note{La mesa simulada está inclinada; la altura correcta en todas las casillas es 0 mm. Elaboración propia.}
\\end{table}

\\begin{table}[htbp]
\\caption{Verificación numérica de las matrices del escáner}
\\label{tab:verif3d}
\\begin{tabular}{ll}
\\toprule
Propiedad & Desviación máxima \\\\
\\midrule
$A = T\\,F\\,E$ & ${cient(difA)} \\\\
$|\\det V - 1|$, con $V = R_x(\\varphi)R_z(\\theta)$ & ${cient(maxDet)} \\\\
$\\max |V^{T}V - I|$ & ${cient(maxOrto)} \\\\
Plano conocido recuperado por mínimos cuadrados & ${cient(difPlano)} \\\\
\\bottomrule
\\end{tabular}
\\note{Las rotaciones se evaluaron en 1224 combinaciones de $\\theta$ y $\\varphi$. Elaboración propia.}
\\end{table}
`);

writeFileSync(join(docs, 'tablas', 'cifras3d.tex'), `% Generado por experimentos/generar3d.ts. No editar a mano.
\\newcommand{\\simNX}{${NX}}
\\newcommand{\\simNY}{${NY}}
\\newcommand{\\simPaso}{${PASO}}
\\newcommand{\\simH}{${H}}
\\newcommand{\\simTazaMax}{${coma(resumen.taza_max!)}}
\\newcommand{\\simTazaSin}{${coma(resumen.taza_rmse_sin!)}}
\\newcommand{\\simTazaCon}{${coma(resumen.taza_rmse_con!)}}
\\newcommand{\\simCajaMax}{${coma(resumen.caja_max!)}}
\\newcommand{\\simCajaCon}{${coma(resumen.caja_rmse_con!)}}
\\newcommand{\\simLibrosCon}{${coma(resumen.libros_rmse_con!)}}
\\newcommand{\\simMesaSin}{${coma(vaciaSin.rmse, 2)}}
\\newcommand{\\simMesaCon}{${coma(vaciaCon.rmse, 2)}}
\\newcommand{\\simPlanoA}{${coma(resumen.taza_a!, 4)}}
\\newcommand{\\simPlanoB}{${coma(resumen.taza_b!, 4)}}
`);

console.log(filasTabla.join('\n'));
console.log('mesa vacía', vaciaSin.rmse.toFixed(2), '->', vaciaCon.rmse.toFixed(2), 'a', planoVacia.a.toFixed(4), 'b', planoVacia.b.toFixed(4));
