// Genera los datos y las tablas del informe a partir del simulador.
// Uso:  node experimentos/generar.mjs
// Salida: docs/datos/*.csv (para pgfplots) y docs/tablas/*.tex
// Las cifras del informe salen de aquí: no se escriben a mano en el .tex.

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

// Semilla fija: el ruido del simulador es reproducible
let semilla = 20261028;
Math.random = () => {
  semilla = (semilla + 0x6d2b79f5) | 0;
  let t = Math.imul(semilla ^ (semilla >>> 15), 1 | semilla);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const { procesarLectura, rotacion, homogenea, inversaHomogenea, multiplicarMat, multiplicarMatVec, det2 } =
  await import('../web/js/matematica.js');
const { medirSimulado, POSES_REALES, DESFASE_LENTE_MM } = await import('../web/js/simulador.js');

const raiz = join(dirname(fileURLToPath(import.meta.url)), '..', 'docs');
mkdirSync(join(raiz, 'datos'), { recursive: true });
mkdirSync(join(raiz, 'tablas'), { recursive: true });

const d = DESFASE_LENTE_MM;
const ANGULOS = Array.from({ length: 91 }, (_, i) => i * 2);
const n2 = (v) => Math.hypot(v[0], v[1]);
const csv = (nombre, cabecera, filas) =>
  writeFileSync(join(raiz, 'datos', nombre), [cabecera, ...filas.map((f) => f.map((v) => +v.toFixed(3)).join(','))].join('\n') + '\n');
const coma = (v, dec = 1) => v.toFixed(dec).replace('.', ',');

// ---------- 1. Escaneos con ruido (lo que vería el prototipo) ----------
function escanear(pos, conRuido = true) {
  const pose = POSES_REALES[pos];
  return ANGULOS.map((theta) => ({ theta, r: medirSimulado(pose, theta, conRuido) }))
    .filter((m) => m.r > 0)
    .map((m) => ({ ...m, ...procesarLectura({ r: m.r, thetaServo: m.theta, d, invertir: false, pose }) }));
}
const p1 = escanear(1);
const p2 = escanear(2);
csv('p1_local.csv', 'x,y', p1.map((p) => p.local));
csv('p2_local.csv', 'x,y', p2.map((p) => p.local));
csv('fusion_p1.csv', 'x,y', p1.map((p) => p.global));
csv('fusion_p2.csv', 'x,y', p2.map((p) => p.global));

// Pose mal medida: 8° de error en α y 80 mm en dx
const poseMala = { ...POSES_REALES[2], alpha: POSES_REALES[2].alpha + 8, dx: POSES_REALES[2].dx + 80 };
csv('fusion_mala_p2.csv', 'x,y', p2.map((p) => procesarLectura({ r: p.r, thetaServo: p.theta, d, invertir: false, pose: poseMala }).global));

// ---------- 2. Error de registro frente a error en α ----------
// Sin ruido del sensor, para aislar el efecto de la pose.
// Teoría: ‖(R(α+δ) − R(α)) v‖ = 2|sin(δ/2)|·‖v‖
const exactos = escanear(2, false);
const normaMedia = exactos.reduce((s, p) => s + n2(p.local), 0) / exactos.length;
const filasAlpha = [];
for (let delta = 0; delta <= 20; delta += 1) {
  const pose = { ...POSES_REALES[2], alpha: POSES_REALES[2].alpha + delta };
  const err = exactos.map((p) => {
    const g = procesarLectura({ r: p.r, thetaServo: p.theta, d, invertir: false, pose }).global;
    return n2([g[0] - p.global[0], g[1] - p.global[1]]);
  });
  const medio = err.reduce((a, b) => a + b, 0) / err.length;
  const teorico = 2 * Math.abs(Math.sin((delta * Math.PI) / 360)) * normaMedia;
  filasAlpha.push([delta, medio, teorico, Math.max(...err)]);
}
csv('error_alpha.csv', 'delta,medio,teorico,maximo', filasAlpha);

// ---------- 3. Ruido del sensor: error de los puntos con pose correcta ----------
const errRuido = p2.map((p) => {
  const ideal = exactos.find((q) => q.theta === p.theta);
  return ideal ? n2([p.global[0] - ideal.global[0], p.global[1] - ideal.global[1]]) : null;
}).filter((v) => v !== null);
const ruidoMedio = errRuido.reduce((a, b) => a + b, 0) / errRuido.length;

// ---------- 4. Propiedades numéricas verificadas ----------
let maxDet = 0, maxNorma = 0, maxOrto = 0, maxInv = 0;
for (let a = 0; a < 360; a += 0.5) {
  const R = rotacion(a);
  maxDet = Math.max(maxDet, Math.abs(det2(R) - 1));
  const v = [523.7, -118.2];
  maxNorma = Math.max(maxNorma, Math.abs(n2(multiplicarMatVec(R, v)) - n2(v)));
  const Rt = [[R[0][0], R[1][0]], [R[0][1], R[1][1]]];
  const I2 = multiplicarMat(Rt, R);
  maxOrto = Math.max(maxOrto, Math.abs(I2[0][0] - 1), Math.abs(I2[0][1]), Math.abs(I2[1][0]), Math.abs(I2[1][1] - 1));
  const I3 = multiplicarMat(inversaHomogenea(a, 250, 1100), homogenea(a, 250, 1100));
  I3.forEach((fila, i) => fila.forEach((x, j) => { maxInv = Math.max(maxInv, Math.abs(x - (i === j ? 1 : 0))); }));
}
const cient = (v) => {
  if (v === 0) return '$0$';
  const [m, e] = v.toExponential(1).split('e');
  return `$${m.replace('.', '{,}')} \\times 10^{${Number(e)}}$`;
};

writeFileSync(join(raiz, 'tablas', 'propiedades.tex'), `% Generado por experimentos/generar.mjs. No editar a mano.
\\begin{table}[htbp]
\\caption{Verificación numérica de las propiedades de las matrices implementadas}
\\label{tab:propiedades}
\\begin{tabular}{lll}
\\toprule
Propiedad & Expresión & Desviación máxima \\\\
\\midrule
Determinante unitario & $|\\det R(\\theta) - 1|$ & ${cient(maxDet)} \\\\
Conservación de la norma & $\\bigl|\\,\\|R\\mathbf{v}\\| - \\|\\mathbf{v}\\|\\,\\bigr|$ (mm) & ${cient(maxNorma)} \\\\
Ortogonalidad & $\\max |R^{T}R - I|$ & ${cient(maxOrto)} \\\\
Inversa homogénea & $\\max |T^{-1}T - I|$ & ${cient(maxInv)} \\\\
\\bottomrule
\\end{tabular}
\\note{Evaluado en 720 ángulos entre $0^\\circ$ y $359{,}5^\\circ$ con aritmética de doble precisión. Elaboración propia.}
\\end{table}
`);

const fila = (d) => filasAlpha.find((f) => f[0] === d);
writeFileSync(join(raiz, 'tablas', 'resultados_sim.tex'), `% Generado por experimentos/generar.mjs. No editar a mano.
\\begin{table}[htbp]
\\caption{Resultados del escaneo y la fusión en el entorno simulado}
\\label{tab:resultados-sim}
\\begin{tabular}{lr}
\\toprule
Medida & Valor \\\\
\\midrule
Lecturas por barrido ($0^\\circ$ a $180^\\circ$, paso $2^\\circ$) & ${ANGULOS.length} \\\\
Puntos con eco desde la posición 1 & ${p1.length} \\\\
Puntos con eco desde la posición 2 & ${p2.length} \\\\
Puntos en el mapa fusionado & ${p1.length + p2.length} \\\\
Distancia media al eje, posición 2 (mm) & ${coma(normaMedia)} \\\\
Error medio por ruido del sensor, pose correcta (mm) & ${coma(ruidoMedio)} \\\\
Error medio con $5^\\circ$ de error en $\\alpha$ (mm) & ${coma(fila(5)[1])} \\\\
Error medio con $10^\\circ$ de error en $\\alpha$ (mm) & ${coma(fila(10)[1])} \\\\
\\bottomrule
\\end{tabular}
\\note{Habitación simulada con ruido uniforme de $\\pm(3 + 0{,}01\\,r)$ mm y 3\\,\\% de lecturas perdidas. Semilla fija. Elaboración propia.}
\\end{table}
`);

// Valores sueltos para el texto
writeFileSync(join(raiz, 'tablas', 'cifras.tex'), `% Generado por experimentos/generar.mjs. No editar a mano.
\\newcommand{\\cifPuntosUno}{${p1.length}}
\\newcommand{\\cifPuntosDos}{${p2.length}}
\\newcommand{\\cifPuntosTotal}{${p1.length + p2.length}}
\\newcommand{\\cifRuidoMedio}{${coma(ruidoMedio)}}
\\newcommand{\\cifNormaMedia}{${coma(normaMedia)}}
\\newcommand{\\cifErrorCinco}{${coma(fila(5)[1])}}
\\newcommand{\\cifErrorDiez}{${coma(fila(10)[1])}}
\\newcommand{\\cifErrorVeinte}{${coma(fila(20)[1])}}
\\newcommand{\\cifDifTeoria}{${cient(Math.max(...filasAlpha.map((f) => Math.abs(f[1] - f[2]))))}}
`);

console.log(`P1 ${p1.length} pts, P2 ${p2.length} pts, ruido ${ruidoMedio.toFixed(2)} mm, ` +
  `error α=5° ${fila(5)[1].toFixed(1)} mm, α=10° ${fila(10)[1].toFixed(1)} mm`);
