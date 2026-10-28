// Hoja de cuaderno: el cálculo de la última medición desglosado paso a paso,
// de la distancia del sensor al punto 3D sombreado en la pantalla.

import katex from 'katex';
import { useMemo, useState } from 'react';
import { cruz, matrizVista, normalizar, punto, sub, type Vec3 } from '../lib/algebra3d.ts';
import { multiplicarMatVec } from '../lib/matematica.ts';
import { escaneo, useEscaneo } from '../lib/escaneo.ts';

const FX = '#b5452a';
const FY = '#2f6f8f';
const FZ = '#3f7d20';
const LUZ: Vec3 = [-0.35, -0.55, 1];

const n = (v: number, dec?: number) => {
  const x = Math.abs(v) < 5e-4 ? 0 : v;
  const d = dec ?? (Math.abs(x) >= 100 ? 1 : Math.abs(x) >= 10 ? 2 : 3);
  return x.toFixed(d).replace('.', '{,}');
};
const llano = (tex: string) => tex.replace(/\{,\}/g, ',');
const mat = (M: number[][], dec?: number) => `\\begin{bmatrix}${M.map((f) => f.map((v) => n(v, dec)).join(' & ')).join(' \\\\ ')}\\end{bmatrix}`;
const col = (v: number[], dec?: number) => mat(v.map((x) => [x]), dec);

function Formula({ tex }: { tex: string }) {
  const html = useMemo(() => katex.renderToString(tex, { displayMode: true, throwOnError: false, strict: false }), [tex]);
  return <div className="formula" dangerouslySetInnerHTML={{ __html: html }} />;
}

function Paso({ numero, titulo, children }: { numero: number; titulo: string; children: React.ReactNode }) {
  return (
    <section className="paso">
      <h3><span className="numero">{numero}</span>{titulo}</h3>
      {children}
    </section>
  );
}

export function Cuaderno() {
  const e = useEscaneo(200);
  const [congelada, setCongelada] = useState<{ i: number; j: number } | null>(null);
  const celda = congelada ?? e.ultima;
  const { puntos, plano } = e.geometria();
  const p = celda && puntos.find((q) => q.i === celda.i && q.j === celda.j);

  const congelar = () => {
    const c = congelada ? null : e.ultima;
    setCongelada(c);
    escaneo.resaltar(c);
  };

  if (!p) {
    return (
      <aside className="cuaderno" aria-label="Cálculo paso a paso">
        <header className="cuaderno-cabeza"><h2>De la medida al punto 3D</h2></header>
        <p className="nota">
          Aquí se desglosa, con los números de cada medición, cómo una distancia del sensor se convierte en un punto de la
          figura 3D: la matriz que lleva la casilla al espacio, el plano de la mesa por mínimos cuadrados, la rotación de
          la vista y el sombreado.
        </p>
        <p className="nota">Conecta el ESP32 o pulsa <b>Simular prototipo</b> y toma la primera medición.</p>
      </aside>
    );
  }

  const { paso: D, altura: H } = e.config;
  const m = e.medidas.get(`${p.i},${p.j}`)!;
  const r = m.r;
  const V = matrizVista(e.vista.theta, e.vista.phi);
  const c: Vec3 = [((e.config.nx - 1) * D) / 2, ((e.config.ny - 1) * D) / 2, H * 0.35];
  const pv: Vec3 = [p.x, p.y, Math.max(0, p.h)];
  const rel = sub(pv, c);
  const vista = multiplicarMatVec(V, rel);

  // Triángulo de esta casilla con sus vecinas (si ya están medidas)
  const vecina = (di: number, dj: number) => puntos.find((q) => q.i === p.i + di && q.j === p.j + dj);
  const b = vecina(1, 0) ?? vecina(-1, 0);
  const cc = vecina(0, 1) ?? vecina(0, -1);
  let tri: { a: Vec3; b: Vec3; c: Vec3; u: Vec3; w: Vec3; nrm: Vec3; brillo: number } | null = null;
  if (b && cc) {
    const A: Vec3 = pv, B: Vec3 = [b.x, b.y, Math.max(0, b.h)], C: Vec3 = [cc.x, cc.y, Math.max(0, cc.h)];
    const u = sub(B, A), w = sub(C, A);
    let nrm = cruz(u, w);
    if (nrm[2] < 0) nrm = [-nrm[0], -nrm[1], -nrm[2]];
    tri = { a: A, b: B, c: C, u, w, nrm, brillo: Math.max(0, punto(normalizar(nrm), normalizar(LUZ))) };
  }

  return (
    <aside className="cuaderno" aria-label="Cálculo paso a paso">
      <header className="cuaderno-cabeza">
        <h2>De la medida al punto 3D</h2>
        <button type="button" className={'congelar' + (congelada ? ' activo' : '')} onClick={congelar}>
          {congelada ? '▶ Seguir la última' : '❚❚ Congelar'}
        </button>
      </header>
      <p className="dato">{congelada ? 'Cálculo congelado en una casilla (marcada en la cuadrícula y en 3D).' : 'Última medición tomada.'}</p>

      <Paso numero={1} titulo="La medición">
        <p className="nota">
          En la casilla (i, j) = ({p.i}, {p.j}) el sensor promedió {m.n} lecturas: r = {llano(n(r, 1))} mm, con una
          desviación de {llano(n(m.desv, 1))} mm.
        </p>
      </Paso>

      <Paso numero={2} titulo="De la casilla al espacio con A">
        <p className="nota">
          Una sola matriz homogénea 4×4 escala la casilla a milímetros (Δ = {D} mm), voltea el eje del sensor hacia abajo
          y lo sube a la altura H = {H} mm:
        </p>
        <Formula tex={`\\begin{bmatrix} \\textcolor{${FX}}{x} \\\\ \\textcolor{${FY}}{y} \\\\ \\textcolor{${FZ}}{z} \\\\ 1 \\end{bmatrix} = \\begin{bmatrix} \\textcolor{${FX}}{${D}} & \\textcolor{${FX}}{0} & \\textcolor{${FX}}{0} & \\textcolor{${FX}}{0} \\\\ \\textcolor{${FY}}{0} & \\textcolor{${FY}}{${D}} & \\textcolor{${FY}}{0} & \\textcolor{${FY}}{0} \\\\ \\textcolor{${FZ}}{0} & \\textcolor{${FZ}}{0} & \\textcolor{${FZ}}{-1} & \\textcolor{${FZ}}{${H}} \\\\ 0 & 0 & 0 & 1 \\end{bmatrix}\\begin{bmatrix} ${p.i} \\\\ ${p.j} \\\\ ${n(r, 1)} \\\\ 1 \\end{bmatrix}`} />
        <Formula tex={`\\begin{aligned}
          \\textcolor{${FX}}{x} &= ${D}\\cdot ${p.i} + 0 + 0 + 0 = \\textcolor{${FX}}{${n(p.x, 0)}}\\ \\text{mm} \\\\
          \\textcolor{${FY}}{y} &= 0 + ${D}\\cdot ${p.j} + 0 + 0 = \\textcolor{${FY}}{${n(p.y, 0)}}\\ \\text{mm} \\\\
          \\textcolor{${FZ}}{z} &= 0 + 0 - 1\\cdot ${n(r, 1)} + ${H}\\cdot 1 = \\textcolor{${FZ}}{${n(p.z, 1)}}\\ \\text{mm}
        \\end{aligned}`} />
        <p className="nota">
          A = T·F·E: E escala, F = diag(1, 1, −1) es una reflexión (det F = −1) y T traslada en z.
        </p>
      </Paso>

      <Paso numero={3} titulo="El plano de la mesa (mínimos cuadrados)">
        {plano ? (
          <>
            <p className="nota">
              La mesa no está perfectamente nivelada. Con {plano.n} casillas del borde se busca el plano z = a·x + b·y + c
              que mejor las ajusta, resolviendo las ecuaciones normales (MᵀM)·[a b c]ᵀ = Mᵀz:
            </p>
            <Formula tex={`${mat(plano.MtM, 0)}\\begin{bmatrix} a \\\\ b \\\\ c \\end{bmatrix} = ${col(plano.Mtz, 0)}`} />
            <Formula tex={`a = ${n(plano.a, 4)},\\quad b = ${n(plano.b, 4)},\\quad c = ${n(plano.c, 2)}`} />
            <p className="nota">Altura del objeto en este punto, ya sin la inclinación de la mesa:</p>
            <Formula tex={`h = z - (a x + b y + c) = ${n(p.z, 1)} - (${n(plano.a * p.x + plano.b * p.y + plano.c, 2)}) = ${n(p.h, 1)}\\ \\text{mm}`} />
          </>
        ) : (
          <p className="nota">Cuando haya al menos 6 casillas medidas en el borde de la cuadrícula, aquí se calcula el plano de la mesa.</p>
        )}
      </Paso>

      <Paso numero={4} titulo="Girar la vista 3D">
        <p className="nota">
          Para ver la figura desde cualquier ángulo, cada punto se rota con V = Rx(φ)·Rz(θ) y se proyecta con Π, que
          descarta la profundidad. Ahora θ = {Math.round(((e.vista.theta % 360) + 360) % 360)}° y φ = {Math.round(e.vista.phi)}°:
        </p>
        <Formula tex={`V = ${mat(V)},\\qquad \\Pi = \\begin{bmatrix} 1 & 0 & 0 \\\\ 0 & 0 & 1 \\end{bmatrix}`} />
        <Formula tex={`\\Pi\\, V \\,(p - c) = \\Pi ${col(vista, 1)} = ${col([vista[0]!, vista[2]!], 1)}`} />
      </Paso>

      <Paso numero={5} titulo="Sombrear la superficie">
        {tri ? (
          <>
            <p className="nota">Con dos vecinas medidas se forma un triángulo. Su normal sale del producto vectorial:</p>
            <Formula tex={`\\vec n = (B - A)\\times(C - A) = ${col(tri.u, 1)}\\times${col(tri.w, 1)} = ${col(tri.nrm, 1)}`} />
            <p className="nota">Y qué tan iluminado se ve, del producto punto con la dirección de la luz:</p>
            <Formula tex={`\\text{brillo} = \\hat n \\cdot \\hat \\ell = ${n(tri.brillo)}`} />
          </>
        ) : (
          <p className="nota">Cuando la casilla tenga vecinas medidas, aquí aparece el triángulo de superficie, su normal y su brillo.</p>
        )}
      </Paso>
    </aside>
  );
}
