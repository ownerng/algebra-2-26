// Hoja de cuaderno: el cálculo de la última foto desglosado paso a paso, con
// los números de un píxel del objeto (el del centro de la silueta).

import katex from 'katex';
import { useMemo } from 'react';
import { cruz, matrizVista, normalizar, punto, sub, type Vec3 } from '../lib/algebra3d.ts';
import { multiplicarMatVec } from '../lib/matematica.ts';
import { escaneo } from '../lib/escaneo.ts';
import { useCamara } from '../lib/camara.ts';
import { aPlano, LUZ } from './Vista3D.tsx';

const n = (v: number, dec?: number) => {
  const x = Math.abs(v) < 5e-4 ? 0 : v;
  const d = dec ?? (Math.abs(x) >= 100 ? 1 : Math.abs(x) >= 10 ? 2 : 3);
  return x.toFixed(d).replace('.', '{,}');
};
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
  const c = useCamara(300);
  const m = c.modelo;

  if (!m) {
    return (
      <aside className="cuaderno" aria-label="Cálculos">
        <header className="cuaderno-cabeza"><h2>De la foto al objeto 3D</h2></header>
        <p className="nota">
          Aquí se desglosa, con los números de la última foto, cómo se separa el objeto del fondo, cómo se obtiene su tamaño
          real con la matriz de la cámara y la distancia del sensor, y cómo se gira y se sombrea en 3D.
        </p>
        <p className="nota">Toma una foto para ver los cálculos.</p>
      </aside>
    );
  }

  // Píxel de referencia: el más alejado del borde de la silueta (el centro del objeto)
  let ref = 0;
  for (let k = 1; k < m.altura.length; k++) if (m.altura[k]! > m.altura[ref]!) ref = k;
  const u = ref % m.ancho, v = (ref - u) / m.ancho;
  const p = [m.rgb[ref * 4]!, m.rgb[ref * 4 + 1]!, m.rgb[ref * 4 + 2]!];
  const d = Math.hypot(p[0]! - m.fondo[0], p[1]! - m.fondo[1], p[2]! - m.fondo[2]);
  const f = m.K[0]![0]!, cx = m.K[0]![2]!;
  const du = m.caja.u1 - m.caja.u0 + 1;
  const X0 = ((m.caja.u0 - cx) * m.Z) / f, X1 = ((m.caja.u1 + 1 - cx) * m.Z) / f;
  const h = m.altura[ref]!;

  const V = matrizVista(escaneo.vista.theta, escaneo.vista.phi);
  const [px, py] = aPlano(m, u, v);
  const P: Vec3 = [px, py, h];
  const vista = multiplicarMatVec(V, P);

  // Triángulo con dos vecinos a 3 píxeles
  const paso = 3;
  const punto3 = (uu: number, vv: number): Vec3 => { const [x, y] = aPlano(m, uu, vv); return [x, y, m.altura[vv * m.ancho + uu]!]; };
  const B = punto3(Math.min(m.ancho - 1, u + paso), v), C = punto3(u, Math.max(0, v - paso));
  const U = sub(B, P), W = sub(C, P);
  let N = cruz(U, W);
  if (N[2] < 0) N = [-N[0], -N[1], -N[2]];
  const brillo = Math.max(0, punto(normalizar(N), normalizar(LUZ)));

  return (
    <aside className="cuaderno" aria-label="Cálculos">
      <header className="cuaderno-cabeza"><h2>De la foto al objeto 3D</h2></header>
      <p className="dato">Números de la última foto, en el píxel ({u}, {v}) del centro del objeto.</p>

      <Paso numero={1} titulo="La foto es una matriz">
        <p className="nota">
          La foto reducida tiene {m.ancho} × {m.alto} píxeles. Cada píxel es un vector de ℝ³ con su rojo, verde y azul.
          Este píxel vale:
        </p>
        <Formula tex={`\\vec p = ${col(p, 0)}`} />
      </Paso>

      <Paso numero={2} titulo="Separar el objeto del fondo">
        <p className="nota">El color del fondo es la mediana de los píxeles del borde de la foto. Qué tan distinto es el píxel del fondo lo dice la norma de la resta:</p>
        <Formula tex={`\\|\\vec p - \\vec f\\| = \\left\\| ${col(p, 0)} - ${col(m.fondo, 0)} \\right\\| = ${n(d, 1)}`} />
        <p className="nota">
          El umbral de Otsu, que separa mejor los dos grupos del histograma, salió en {m.umbral}. Como {n(d, 1).replace('{,}', ',')} {d > m.umbral ? '>' : '≤'} {m.umbral},
          el píxel es del objeto. La silueta tiene {m.areaPx} píxeles.
        </p>
      </Paso>

      <Paso numero={3} titulo="Tamaño real: matriz de la cámara">
        <p className="nota">
          La cámara proyecta un punto (X, Y, Z) en el píxel (u, v) con la matriz K, donde f es la distancia focal en píxeles
          (sale del ángulo de visión de {c.anguloVision}°):
        </p>
        <Formula tex={`\\begin{bmatrix} u \\\\ v \\\\ 1 \\end{bmatrix} \\sim ${mat(m.K, 1)}\\begin{bmatrix} X \\\\ Y \\\\ Z \\end{bmatrix}`} />
        <p className="nota">
          {m.zDelSensor ? `El sensor midió Z = ${Math.round(m.Z)} mm.` : `Sin sensor, se supone Z = ${Math.round(m.Z)} mm.`} Despejando,
          X = (u − cₓ)·Z / f. El objeto va de u = {m.caja.u0} a u = {m.caja.u1 + 1}:
        </p>
        <Formula tex={`\\text{ancho} = X_1 - X_0 = \\frac{${du}\\cdot ${n(m.Z, 0)}}{${n(f, 1)}} = ${n(X1 - X0, 1)}\\ \\text{mm}`} />
      </Paso>

      <Paso numero={4} titulo="Volumen estimado">
        <p className="nota">
          Una foto no trae profundidad, así que el volumen se estima: la altura crece con la distancia al borde de la
          silueta. En el centro del objeto, con base de {c.grosor} mm:
        </p>
        <Formula tex={`h = ${n(c.grosor, 1)} + h_{\\max}\\sqrt{d/d_{\\max}} = ${n(h, 1)}\\ \\text{mm}`} />
      </Paso>

      <Paso numero={5} titulo="Girar y proyectar la vista 3D">
        <p className="nota">Cada punto se rota con V = Rx(φ)·Rz(θ) y se proyecta con Π, que descarta la profundidad:</p>
        <Formula tex={`V = ${mat(V)},\\qquad \\Pi = \\begin{bmatrix} 1 & 0 & 0 \\\\ 0 & 0 & 1 \\end{bmatrix}`} />
        <Formula tex={`\\Pi\\, V ${col(P, 1)} = \\Pi ${col(vista, 1)} = ${col([vista[0]!, vista[2]!], 1)}`} />
      </Paso>

      <Paso numero={6} titulo="Sombrear la superficie">
        <p className="nota">Con dos puntos vecinos se forma un triángulo; su normal sale del producto vectorial y su brillo, del producto punto con la luz:</p>
        <Formula tex={`\\vec n = (B - A)\\times(C - A) = ${col(U, 1)}\\times${col(W, 1)} = ${col(N, 1)}`} />
        <Formula tex={`\\text{brillo} = \\hat n \\cdot \\hat \\ell = ${n(brillo)}`} />
      </Paso>
    </aside>
  );
}
