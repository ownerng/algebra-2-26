// Hoja de cuaderno: el cálculo de una lectura desglosado paso a paso, de la
// distancia del sensor al punto del mapa. Se puede congelar en una lectura
// para explicarla con calma.

import katex from 'katex';
import { useMemo, useState } from 'react';
import { det2, procesarLectura, rotacion } from '../lib/matematica.ts';
import { escaner, useEscaner, type Lectura } from '../lib/escaner.ts';

// Fila 1 de cada matriz y la coordenada x en un color; fila 2 y la y en otro
const FX = '#b5452a';
const FY = '#2f6f8f';

const n = (v: number, dec?: number) => {
  const x = Math.abs(v) < 5e-4 ? 0 : v;
  const d = dec ?? (Math.abs(x) >= 10 ? 1 : 3);
  return x.toFixed(d).replace('.', '{,}');
};
const sinComa = (tex: string) => tex.replace('{,}', ',');
const signo = (v: number, tex: string) => (v < 0 ? `- ${tex.replace('-', '')}` : `+ ${tex}`);

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
  const e = useEscaner(140);
  const [congelada, setCongelada] = useState<Lectura | null>(null);
  const u = congelada ?? e.ultimaValida;

  const congelar = () => {
    const l = congelada ? null : e.ultimaValida;
    setCongelada(l);
    escaner.resaltar(l);
  };

  if (!u) {
    return (
      <aside className="cuaderno" aria-label="Cálculo paso a paso">
        <header className="cuaderno-cabeza"><h2>De la lectura al mapa</h2></header>
        <p className="nota">
          Aquí se desglosa, con los números de cada medición, cómo una distancia del sensor se convierte en un punto
          del mapa: primero con la matriz de rotación R(θ) y después con la matriz homogénea T.
        </p>
        <p className="nota">Conecta el ESP32 o pulsa <b>Simular prototipo</b> para ver la primera lectura.</p>
      </aside>
    );
  }

  const pose = e.poses[u.posicion];
  const c = procesarLectura({ r: u.r, thetaServo: u.thetaServo, d: e.d, invertir: e.invertir, pose });
  const [cos, sin] = [c.R[0]![0]!, c.R[1]![0]!];
  const rd = u.r + e.d;
  const [xl, yl] = [c.local[0]!, c.local[1]!];
  const [xg, yg] = [c.global[0]!, c.global[1]!];
  const [ca, sa] = [c.T[0]![0]!, c.T[1]![0]!];
  const th = c.theta;
  const a = pose.alpha;

  return (
    <aside className="cuaderno" aria-label="Cálculo paso a paso">
      <header className="cuaderno-cabeza">
        <h2>De la lectura al mapa</h2>
        <button type="button" className={'congelar' + (congelada ? ' activo' : '')} onClick={congelar}>
          {congelada ? '▶ Seguir en vivo' : '❚❚ Congelar cálculo'}
        </button>
      </header>
      <p className="dato">
        {congelada ? 'Cálculo congelado en una lectura (marcada en el radar).' : `En vivo · ${e.lecturasPorSegundo()} lecturas/s`}
      </p>

      <Paso numero={1} titulo="La lectura del sensor">
        <p className="nota">
          Desde la posición {u.posicion}, el servo apunta a θ = {th}° y el sensor mide r = {u.r} mm. El lente está a
          d = {e.d} mm del eje de giro, así que el punto queda a r + d del eje:
        </p>
        <Formula tex={`r + d = ${u.r} + ${e.d} = ${rd}\\ \\text{mm}`} />
      </Paso>

      <Paso numero={2} titulo="Matriz de rotación R(θ)">
        <p className="nota">El vector (r + d, 0) apunta sobre el eje x. Para girarlo θ grados se usa:</p>
        <Formula tex={`R(${th}^\\circ) = \\begin{bmatrix} \\textcolor{${FX}}{\\cos ${th}^\\circ} & \\textcolor{${FX}}{-\\sin ${th}^\\circ} \\\\ \\textcolor{${FY}}{\\sin ${th}^\\circ} & \\textcolor{${FY}}{\\cos ${th}^\\circ} \\end{bmatrix} = \\begin{bmatrix} \\textcolor{${FX}}{${n(cos)}} & \\textcolor{${FX}}{${n(-sin)}} \\\\ \\textcolor{${FY}}{${n(sin)}} & \\textcolor{${FY}}{${n(cos)}} \\end{bmatrix}`} />
      </Paso>

      <Paso numero={3} titulo="Multiplicar: fila por columna">
        <p className="nota">Cada coordenada sale de multiplicar una fila de R por el vector y sumar:</p>
        <Formula tex={`\\begin{aligned}
          \\textcolor{${FX}}{x_L} &= \\textcolor{${FX}}{${n(cos)}}\\cdot ${rd} ${signo(-sin, `\\textcolor{${FX}}{${n(-sin)}}`)}\\cdot 0 = \\textcolor{${FX}}{${n(xl)}}\\ \\text{mm} \\\\
          \\textcolor{${FY}}{y_L} &= \\textcolor{${FY}}{${n(sin)}}\\cdot ${rd} ${signo(cos, `\\textcolor{${FY}}{${n(cos)}}`)}\\cdot 0 = \\textcolor{${FY}}{${n(yl)}}\\ \\text{mm}
        \\end{aligned}`} />
        <p className="nota">Comprobación: la rotación no cambia la distancia.</p>
        <Formula tex={`\\sqrt{${n(xl)}^2 + ${n(yl)}^2} = ${n(Math.hypot(xl, yl))} = r + d \\;\\checkmark`} />
      </Paso>

      <Paso numero={4} titulo="Llevar el punto al mapa con T">
        {u.posicion === 1 ? (
          <>
            <p className="nota">
              La posición 1 es el origen del mapa: no se movió ni giró (α = 0°, d<sub>x</sub> = d<sub>y</sub> = 0), así que
              T es la identidad y el punto queda igual.
            </p>
            <Formula tex={`\\begin{bmatrix} x_G \\\\ y_G \\\\ 1 \\end{bmatrix} = \\begin{bmatrix} 1 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1 \\end{bmatrix}\\begin{bmatrix} ${n(xl)} \\\\ ${n(yl)} \\\\ 1 \\end{bmatrix} = \\begin{bmatrix} ${n(xg)} \\\\ ${n(yg)} \\\\ 1 \\end{bmatrix}`} />
          </>
        ) : (
          <>
            <p className="nota">
              El aparato se movió d<sub>x</sub> = {pose.dx} mm, d<sub>y</sub> = {pose.dy} mm y giró α = {a}°. La matriz
              homogénea hace el giro y el desplazamiento en un solo producto:
            </p>
            <Formula tex={`T = \\begin{bmatrix} \\textcolor{${FX}}{${n(ca)}} & \\textcolor{${FX}}{${n(-sa)}} & \\textcolor{${FX}}{${n(pose.dx, 0)}} \\\\ \\textcolor{${FY}}{${n(sa)}} & \\textcolor{${FY}}{${n(ca)}} & \\textcolor{${FY}}{${n(pose.dy, 0)}} \\\\ 0 & 0 & 1 \\end{bmatrix}`} />
            <Formula tex={`\\begin{aligned}
              \\textcolor{${FX}}{x_G} &= ${n(ca)}\\cdot ${n(xl)} ${signo(-sa, n(-sa))}\\cdot ${n(yl)} + ${n(pose.dx, 0)}\\cdot 1 = \\textcolor{${FX}}{${n(xg)}} \\\\
              \\textcolor{${FY}}{y_G} &= ${n(sa)}\\cdot ${n(xl)} ${signo(ca, n(ca))}\\cdot ${n(yl)} + ${n(pose.dy, 0)}\\cdot 1 = \\textcolor{${FY}}{${n(yg)}}
            \\end{aligned}`} />
          </>
        )}
      </Paso>

      <Paso numero={5} titulo="Resultado">
        <p className="nota">
          El punto ({sinComa(n(xg))}; {sinComa(n(yg))}) mm se agrega al mapa. Lleva {e.puntos()[1].length + e.puntos()[2].length} puntos
          en total.
        </p>
        {u.posicion === 2 && (
          <p className="nota">
            det R(α) = {sinComa(n(det2(rotacion(a))))}: el giro no deforma el mapa, por eso los puntos de P2 encajan con los
            de P1 si α, d<sub>x</sub> y d<sub>y</sub> están bien medidos.
          </p>
        )}
      </Paso>
    </aside>
  );
}
