// Hoja de cuaderno con las matrices de la lectura actual, escritas con KaTeX.

import katex from 'katex';
import { useMemo } from 'react';
import { det2, procesarLectura, rotacion, type Matriz, type Vector } from '../lib/matematica.ts';
import { useEscaner } from '../lib/escaner.ts';

const n = (v: number) => {
  const x = Math.abs(v) < 5e-4 ? 0 : v;
  return (Math.abs(x) >= 10 ? x.toFixed(1) : x.toFixed(3)).replace('.', '{,}');
};
const mat = (M: Matriz) => `\\begin{bmatrix}${M.map((f) => f.map(n).join(' & ')).join(' \\\\ ')}\\end{bmatrix}`;
const col = (v: Vector) => mat(v.map((x) => [x]));
const marcar = (tex: string) => `\\colorbox{#ffe08a}{$${tex}$}`;

function Formula({ tex }: { tex: string }) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: true, throwOnError: false, strict: false }),
    [tex],
  );
  return <div className="formula" dangerouslySetInnerHTML={{ __html: html }} />;
}

export function Cuaderno() {
  const e = useEscaner(140);
  const u = e.ultimaValida;
  const pose2 = e.poses[2];

  const calculo = u && procesarLectura({ r: u.r, thetaServo: u.thetaServo, d: e.d, invertir: e.invertir, pose: e.poses[u.posicion] });
  const R2 = rotacion(pose2.alpha);
  const [c, s] = [R2[0]![0]!, R2[1]![0]!];

  return (
    <aside className="cuaderno" aria-label="Cálculo de matrices en tiempo real">
      <header className="cuaderno-cabeza">
        <h2>Cálculo en vivo</h2>
        <span className="ritmo">{e.lecturasPorSegundo()} lecturas/s</span>
      </header>

      <section className="paso">
        <h3><span className="numero">1</span>Rotación: polares → cartesianas</h3>
        <Formula tex={'\\begin{bmatrix} x_L \\\\ y_L \\end{bmatrix} = \\begin{bmatrix} \\cos\\theta & -\\sin\\theta \\\\ \\sin\\theta & \\cos\\theta \\end{bmatrix}\\begin{bmatrix} r + d \\\\ 0 \\end{bmatrix}'} />
        {calculo && u ? (
          <>
            <p className="dato">Posición {u.posicion} · θ = {calculo.theta}° · r = {u.r} mm · d = {e.d} mm</p>
            <Formula tex={`${marcar(col(calculo.local))} = ${mat(calculo.R)}${col(calculo.vectorPolar)}`} />
            <p className="nota">
              ‖(x<sub>L</sub>, y<sub>L</sub>)‖ = {n(Math.hypot(calculo.local[0]!, calculo.local[1]!)).replace('{,}', ',')} mm = r + d.
              La rotación cambia la dirección, no la distancia.
            </p>
          </>
        ) : (
          <p className="nota">Esperando la primera lectura con eco…</p>
        )}
      </section>

      <section className="paso">
        <h3><span className="numero">2</span>Matriz homogénea: local → global</h3>
        <Formula tex={'\\begin{bmatrix} x_G \\\\ y_G \\\\ 1 \\end{bmatrix} = \\begin{bmatrix} \\cos\\alpha & -\\sin\\alpha & d_x \\\\ \\sin\\alpha & \\cos\\alpha & d_y \\\\ 0 & 0 & 1 \\end{bmatrix}\\begin{bmatrix} x_L \\\\ y_L \\\\ 1 \\end{bmatrix}'} />
        {calculo && u && (
          <>
            <p className="dato">
              Pose de P{u.posicion}: α = {e.poses[u.posicion].alpha}°, d<sub>x</sub> = {e.poses[u.posicion].dx} mm, d<sub>y</sub> = {e.poses[u.posicion].dy} mm
            </p>
            <Formula tex={`${marcar(col(calculo.global))} = ${mat(calculo.T)}${col(calculo.homLocal)}`} />
            <p className="nota">
              {u.posicion === 1
                ? 'La posición 1 define el sistema global: T = I y el punto no cambia.'
                : `El punto medido desde P2 cae en (${n(calculo.global[0]!).replace('{,}', ',')}; ${n(calculo.global[1]!).replace('{,}', ',')}) mm del mapa, junto a los de P1.`}
            </p>
          </>
        )}
      </section>

      <section className="paso">
        <h3><span className="numero">3</span>Cambio de base</h3>
        <p className="nota">Las columnas de R(α) son los ejes de P2 escritos en la base global: las flechas roja y verde del radar.</p>
        <Formula tex={`R(${pose2.alpha}^\\circ) = \\begin{bmatrix} \\textcolor{#c2412d}{${n(c)}} & \\textcolor{#3f7d20}{${n(-s)}} \\\\ \\textcolor{#c2412d}{${n(s)}} & \\textcolor{#3f7d20}{${n(c)}} \\end{bmatrix}`} />
        <p className="nota">
          det R = {n(det2(R2)).replace('{,}', ',')} y x₂ · y₂ = {n(c * -s + s * c).replace('{,}', ',')}: base ortonormal, por eso el mapa no se deforma al rotarlo.
        </p>
      </section>
    </aside>
  );
}
