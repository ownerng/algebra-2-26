// La cuadrícula vista desde arriba: dónde poner la caja, qué casillas ya se
// midieron (color según la altura) y el botón para medir.

import { escaneo, useEscaneo } from '../lib/escaneo.ts';

const CELDA = 32;

export function Cuadricula() {
  const e = useEscaneo(60);
  const { nx, ny, paso, altura } = e.config;
  const { puntos } = e.geometria();
  const porCasilla = new Map(puntos.map((p) => [`${p.i},${p.j}`, p]));
  const maxH = Math.max(10, ...puntos.map((p) => p.h));
  const terminado = e.cursor >= e.total;
  const [ci, cj] = e.casillaActual;
  const conectado = e.tipoFuente !== 'ninguna';
  const vivo = e.vivo && performance.now() - e.vivo.t < 1500 ? e.vivo.r : null;
  const medidas = e.medidas.size;

  return (
    <section className="cuadricula" aria-label="Cuadrícula de escaneo">
      <header className="placa-titulo">
        <span>Cuadrícula sobre la mesa</span>
        <span>{medidas} de {e.total} casillas</span>
      </header>
      <div className="progreso" aria-hidden="true"><i style={{ width: `${(100 * medidas) / e.total}%` }} /></div>

      <div className="cuadricula-cuerpo">
        <svg viewBox={`-26 -8 ${nx * CELDA + 34} ${ny * CELDA + 30}`} role="img" aria-label="Casillas medidas y casilla actual">
          {Array.from({ length: ny }, (_, j) => Array.from({ length: nx }, (_, i) => {
            const p = porCasilla.get(`${i},${j}`);
            const t = p ? Math.min(1, Math.max(0, p.h) / maxH) : 0;
            const actual = !terminado && i === ci && j === cj;
            const res = e.resaltada && e.resaltada.i === i && e.resaltada.j === j;
            // y crece hacia arriba, como en la mesa vista desde arriba
            const yPant = (ny - 1 - j) * CELDA;
            return (
              <g key={`${i},${j}`} onClick={() => escaneo.irA(i, j)} className="casilla">
                <rect x={i * CELDA + 1} y={yPant + 1} width={CELDA - 2} height={CELDA - 2} rx="3"
                  className={p ? (p.mesa ? 'medida mesa' : 'medida') : 'vacia'}
                  style={p ? { fill: `rgb(${Math.round(70 + t * 185)},${Math.round(48 + t * 166)},${Math.round(20 + t * 100)})` } : undefined} />
                {p && <text x={i * CELDA + CELDA / 2} y={yPant + CELDA / 2 + 4} className="casilla-texto">{Math.round(Math.max(0, p.h))}</text>}
                {actual && <rect x={i * CELDA - 1} y={yPant - 1} width={CELDA + 2} height={CELDA + 2} rx="4" className="cursor" />}
                {res && <circle cx={i * CELDA + CELDA / 2} cy={yPant + CELDA / 2} r={CELDA / 2 - 2} className="resaltada" />}
              </g>
            );
          }))}
          {Array.from({ length: nx }, (_, i) => <text key={'x' + i} x={i * CELDA + CELDA / 2} y={ny * CELDA + 14} className="eje">{i}</text>)}
          {Array.from({ length: ny }, (_, j) => <text key={'y' + j} x={-12} y={(ny - 1 - j) * CELDA + CELDA / 2 + 4} className="eje">{j}</text>)}
          <text x={nx * CELDA + 4} y={ny * CELDA + 14} className="eje-nombre">i →</text>
          <text x={-22} y={-1} className="eje-nombre">j ↑</text>
        </svg>

        <div className="instruccion">
          {terminado ? (
            <>
              <strong>¡Cuadrícula completa!</strong>
              <span>Gira el objeto en la vista 3D y guárdalo en <b>Ajustes y archivos</b>.</span>
            </>
          ) : (
            <>
              <span className="paso-chico">Pon la caja en la casilla</span>
              <strong className="casilla-grande">({ci}, {cj})</strong>
              <span>x = {ci * paso} mm · y = {cj * paso} mm</span>
            </>
          )}
          <button type="button" className="tecla grande ambar medir" disabled={!conectado || e.midiendo || terminado} onClick={() => escaneo.medir()}>
            {e.midiendo ? 'Midiendo…' : '● Medir'}
          </button>
          <span className="ayuda">o la barra espaciadora, o el botón BOOT de la caja</span>
          <div className="fila-botones">
            <button type="button" className="tecla chica" disabled={!medidas} onClick={() => escaneo.deshacer()}>↶ Deshacer</button>
            <button type="button" className="tecla chica" disabled={!medidas} onClick={() => escaneo.reiniciar()}>Empezar de nuevo</button>
          </div>
          {conectado && (
            <p className="vivo">
              Debajo del sensor: <b>{vivo && vivo > 0 ? `${vivo} mm` : 'sin eco'}</b>
              {vivo && vivo > 0 ? <> → altura ≈ {Math.round(altura - vivo)} mm</> : null}
            </p>
          )}
        </div>
      </div>
      <p className="ayuda">
        Cada cuadro mide {paso} mm. El recorrido va en zigzag: la fila 0 de izquierda a derecha, la fila 1 de derecha a
        izquierda, y así. El número de cada casilla es la altura medida en mm; las casillas celestes del borde se usan
        para encontrar el plano de la mesa. Toca una casilla para ver su cálculo o volver a medirla.
      </p>
    </section>
  );
}
