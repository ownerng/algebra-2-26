// La cuadrícula vista desde arriba. Con un solo botón empieza el escaneo
// continuo: una marca recorre cada fila y la caja se desliza siguiéndola.

import { CUENTA_MS, escaneo, useEscaneo } from '../lib/escaneo.ts';

const CELDA = 32;
const VELOCIDADES = [15, 20, 30];

export function Cuadricula() {
  const e = useEscaneo(40);
  const { nx, ny, paso } = e.config;
  const { puntos } = e.geometria();
  const porCasilla = new Map(puntos.map((p) => [`${p.i},${p.j}`, p]));
  const maxH = Math.max(10, ...puntos.map((p) => p.h));
  const conectado = e.tipoFuente !== 'ninguna';
  const b = e.barrido;
  const vivo = e.vivo && performance.now() - e.vivo.t < 1500 ? e.vivo.r : null;
  const segundosFila = Math.round(e.largoFila / e.velocidad);

  // Marca del escaneo continuo en coordenadas del dibujo
  let marca: { x: number; y: number; derecha: boolean } | null = null;
  if (b && b.fase !== 'fin') {
    const [x, y] = e.posicionMarca();
    marca = { x: (x / paso) * CELDA + CELDA / 2, y: (ny - 1 - y / paso) * CELDA + CELDA / 2, derecha: b.fila % 2 === 0 };
  }
  const cuenta = b?.fase === 'cuenta' ? Math.max(1, Math.ceil((CUENTA_MS - (performance.now() - b.tFase)) / 1000)) : 0;

  return (
    <section className="cuadricula" aria-label="Cuadrícula de escaneo">
      <header className="placa-titulo">
        <span>Cuadrícula sobre la mesa</span>
        <span>{b && b.fase !== 'fin' ? `fila ${b.fila + 1} de ${ny}` : `${e.medidas.size} de ${e.total} casillas`}</span>
      </header>
      <div className="progreso" aria-hidden="true">
        <i style={{ width: `${b ? (100 * (b.fase === 'fin' ? ny : b.fila + e.recorrido() / Math.max(1, e.largoFila))) / ny : (100 * e.medidas.size) / e.total}%` }} />
      </div>

      <div className="cuadricula-cuerpo">
        <svg viewBox={`-26 -8 ${nx * CELDA + 34} ${ny * CELDA + 30}`} role="img" aria-label="Casillas medidas y posición de la caja">
          {Array.from({ length: ny }, (_, j) => Array.from({ length: nx }, (_, i) => {
            const p = porCasilla.get(`${i},${j}`);
            const t = p ? Math.min(1, Math.max(0, p.h) / maxH) : 0;
            const res = e.resaltada && e.resaltada.i === i && e.resaltada.j === j;
            const yPant = (ny - 1 - j) * CELDA;
            const filaActiva = b && b.fase !== 'fin' && b.fila === j;
            return (
              <g key={`${i},${j}`} onClick={() => escaneo.irA(i, j)} className="casilla">
                <rect x={i * CELDA + 1} y={yPant + 1} width={CELDA - 2} height={CELDA - 2} rx="3"
                  className={(p ? (p.mesa ? 'medida mesa' : 'medida') : 'vacia') + (filaActiva ? ' fila-activa' : '')}
                  style={p ? { fill: `rgb(${Math.round(70 + t * 185)},${Math.round(48 + t * 166)},${Math.round(20 + t * 100)})` } : undefined} />
                {p && <text x={i * CELDA + CELDA / 2} y={yPant + CELDA / 2 + 4} className="casilla-texto">{Math.round(Math.max(0, p.h))}</text>}
                {res && <circle cx={i * CELDA + CELDA / 2} cy={yPant + CELDA / 2} r={CELDA / 2 - 2} className="resaltada" />}
              </g>
            );
          }))}
          {marca && (
            <g className="marca-caja" transform={`translate(${marca.x} ${marca.y})`}>
              <rect x={-CELDA * 0.55} y={-CELDA * 0.55} width={CELDA * 1.1} height={CELDA * 1.1} rx="5" />
              <path d={marca.derecha ? 'M 22 -6 L 32 0 L 22 6 Z' : 'M -22 -6 L -32 0 L -22 6 Z'} />
            </g>
          )}
          {Array.from({ length: nx }, (_, i) => <text key={'x' + i} x={i * CELDA + CELDA / 2} y={ny * CELDA + 14} className="eje">{i}</text>)}
          {Array.from({ length: ny }, (_, j) => <text key={'y' + j} x={-12} y={(ny - 1 - j) * CELDA + CELDA / 2 + 4} className="eje">{j}</text>)}
          <text x={nx * CELDA + 4} y={ny * CELDA + 14} className="eje-nombre">x →</text>
          <text x={-22} y={-1} className="eje-nombre">y ↑</text>
        </svg>

        <div className="instruccion">
          {!b && (
            <>
              <span className="paso-chico">Pon el objeto en el centro y la caja en la esquina (0, 0)</span>
              <button type="button" className="tecla grande ambar medir" disabled={!conectado} onClick={() => escaneo.escanearContinuo()}>▶ Escanear</button>
              <span className="ayuda">Después solo desliza la caja siguiendo la marca. Cada fila dura unos {segundosFila} s.</span>
            </>
          )}
          {b?.fase === 'cuenta' && (
            <>
              <span className="paso-chico">Fila {b.fila + 1} de {ny}: pon la caja al {b.fila % 2 === 0 ? 'inicio izquierdo' : 'inicio derecho'}</span>
              <strong className="casilla-grande">{cuenta}</strong>
              <button type="button" className="tecla chica" onClick={() => escaneo.detenerContinuo()}>■ Detener</button>
            </>
          )}
          {b?.fase === 'fila' && (
            <>
              <span className="paso-chico">Fila {b.fila + 1} de {ny}</span>
              <strong className="casilla-grande">{b.fila % 2 === 0 ? 'desliza →' : '← desliza'}</strong>
              <span className="ayuda">sigue la marca a velocidad constante</span>
              <button type="button" className="tecla chica" onClick={() => escaneo.detenerContinuo()}>■ Detener</button>
            </>
          )}
          {b?.fase === 'fin' && (
            <>
              <strong>¡Escaneo completo!</strong>
              <span>Gira la figura en 3D y guárdala en <b>Guardar y exportar</b>.</span>
              <button type="button" className="tecla grande ambar medir" onClick={() => escaneo.escanearContinuo()}>▶ Escanear otra vez</button>
            </>
          )}
          <div className="velocidades" role="radiogroup" aria-label="Velocidad de la marca">
            {VELOCIDADES.map((v) => (
              <button key={v} type="button" role="radio" aria-checked={e.velocidad === v} disabled={!!b && b.fase !== 'fin'}
                className={'alcance' + (e.velocidad === v ? ' activo' : '')} onClick={() => escaneo.cambiar({ velocidad: v })}>
                {v === 15 ? 'lento' : v === 20 ? 'normal' : 'rápido'}
              </button>
            ))}
          </div>
          {conectado && (
            <p className="vivo">
              Debajo del sensor: <b>{vivo && vivo > 0 ? `${vivo} mm` : 'sin eco'}</b>
            </p>
          )}
        </div>
      </div>
      <p className="ayuda">
        Cada cuadro mide {paso} mm. El escaneo va en zigzag: la fila 0 de izquierda a derecha, la fila 1 de derecha a
        izquierda, y así. El número de cada casilla es la altura en mm. Para repetir una casilla, tócala y usa{' '}
        <button type="button" className="enlace" disabled={!conectado || e.midiendo || !!(b && b.fase !== 'fin')} onClick={() => escaneo.medir()}>medir solo esta casilla</button>.
      </p>
    </section>
  );
}
