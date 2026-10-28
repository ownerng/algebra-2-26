// Control del escaneo sobre la figura 3D: un botón, la barra que indica dónde
// debe ir la caja y un mapa pequeño de la cuadrícula.

import { CUENTA_MS, escaneo, useEscaneo } from '../lib/escaneo.ts';

const MUY_CERCA_MM = 60;

function MiniMapa() {
  const e = useEscaneo(80);
  const { nx, ny, paso } = e.config;
  const { puntos } = e.geometria();
  const maxH = Math.max(10, ...puntos.map((p) => p.h));
  const C = 12;
  const b = e.barrido;
  let marca: [number, number] | null = null;
  if (b && b.fase !== 'fin') {
    const [x, y] = e.posicionMarca();
    marca = [(x / paso) * C + C / 2, (ny - 1 - y / paso) * C + C / 2];
  }
  return (
    <svg className="minimapa" viewBox={`-2 -2 ${nx * C + 4} ${ny * C + 4}`} aria-label="Mapa de la cuadrícula">
      {Array.from({ length: ny }, (_, j) => Array.from({ length: nx }, (_, i) => {
        const p = puntos.find((q) => q.i === i && q.j === j);
        const t = p ? Math.min(1, Math.max(0, p.h) / maxH) : 0;
        return (
          <rect key={`${i},${j}`} x={i * C + 0.5} y={(ny - 1 - j) * C + 0.5} width={C - 1} height={C - 1} rx="1.5"
            onClick={() => escaneo.irA(i, j)}
            style={{ fill: p ? `rgb(${Math.round(70 + t * 185)},${Math.round(48 + t * 166)},${Math.round(20 + t * 100)})` : '#1c1710' }} />
        );
      }))}
      {marca && <circle cx={marca[0]} cy={marca[1]} r={C * 0.55} className="minimapa-marca" />}
    </svg>
  );
}

export function Control() {
  const e = useEscaneo(40);
  const conectado = e.tipoFuente !== 'ninguna';
  const b = e.barrido;
  const vivo = e.vivo && performance.now() - e.vivo.t < 1500 ? e.vivo.r : null;
  const muyCerca = !b && e.tipoFuente === 'serial' && vivo !== null && vivo > 0 && vivo < MUY_CERCA_MM;
  const avance = b?.fase === 'fila' ? e.recorrido() / Math.max(1, e.largoFila) : 0;
  const haciaDerecha = (b?.fila ?? 0) % 2 === 0;
  const cuenta = b?.fase === 'cuenta' ? Math.max(1, Math.ceil((CUENTA_MS - (performance.now() - b.tFase)) / 1000)) : 0;

  return (
    <>
      {e.medidas.size > 0 || b ? <MiniMapa /> : null}

      <div className="control">
        {!conectado && (
          <div className="control-centro">
            <p className="control-titulo">Escáner 3D</p>
            <div className="fila-botones centrados">
              <button type="button" className="tecla grande" onClick={() => escaneo.conectar('serial')}>Conectar ESP32</button>
              <button type="button" className="tecla grande ambar" onClick={() => escaneo.conectar('simulacion')}>▶ Simular</button>
            </div>
          </div>
        )}

        {conectado && !b && (
          <div className="control-centro">
            <p className="distancia">{vivo && vivo > 0 ? <>{vivo}<small> mm</small></> : <small>sin eco</small>}</p>
            <p className="control-ayuda">distancia del sensor a lo que tiene debajo</p>
            {muyCerca ? (
              <p className="aviso-cerca">La caja está a solo {Math.round(vivo! / 10)} cm de la mesa. Súbela a unos 15 cm sobre la cartulina.</p>
            ) : (
              <p className="control-ayuda">Pon el objeto en el centro de la cartulina y la caja en la esquina de abajo a la izquierda.</p>
            )}
            <button type="button" className="tecla enorme ambar" onClick={() => escaneo.escanearContinuo()}>▶ Escanear</button>
          </div>
        )}

        {b && b.fase !== 'fin' && (
          <div className="control-barrido">
            <div className="control-fila">
              <span>Fila {b.fila + 1} de {e.config.ny}</span>
              <span>{b.fase === 'cuenta' ? `empieza en ${cuenta}…` : haciaDerecha ? 'desliza la caja →' : '← desliza la caja'}</span>
              <button type="button" className="tecla chica" onClick={() => escaneo.detenerContinuo()}>■ Detener</button>
            </div>
            <div className="riel" aria-label="Dónde debe ir la caja en la fila">
              <div className="riel-marca" style={{ left: `calc(${haciaDerecha ? avance : 1 - avance} * (100% - 64px))` }}>
                {b.fase === 'cuenta' ? cuenta : haciaDerecha ? '→' : '←'}
              </div>
            </div>
          </div>
        )}

        {b?.fase === 'fin' && (
          <div className="control-centro">
            <p className="control-titulo">¡Escaneo completo!</p>
            <p className="control-ayuda">Gira la figura con el mouse. Para guardarla, abre Ajustes.</p>
            <button type="button" className="tecla grande ambar" onClick={() => escaneo.escanearContinuo()}>▶ Escanear otra vez</button>
          </div>
        )}
      </div>
    </>
  );
}
