// Vista previa de la cámara con una cuadrícula para encuadrar el objeto y el
// recuadro central donde el servidor lo busca. Dos modos: una sola foto, o
// cinco vistas (frontal, laterales, trasera y superior) para la envolvente visual.
// Si el tablero de marcadores se ve en todas las fotos, se pueden sumar vistas
// extra desde cualquier ángulo.

import { useState } from 'react';
import { MARCO, URL_TABLERO, VIDEO, VISTAS, type NombreVista } from '../lib/api.ts';

export type Modo = 'una' | 'vistas';

const FIJAS: Record<NombreVista, string> = {
  frontal: 'Frontal', izquierda: 'Lateral izquierdo', trasera: 'Trasera', derecha: 'Lateral derecho', superior: 'Superior',
};

export const NOMBRES = (n: string) => FIJAS[n as NombreVista] ?? `Extra ${n.slice('extra-'.length)}`;

const PISTAS: Record<NombreVista, string> = {
  frontal: 'De frente al objeto, a la altura de su centro.',
  izquierda: 'Muévete a tu izquierda hasta ver el objeto de lado, a la misma distancia.',
  trasera: 'Sigue rodeando el objeto hasta quedar detrás de él, mirando hacia donde estabas al principio.',
  derecha: 'Sigue rodeando el objeto hasta verlo por el otro lado.',
  superior: 'Desde arriba, mirando hacia abajo, con la parte de arriba del celular hacia el fondo.',
};
const PISTA_EXTRA = 'Desde cualquier otro ángulo, con el tablero a la vista.';

interface Props {
  conexion: number;
  procesando: string;          // texto mientras el servidor trabaja ('' si no)
  error: string;
  modo: Modo;
  cambiarModo: (m: Modo) => void;
  tomarFoto: () => void;
  recortes: Record<string, string>;
  tableros: Record<string, boolean>;   // vistas en que se detectó el tablero
  extras: string[];
  agregarExtra: () => void;
  paso: string | null;
  elegirPaso: (n: string) => void;
  tomarVista: () => void;
  reconstruir: () => void;
  reiniciar: () => void;
}

export function Encuadre(p: Props) {
  const [proporcion, setProporcion] = useState(4 / 3);
  const [x0, y0, x1, y1] = MARCO.map((f) => f * 100) as [number, number, number, number];
  const esquina = 4;
  const todas = [...VISTAS, ...p.extras];
  const tomadas = todas.filter((n) => p.recortes[n]).length;
  const completas = tomadas === todas.length;
  const conTablero = tomadas > 0 && todas.every((n) => !p.recortes[n] || p.tableros[n]);
  const puedeReconstruir = tomadas >= 2 && (!!p.recortes.frontal || conTablero);
  const ocupado = !!p.procesando;

  return (
    <div className="encuadre">
      <div className="selector" role="group" aria-label="Modo de captura">
        <button type="button" aria-pressed={p.modo === 'una'} disabled={ocupado} onClick={() => p.cambiarModo('una')}>1 foto</button>
        <button type="button" aria-pressed={p.modo === 'vistas'} disabled={ocupado} onClick={() => p.cambiarModo('vistas')}>{VISTAS.length} vistas</button>
      </div>

      <div className="cuadro-caja">
        <div className="cuadro" style={{ '--ar': proporcion } as React.CSSProperties}>
          <img key={p.conexion} src={`${VIDEO}?c=${p.conexion}`} alt="Imagen de la cámara"
            onLoad={(ev) => { const i = ev.currentTarget; if (i.naturalWidth) setProporcion(i.naturalWidth / i.naturalHeight); }} />
          <svg className="rejilla" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
            {[1, 2].map((k) => <line key={'v' + k} x1={(100 * k) / 3} y1={0} x2={(100 * k) / 3} y2={100} />)}
            {[1, 2].map((k) => <line key={'h' + k} x1={0} y1={(100 * k) / 3} x2={100} y2={(100 * k) / 3} />)}
            <rect className="marco" x={x0} y={y0} width={x1 - x0} height={y1 - y0} />
            <path className="esquinas" d={[
              `M${x0} ${y0 + esquina}V${y0}H${x0 + esquina}`, `M${x1 - esquina} ${y0}H${x1}V${y0 + esquina}`,
              `M${x0} ${y1 - esquina}V${y1}H${x0 + esquina}`, `M${x1 - esquina} ${y1}H${x1}V${y1 - esquina}`,
            ].join('')} />
            <path className="centro" d="M48.5 50H51.5M50 48.5V51.5" />
          </svg>
          {p.modo === 'vistas' && p.paso && <span className="etiqueta-vista">{NOMBRES(p.paso)}</span>}
          {ocupado && <div className="procesando"><span className="giro" aria-hidden="true" />{p.procesando}</div>}
        </div>
      </div>

      <div className="acciones">
        {p.modo === 'vistas' && (
          <div className="vistas" role="group" aria-label="Vistas">
            {todas.map((n, k) => (
              <button key={n} type="button" className={'vista' + (p.paso === n ? ' actual' : '') + (p.recortes[n] ? ' lista' : '')}
                disabled={ocupado} onClick={() => p.elegirPaso(n)}
                title={(p.recortes[n] ? `Repetir: ${NOMBRES(n)}` : NOMBRES(n)) + (p.tableros[n] ? ' (tablero detectado)' : '')}>
                {p.recortes[n] ? <img src={p.recortes[n]} alt="" /> : <span className="numero">{k + 1}</span>}
                <small>{NOMBRES(n)}{p.tableros[n] ? ' ▦' : ''}</small>
              </button>
            ))}
          </div>
        )}
        {p.error && <p className="aviso" role="alert">{p.error}</p>}
        <p className="pista">
          {p.modo === 'una' ? 'Pon el objeto dentro del recuadro'
            : p.paso ? PISTAS[p.paso as NombreVista] ?? PISTA_EXTRA : 'Listo: ya puedes reconstruir (o toca una vista para repetirla).'}
        </p>
        {p.modo === 'vistas' && (
          <p className="pista">
            {conTablero ? 'Tablero detectado: poses exactas y medidas reales en mm. ' : 'Más precisión: pon el objeto sobre el '}
            <a href={URL_TABLERO} target="_blank" rel="noreferrer">{conTablero ? 'Ver tablero' : 'tablero impreso'}</a>.
          </p>
        )}
        <div className="fila-botones">
          {p.modo === 'una' && (
            <button type="button" className="boton disparo" disabled={ocupado} onClick={p.tomarFoto}>{ocupado ? 'Procesando…' : 'Tomar foto'}</button>
          )}
          {p.modo === 'vistas' && (
            <>
              {tomadas > 0 && <button type="button" className="boton" disabled={ocupado} onClick={p.reiniciar}>Empezar de nuevo</button>}
              {p.paso && (
                <button type="button" className={'boton' + (completas ? '' : ' disparo')} disabled={ocupado} onClick={p.tomarVista}>
                  Tomar foto · {NOMBRES(p.paso)} ({todas.indexOf(p.paso) + 1}/{todas.length})
                </button>
              )}
              {conTablero && !p.paso && (
                <button type="button" className="boton" disabled={ocupado} onClick={p.agregarExtra}>+ Vista extra</button>
              )}
              {puedeReconstruir && (
                <button type="button" className={'boton' + (completas ? ' disparo' : '')} disabled={ocupado} onClick={p.reconstruir}>
                  Reconstruir 3D{completas ? '' : ` (${tomadas} vistas)`}
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
