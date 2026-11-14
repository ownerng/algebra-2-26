// Vista previa de la cámara con una cuadrícula para encuadrar el objeto y el
// recuadro central donde el servidor lo busca. Dos modos: una sola foto (el
// procesamiento alto/bajo decide cómo se reconstruye) o calibrar la cámara con el tablero.

import { useEffect, useState } from 'react';
import { MARCO, VIDEO, estadoCamara, orientarCamara } from '../lib/api.ts';
import { Calibracion } from './Calibracion.tsx';

export type Modo = 'una' | 'calibrar';

interface Props {
  conexion: number;
  procesando: string;          // texto mientras el servidor trabaja ('' si no)
  error: string;
  modo: Modo;
  cambiarModo: (m: Modo) => void;
  tomarFoto: (foto?: Blob) => void;   // sin foto: el cuadro actual de la cámara
}

export function Encuadre(p: Props) {
  const [proporcion, setProporcion] = useState(4 / 3);
  const [giro, setGiro] = useState({ espejo: false, voltear: false });
  useEffect(() => { estadoCamara().then((e) => setGiro({ espejo: e.espejo, voltear: e.voltear })).catch(() => {}); }, [p.conexion]);
  const girar = (cambio: Partial<typeof giro>) => {
    const nuevo = { ...giro, ...cambio };
    setGiro(nuevo);
    orientarCamara(nuevo.espejo, nuevo.voltear).catch(() => {});
  };
  const [x0, y0, x1, y1] = MARCO.map((f) => f * 100) as [number, number, number, number];
  const esquina = 4;
  const ocupado = !!p.procesando;

  return (
    <div className="encuadre">
      <div className="selector" role="group" aria-label="Modo de captura">
        <button type="button" aria-pressed={p.modo === 'una'} disabled={ocupado} onClick={() => p.cambiarModo('una')}>1 foto</button>
        <button type="button" aria-pressed={p.modo === 'calibrar'} disabled={ocupado} onClick={() => p.cambiarModo('calibrar')}>Calibrar</button>
      </div>

      <div className="selector" role="group" aria-label="Orientación de la imagen">
        <button type="button" aria-pressed={giro.espejo} onClick={() => girar({ espejo: !giro.espejo })}>Espejo ⇋</button>
        <button type="button" aria-pressed={giro.voltear} onClick={() => girar({ voltear: !giro.voltear })}>Voltear ⇅</button>
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
          {ocupado && <div className="procesando"><span className="giro" aria-hidden="true" />{p.procesando}</div>}
        </div>
      </div>

      <div className="acciones">
        {p.modo === 'calibrar' ? <Calibracion /> : (
          <>
            {p.error && <p className="aviso" role="alert">{p.error}</p>}
            <p className="pista">Pon el objeto dentro del recuadro</p>
            <div className="fila-botones">
              <label className="boton" aria-disabled={ocupado}>
                Subir foto
                <input type="file" accept="image/*" hidden disabled={ocupado}
                  onChange={(ev) => { const f = ev.currentTarget.files?.[0]; ev.currentTarget.value = ''; if (f) p.tomarFoto(f); }} />
              </label>
              <button type="button" className="boton disparo" disabled={ocupado} onClick={() => p.tomarFoto()}>{ocupado ? 'Procesando…' : 'Tomar foto'}</button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
