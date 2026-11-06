// Modo habitación: una foto de todo el cuarto (de la cámara o subida desde el
// equipo) y el servidor la convierte en una maqueta 3D (servidor/escaner/habitacion.py).

import { useState } from 'react';
import { VIDEO } from '../lib/api.ts';

interface Props {
  conectada: boolean;
  conexion: number;
  procesando: string;
  error: string;
  enviar: (foto?: Blob) => void;   // sin foto: el cuadro actual de la cámara
}

export function Habitacion(p: Props) {
  const [proporcion, setProporcion] = useState(16 / 9);
  const ocupado = !!p.procesando;

  return (
    <div className="encuadre">
      <div className="cuadro-caja">
        {p.conectada ? (
          <div className="cuadro" style={{ '--ar': proporcion } as React.CSSProperties}>
            <img key={p.conexion} src={`${VIDEO}?c=${p.conexion}`} alt="Imagen de la cámara"
              onLoad={(ev) => { const i = ev.currentTarget; if (i.naturalWidth) setProporcion(i.naturalWidth / i.naturalHeight); }} />
            <svg className="rejilla" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              {[1, 2].map((k) => <line key={'v' + k} x1={(100 * k) / 3} y1={0} x2={(100 * k) / 3} y2={100} />)}
              {[1, 2].map((k) => <line key={'h' + k} x1={0} y1={(100 * k) / 3} x2={100} y2={(100 * k) / 3} />)}
            </svg>
            {ocupado && <div className="procesando"><span className="giro" aria-hidden="true" />{p.procesando}</div>}
          </div>
        ) : (
          <p className="pista">{ocupado ? p.procesando : 'Sin cámara: sube una foto de la habitación.'}</p>
        )}
      </div>

      <div className="acciones">
        {p.error && <p className="aviso" role="alert">{p.error}</p>}
        <p className="pista">Desde una esquina o la puerta, celular horizontal a la altura del pecho, que se vea el piso.</p>
        <div className="fila-botones">
          <label className={'boton' + (p.conectada ? '' : ' disparo')} aria-disabled={ocupado}>
            Subir foto
            <input type="file" accept="image/*" hidden disabled={ocupado}
              onChange={(ev) => { const f = ev.currentTarget.files?.[0]; ev.currentTarget.value = ''; if (f) p.enviar(f); }} />
          </label>
          {p.conectada && (
            <button type="button" className="boton disparo" disabled={ocupado} onClick={() => p.enviar()}>
              {ocupado ? 'Procesando…' : 'Tomar foto'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
