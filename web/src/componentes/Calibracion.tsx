// Calibración de la cámara con el tablero impreso: varias fotos desde ángulos
// distintos y el servidor calcula K y la distorsión del lente (calibracion.py).

import { useEffect, useState } from 'react';
import { borrarCalibracion, calcularCalibracion, estadoCalibracion, fotoCalibracion, URL_TABLERO, type EstadoCalibracion } from '../lib/api.ts';

export function Calibracion() {
  const [estado, setEstado] = useState<EstadoCalibracion | null>(null);
  const [ocupado, setOcupado] = useState('');
  const [error, setError] = useState('');

  useEffect(() => { estadoCalibracion().then(setEstado).catch(() => {}); }, []);

  const hacer = async (texto: string, tarea: () => Promise<EstadoCalibracion>) => {
    setOcupado(texto); setError('');
    try { setEstado(await tarea()); } catch (e) { setError((e as Error).message); }
    setOcupado('');
  };

  const fotos = estado?.fotos ?? 0;
  const minimo = estado?.minimo ?? 5;

  return (
    <>
      <p className="pista">
        Muestra el <a href={URL_TABLERO} target="_blank" rel="noreferrer">tablero impreso</a> desde ángulos distintos:
        inclinado, cerca, lejos y en cada esquina de la imagen. Mínimo {minimo} fotos; 10–15 es mejor.
      </p>
      {estado?.calibrada && (
        <p className="pista">Calibrada ({estado.resolucion}): f = {estado.f_px} px, error {estado.error_px} px.</p>
      )}
      {error && <p className="aviso" role="alert">{error}</p>}
      <div className="fila-botones">
        {(fotos > 0 || estado?.calibrada) && (
          <button type="button" className="boton" disabled={!!ocupado} onClick={() => void hacer('Borrando…', borrarCalibracion)}>Borrar</button>
        )}
        <button type="button" className={'boton' + (fotos >= minimo ? '' : ' disparo')} disabled={!!ocupado}
          onClick={() => void hacer('Buscando el tablero…', fotoCalibracion)}>
          {ocupado === 'Buscando el tablero…' ? ocupado : `Capturar (${fotos})`}
        </button>
        {fotos >= minimo && (
          <button type="button" className="boton disparo" disabled={!!ocupado} onClick={() => void hacer('Calculando…', calcularCalibracion)}>
            {ocupado === 'Calculando…' ? ocupado : 'Calcular'}
          </button>
        )}
      </div>
    </>
  );
}
