// Guía de uso en cuatro pasos; cada uno se marca solo cuando está hecho.

import { escaneo, useEscaneo } from '../lib/escaneo.ts';

export function Guia() {
  const e = useEscaneo(300);
  const conectado = e.tipoFuente !== 'ninguna';
  const b = e.barrido;
  const pasos = [
    {
      titulo: 'Conectar',
      texto: 'Conecta el ESP32 por USB o simula el prototipo.',
      hecho: conectado,
      accion: !conectado ? { texto: 'Simular', fn: () => escaneo.conectar('simulacion') } : null,
    },
    {
      titulo: 'Poner el objeto',
      texto: 'En el centro de la cuadrícula, con el borde libre.',
      hecho: !!b || e.medidas.size > 0,
      accion: null,
    },
    {
      titulo: 'Escanear',
      texto: b && b.fase !== 'fin' ? `Desliza la caja siguiendo la marca (fila ${b.fila + 1} de ${e.config.ny}).` : 'Un botón: la marca recorre las filas y la figura se arma sola.',
      hecho: b?.fase === 'fin' || e.medidas.size >= e.total,
      accion: conectado && !b ? { texto: '▶ Escanear', fn: () => escaneo.escanearContinuo() } : null,
    },
    {
      titulo: 'Ver y guardar',
      texto: 'Gira la figura en 3D y guárdala o expórtala.',
      hecho: false,
      accion: null,
    },
  ];
  const actual = pasos.findIndex((p) => !p.hecho);

  return (
    <nav className="guia" aria-label="Pasos para escanear un objeto">
      {pasos.map((p, i) => (
        <div key={p.titulo} className={'guia-paso' + (p.hecho ? ' hecho' : '') + (i === actual ? ' actual' : '')}>
          <span className="guia-num">{p.hecho ? '✓' : i + 1}</span>
          <div>
            <strong>{p.titulo}</strong>
            <span>{p.texto}</span>
          </div>
          {i === actual && p.accion && (
            <button type="button" className="tecla chica" onClick={p.accion.fn}>{p.accion.texto}</button>
          )}
        </div>
      ))}
    </nav>
  );
}
