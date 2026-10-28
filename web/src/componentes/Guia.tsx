// Guía de uso en cuatro pasos; cada uno se marca solo cuando está hecho.

import { escaneo, useEscaneo } from '../lib/escaneo.ts';

export function Guia() {
  const e = useEscaneo(300);
  const conectado = e.tipoFuente !== 'ninguna';
  const pasos = [
    {
      titulo: 'Conectar',
      texto: 'Conecta el ESP32 por USB o simula el prototipo.',
      hecho: conectado,
      accion: !conectado ? { texto: 'Simular', fn: () => escaneo.conectar('simulacion') } : null,
    },
    {
      titulo: 'Medir la mesa',
      texto: 'Con la mesa vacía, mide la altura H de la caja.',
      hecho: e.alturaMedida,
      accion: conectado && !e.alturaMedida ? { texto: 'Medir la mesa', fn: () => escaneo.medirMesa() } : null,
    },
    {
      titulo: 'Recorrer la cuadrícula',
      texto: `Pon el objeto y mide casilla por casilla (${e.medidas.size}/${e.total}).`,
      hecho: e.medidas.size >= e.total,
      accion: conectado && e.medidas.size < e.total ? { texto: 'Medir', fn: () => escaneo.medir() } : null,
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
