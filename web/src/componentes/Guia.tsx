// Guía de uso en cuatro pasos. Cada paso se marca solo cuando está hecho,
// para que quien mira la demostración sepa en qué parte va el proceso.

import { escaner, useEscaner } from '../lib/escaner.ts';

export function Guia() {
  const e = useEscaner(300);
  const puntos = e.puntos();
  const pasos = [
    {
      titulo: 'Conectar',
      texto: 'Conecta el ESP32 por USB o simula el prototipo.',
      hecho: e.tipoFuente !== 'ninguna',
      accion: e.tipoFuente === 'ninguna' ? { texto: 'Simular', fn: () => escaner.conectar('simulacion') } : null,
    },
    {
      titulo: 'Escanear desde P1',
      texto: 'Deja que el servo barra una o dos veces. P1 es el origen del mapa.',
      hecho: puntos[1].length > 20,
      accion: e.posicionActiva !== 1 ? { texto: 'Usar P1', fn: () => escaner.cambiar({ posicionActiva: 1 }) } : null,
    },
    {
      titulo: 'Mover el aparato',
      texto: 'Muévelo a otro lugar y escribe cuánto se desplazó y giró.',
      hecho: e.posicionActiva === 2 || puntos[2].length > 0,
      accion: e.posicionActiva !== 2 ? { texto: 'Ya lo moví', fn: () => escaner.cambiar({ posicionActiva: 2 }) } : null,
    },
    {
      titulo: 'Escanear desde P2',
      texto: 'Los puntos nuevos se llevan al mapa con la matriz T y se suman a los de P1.',
      hecho: puntos[2].length > 20,
      accion: null,
    },
  ];
  const actual = pasos.findIndex((p) => !p.hecho);

  return (
    <nav className="guia" aria-label="Pasos para hacer un mapa">
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
