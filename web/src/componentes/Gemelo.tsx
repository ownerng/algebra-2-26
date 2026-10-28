// Gemelo del prototipo: vista superior del aparato. Con el ESP32 conectado
// muestra lo que hace el aparato real; en simulación, lo que debería hacer.

import { escaner, useEscaner, COLORES } from '../lib/escaner.ts';

const PX_POR_MM = 135 / 1200; // el alcance de 1,2 m ocupa 135 px
const CX = 160;
const CY = 182;

export function Gemelo() {
  const e = useEscaner();
  const u = e.ultima;
  const conectado = e.tipoFuente !== 'ninguna';
  const theta = u?.thetaServo ?? 90;
  const r = u && u.r > 0 ? u.r : null;
  const color = COLORES[e.posicionActiva];
  const largo = r ? Math.min(r + e.d, 1200) * PX_POR_MM : 135;
  const estado = !conectado ? 'apagado' : e.servo === 'pausado' ? 'en pausa' : 'barriendo';

  return (
    <section className="gemelo" aria-label="Prototipo">
      <header className="placa-titulo">
        <span>Prototipo</span>
        <span className={'etiqueta ' + e.tipoFuente}>
          {e.tipoFuente === 'simulacion' ? 'simulado' : e.tipoFuente === 'serial' ? 'en vivo' : 'sin señal'}
        </span>
      </header>
      <p className="ayuda">Lo que está haciendo el aparato en este momento: hacia dónde apunta el sensor y qué distancia mide.</p>

      <svg viewBox="-14 0 348 214" role="img" aria-label={`Servo en ${theta} grados${r ? `, objeto a ${r} milímetros` : ''}`}>
        {/* Base de MDF y protoboard */}
        <rect x="18" y="128" width="284" height="80" rx="6" className="mdf" />
        <rect x="200" y="150" width="88" height="44" rx="3" className="protoboard" />
        {Array.from({ length: 8 }, (_, i) => (
          <line key={i} x1={208 + i * 10} y1="156" x2={208 + i * 10} y2="188" className="protoboard-huecos" />
        ))}
        <rect x="216" y="160" width="56" height="24" rx="2" className="esp32" />
        <text x="244" y="175" className="esp32-texto">ESP32</text>

        {/* Arco de barrido 0–180° */}
        <path d={`M ${CX - 135} ${CY} A 135 135 0 0 1 ${CX + 135} ${CY}`} className="arco" />
        {[0, 30, 60, 90, 120, 150, 180].map((t) => {
          const a = (t * Math.PI) / 180;
          return (
            <g key={t}>
              <line x1={CX + Math.cos(a) * 129} y1={CY - Math.sin(a) * 129} x2={CX + Math.cos(a) * 135} y2={CY - Math.sin(a) * 135} className="marca" />
              <text x={CX + Math.cos(a) * 148} y={CY - Math.sin(a) * 148 + 4} className="marca-texto">{t}°</text>
            </g>
          );
        })}

        {/* Cuerpo del servo */}
        <rect x={CX - 16} y={CY - 12} width="54" height="26" rx="3" className="servo" />
        <rect x={CX - 24} y={CY - 4} width="70" height="9" rx="2" className="servo-orejas" />
        <text x={CX + 18} y={CY + 30} className="servo-texto">SG90</text>

        {/* Parte que gira: brazo, sensor y haz */}
        <g className="giro" style={{ transform: `rotate(${-theta}deg)`, transformOrigin: `${CX}px ${CY}px` }}>
          {conectado && (
            <>
              <line x1={CX + 22} y1={CY} x2={CX + largo} y2={CY} className="haz" style={{ stroke: color, color }} />
              {r && <circle cx={CX + largo} cy={CY} r="4.5" className="impacto" style={{ fill: color }} />}
            </>
          )}
          <rect x={CX - 5} y={CY - 5} width="26" height="10" rx="5" className="horn" />
          <rect x={CX + 8} y={CY - 11} width="12" height="22" rx="1.5" className="sensor" />
          <rect x={CX + 19} y={CY - 4} width="3" height="8" className="lente" />
          <circle cx={CX} cy={CY} r="3" className="tornillo" />
        </g>
      </svg>

      <dl className="lecturas">
        <div><dt>ángulo servo</dt><dd>{conectado && u ? `${theta}°` : '—'}</dd></div>
        <div><dt>distancia</dt><dd>{!conectado || !u ? '—' : r ? `${r} mm` : 'sin eco'}</dd></div>
        <div><dt>estado</dt><dd>{estado}</dd></div>
      </dl>

      {!conectado && (
        <button type="button" className="boton-simular" onClick={() => escaner.conectar('simulacion')}>
          ▶ Simular prototipo
        </button>
      )}
    </section>
  );
}
