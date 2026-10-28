import { useEffect, useState } from 'react';
import { ConexionSerial } from './lib/conexion.ts';
import { escaner, useEscaner } from './lib/escaner.ts';
import { Radar } from './componentes/Radar.tsx';
import { Vista3D } from './componentes/Vista3D.tsx';
import { Gemelo } from './componentes/Gemelo.tsx';
import { Cuaderno } from './componentes/Cuaderno.tsx';
import { Controles } from './componentes/Controles.tsx';

const SERIAL = ConexionSerial.disponible();

function Cabecera() {
  const e = useEscaner(100);
  const conectado = e.tipoFuente !== 'ninguna';
  return (
    <header className="cabecera">
      <div className="placa">
        <span className="tornillo-placa" aria-hidden="true" />
        <div>
          <h1>LIDAR-01</h1>
          <p>Escáner 2D · ESP32 · VL53L0X · SG90</p>
        </div>
      </div>

      <div className="botonera">
        {!conectado ? (
          <>
            <button type="button" className="tecla grande" disabled={!SERIAL} onClick={() => escaner.conectar('serial')}
              title={SERIAL ? 'Elegir el puerto del ESP32' : 'Este navegador no tiene Web Serial. Usa Chrome, Chromium o Edge.'}>
              Conectar ESP32
            </button>
            <button type="button" className="tecla grande ambar" onClick={() => escaner.conectar('simulacion')}>▶ Simular prototipo</button>
          </>
        ) : (
          <>
            <button type="button" className="tecla" onClick={() => escaner.enviar('S')}>▶ Barrer</button>
            <button type="button" className="tecla" onClick={() => escaner.enviar('P')}>❚❚ Pausa</button>
            <button type="button" className="tecla" onClick={() => escaner.enviar('C')}>⊙ 90°</button>
            <button type="button" className="tecla" onClick={() => escaner.desconectar()}>Desconectar</button>
          </>
        )}
        <span className={'piloto ' + e.estadoConexion} role="status">
          <i aria-hidden="true" />
          {e.estadoConexion === 'error' ? e.mensaje : conectado ? `${e.mensaje} · ${e.servo}` : e.mensaje}
        </span>
      </div>
    </header>
  );
}

function Pantalla() {
  const [modo, setModo] = useState<'2d' | '3d'>(() => (location.hash === '#demo-3d' ? '3d' : '2d'));
  const [encuadre, setEncuadre] = useState(0);
  const [girar, setGirar] = useState(true);
  const e = useEscaner(150);
  const u = e.ultima;
  const theta = u ? (e.invertir ? 180 - u.thetaServo : u.thetaServo) : null;

  return (
    <section className="pantalla" aria-label="Mapa">
      <div className="bisel">
        <div className="modos" role="tablist">
          {(['2d', '3d'] as const).map((m) => (
            <button key={m} type="button" role="tab" aria-selected={modo === m} className={'modo' + (modo === m ? ' activo' : '')} onClick={() => setModo(m)}>
              {m === '2d' ? 'Radar 2D' : 'Mapa 3D'}
            </button>
          ))}
          <span className="hud">
            {u ? `P${u.posicion}  θ ${String(theta).padStart(3)}°  r ${u.r > 0 ? String(u.r).padStart(4) + ' mm' : ' sin eco'}` : 'θ —   r —'}
          </span>
          {modo === '2d'
            ? <button type="button" className="tecla chica" onClick={() => setEncuadre((n) => n + 1)}>Encuadrar</button>
            : <label className="casilla"><input type="checkbox" checked={girar} onChange={(ev) => setGirar(ev.target.checked)} /> girar</label>}
        </div>
        <div className="cristal">
          {modo === '2d' ? <Radar encuadre={encuadre} /> : <Vista3D girar={girar} />}
          {!u && (
            <div className="aviso">
              <strong>Sin señal</strong>
              <span>Conecta el ESP32 por USB o pulsa <b>Simular prototipo</b> para ver lo que debería medir en una habitación de prueba.</span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}

export function App() {
  useEffect(() => {
    const tecla = (ev: KeyboardEvent) => {
      if ((ev.target as HTMLElement).matches('input')) return;
      if (ev.key === '1' || ev.key === '2') escaner.cambiar({ posicionActiva: Number(ev.key) as 1 | 2 });
      if (ev.key === ' ' && escaner.fuente) {
        ev.preventDefault();
        escaner.enviar(escaner.servo === 'pausado' ? 'S' : 'P');
      }
    };
    document.addEventListener('keydown', tecla);
    // #demo arranca la simulación sin tocar nada (para ensayar la presentación);
    // #demo-fusion además pasa sola a P2 después de dos barridos desde P1.
    if (location.hash.startsWith('#demo')) void escaner.conectar('simulacion');
    const paso = location.hash === '#demo-fusion' ? setTimeout(() => escaner.cambiar({ posicionActiva: 2 }), 8500) : undefined;
    return () => { document.removeEventListener('keydown', tecla); clearTimeout(paso); };
  }, []);

  return (
    <div className="instrumento">
      <Cabecera />
      <main className="cuerpo">
        <div className="columna-principal">
          <Pantalla />
          <div className="fila-inferior">
            <Gemelo />
            <Controles />
          </div>
        </div>
        <Cuaderno />
      </main>
    </div>
  );
}
