import { useEffect } from 'react';
import { escaneo, useEscaneo } from './lib/escaneo.ts';
import { Guia } from './componentes/Guia.tsx';
import { Vista3D } from './componentes/Vista3D.tsx';
import { Cuadricula } from './componentes/Cuadricula.tsx';
import { Ajustes } from './componentes/Ajustes.tsx';
import { Cuaderno } from './componentes/Cuaderno.tsx';

function Cabecera() {
  const e = useEscaneo(150);
  const conectado = e.tipoFuente !== 'ninguna';
  return (
    <header className="cabecera">
      <div className="placa">
        <span className="tornillo-placa" aria-hidden="true" />
        <div>
          <h1>LIDAR-3D</h1>
          <p>Escáner 3D por cuadrícula · ESP32 · VL53L0X</p>
        </div>
      </div>
      <div className="botonera">
        {!conectado ? (
          <>
            <button type="button" className="tecla grande" onClick={() => escaneo.conectar('serial')}
              title="Usa el puente USB; sin puente, Web Serial (Chrome, Chromium o Edge)">Conectar ESP32</button>
            <button type="button" className="tecla grande ambar" onClick={() => escaneo.conectar('simulacion')}>▶ Simular prototipo</button>
          </>
        ) : (
          <button type="button" className="tecla" onClick={() => escaneo.desconectar()}>Desconectar</button>
        )}
        <span className={'piloto ' + e.estadoConexion} role="status">
          <i aria-hidden="true" />
          {e.mensaje}{conectado && e.estadoConexion === 'conectado' ? (e.sensorListo ? ' · sensor listo' : ' · esperando el sensor') : ''}
        </span>
      </div>
    </header>
  );
}

export function App() {
  useEffect(() => {
    const tecla = (ev: KeyboardEvent) => {
      if ((ev.target as HTMLElement).matches('input, select, textarea')) return;
      if (ev.key === ' ') { ev.preventDefault(); escaneo.escanearContinuo(); }
      if (ev.key === 'Enter') { ev.preventDefault(); escaneo.medir(); }
      if (ev.key === 'Backspace' || (ev.key === 'z' && ev.ctrlKey)) { ev.preventDefault(); escaneo.deshacer(); }
    };
    document.addEventListener('keydown', tecla);
    // #demo arranca la simulación; #demo-auto además hace el escaneo continuo
    // (en la simulación la caja sigue la marca sola)
    if (location.hash.startsWith('#demo')) {
      void escaneo.conectar('simulacion').then(() => {
        if (location.hash === '#demo-auto') escaneo.escanearContinuo();
      });
    }
    return () => document.removeEventListener('keydown', tecla);
  }, []);

  return (
    <div className="instrumento">
      <Cabecera />
      <Guia />
      <main className="cuerpo">
        <div className="columna-principal">
          <div className="fila-superior">
          <Cuadricula />
          <section className="pantalla" aria-label="Vista 3D">
            <div className="bisel">
              <div className="modos">
                <span className="modo activo">Figura 3D</span>
                <label className="casilla-check"><input type="checkbox" defaultChecked onChange={(ev) => { escaneo.vista.girar = ev.target.checked; }} /> girar sola</label>
              </div>
              <div className="cristal"><Vista3D /></div>
            </div>
          </section>
          </div>
          <Ajustes />
        </div>
        <Cuaderno />
      </main>
    </div>
  );
}
