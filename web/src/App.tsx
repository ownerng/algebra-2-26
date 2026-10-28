import { useEffect, useState } from 'react';
import { escaneo, useEscaneo } from './lib/escaneo.ts';
import { camara } from './lib/camara.ts';
import { Escaner } from './componentes/Escaner.tsx';
import { Ajustes } from './componentes/Ajustes.tsx';
import { Cuaderno } from './componentes/Cuaderno.tsx';

type Cajon = 'calculos' | 'ajustes' | null;

function Barra({ cajon, abrir }: { cajon: Cajon; abrir: (c: Cajon) => void }) {
  const e = useEscaneo(300);
  const sensor = e.tipoFuente === 'serial' && e.estadoConexion === 'conectado';
  const vivo = e.vivo && performance.now() - e.vivo.t < 1500 && e.vivo.r > 0 ? e.vivo.r : null;
  return (
    <header className="barra">
      <span className="marca">LIDAR-3D</span>
      <button type="button" className={'piloto ' + (sensor ? 'conectado' : e.estadoConexion === 'error' ? 'error' : '')}
        onClick={() => { if (!sensor) void escaneo.conectar('serial'); }} title={sensor ? 'Distancia que mide el VL53L0X' : 'Conectar el sensor'}>
        <i aria-hidden="true" />{sensor ? `sensor · ${vivo ? `${vivo} mm` : 'sin eco'}` : e.estadoConexion === 'error' ? e.mensaje : 'Conectar sensor'}
      </button>
      <div className="barra-derecha">
        <button type="button" className={'pestana' + (cajon === 'calculos' ? ' activa' : '')} onClick={() => abrir(cajon === 'calculos' ? null : 'calculos')}>Cálculos</button>
        <button type="button" className={'pestana' + (cajon === 'ajustes' ? ' activa' : '')} onClick={() => abrir(cajon === 'ajustes' ? null : 'ajustes')}>Ajustes</button>
      </div>
    </header>
  );
}

export function App() {
  const [cajon, setCajon] = useState<Cajon>(null);

  useEffect(() => {
    const tecla = (ev: KeyboardEvent) => {
      if ((ev.target as HTMLElement).matches('input, select, textarea')) return;
      if (ev.key === ' ') { ev.preventDefault(); if (camara.vista === 'camara') void camara.tomarFoto(); else camara.cambiar({ vista: 'camara' }); }
      if (ev.key === 'Escape') setCajon(null);
    };
    document.addEventListener('keydown', tecla);
    // El sensor se conecta solo si el puente USB está corriendo
    void escaneo.conectar('serial');
    return () => document.removeEventListener('keydown', tecla);
  }, []);

  return (
    <div className="app-min">
      <Barra cajon={cajon} abrir={setCajon} />
      <main className="escena">
        <Escaner />
        {cajon && (
          <aside className={'cajon ' + cajon} aria-label={cajon === 'calculos' ? 'Cálculos' : 'Ajustes'}>
            <button type="button" className="cajon-cerrar" onClick={() => setCajon(null)} aria-label="Cerrar">✕</button>
            {cajon === 'calculos' ? <Cuaderno /> : <Ajustes />}
          </aside>
        )}
      </main>
    </div>
  );
}
