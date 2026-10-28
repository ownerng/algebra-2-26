import { useEffect, useState } from 'react';
import { escaneo, useEscaneo } from './lib/escaneo.ts';
import { Vista3D } from './componentes/Vista3D.tsx';
import { Control } from './componentes/Control.tsx';
import { Ajustes } from './componentes/Ajustes.tsx';
import { Cuaderno } from './componentes/Cuaderno.tsx';
import { Camara } from './componentes/Camara.tsx';

type Cajon = 'calculos' | 'ajustes' | 'camara' | null;

function Barra({ cajon, abrir }: { cajon: Cajon; abrir: (c: Cajon) => void }) {
  const e = useEscaneo(200);
  const conectado = e.tipoFuente !== 'ninguna';
  return (
    <header className="barra">
      <span className="marca">LIDAR-3D</span>
      <span className={'piloto ' + e.estadoConexion} role="status"><i aria-hidden="true" />{e.mensaje}</span>
      <div className="barra-derecha">
        <button type="button" className={'pestana' + (cajon === 'camara' ? ' activa' : '')} onClick={() => abrir(cajon === 'camara' ? null : 'camara')}>Cámara</button>
        <button type="button" className={'pestana' + (cajon === 'calculos' ? ' activa' : '')} onClick={() => abrir(cajon === 'calculos' ? null : 'calculos')}>Cálculos</button>
        <button type="button" className={'pestana' + (cajon === 'ajustes' ? ' activa' : '')} onClick={() => abrir(cajon === 'ajustes' ? null : 'ajustes')}>Ajustes</button>
        {conectado && <button type="button" className="pestana" onClick={() => escaneo.desconectar()}>Desconectar</button>}
      </div>
    </header>
  );
}

export function App() {
  const [cajon, setCajon] = useState<Cajon>(null);

  useEffect(() => {
    const tecla = (ev: KeyboardEvent) => {
      if ((ev.target as HTMLElement).matches('input, select, textarea')) return;
      if (ev.key === ' ') { ev.preventDefault(); escaneo.escanearContinuo(); }
      if (ev.key === 'Escape') setCajon(null);
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
    <div className="app-min">
      <Barra cajon={cajon} abrir={setCajon} />
      <main className="escena">
        <Vista3D />
        <Control />
        {cajon && (
          <aside className={'cajon ' + cajon} aria-label={cajon === 'calculos' ? 'Cálculos' : cajon === 'camara' ? 'Cámara' : 'Ajustes'}>
            <button type="button" className="cajon-cerrar" onClick={() => setCajon(null)} aria-label="Cerrar">✕</button>
            {cajon === 'calculos' ? <Cuaderno /> : cajon === 'camara' ? <Camara /> : <Ajustes />}
          </aside>
        )}
      </main>
    </div>
  );
}
