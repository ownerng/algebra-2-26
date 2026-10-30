import { useEffect, useState } from 'react';
import { borrarVistas, escanear, estadoCamara, reconstruir, tomarVista, URL_OBJ, VISTAS, type Modelo, type NombreVista } from './lib/api.ts';
import { PanelCamara } from './componentes/PanelCamara.tsx';
import { Encuadre, NOMBRES, type Modo } from './componentes/Encuadre.tsx';
import { Escena3D, type Mostrar } from './componentes/Escena3D.tsx';

const miles = (n: number) => n.toLocaleString('es');

export function App() {
  const [panel, setPanel] = useState(false);
  const [conectada, setConectada] = useState(false);
  const [conexion, setConexion] = useState(0);   // cambia en cada conexión: recarga el video
  const [fase, setFase] = useState<'encuadre' | '3d'>('encuadre');
  const [modelo, setModelo] = useState<Modelo | null>(null);
  const [mostrar, setMostrar] = useState<Mostrar>('malla');
  const [procesando, setProcesando] = useState('');
  const [error, setError] = useState('');
  // Modo de varias vistas: recorte de cada vista tomada y la que toca ahora
  const [modo, setModo] = useState<Modo>('una');
  const [recortes, setRecortes] = useState<Partial<Record<NombreVista, string>>>({});
  const [paso, setPaso] = useState<NombreVista | null>('frontal');

  // Estado de la cámara: al abrir la página y cada pocos segundos
  useEffect(() => {
    let primera = true;
    const revisar = async () => {
      try {
        const e = await estadoCamara();
        setConectada(e.conectada);
        if (primera && !e.conectada) setPanel(true);
      } catch {
        setConectada(false);
        if (primera) setPanel(true);
      }
      primera = false;
    };
    void revisar();
    const id = setInterval(revisar, 3000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    const tecla = (ev: KeyboardEvent) => { if (ev.key === 'Escape') setPanel(false); };
    document.addEventListener('keydown', tecla);
    return () => document.removeEventListener('keydown', tecla);
  }, []);

  // Ejecuta una petición al servidor mostrando `texto` mientras tanto
  const trabajar = async (texto: string, tarea: () => Promise<void>) => {
    setProcesando(texto); setError('');
    try { await tarea(); } catch (e) { setError((e as Error).message); }
    setProcesando('');
  };

  const verModelo = (m: Modelo) => { setModelo(m); setFase('3d'); };

  const tomarFoto = () => trabajar('Segmentando y reconstruyendo…', async () => verModelo(await escanear()));

  const tomarVistaActual = () => trabajar(`Segmentando la vista ${paso ? NOMBRES[paso].toLowerCase() : ''}…`, async () => {
    if (!paso) return;
    const recorte = await tomarVista(paso);
    const nuevos = { ...recortes, [paso]: recorte };
    setRecortes(nuevos);
    setPaso(VISTAS.find((n) => !nuevos[n]) ?? null);
  });

  const reconstruir3D = () => trabajar('Tallando voxeles con las vistas…', async () => verModelo(await reconstruir()));

  const reiniciarVistas = () => { void borrarVistas().catch(() => {}); setRecortes({}); setPaso('frontal'); setError(''); };

  const cambiarModo = (m: Modo) => { setModo(m); setError(''); if (m === 'vistas' && !recortes.frontal) reiniciarVistas(); };

  return (
    <div className="app">
      <header className="cabecera">
        <button type="button" className={'boton-camara' + (panel ? ' activo' : '')} aria-expanded={panel} onClick={() => setPanel(!panel)}>
          <i className={'punto' + (conectada ? ' ok' : '')} aria-hidden="true" />Cámara
        </button>
      </header>

      <main className="escenario">
        <Escena3D modelo={fase === '3d' ? modelo : null} mostrar={mostrar} />

        {fase === 'encuadre' && (conectada
          ? <Encuadre conexion={conexion} procesando={procesando} error={error} modo={modo} cambiarModo={cambiarModo}
              tomarFoto={() => void tomarFoto()} recortes={recortes} paso={paso} elegirPaso={setPaso}
              tomarVista={() => void tomarVistaActual()} reconstruir={() => void reconstruir3D()} reiniciar={reiniciarVistas} />
          : <div className="vacio"><p>Conecta la cámara para encuadrar el objeto.</p>
              <button type="button" className="boton primario" onClick={() => setPanel(true)}>Configurar cámara</button></div>)}

        {fase === '3d' && modelo && (
          <>
            <div className="ficha">
              <img src={modelo.recorte} alt="Objeto segmentado" />
              <dl>
                <dt>V</dt><dd>{miles(modelo.N)} × 3</dd>
                <dt>F</dt><dd>{miles(modelo.M)} × 3</dd>
                <dt>Tamaño</dt><dd>{modelo.tamanoMm.map((x) => Math.round(x)).join(' × ')} mm</dd>
                <dt>Tiempo</dt><dd>{(Object.values(modelo.tiemposMs).reduce((a, b) => a + b, 0) / 1000).toFixed(1)} s</dd>
              </dl>
              <small>{modelo.metodo}</small>
            </div>
            <div className="barra-3d">
              <div className="selector" role="group" aria-label="Mostrar">
                {(['malla', 'puntos', 'ambos'] as const).map((m) => (
                  <button key={m} type="button" aria-pressed={mostrar === m} onClick={() => setMostrar(m)}>
                    {m === 'malla' ? 'Malla' : m === 'puntos' ? 'Puntos' : 'Ambos'}
                  </button>
                ))}
              </div>
              <a className="boton" href={URL_OBJ} download="objeto.obj">.obj</a>
              <button type="button" className="boton disparo" onClick={() => { setError(''); setFase('encuadre'); }}>{modo === 'vistas' ? 'Volver a las vistas' : 'Nueva foto'}</button>
            </div>
          </>
        )}

        {panel && (
          <PanelCamara conectada={conectada} cerrar={() => setPanel(false)}
            alConectar={() => { setConectada(true); setConexion((c) => c + 1); setFase('encuadre'); setPanel(false); }} />
        )}
      </main>
    </div>
  );
}
