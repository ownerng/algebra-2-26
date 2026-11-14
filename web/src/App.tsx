import { useEffect, useState } from 'react';
import { escanear, escanearHabitacion, estadoCamara, generar3D, URL_OBJ, type Modelo } from './lib/api.ts';
import { PanelCamara } from './componentes/PanelCamara.tsx';
import { Encuadre, type Modo } from './componentes/Encuadre.tsx';
import { Escena3D, type Mostrar } from './componentes/Escena3D.tsx';
import { Habitacion } from './componentes/Habitacion.tsx';

const miles = (n: number) => n.toLocaleString('es');

export function App() {
  const [panel, setPanel] = useState(false);
  const [conectada, setConectada] = useState(false);
  const [conexion, setConexion] = useState(0);   // cambia en cada conexión: recarga el video
  const [fase, setFase] = useState<'encuadre' | '3d'>('encuadre');
  const [modelo, setModelo] = useState<Modelo | null>(null);
  const [mostrar, setMostrar] = useState<Mostrar>('malla');
  const [tipo, setTipo] = useState<'objetos' | 'habitaciones'>('objetos');
  const [girar, setGirar] = useState(false);
  // Procesamiento alto: IA generativa en fal.ai (objetos completos; en habitaciones, los muebles)
  const [calidad, setCalidad] = useState<'bajo' | 'alto'>('bajo');
  const alto = calidad === 'alto';
  const [procesando, setProcesando] = useState('');
  const [error, setError] = useState('');
  const [modo, setModo] = useState<Modo>('una');

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

  const verModelo = (m: Modelo) => { setModelo(m); setGirar(false); setFase('3d'); };

  const tomarFoto = (foto?: Blob) => alto
    ? trabajar('Generando modelo completo (1–4 min)…', async () => verModelo(await generar3D(foto)))
    : trabajar('Segmentando y reconstruyendo…', async () => verModelo(await escanear(foto)));

  const tomarHabitacion = (foto?: Blob) => trabajar(alto ? 'Generando habitación completa (3–5 min)…' : 'Calculando profundidad y relleno…',
    async () => verModelo(await escanearHabitacion(foto, alto)));

  const cambiarModo = (m: Modo) => { setModo(m); setError(''); };

  return (
    <div className="app">
      <header className="cabecera">
        <button type="button" className={'boton-camara' + (panel ? ' activo' : '')} aria-expanded={panel} onClick={() => setPanel(!panel)}>
          <i className={'punto' + (conectada ? ' ok' : '')} aria-hidden="true" />Cámara
        </button>
        <div className="selector" role="group" aria-label="Qué escanear">
          {(['objetos', 'habitaciones'] as const).map((t) => (
            <button key={t} type="button" aria-pressed={tipo === t} disabled={!!procesando}
              onClick={() => { setTipo(t); setError(''); setFase('encuadre'); }}>
              {t === 'objetos' ? 'Objetos' : 'Habitaciones'}
            </button>
          ))}
        </div>
        <label className="calidad">
          Procesamiento
          <select value={alto ? 'alto' : 'bajo'} disabled={!!procesando} onChange={(ev) => setCalidad(ev.currentTarget.value as 'bajo' | 'alto')}>
            <option value="bajo">Bajo</option>
            <option value="alto">Alto</option>
          </select>
        </label>
      </header>

      <main className="escenario">
        <Escena3D modelo={fase === '3d' ? modelo : null} mostrar={mostrar} girar={girar} />

        {fase === 'encuadre' && tipo === 'habitaciones' && (
          <Habitacion conectada={conectada} conexion={conexion} procesando={procesando} error={error} enviar={(f) => void tomarHabitacion(f)} />
        )}

        {fase === 'encuadre' && tipo === 'objetos' && (conectada
          ? <Encuadre conexion={conexion} procesando={procesando} error={error} modo={modo} cambiarModo={cambiarModo}
              tomarFoto={(f) => void tomarFoto(f)} />
          : <div className="vacio"><p>Conecta la cámara para encuadrar el objeto, o sube una foto.</p>
              <div className="fila-botones">
                <button type="button" className="boton primario" onClick={() => setPanel(true)}>Configurar cámara</button>
                <label className="boton" aria-disabled={!!procesando}>
                  {procesando ? 'Procesando…' : 'Subir foto'}
                  <input type="file" accept="image/*" hidden disabled={!!procesando}
                    onChange={(ev) => { const f = ev.currentTarget.files?.[0]; ev.currentTarget.value = ''; if (f) void tomarFoto(f); }} />
                </label>
              </div>
              {error && <p className="aviso" role="alert">{error}</p>}</div>)}

        {fase === '3d' && modelo && (
          <>
            <div className="ficha">
              <img src={modelo.recorte} alt="Objeto segmentado" />
              <dl>
                <dt>V</dt><dd>{miles(modelo.N)} × 3</dd>
                <dt>F</dt><dd>{miles(modelo.M)} × 3</dd>
                <dt>Tamaño</dt><dd>{modelo.glb ? '—' : modelo.vista
                  ? modelo.tamanoMm.map((x) => (x / 1000).toFixed(1)).join('×') + ' m'
                  : modelo.tamanoMm.map((x) => Math.round(x)).join(' × ') + ' mm'}</dd>
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
              {modelo.vista && (
                <button type="button" className="boton" aria-pressed={girar} onClick={() => setGirar(!girar)}>{girar ? 'Detener giro' : 'Girar'}</button>
              )}
              {modelo.glb
                ? <a className="boton" href={modelo.glb} download="objeto.glb">.glb</a>
                : <a className="boton" href={URL_OBJ} download={modelo.vista ? 'habitacion.obj' : 'objeto.obj'}>.obj</a>}
              <button type="button" className="boton disparo" onClick={() => { setError(''); setFase('encuadre'); }}>
                Nueva foto
              </button>
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
