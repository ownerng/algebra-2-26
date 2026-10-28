// Panel de la cámara: elegir DroidCam, marcar las 4 esquinas de la cuadrícula,
// tomar la foto de la cartulina vacía y la del objeto, y ver la silueta.

import { useEffect, useRef } from 'react';
import { camara, DROIDCAM, useCamara } from '../lib/camara.ts';

const NOMBRES_ESQUINA = ['abajo a la izquierda', 'abajo a la derecha', 'arriba a la derecha', 'arriba a la izquierda'];

function Silueta() {
  const c = useCamara();
  const lienzo = useRef<HTMLCanvasElement>(null);
  const m = c.modelo;
  useEffect(() => {
    if (!m || !lienzo.current) return;
    const cv = lienzo.current;
    cv.width = m.ancho; cv.height = m.alto;
    const g = cv.getContext('2d')!;
    const img = g.createImageData(m.ancho, m.alto);
    for (let k = 0; k < m.mascara.length; k++) {
      const o = m.mascara[k]!;
      // El objeto con sus colores reales; el fondo, oscurecido
      img.data[k * 4] = o ? m.rgb[k * 4]! : m.rgb[k * 4]! * 0.25;
      img.data[k * 4 + 1] = o ? m.rgb[k * 4 + 1]! : m.rgb[k * 4 + 1]! * 0.25;
      img.data[k * 4 + 2] = o ? m.rgb[k * 4 + 2]! : m.rgb[k * 4 + 2]! * 0.25;
      img.data[k * 4 + 3] = 255;
    }
    g.putImageData(img, 0, 0);
  }, [m]);
  if (!m) return null;
  return (
    <div className="silueta">
      <canvas ref={lienzo} aria-label="Silueta del objeto vista desde arriba, enderezada" />
      <p className="ayuda">
        {m.caja
          ? <>Objeto: <b>{Math.round(m.caja.xMax - m.caja.xMin)} × {Math.round(m.caja.yMax - m.caja.yMin)} mm</b>, área {Math.round(m.areaMm2)} mm² (umbral {m.umbral}).</>
          : 'No se encontró ningún objeto: revisa que la foto vacía y la del objeto se tomaron con la cámara quieta.'}
      </p>
    </div>
  );
}

export function Camara() {
  const c = useCamara();
  const video = useRef<HTMLVideoElement>(null);
  const imagen = useRef<HTMLImageElement>(null);

  useEffect(() => { void camara.listar(); }, []);
  useEffect(() => {
    if (video.current && c.stream && video.current.srcObject !== c.stream) {
      video.current.srcObject = c.stream;
      void video.current.play();
    }
    camara.video = video.current;
    camara.imagen = imagen.current;
  });

  // Tamaño real de la imagen que se está viendo (video de la webcam o MJPEG del celular)
  const tam = (): [number, number] => {
    if (c.usaDroidcam) return [imagen.current?.naturalWidth ?? 0, imagen.current?.naturalHeight ?? 0];
    return [video.current?.videoWidth ?? 0, video.current?.videoHeight ?? 0];
  };
  const tocar = (ev: React.MouseEvent<HTMLDivElement>) => {
    const [w, h] = tam();
    const el = c.usaDroidcam ? imagen.current : video.current;
    if (!el || !w) return;
    const r = el.getBoundingClientRect();
    camara.marcarEsquina([((ev.clientX - r.left) / r.width) * w, ((ev.clientY - r.top) / r.height) * h]);
  };

  const [vw, vh] = tam();
  const listo4 = c.esquinas.length === 4;

  return (
    <div className="ajustes-panel camara-panel">
      <h2>Cámara</h2>
      <p className="ayuda">
        El celular con DroidCam mira la cartulina desde arriba y no se mueve. La cámara saca la silueta exacta del objeto;
        el sensor, su altura.
      </p>

      <section>
        <h3>1 · Elegir la cámara</h3>
        <div className="fila-botones">
          <select value={c.deviceId} onChange={(ev) => void camara.iniciar(ev.target.value)} aria-label="Cámara">
            <option value={DROIDCAM}>Celular por WiFi (DroidCam)</option>
            {c.dispositivos.map((d, k) => <option key={d.deviceId || k} value={d.deviceId}>{d.label || `Cámara ${k + 1}`}</option>)}
          </select>
          {!c.usaDroidcam && <button type="button" className="tecla chica" onClick={() => void camara.iniciar()}>{c.stream ? 'Reiniciar' : 'Encender'}</button>}
        </div>
        {c.usaDroidcam && (
          <label className="campo-fila" htmlFor="ip">
            <span className="campo-nombre">IP del celular<small>la que muestra DroidCam (WiFi IP)</small></span>
            <span className="campo-valor"><input id="ip" className="texto" placeholder="192.168.1.5" value={c.ipCelular} onChange={(ev) => camara.cambiarIp(ev.target.value)} style={{ width: 140 }} /></span>
          </label>
        )}
        {c.error && <p className="aviso-cerca">{c.error}</p>}
      </section>

      <div className="video-marco" onClick={tocar} role="button" tabIndex={0} aria-label="Imagen de la cámara: toca las esquinas de la cuadrícula">
        {c.usaDroidcam
          ? (c.ipCelular.trim() && <img ref={imagen} src={c.vistaDroidcam} crossOrigin="anonymous" alt="Cámara del celular" onLoad={() => camara.cambiarIp(c.ipCelular)} />)
          : <video ref={video} muted playsInline />}
        {vw > 0 && (
          <svg viewBox={`0 0 ${vw} ${vh}`} preserveAspectRatio="none">
            {c.esquinas.length > 1 && (
              <polygon points={c.esquinas.map((p) => p.join(',')).join(' ')} className={listo4 ? 'cuadro-listo' : 'cuadro'} />
            )}
            {c.esquinas.map((p, k) => (
              <g key={k}>
                <circle cx={p[0]} cy={p[1]} r={vw / 90} className="esquina" />
                <text x={p[0] + vw / 60} y={p[1] - vw / 80} className="esquina-texto" style={{ fontSize: vw / 35 }}>{k + 1}</text>
              </g>
            ))}
          </svg>
        )}
        {!c.encendida && <span className="video-vacio">{c.usaDroidcam ? 'Escribe la IP del celular' : 'Enciende la cámara'}</span>}
      </div>

      <section>
        <h3>2 · Marcar las esquinas de la cuadrícula</h3>
        <p className="ayuda">
          {listo4
            ? 'Esquinas listas. Si moviste el celular, vuelve a marcarlas.'
            : <>Toca en la imagen la esquina <b>{NOMBRES_ESQUINA[c.esquinas.length]}</b> de la cuadrícula ({c.esquinas.length} de 4).</>}
        </p>
        {c.esquinas.length > 0 && <button type="button" className="tecla chica" onClick={() => camara.borrarEsquinas()}>Borrar esquinas</button>}
      </section>

      <section>
        <h3>3 · Fotos</h3>
        <div className="fila-botones">
          <button type="button" className={'tecla chica' + (c.fondo ? ' hecho' : '')} disabled={!c.encendida} onClick={() => void camara.tomarFondo()}>
            {c.fondo ? '✓ ' : ''}Cartulina vacía
          </button>
          <button type="button" className={'tecla chica' + (c.foto ? ' hecho' : '')} disabled={!c.encendida || !c.fondo} onClick={() => void camara.tomarFoto()}>
            {c.foto ? '✓ ' : ''}Con el objeto
          </button>
        </div>
        <p className="ayuda">Primero sin nada sobre la cartulina; después pon el objeto y toma la segunda. No muevas el celular entre las dos.</p>
      </section>

      <section>
        <h3>4 · Resultado</h3>
        <label className="campo-fila" htmlFor="grosor">
          <span className="campo-nombre">Grosor mínimo<small>para objetos planos, como llaves</small></span>
          <span className="campo-valor">
            <input id="grosor" type="number" min={0.5} step={0.5} value={c.grosor} onChange={(ev) => camara.cambiarGrosor(Number(ev.target.value) || 1)} />
            <span className="unidad">mm</span>
          </span>
        </label>
        <Silueta />
        {!c.modelo && <p className="ayuda">{listo4 ? 'Toma las dos fotos para ver la silueta.' : 'Marca las 4 esquinas y toma las dos fotos.'}</p>}
      </section>
    </div>
  );
}
