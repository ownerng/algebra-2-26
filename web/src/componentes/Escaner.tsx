// Pantalla principal: lo que ve la cámara, un botón para tomar la foto y, al
// tomarla, el objeto en 3D.

import { useEffect, useRef } from 'react';
import { camara, DROIDCAM, useCamara } from '../lib/camara.ts';
import { Vista3D } from './Vista3D.tsx';

const coma = (v: number) => v.toFixed(0);

export function Escaner() {
  const c = useCamara(100);
  const video = useRef<HTMLVideoElement>(null);

  useEffect(() => { void camara.listar(); void camara.revisarIA(); }, []);
  useEffect(() => {
    if (video.current && c.stream && video.current.srcObject !== c.stream) {
      video.current.srcObject = c.stream;
      void video.current.play();
    }
    camara.video = video.current;
  });

  const m = c.modelo;

  if (c.vista === '3d' && m) {
    return (
      <>
        <Vista3D />
        <div className="control">
          <div className="control-centro">
            <p className="medida">{coma(m.caja.anchoMm)} × {coma(m.caja.altoMm)} <small>mm</small></p>
            <p className="control-ayuda">
              {m.zDelSensor
                ? <>Tamaño real: el sensor midió el objeto a {coma(m.Z)} mm.</>
                : <>Tamaño aproximado: sin sensor se supone que el objeto está a {coma(m.Z)} mm (cámbialo en Ajustes).</>}
              {' '}{c.ia.glb ? 'Forma completa: TRELLIS 2 (IA de pesos abiertos).' : m.conIA ? 'La profundidad la estimó una red neuronal (Depth Anything V2).' : 'El volumen es una estimación a partir de la silueta.'}
            </p>
            {c.ia.estado && c.ia.estado !== 'listo' && (
              <p className={c.ia.estado === 'error' ? 'aviso-cerca' : 'control-ayuda ia-estado'}>
                {c.ia.estado === 'error'
                  ? `TRELLIS 2 no pudo generar el modelo: ${c.ia.mensaje}. Se muestra el relieve de la foto.`
                  : `Generando el modelo completo con TRELLIS 2… ${c.ia.estado === 'en cola' ? `(en cola${c.ia.mensaje ? ', ' + c.ia.mensaje : ''})` : ''} ${c.ia.segundos ? c.ia.segundos + ' s' : ''}`}
              </p>
            )}
            <button type="button" className="tecla grande ambar" onClick={() => camara.cambiar({ vista: 'camara' })}>📷 Tomar otra foto</button>
          </div>
        </div>
      </>
    );
  }

  const sinIp = c.usaDroidcam && !c.ipCelular.trim();
  return (
    <>
      <div className="visor">
        {c.usaDroidcam
          ? (!sinIp && <img src={c.vistaDroidcam} crossOrigin="anonymous" alt="Imagen de la cámara del celular" />)
          : <video ref={video} muted playsInline />}
        {(sinIp || (!c.usaDroidcam && !c.stream)) && (
          <div className="visor-vacio">
            <p className="control-titulo">Conecta la cámara</p>
            <label className="campo-fila" htmlFor="camara-sel">
              <span className="campo-nombre">Cámara</span>
              <select id="camara-sel" value={c.deviceId} onChange={(ev) => void camara.elegir(ev.target.value)}>
                <option value={DROIDCAM}>Celular por WiFi (DroidCam)</option>
                {c.dispositivos.map((d, k) => <option key={d.deviceId || k} value={d.deviceId}>{d.label || `Cámara del computador ${k + 1}`}</option>)}
              </select>
            </label>
            {c.usaDroidcam && (
              <label className="campo-fila" htmlFor="ip">
                <span className="campo-nombre">IP del celular<small>la "WiFi IP" que muestra DroidCam</small></span>
                <input id="ip" className="texto" placeholder="192.168.1.5" value={c.ipCelular} onChange={(ev) => camara.cambiarIp(ev.target.value)} />
              </label>
            )}
          </div>
        )}
        {!sinIp && (c.usaDroidcam || c.stream) && <div className="mira" aria-hidden="true" />}
      </div>

      {!sinIp && (c.usaDroidcam || c.stream) && (
        <div className="control">
          <div className="control-centro">
            {c.error && <p className="aviso-cerca">{c.error}</p>}
            <p className="control-ayuda">Apunta al objeto para que quede en el centro y ocupe buena parte de la imagen.</p>
            <button type="button" className="tecla enorme ambar" disabled={c.capturando} onClick={() => void camara.tomarFoto()}>
              {c.capturando ? 'Procesando…' : '📷 Tomar foto'}
            </button>
            {c.estadoTexto && <p className="control-ayuda">{c.estadoTexto}</p>}
          </div>
        </div>
      )}
    </>
  );
}
