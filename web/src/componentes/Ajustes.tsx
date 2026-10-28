// Ajustes (panel lateral): cámara, sensor de distancia, forma 3D y exportar.

import { camara, DROIDCAM, useCamara } from '../lib/camara.ts';
import { escaneo } from '../lib/escaneo.ts';
import { caras } from './Vista3D.tsx';

function descargar(nombre: string, contenido: string, tipo: string) {
  const url = URL.createObjectURL(new Blob([contenido], { type: tipo }));
  const a = Object.assign(document.createElement('a'), { href: url, download: nombre });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// Malla en formato OBJ (se abre en cualquier visor 3D)
function obj(): string {
  const m = camara.modelo;
  if (!m) return '';
  const lineas = ['# Escáner 3D: silueta de la cámara y tamaño con el VL53L0X (unidades: mm)'];
  let n = 0;
  for (const c of caras(m)) {
    for (const p of c.pts) lineas.push(`v ${p[0].toFixed(2)} ${p[1].toFixed(2)} ${p[2].toFixed(2)}`);
    lineas.push(`f ${n + 1} ${n + 2} ${n + 3} ${n + 4}`);
    n += 4;
  }
  return lineas.join('\n') + '\n';
}

function Numero({ id, etiqueta, ayuda, unidad, valor, min, max, paso, alCambiar }:
  { id: string; etiqueta: string; ayuda: string; unidad: string; valor: number; min: number; max: number; paso: number; alCambiar: (v: number) => void }) {
  return (
    <label className="campo-fila" htmlFor={id}>
      <span className="campo-nombre">{etiqueta}<small>{ayuda}</small></span>
      <span className="campo-valor">
        <input id={id} type="number" min={min} max={max} step={paso} value={valor} onChange={(ev) => { const v = Number(ev.target.value); if (v >= min && v <= max) alCambiar(v); }} />
        <span className="unidad">{unidad}</span>
      </span>
    </label>
  );
}

export function Ajustes() {
  const c = useCamara(150);
  const sensor = escaneo.tipoFuente === 'serial';
  const vivo = escaneo.vivo && performance.now() - escaneo.vivo.t < 1500 && escaneo.vivo.r > 0 ? escaneo.vivo.r : null;

  return (
    <div className="ajustes-panel">
      <h2>Ajustes</h2>

      <section>
        <h3>Cámara</h3>
        <select value={c.deviceId} onChange={(ev) => void camara.elegir(ev.target.value)} aria-label="Cámara">
          <option value={DROIDCAM}>Celular por WiFi (DroidCam)</option>
          {c.dispositivos.map((d, k) => <option key={d.deviceId || k} value={d.deviceId}>{d.label || `Cámara del computador ${k + 1}`}</option>)}
        </select>
        {c.usaDroidcam && (
          <label className="campo-fila" htmlFor="ip-aj">
            <span className="campo-nombre">IP del celular<small>la "WiFi IP" de DroidCam</small></span>
            <input id="ip-aj" className="texto" placeholder="192.168.1.5" value={c.ipCelular} onChange={(ev) => camara.cambiarIp(ev.target.value)} />
          </label>
        )}
        <Numero id="angulo" etiqueta="Ángulo de visión" ayuda="horizontal; la mayoría de celulares, 60–70°" unidad="°" min={20} max={120} paso={1}
          valor={c.anguloVision} alCambiar={(v) => camara.cambiar({ anguloVision: v })} />
      </section>

      <section>
        <h3>Sensor de distancia (tamaño real)</h3>
        <p className="ayuda">
          El VL53L0X va junto a la cámara del celular y mide a qué distancia está el objeto. Con esa distancia, cada píxel
          de la foto se convierte en milímetros.
        </p>
        {sensor
          ? <p className="control-ayuda">Sensor conectado · mide <b>{vivo ? `${vivo} mm` : 'sin eco'}</b></p>
          : <button type="button" className="tecla chica" onClick={() => void escaneo.conectar('serial')}>Conectar el sensor (ESP32)</button>}
        <Numero id="dist" etiqueta="Distancia sin sensor" ayuda="se usa si el sensor no está conectado" unidad="mm" min={50} max={3000} paso={10}
          valor={c.distanciaManual} alCambiar={(v) => camara.cambiar({ distanciaManual: v })} />
      </section>

      <section>
        <h3>Forma 3D</h3>
        <Numero id="grosor" etiqueta="Grosor de la base" ayuda="para objetos planos, como llaves" unidad="mm" min={0.5} max={50} paso={0.5}
          valor={c.grosor} alCambiar={(v) => camara.cambiar({ grosor: v })} />
        <label className="campo-fila" htmlFor="relieve">
          <span className="campo-nombre">Relieve<small>0 = plano · más alto en el centro de la silueta</small></span>
          <input id="relieve" type="range" min={0} max={1} step={0.05} value={c.relieve} onChange={(ev) => camara.cambiar({ relieve: Number(ev.target.value) })} />
        </label>
        <label className="casilla-check"><input type="checkbox" defaultChecked={escaneo.vista.girar} onChange={(ev) => { escaneo.vista.girar = ev.target.checked; }} /> Girar la figura sola</label>
      </section>

      <section>
        <h3>Exportar</h3>
        <button type="button" className="tecla chica" disabled={!c.modelo} onClick={() => descargar('objeto.obj', obj(), 'model/obj')}>Modelo 3D (.obj)</button>
        <p className="ayuda">El .obj se abre en el Visor 3D de Windows, en Blender o en cualquier visor en línea.</p>
      </section>
    </div>
  );
}
