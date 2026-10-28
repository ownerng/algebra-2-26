// Mandos: desde qué posición se mide, cuánto se movió el aparato entre las
// dos posiciones y, plegados, los ajustes y archivos.

import { useRef } from 'react';
import { escaner, useEscaner, type Posicion } from '../lib/escaner.ts';

function descargar(nombre: string, contenido: string, tipo: string) {
  const url = URL.createObjectURL(new Blob([contenido], { type: tipo }));
  const a = Object.assign(document.createElement('a'), { href: url, download: nombre });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportarCSV() {
  const filas = ['posicion,theta_servo,theta,r_mm,x_local_mm,y_local_mm,x_global_mm,y_global_mm'];
  const puntos = escaner.puntos();
  for (const id of [1, 2] as Posicion[]) {
    for (const p of puntos[id]) {
      filas.push([p.id, p.thetaServo, p.theta, p.r, p.xl, p.yl, p.x, p.y].map((v) => +v.toFixed(2)).join(','));
    }
  }
  descargar('escaneo-lidar.csv', filas.join('\n'), 'text/csv');
}

function Numero({ id, etiqueta, ayuda, unidad, valor, paso, alCambiar }:
  { id: string; etiqueta: string; ayuda: string; unidad: string; valor: number; paso: number; alCambiar: (v: number) => void }) {
  return (
    <label className="campo-fila" htmlFor={id}>
      <span className="campo-nombre">{etiqueta}<small>{ayuda}</small></span>
      <span className="campo-valor">
        <input id={id} type="number" step={paso} value={valor} onChange={(ev) => alCambiar(Number(ev.target.value) || 0)} />
        <span className="unidad">{unidad}</span>
      </span>
    </label>
  );
}

export function Controles() {
  const e = useEscaner(60);
  const archivo = useRef<HTMLInputElement>(null);
  const puntos = e.puntos();
  const pose2 = e.poses[2];

  const cargar = async (lista: FileList | null) => {
    const f = lista?.[0];
    if (!f) return;
    try {
      const datos = JSON.parse(await f.text());
      if (datos.formato !== 'escaner-lidar-v1') throw new Error('formato');
      escaner.cargarSesion(datos);
    } catch {
      alert('Ese archivo no es una sesión guardada por esta página.');
    }
    if (archivo.current) archivo.current.value = '';
  };

  return (
    <section className="controles" aria-label="Mandos">
      <div className="mando">
        <h3 className="placa-titulo">¿Desde dónde mides?</h3>
        <div className="selector" role="radiogroup" aria-label="Posición activa">
          {([1, 2] as Posicion[]).map((id) => (
            <button key={id} type="button" role="radio" aria-checked={e.posicionActiva === id}
              className={'opcion p' + id + (e.posicionActiva === id ? ' activa' : '')}
              onClick={() => escaner.cambiar({ posicionActiva: id })}>
              <span className="led" />P{id}
              <small>{puntos[id].length} puntos</small>
            </button>
          ))}
        </div>
        <p className="ayuda">
          <b>P1</b> es el primer lugar donde pones el aparato: ahí está el origen del mapa. <b>P2</b> es el segundo
          lugar, después de moverlo. Cada lectura se guarda en la posición elegida.
        </p>
      </div>

      <div className="mando">
        <h3 className="placa-titulo">¿Cuánto moviste el aparato? (de P1 a P2)</h3>
        <Numero id="dx" etiqueta="A la derecha" ayuda="dx" unidad="mm" paso={10} valor={pose2.dx} alCambiar={(v) => escaner.cambiarPose2({ dx: v })} />
        <Numero id="dy" etiqueta="Hacia adelante" ayuda="dy" unidad="mm" paso={10} valor={pose2.dy} alCambiar={(v) => escaner.cambiarPose2({ dy: v })} />
        <Numero id="alpha" etiqueta="Giro" ayuda="α, antihorario" unidad="°" paso={1} valor={pose2.alpha} alCambiar={(v) => escaner.cambiarPose2({ alpha: v })} />
        <input type="range" min={-180} max={180} step={1} value={pose2.alpha} aria-label="Giro α de P2"
          onChange={(ev) => escaner.cambiarPose2({ alpha: Number(ev.target.value) })} />
        <p className="ayuda">
          Mide con cinta desde donde estaba el eje del servo en P1 hasta donde quedó en P2. Estos tres números arman la
          matriz T. Si el mapa de P2 no encaja con el de P1, alguno está mal medido.
        </p>
      </div>

      <details className="mando plegable">
        <summary className="placa-titulo">Ajustes y archivos</summary>
        <Numero id="desfase" etiqueta="Del eje al lente" ayuda="d, se suma a cada lectura" unidad="mm" paso={1} valor={e.d} alCambiar={(v) => escaner.cambiar({ d: v })} />
        <label className="casilla"><input type="checkbox" checked={e.invertir} onChange={(ev) => escaner.cambiar({ invertir: ev.target.checked })} /> El servo gira al revés (usar 180° − θ)</label>
        <label className="casilla"><input type="checkbox" checked={e.grabando} onChange={(ev) => escaner.cambiar({ grabando: ev.target.checked })} /> Guardar los puntos en el mapa</label>
        {e.tipoFuente === 'simulacion' && (
          <label className="casilla"><input type="checkbox" checked={e.verHabitacion} onChange={(ev) => escaner.cambiar({ verHabitacion: ev.target.checked })} /> Mostrar la habitación real de la simulación</label>
        )}
        <div className="fila-botones">
          <button type="button" className="tecla chica" onClick={() => escaner.borrar(e.posicionActiva)}>Borrar puntos de P{e.posicionActiva}</button>
          <button type="button" className="tecla chica" onClick={exportarCSV}>Exportar CSV</button>
          <button type="button" className="tecla chica" onClick={() => descargar('sesion-lidar.json', JSON.stringify(escaner.sesion(), null, 2), 'application/json')}>Guardar sesión</button>
          <button type="button" className="tecla chica" onClick={() => archivo.current?.click()}>Abrir sesión</button>
          <input ref={archivo} type="file" accept="application/json,.json" hidden onChange={(ev) => cargar(ev.target.files)} />
        </div>
        <p className="ayuda">Teclas: <kbd>1</kbd> <kbd>2</kbd> cambian de posición · <kbd>espacio</kbd> pausa el barrido</p>
      </details>
    </section>
  );
}
