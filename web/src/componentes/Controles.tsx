// Panel de mandos: posición activa, pose de P2, calibración y archivos.

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

function Numero({ id, etiqueta, unidad, valor, paso, alCambiar }:
  { id: string; etiqueta: string; unidad: string; valor: number; paso: number; alCambiar: (v: number) => void }) {
  return (
    <label className="campo" htmlFor={id}>
      <span>{etiqueta}</span>
      <input id={id} type="number" step={paso} value={valor} onChange={(ev) => alCambiar(Number(ev.target.value) || 0)} />
      <span className="unidad">{unidad}</span>
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
        <h3 className="placa-titulo">Posición del aparato</h3>
        <div className="selector" role="radiogroup" aria-label="Posición activa">
          {([1, 2] as Posicion[]).map((id) => (
            <button key={id} type="button" role="radio" aria-checked={e.posicionActiva === id}
              className={'opcion p' + id + (e.posicionActiva === id ? ' activa' : '')}
              onClick={() => escaner.cambiar({ posicionActiva: id })}>
              <span className="led" />P{id}
              <small>{puntos[id].length} pts</small>
            </button>
          ))}
        </div>
        <p className="ayuda">P1 es el origen del mapa. Mueve el aparato, mide cuánto se desplazó y giró, y escanea en P2.</p>
      </div>

      <div className="mando">
        <h3 className="placa-titulo">Pose de P2</h3>
        <div className="campos">
          <Numero id="dx" etiqueta="dx" unidad="mm" paso={10} valor={pose2.dx} alCambiar={(v) => escaner.cambiarPose2({ dx: v })} />
          <Numero id="dy" etiqueta="dy" unidad="mm" paso={10} valor={pose2.dy} alCambiar={(v) => escaner.cambiarPose2({ dy: v })} />
          <Numero id="alpha" etiqueta="α" unidad="°" paso={1} valor={pose2.alpha} alCambiar={(v) => escaner.cambiarPose2({ alpha: v })} />
        </div>
        <input type="range" min={-180} max={180} step={1} value={pose2.alpha} aria-label="Ángulo α de P2"
          onChange={(ev) => escaner.cambiarPose2({ alpha: Number(ev.target.value) })} />
      </div>

      <div className="mando">
        <h3 className="placa-titulo">Calibración</h3>
        <div className="campos">
          <Numero id="desfase" etiqueta="d" unidad="mm" paso={1} valor={e.d} alCambiar={(v) => escaner.cambiar({ d: v })} />
        </div>
        <label className="casilla"><input type="checkbox" checked={e.invertir} onChange={(ev) => escaner.cambiar({ invertir: ev.target.checked })} /> invertir ángulo (180° − θ)</label>
        <label className="casilla"><input type="checkbox" checked={e.grabando} onChange={(ev) => escaner.cambiar({ grabando: ev.target.checked })} /> guardar puntos</label>
        {e.tipoFuente === 'simulacion' && (
          <label className="casilla"><input type="checkbox" checked={e.verHabitacion} onChange={(ev) => escaner.cambiar({ verHabitacion: ev.target.checked })} /> ver habitación real</label>
        )}
      </div>

      <div className="mando">
        <h3 className="placa-titulo">Datos</h3>
        <div className="fila-botones">
          <button type="button" className="tecla" onClick={() => escaner.borrar(e.posicionActiva)}>Borrar P{e.posicionActiva}</button>
          <button type="button" className="tecla" onClick={exportarCSV}>CSV</button>
          <button type="button" className="tecla" onClick={() => descargar('sesion-lidar.json', JSON.stringify(escaner.sesion(), null, 2), 'application/json')}>Guardar</button>
          <button type="button" className="tecla" onClick={() => archivo.current?.click()}>Cargar</button>
          <input ref={archivo} type="file" accept="application/json,.json" hidden onChange={(ev) => cargar(ev.target.files)} />
        </div>
        <p className="ayuda">Teclas: <kbd>1</kbd> <kbd>2</kbd> posición · <kbd>espacio</kbd> pausa</p>
      </div>
    </section>
  );
}
