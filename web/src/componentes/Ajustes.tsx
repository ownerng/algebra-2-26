// Ajustes (en un panel lateral): cuadrícula, velocidad, simulación y archivos.

import { useRef, useState } from 'react';
import { escaneo, useEscaneo, type EscaneoGuardado } from '../lib/escaneo.ts';
import { OBJETOS, type ObjetoSimulado } from '../lib/simulador3d.ts';

function descargar(nombre: string, contenido: string, tipo: string) {
  const url = URL.createObjectURL(new Blob([contenido], { type: tipo }));
  const a = Object.assign(document.createElement('a'), { href: url, download: nombre });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function Numero({ id, etiqueta, unidad, valor, min, alCambiar }:
  { id: string; etiqueta: string; unidad: string; valor: number; min: number; alCambiar: (v: number) => void }) {
  return (
    <label className="campo-fila" htmlFor={id}>
      <span className="campo-nombre">{etiqueta}</span>
      <span className="campo-valor">
        <input id={id} type="number" min={min} value={valor} onChange={(ev) => { const v = Number(ev.target.value); if (v >= min) alCambiar(v); }} />
        <span className="unidad">{unidad}</span>
      </span>
    </label>
  );
}

export function Ajustes() {
  const e = useEscaneo(150);
  const [nombre, setNombre] = useState('Objeto');
  const archivo = useRef<HTMLInputElement>(null);
  const guardados = e.guardados();
  const escaneando = !!e.barrido && e.barrido.fase !== 'fin';
  const { nx, ny, paso } = e.config;

  const abrir = async (lista: FileList | null) => {
    const f = lista?.[0];
    if (!f) return;
    try {
      const d = JSON.parse(await f.text()) as EscaneoGuardado;
      if (d.formato !== 'escaner-3d-v1') throw new Error('formato');
      escaneo.cargar(d);
    } catch {
      alert('Ese archivo no es un escaneo guardado por esta página.');
    }
    if (archivo.current) archivo.current.value = '';
  };

  return (
    <div className="ajustes-panel">
      <h2>Ajustes</h2>

      <section>
        <h3>Cuadrícula</h3>
        <p className="ayuda">Área: {(nx * paso) / 10} × {(ny * paso) / 10} cm. Dibújala igual en la cartulina.</p>
        <Numero id="nx" etiqueta="Columnas" unidad="" min={2} valor={nx} alCambiar={(v) => escaneo.cambiarConfig({ nx: Math.min(30, v) })} />
        <Numero id="ny" etiqueta="Filas" unidad="" min={2} valor={ny} alCambiar={(v) => escaneo.cambiarConfig({ ny: Math.min(30, v) })} />
        <Numero id="paso" etiqueta="Lado del cuadro" unidad="mm" min={5} valor={paso} alCambiar={(v) => escaneo.cambiarConfig({ paso: v })} />
      </section>

      <section>
        <h3>Velocidad de la marca</h3>
        <div className="velocidades" role="radiogroup" aria-label="Velocidad">
          {[20, 30, 40].map((v) => (
            <button key={v} type="button" role="radio" aria-checked={e.velocidad === v} disabled={escaneando}
              className={'alcance' + (e.velocidad === v ? ' activo' : '')} onClick={() => escaneo.cambiar({ velocidad: v })}>
              {v === 20 ? 'lenta' : v === 30 ? 'normal' : 'rápida'} · {v / 10} cm/s
            </button>
          ))}
        </div>
        <label className="casilla-check"><input type="checkbox" defaultChecked={escaneo.vista.girar} onChange={(ev) => { escaneo.vista.girar = ev.target.checked; }} /> Girar la figura sola</label>
        <label className="casilla-check"><input type="checkbox" checked={e.corregirMesa} onChange={(ev) => escaneo.cambiar({ corregirMesa: ev.target.checked })} /> Corregir la inclinación de la mesa</label>
      </section>

      {e.tipoFuente === 'simulacion' && (
        <section>
          <h3>Objeto simulado</h3>
          <select value={e.objeto} onChange={(ev) => escaneo.cambiar({ objeto: ev.target.value as ObjetoSimulado })} aria-label="Objeto simulado">
            {Object.entries(OBJETOS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
          </select>
        </section>
      )}

      <section>
        <h3>Guardar y exportar</h3>
        <div className="fila-botones">
          <input className="texto" value={nombre} onChange={(ev) => setNombre(ev.target.value)} aria-label="Nombre del escaneo" />
          <button type="button" className="tecla chica" disabled={!e.medidas.size} onClick={() => escaneo.guardarComo(nombre || 'Escaneo')}>Guardar</button>
        </div>
        {guardados.length > 0 && (
          <ul className="guardados">
            {guardados.map((g) => (
              <li key={g.nombre}>
                <span>{g.nombre} <small>{g.medidas.length} pts</small></span>
                <button type="button" className="tecla chica" onClick={() => escaneo.cargar(g)}>Abrir</button>
                <button type="button" className="tecla chica" onClick={() => escaneo.borrarGuardado(g.nombre)} aria-label={`Borrar ${g.nombre}`}>✕</button>
              </li>
            ))}
          </ul>
        )}
        <div className="fila-botones">
          <button type="button" className="tecla chica" disabled={!e.medidas.size} onClick={() => descargar(`${nombre || 'escaneo'}.json`, JSON.stringify(escaneo.datos(nombre), null, 2), 'application/json')}>Descargar .json</button>
          <button type="button" className="tecla chica" disabled={!e.medidas.size} onClick={() => descargar(`${nombre || 'escaneo'}.obj`, escaneo.obj(), 'model/obj')}>Modelo 3D .obj</button>
          <button type="button" className="tecla chica" onClick={() => archivo.current?.click()}>Abrir .json</button>
          <input ref={archivo} type="file" accept="application/json,.json" hidden onChange={(ev) => abrir(ev.target.files)} />
        </div>
      </section>
    </div>
  );
}
