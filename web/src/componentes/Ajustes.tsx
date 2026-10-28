// Ajustes del escaneo: tamaño de la cuadrícula, altura de la caja sobre la
// mesa, simulación y archivos (guardar, abrir y exportar la figura).

import { useRef, useState } from 'react';
import { escaneo, useEscaneo, type EscaneoGuardado } from '../lib/escaneo.ts';
import { OBJETOS, type ObjetoSimulado } from '../lib/simulador3d.ts';

function descargar(nombre: string, contenido: string, tipo: string) {
  const url = URL.createObjectURL(new Blob([contenido], { type: tipo }));
  const a = Object.assign(document.createElement('a'), { href: url, download: nombre });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function Numero({ id, etiqueta, ayuda, unidad, valor, min, alCambiar }:
  { id: string; etiqueta: string; ayuda: string; unidad: string; valor: number; min: number; alCambiar: (v: number) => void }) {
  return (
    <label className="campo-fila" htmlFor={id}>
      <span className="campo-nombre">{etiqueta}<small>{ayuda}</small></span>
      <span className="campo-valor">
        <input id={id} type="number" min={min} value={valor} onChange={(ev) => { const v = Number(ev.target.value); if (v >= min) alCambiar(v); }} />
        <span className="unidad">{unidad}</span>
      </span>
    </label>
  );
}

export function Ajustes() {
  const e = useEscaneo(120);
  const [nombre, setNombre] = useState('Taza');
  const archivo = useRef<HTMLInputElement>(null);
  const guardados = e.guardados();
  const enCurso = e.medidas.size === 0 ? e.recuperarEnCurso() : null;

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
    <section className="ajustes" aria-label="Ajustes y archivos">
      <div className="mando">
        <h3 className="placa-titulo">1 · Altura de la caja sobre la mesa</h3>
        <p className="ayuda">
          Pon la caja en su soporte sobre la mesa <b>vacía</b> y pulsa <b>Medir la mesa</b>: esa distancia es H. La
          altura de cada punto es H − r.
        </p>
        <Numero id="altura" etiqueta="Altura H" ayuda={e.alturaMedida ? 'medida con el sensor' : 'escrita a mano'} unidad="mm" min={20}
          valor={e.config.altura} alCambiar={(v) => escaneo.cambiarConfig({ altura: v })} />
        <button type="button" className="tecla chica" disabled={!e.fuente || e.midiendo} onClick={() => escaneo.medirMesa()}>
          {e.calibrando ? 'Midiendo la mesa…' : 'Medir la mesa'}
        </button>
      </div>

      <div className="mando">
        <h3 className="placa-titulo">2 · Tamaño de la cuadrícula</h3>
        <Numero id="nx" etiqueta="Columnas" ayuda="casillas en x (i)" unidad="" min={2} valor={e.config.nx} alCambiar={(v) => escaneo.cambiarConfig({ nx: Math.min(30, v) })} />
        <Numero id="ny" etiqueta="Filas" ayuda="casillas en y (j)" unidad="" min={2} valor={e.config.ny} alCambiar={(v) => escaneo.cambiarConfig({ ny: Math.min(30, v) })} />
        <Numero id="paso" etiqueta="Lado del cuadro" ayuda="Δ, igual al dibujado" unidad="mm" min={5} valor={e.config.paso} alCambiar={(v) => escaneo.cambiarConfig({ paso: v })} />
        <label className="casilla-check"><input type="checkbox" checked={e.corregirMesa} onChange={(ev) => escaneo.cambiar({ corregirMesa: ev.target.checked })} /> Corregir la inclinación de la mesa (mínimos cuadrados)</label>
        <p className="ayuda">Cambiar las columnas o filas borra el escaneo actual.</p>
      </div>

      {e.tipoFuente === 'simulacion' && (
        <div className="mando">
          <h3 className="placa-titulo">Simulación</h3>
          <label className="campo-fila" htmlFor="objeto">
            <span className="campo-nombre">Objeto<small>sobre la mesa virtual</small></span>
            <select id="objeto" value={e.objeto} onChange={(ev) => escaneo.cambiar({ objeto: ev.target.value as ObjetoSimulado })}>
              {Object.entries(OBJETOS).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
            </select>
          </label>
          <button type="button" className="tecla chica" onClick={() => escaneo.escanearAutomatico()}>
            {e.automatico ? '❚❚ Detener escaneo automático' : '▶ Escaneo automático'}
          </button>
          <p className="ayuda">En la simulación la caja puede recorrer la cuadrícula sola para ver cómo se arma la figura.</p>
        </div>
      )}

      <details className="mando plegable">
        <summary className="placa-titulo">3 · Guardar y exportar la figura</summary>
        <div className="fila-botones">
          <input id="nombre" className="texto" value={nombre} onChange={(ev) => setNombre(ev.target.value)} aria-label="Nombre del escaneo" />
          <button type="button" className="tecla chica" disabled={!e.medidas.size} onClick={() => escaneo.guardarComo(nombre || 'Escaneo')}>Guardar aquí</button>
        </div>
        {guardados.length > 0 && (
          <ul className="guardados">
            {guardados.map((g) => (
              <li key={g.nombre}>
                <span>{g.nombre} <small>{g.medidas.length} pts · {new Date(g.fecha).toLocaleString('es-CO', { dateStyle: 'short', timeStyle: 'short' })}</small></span>
                <button type="button" className="tecla chica" onClick={() => escaneo.cargar(g)}>Abrir</button>
                <button type="button" className="tecla chica" onClick={() => escaneo.borrarGuardado(g.nombre)}>✕</button>
              </li>
            ))}
          </ul>
        )}
        {enCurso && (
          <button type="button" className="tecla chica" onClick={() => escaneo.cargar(enCurso)}>Recuperar el escaneo sin terminar ({enCurso.medidas.length} pts)</button>
        )}
        <div className="fila-botones">
          <button type="button" className="tecla chica" disabled={!e.medidas.size} onClick={() => descargar(`${nombre || 'escaneo'}.obj`, escaneo.obj(), 'model/obj')}>Exportar 3D (.obj)</button>
          <button type="button" className="tecla chica" disabled={!e.medidas.size} onClick={() => descargar(`${nombre || 'escaneo'}.csv`, escaneo.csv(), 'text/csv')}>CSV</button>
          <button type="button" className="tecla chica" disabled={!e.medidas.size} onClick={() => descargar(`${nombre || 'escaneo'}.json`, JSON.stringify(escaneo.datos(nombre), null, 2), 'application/json')}>Archivo .json</button>
          <button type="button" className="tecla chica" onClick={() => archivo.current?.click()}>Abrir .json</button>
          <input ref={archivo} type="file" accept="application/json,.json" hidden onChange={(ev) => abrir(ev.target.files)} />
        </div>
        <p className="ayuda">“Guardar aquí” queda en este navegador. El .obj se abre en cualquier visor 3D (por ejemplo, el de Windows o Blender).</p>
      </details>
    </section>
  );
}
