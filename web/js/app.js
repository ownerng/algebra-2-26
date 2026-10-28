import { procesarLectura, rotacion, det2 } from './matematica.js';
import { ConexionSerial } from './conexion.js';
import { Simulador } from './simulador.js';
import { Radar2D } from './radar2d.js';

const COLORES = { 1: '#26e7ff', 2: '#ff3fa4' };
const $ = (id) => document.getElementById(id);

const estado = {
  posicionActiva: 1,
  grabando: true,
  d: 15,
  invertir: false,
  poses: { 1: { dx: 0, dy: 0, alpha: 0 }, 2: { dx: 250, dy: 1100, alpha: 180 } },
  escaneos: { 1: new Map(), 2: new Map() },  // thetaServo -> { r, t }
  ultima: null,                               // { r, thetaServo, posicion, t }
  direccion: 1,
  servo: 'listo',
  version: 0,
  lecturas: [],                               // marcas de tiempo para lecturas/s
};

let conexion = null;
let vista3d = null;
let vistaActual = '2d';
const radar = new Radar2D($('radar'));

// ---------- Datos entrantes ----------

function alDato(dato) {
  if (dato.error) return ponerEstado('error', dato.error);
  if (dato.estado) {
    estado.servo = dato.estado;
    const textos = { listo: 'listo', escaneando: 'escaneando', pausado: 'pausado' };
    $('estadoTexto').textContent = `${conexion instanceof Simulador ? 'Demo' : 'ESP32'} · ${textos[dato.estado] ?? dato.estado}`;
    return;
  }
  if (typeof dato.theta !== 'number' || typeof dato.r !== 'number') return;

  const ahora = performance.now();
  const anterior = estado.ultima;
  if (anterior && dato.theta !== anterior.thetaServo) estado.direccion = dato.theta > anterior.thetaServo ? 1 : -1;
  estado.ultima = { r: dato.r, thetaServo: dato.theta, posicion: estado.posicionActiva, t: ahora };
  if (dato.r > 0) estado.ultimaValida = estado.ultima;
  estado.lecturas.push(ahora);

  if (estado.grabando) {
    const mapa = estado.escaneos[estado.posicionActiva];
    if (dato.r > 0) mapa.set(dato.theta, { r: dato.r, t: ahora });
    else mapa.delete(dato.theta);
    estado.version++;
  }
  $('aviso').hidden = true;
}

function ponerEstado(tipo, texto) {
  const el = $('estado');
  el.className = 'estado ' + tipo;
  $('estadoTexto').textContent = texto;
  const conectado = tipo === 'conectado' || (tipo === 'error' && conexion);
  $('btnDesconectar').hidden = !conectado;
  $('btnConectar').hidden = conectado;
  $('btnDemo').hidden = conectado;
  for (const id of ['btnIniciar', 'btnPausar', 'btnCentrar']) $(id).disabled = !conectado;
}

async function iniciarConexion(tipo) {
  if (conexion) await conexion.desconectar();
  const opciones = { alDato, alEstado: ponerEstado, posicionActiva: () => estado.posicionActiva };
  conexion = tipo === 'serial' ? new ConexionSerial(opciones) : new Simulador(opciones);
  try {
    await conexion.conectar();
  } catch (error) {
    conexion = null;
    if (error.name === 'NotFoundError') return ponerEstado('desconectado', 'No elegiste ningún puerto');
    ponerEstado('error', 'No se pudo abrir el puerto. ¿Está abierto el monitor serial del Arduino IDE?');
  }
}

// ---------- Geometría derivada ----------

let cache = { version: -1 };
function puntosGlobales() {
  if (cache.version === estado.version) return cache;
  const escaneos = [1, 2].map((id) => {
    const pose = estado.poses[id];
    const puntos = [...estado.escaneos[id].entries()]
      .map(([thetaServo, { r, t }]) => {
        const c = procesarLectura({ r, thetaServo, d: estado.d, invertir: estado.invertir, pose });
        return { id, thetaServo, theta: c.theta, r, t, xl: c.local[0], yl: c.local[1], x: c.global[0], y: c.global[1] };
      })
      .sort((a, b) => a.theta - b.theta);
    return { id, color: COLORES[id], puntos };
  });
  cache = { version: estado.version, escaneos };
  $('cuenta1').textContent = escaneos[0].puntos.length + ' puntos';
  $('cuenta2').textContent = escaneos[1].puntos.length + ' puntos';
  return cache;
}

function construirVista(ahora) {
  const { escaneos } = puntosGlobales();
  const poses = [1, 2].map((id) => ({
    id, pose: estado.poses[id], color: COLORES[id], activa: id === estado.posicionActiva, nombre: 'P' + id,
  }));
  let haz = null;
  const u = estado.ultima;
  if (u && ahora - u.t < 1500) {
    const theta = estado.invertir ? 180 - u.thetaServo : u.thetaServo;
    const dir = estado.invertir ? -estado.direccion : estado.direccion;
    haz = { pose: estado.poses[u.posicion], theta, direccion: dir, r: u.r > 0 ? u.r + estado.d : -1, color: COLORES[u.posicion] };
  }
  return { version: estado.version, poses, escaneos, haz, ahora };
}

// ---------- Panel de matrices ----------

const f = (n) => {
  const v = Math.abs(n) < 5e-4 ? 0 : n;
  return Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(3);
};
const matriz = (filas, clase = '') => {
  const cols = filas[0].length;
  const celdas = filas.flat().map((v) => `<span>${typeof v === 'number' ? f(v) : v}</span>`).join('');
  return `<span class="m c${cols} ${clase}">${celdas}</span>`;
};
let panelMostrado = null;
let versionPanel = -1;
const col = (v, clase = '') => matriz(v.map((x) => [x]), clase);

function actualizarPanel() {
  // El panel muestra la última lectura con eco; los ángulos sin objeto solo se ven en el HUD
  const u = estado.ultimaValida ?? estado.ultima;
  if (!u || (u === panelMostrado && estado.version === versionPanel)) return;
  panelMostrado = u;
  versionPanel = estado.version;
  const pos = u.posicion;
  const pose = estado.poses[pos];
  const r = u.r > 0 ? u.r : null;
  const c = procesarLectura({ r: r ?? 0, thetaServo: u.thetaServo, d: estado.d, invertir: estado.invertir, pose });
  const claseRes = 'res' + (pos === 2 ? ' p2' : '');

  $('sub1').innerHTML = `Posición ${pos} · θ = <b>${c.theta}°</b> · r = <b>${r ?? 'sin eco'}${r ? ' mm' : ''}</b> · d = ${estado.d} mm`;
  if (r === null) {
    $('ec1').innerHTML = '<span class="nota">El sensor no detectó nada a menos de 1,2 m en este ángulo.</span>';
    $('ec2').innerHTML = '';
    $('nota1').textContent = '';
    $('nota2').textContent = '';
  } else {
    $('ec1').innerHTML = `${col(c.local, claseRes)}<span class="op">=</span>${matriz(c.R)}${col(c.vectorPolar)}`;
    const norma = Math.hypot(c.local[0], c.local[1]);
    $('nota1').innerHTML = `‖(x<sub>L</sub>, y<sub>L</sub>)‖ = <b>${f(norma)} mm</b> = r + d. La rotación no cambia la distancia, solo la dirección.`;
    $('ec2').innerHTML = `${col(c.global, claseRes)}<span class="op">=</span>${matriz(c.T)}${col(c.homLocal)}`;
    $('nota2').innerHTML = pos === 1
      ? 'La posición 1 define el sistema global: α = 0, dx = dy = 0, así que <b>T = I</b> y el punto no cambia.'
      : `El punto medido desde P2 queda en <b>(${f(c.global[0])}, ${f(c.global[1])}) mm</b> del mapa global, junto a los puntos de P1.`;
  }
  $('sub2').innerHTML = `Pose de la posición ${pos}: α = <b>${pose.alpha}°</b>, dx = <b>${pose.dx} mm</b>, dy = <b>${pose.dy} mm</b>`;
  for (const id of ['ec1', 'ec2']) {
    const el = $(id);
    el.classList.remove('destello');
    void el.offsetWidth;
    el.classList.add('destello');
  }
}

function actualizarCambioBase() {
  const R = rotacion(estado.poses[2].alpha);
  const [c, s] = [R[0][0], R[1][0]];
  const marcado = [[`<b class="cx">${f(R[0][0])}</b>`, `<b class="cy">${f(R[0][1])}</b>`], [`<b class="cx">${f(R[1][0])}</b>`, `<b class="cy">${f(R[1][1])}</b>`]];
  $('ec3').innerHTML = `<span class="op">R(${estado.poses[2].alpha}°) =</span>${matriz(marcado)}`;
  const punto = c * -s + s * c;
  $('nota3').innerHTML = `x₂ = <b class="cx">(${f(c)}, ${f(s)})</b>, y₂ = <b class="cy">(${f(-s)}, ${f(c)})</b>. ` +
    `det R = <b>${f(det2(R))}</b> y x₂ · y₂ = <b>${f(punto)}</b>: base ortonormal, por eso el mapa no se deforma al rotarlo.`;
}

// ---------- Bucle de dibujo ----------

let ultimoPanel = -Infinity;
function cuadro() {
  // Mismo reloj que las lecturas (performance.now), no la marca del rAF
  const ahora = performance.now();
  const vista = construirVista(ahora);
  if (vistaActual === '2d') radar.dibujar(vista);
  else vista3d?.dibujar(vista);

  if (ahora - ultimoPanel > 120) {
    ultimoPanel = ahora;
    actualizarPanel();
    estado.lecturas = estado.lecturas.filter((t) => ahora - t < 1000);
    $('ritmo').textContent = estado.lecturas.length + ' lecturas/s';
    const u = estado.ultima;
    if (u) {
      const th = estado.invertir ? 180 - u.thetaServo : u.thetaServo;
      $('hud').textContent = `P${u.posicion}   θ = ${String(th).padStart(3)}°   r = ${u.r > 0 ? String(u.r).padStart(4) + ' mm' : ' sin eco'}`;
    }
  }
  requestAnimationFrame(cuadro);
}

// ---------- Controles ----------

function elegirPosicion(id) {
  estado.posicionActiva = id;
  document.querySelectorAll('.pos[data-pos]').forEach((el) => el.classList.toggle('activa', el.dataset.pos === String(id)));
  document.querySelector(`input[name="posicion"][value="${id}"]`).checked = true;
}

function leerPose() {
  const n = (id) => Number($(id).value) || 0;
  estado.poses[2] = { dx: n('dx'), dy: n('dy'), alpha: n('alpha') };
  $('alphaSlider').value = estado.poses[2].alpha;
  estado.version++;
  actualizarCambioBase();
}

$('btnConectar').addEventListener('click', () => iniciarConexion('serial'));
$('btnDemo').addEventListener('click', () => iniciarConexion('demo'));
$('btnDesconectar').addEventListener('click', async () => { await conexion?.desconectar(); conexion = null; });
$('btnIniciar').addEventListener('click', () => conexion?.enviar('S'));
$('btnPausar').addEventListener('click', () => conexion?.enviar('P'));
$('btnCentrar').addEventListener('click', () => conexion?.enviar('C'));
$('btnEncuadrar').addEventListener('click', () => radar.encuadrar([estado.poses[1], estado.poses[2]]));

if (!ConexionSerial.disponible()) {
  $('btnConectar').disabled = true;
  $('btnConectar').title = 'Este navegador no tiene Web Serial. Usa Chrome, Chromium o Edge.';
}

document.querySelectorAll('input[name="posicion"]').forEach((r) => r.addEventListener('change', () => elegirPosicion(Number(r.value))));
for (const id of ['dx', 'dy', 'alpha']) $(id).addEventListener('input', leerPose);
$('alphaSlider').addEventListener('input', (e) => { $('alpha').value = e.target.value; leerPose(); });
$('desfase').addEventListener('input', (e) => { estado.d = Number(e.target.value) || 0; estado.version++; });
$('invertir').addEventListener('change', (e) => { estado.invertir = e.target.checked; estado.version++; });
$('grabar').addEventListener('change', (e) => { estado.grabando = e.target.checked; });
document.querySelectorAll('[data-limpiar]').forEach((b) => b.addEventListener('click', () => {
  estado.escaneos[b.dataset.limpiar].clear();
  estado.version++;
}));

document.querySelectorAll('.pestana').forEach((tab) => tab.addEventListener('click', async () => {
  vistaActual = tab.dataset.vista;
  document.querySelectorAll('.pestana').forEach((t) => {
    t.classList.toggle('activa', t === tab);
    t.setAttribute('aria-selected', t === tab);
  });
  const es3d = vistaActual === '3d';
  $('radar').hidden = es3d;
  $('contenedor3d').hidden = !es3d;
  $('btnEncuadrar').hidden = es3d;
  $('chkRotar').hidden = !es3d;
  if (es3d && !vista3d) {
    const { Vista3D } = await import('./vista3d.js');
    vista3d = new Vista3D($('contenedor3d'));
    vista3d.rotarSolo = $('rotar').checked;
  }
  if (vista3d) {
    vista3d.visible = es3d;
    vista3d.version = -1;
    vista3d.redimensionar();
  }
}));
$('rotar').addEventListener('change', (e) => { if (vista3d) vista3d.rotarSolo = e.target.checked; });

document.addEventListener('keydown', (e) => {
  if (e.target.matches('input')) return;
  if (e.key === '1' || e.key === '2') elegirPosicion(Number(e.key));
  if (e.key === ' ' && conexion) {
    e.preventDefault();
    conexion.enviar(estado.servo === 'pausado' ? 'S' : 'P');
  }
});

// ---------- Archivos ----------

function descargar(nombre, contenido, tipo) {
  const url = URL.createObjectURL(new Blob([contenido], { type: tipo }));
  const a = Object.assign(document.createElement('a'), { href: url, download: nombre });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

$('btnCSV').addEventListener('click', () => {
  const filas = ['posicion,theta_servo,theta,r_mm,x_local_mm,y_local_mm,x_global_mm,y_global_mm'];
  for (const esc of puntosGlobales().escaneos) {
    for (const p of esc.puntos) {
      filas.push([p.id, p.thetaServo, p.theta, p.r, p.xl, p.yl, p.x, p.y].map((v) => (typeof v === 'number' ? +v.toFixed(2) : v)).join(','));
    }
  }
  descargar('escaneo-lidar.csv', filas.join('\n'), 'text/csv');
});

$('btnGuardar').addEventListener('click', () => {
  const datos = {
    formato: 'escaner-lidar-v1',
    d: estado.d,
    invertir: estado.invertir,
    poses: estado.poses,
    escaneos: { 1: [...estado.escaneos[1]].map(([theta, { r }]) => ({ theta, r })), 2: [...estado.escaneos[2]].map(([theta, { r }]) => ({ theta, r })) },
  };
  descargar('sesion-lidar.json', JSON.stringify(datos, null, 2), 'application/json');
});

$('archivoCargar').addEventListener('change', async (e) => {
  const archivo = e.target.files[0];
  if (!archivo) return;
  try {
    const datos = JSON.parse(await archivo.text());
    if (datos.formato !== 'escaner-lidar-v1') throw new Error('formato');
    estado.d = datos.d; $('desfase').value = datos.d;
    estado.invertir = datos.invertir; $('invertir').checked = datos.invertir;
    $('dx').value = datos.poses[2].dx; $('dy').value = datos.poses[2].dy; $('alpha').value = datos.poses[2].alpha;
    for (const id of [1, 2]) {
      estado.escaneos[id] = new Map(datos.escaneos[id].map(({ theta, r }) => [theta, { r, t: 0 }]));
    }
    leerPose();
    $('aviso').hidden = true;
    radar.encuadrar([estado.poses[1], estado.poses[2]]);
  } catch {
    ponerEstado('error', 'Ese archivo no es una sesión guardada por esta página');
  }
  e.target.value = '';
});

// ---------- Inicio ----------

elegirPosicion(1);
leerPose();
radar.encuadrar([estado.poses[1], estado.poses[2]]);
requestAnimationFrame(cuadro);

// index.html#demo arranca el simulador sin tocar nada (útil para ensayar la presentación)
if (location.hash.startsWith('#demo')) iniciarConexion('demo');
if (location.hash === '#demo-3d') document.querySelector('.pestana[data-vista="3d"]').click();
