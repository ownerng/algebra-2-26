// De la foto a una malla 3D: cada píxel del objeto se retroproyecta al espacio
// con la inversa de la matriz de la cámara,  P = z·K⁻¹·[u, v, 1]ᵀ,  y los
// píxeles vecinos se unen en triángulos. Se agregan una cara trasera y paredes
// en el contorno para que el objeto sea un sólido cerrado.
//
// Ejes del objeto: x a la derecha, y hacia arriba de la foto, z hacia la cámara
// (la altura sobre la base). Unidades: mm.

import type { Matriz } from './matematica.ts';
import type { Vec3 } from './algebra3d.ts';
import type { ModeloCamara } from './camara.ts';

const LADO_MAX = 200;   // vértices a lo largo del lado mayor del objeto
const ESPEJO = 0.4;     // la cara trasera repite el relieve del frente, más suave

// K⁻¹ de la cámara estenopeica (K es triangular superior: su inversa es directa)
export function inversaK(K: Matriz): Matriz {
  const f = K[0]![0]!, cx = K[0]![2]!, cy = K[1]![2]!;
  return [[1 / f, 0, -cx / f], [0, 1 / f, -cy / f], [0, 0, 1]];
}

// Distancia de referencia: la del punto más lejano del objeto (queda en la base)
function zFondo(m: ModeloCamara): number {
  let zMax = 0;
  for (let k = 0; k < m.zMm.length; k++) if (m.mascara[k]) zMax = Math.max(zMax, m.zMm[k]!);
  return zMax;
}

// Punto 3D (ejes del objeto) del píxel (u, v)
export function crearPunto3D(m: ModeloCamara): (u: number, v: number) => Vec3 {
  const Ki = inversaK(m.K);
  const zRef = zFondo(m);
  const uc = (m.caja.u0 + m.caja.u1 + 1) / 2, vc = (m.caja.v0 + m.caja.v1 + 1) / 2;
  const xc = (Ki[0]![0]! * uc + Ki[0]![2]!) * m.Z, yc = (Ki[1]![1]! * vc + Ki[1]![2]!) * m.Z;
  return (u, v) => {
    const k = Math.min(m.alto - 1, Math.max(0, Math.round(v))) * m.ancho + Math.min(m.ancho - 1, Math.max(0, Math.round(u)));
    const z = m.zMm[k]!;
    // P = z·K⁻¹·[u, v, 1]ᵀ
    const X = z * (Ki[0]![0]! * u + Ki[0]![2]!), Y = z * (Ki[1]![1]! * v + Ki[1]![2]!);
    return [X - xc, -(Y - yc), zRef - z + m.grosorBase];
  };
}

export interface Malla {
  posiciones: Float32Array;   // x, y, z por vértice
  uvs: Float32Array;          // coordenadas de la foto (textura)
  frente: number[];           // índices de los triángulos del frente
  atras: number[];
  paredes: number[];
}

export function construirMalla(m: ModeloCamara): Malla {
  const { u0, u1, v0, v1 } = m.caja;
  const paso = Math.max(1, Math.ceil(Math.max(u1 - u0 + 1, v1 - v0 + 1) / LADO_MAX));
  const cols = Math.floor((u1 - u0) / paso) + 1, filas = Math.floor((v1 - v0) / paso) + 1;
  const punto = crearPunto3D(m);
  const pos: number[] = [], uv: number[] = [];
  const id = new Int32Array(cols * filas).fill(-1);
  const nuevo = (p: Vec3, u: number, v: number) => {
    pos.push(p[0], p[1], p[2]);
    uv.push((u + 0.5) / m.ancho, 1 - (v + 0.5) / m.alto);
    return pos.length / 3 - 1;
  };

  // Vértices del frente (uno por nodo de la rejilla dentro del objeto)
  for (let f = 0; f < filas; f++) for (let c = 0; c < cols; c++) {
    const u = u0 + c * paso, v = v0 + f * paso;
    if (m.mascara[v * m.ancho + u]) id[f * cols + c] = nuevo(punto(u, v), u, v);
  }
  const nFrente = pos.length / 3;
  // Vértices de atrás: misma x, y; el relieve reflejado hacia abajo
  const base = m.grosorBase;
  for (let i = 0; i < nFrente; i++) {
    pos.push(pos[i * 3]!, pos[i * 3 + 1]!, -(pos[i * 3 + 2]! - base) * ESPEJO);
    uv.push(uv[i * 2]!, uv[i * 2 + 1]!);
  }

  const frente: number[] = [], atras: number[] = [];
  const aristas = new Map<number, [number, number, number]>(); // clave → [a, b, veces]
  const clave = (a: number, b: number) => (a < b ? a * nFrente + b : b * nFrente + a);
  const triangulo = (a: number, b: number, c: number) => {
    frente.push(a, b, c);
    atras.push(c + nFrente, b + nFrente, a + nFrente);
    for (const [x, y] of [[a, b], [b, c], [c, a]] as const) {
      const k = clave(x, y), e = aristas.get(k);
      if (e) e[2]++; else aristas.set(k, [x, y, 1]);
    }
  };
  // Cada celda de la rejilla: dos triángulos si las 4 esquinas son del objeto, uno si son 3
  for (let f = 0; f + 1 < filas; f++) for (let c = 0; c + 1 < cols; c++) {
    const a = id[f * cols + c]!, b = id[f * cols + c + 1]!, d = id[(f + 1) * cols + c]!, e = id[(f + 1) * cols + c + 1]!;
    // La v de la foto crece hacia abajo y la y del objeto hacia arriba: orden (a, d, b) mira a +z
    if (a >= 0 && b >= 0 && d >= 0 && e >= 0) { triangulo(a, d, b); triangulo(b, d, e); }
    else if (a >= 0 && b >= 0 && d >= 0) triangulo(a, d, b);
    else if (b >= 0 && d >= 0 && e >= 0) triangulo(b, d, e);
    else if (a >= 0 && d >= 0 && e >= 0) triangulo(a, d, e);
    else if (a >= 0 && b >= 0 && e >= 0) triangulo(a, e, b);
  }

  // Paredes: las aristas que usa un solo triángulo son el contorno
  const paredes: number[] = [];
  for (const [a, b, veces] of aristas.values()) {
    if (veces !== 1) continue;
    const pa = nuevo([pos[a * 3]!, pos[a * 3 + 1]!, pos[a * 3 + 2]!], 0, 0), pb = nuevo([pos[b * 3]!, pos[b * 3 + 1]!, pos[b * 3 + 2]!], 0, 0);
    const qa = nuevo([pos[(a + nFrente) * 3]!, pos[(a + nFrente) * 3 + 1]!, pos[(a + nFrente) * 3 + 2]!], 0, 0);
    const qb = nuevo([pos[(b + nFrente) * 3]!, pos[(b + nFrente) * 3 + 1]!, pos[(b + nFrente) * 3 + 2]!], 0, 0);
    for (const [n, o] of [[pa, a], [pb, b], [qa, a], [qb, b]] as const) { uv[n * 2] = uv[o * 2]!; uv[n * 2 + 1] = uv[o * 2 + 1]!; }
    paredes.push(pa, qa, pb, pb, qa, qb);
  }
  return { posiciones: new Float32Array(pos), uvs: new Float32Array(uv), frente, atras, paredes };
}

// Malla en formato OBJ (se abre en Blender, en el Visor 3D de Windows, etc.)
export function aObj(malla: Malla): string {
  const l = ['# Escáner 3D: foto + profundidad estimada + escala del VL53L0X (mm)'];
  const p = malla.posiciones, t = malla.uvs;
  for (let i = 0; i < p.length; i += 3) l.push(`v ${p[i]!.toFixed(2)} ${p[i + 1]!.toFixed(2)} ${p[i + 2]!.toFixed(2)}`);
  for (let i = 0; i < t.length; i += 2) l.push(`vt ${t[i]!.toFixed(4)} ${t[i + 1]!.toFixed(4)}`);
  const caras = [...malla.frente, ...malla.atras, ...malla.paredes];
  for (let i = 0; i < caras.length; i += 3) {
    const [a, b, c] = [caras[i]! + 1, caras[i + 1]! + 1, caras[i + 2]! + 1];
    l.push(`f ${a}/${a} ${b}/${b} ${c}/${c}`);
  }
  return l.join('\n') + '\n';
}
