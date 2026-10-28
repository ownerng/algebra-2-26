// Álgebra lineal del escáner 3D. Todo está escrito a mano (sin bibliotecas)
// para que el código siga las ecuaciones del informe.

import { multiplicarMat, multiplicarMatVec, type Matriz, type Vector } from './matematica.ts';

export type Vec3 = [number, number, number];

const aRad = (g: number) => (g * Math.PI) / 180;

// ---------- 1. De la casilla al espacio ----------
// La caja está en la casilla (i, j) a una altura H sobre la mesa y mide r hacia
// abajo. El punto que toca el haz es
//   [x, y, z, 1]ᵀ = A · [i, j, r, 1]ᵀ,   A = [Δ 0 0 0 ; 0 Δ 0 0 ; 0 0 −1 H ; 0 0 0 1]
// A = T(0,0,H) · F · E(Δ, Δ, 1): E escala la cuadrícula a milímetros, F es la
// reflexión que voltea el eje z (el sensor mide hacia abajo, det F = −1) y T
// sube el origen del haz a la altura H del lente sobre la mesa.
export function matrizCasilla(paso: number, altura: number): Matriz {
  return [
    [paso, 0, 0, 0],
    [0, paso, 0, 0],
    [0, 0, -1, altura],
    [0, 0, 0, 1],
  ];
}

export function descomposicionCasilla(paso: number, altura: number) {
  const E = [[paso, 0, 0, 0], [0, paso, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
  const F = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]];
  const T = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, altura], [0, 0, 0, 1]];
  return { E, F, T };
}

export function puntoDeCasilla(i: number, j: number, r: number, paso: number, altura: number): Vec3 {
  const [x, y, z] = multiplicarMatVec(matrizCasilla(paso, altura), [i, j, r, 1]);
  return [x!, y!, z!];
}

// ---------- 2. Plano de la mesa por mínimos cuadrados ----------
// Se busca z = a·x + b·y + c que mejor ajusta los puntos de la mesa:
//   M = [x y 1] (una fila por punto),   (MᵀM) [a b c]ᵀ = Mᵀ z
export interface Plano {
  a: number;
  b: number;
  c: number;
  MtM: Matriz;
  Mtz: Vector;
  n: number;
}

export function resolver3x3(A: Matriz, b: Vector): Vector | null {
  // Eliminación de Gauss con pivoteo parcial
  const M = A.map((fila, i) => [...fila, b[i]!]);
  for (let col = 0; col < 3; col++) {
    let piv = col;
    for (let f = col + 1; f < 3; f++) if (Math.abs(M[f]![col]!) > Math.abs(M[piv]![col]!)) piv = f;
    if (Math.abs(M[piv]![col]!) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv]!, M[col]!];
    for (let f = 0; f < 3; f++) {
      if (f === col) continue;
      const k = M[f]![col]! / M[col]![col]!;
      for (let c = col; c < 4; c++) M[f]![c]! -= k * M[col]![c]!;
    }
  }
  return [0, 1, 2].map((i) => M[i]![3]! / M[i]![i]!);
}

export function ajustarPlano(puntos: Vec3[]): Plano | null {
  if (puntos.length < 3) return null;
  const MtM: Matriz = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const Mtz: Vector = [0, 0, 0];
  for (const [x, y, z] of puntos) {
    const fila = [x, y, 1];
    for (let i = 0; i < 3; i++) {
      Mtz[i]! += fila[i]! * z;
      for (let j = 0; j < 3; j++) MtM[i]![j]! += fila[i]! * fila[j]!;
    }
  }
  const sol = resolver3x3(MtM, Mtz);
  if (!sol) return null;
  return { a: sol[0]!, b: sol[1]!, c: sol[2]!, MtM, Mtz, n: puntos.length };
}

// ---------- 3. Rotaciones y proyección de la vista ----------
export function rotX(grados: number): Matriz {
  const c = Math.cos(aRad(grados)), s = Math.sin(aRad(grados));
  return [[1, 0, 0], [0, c, -s], [0, s, c]];
}

export function rotZ(grados: number): Matriz {
  const c = Math.cos(aRad(grados)), s = Math.sin(aRad(grados));
  return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
}

// V = Rx(φ) · Rz(θ): primero gira el objeto alrededor del eje vertical (θ)
// y luego lo inclina hacia quien mira (φ).
export function matrizVista(theta: number, phi: number): Matriz {
  return multiplicarMat(rotX(phi), rotZ(theta));
}

// Proyección ortogonal sobre la pantalla: Π = [1 0 0 ; 0 0 1] descarta la
// profundidad (y) y deja el ancho (x) y la altura (z).
export const PROYECCION: Matriz = [[1, 0, 0], [0, 0, 1]];

export function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function cruz(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function punto(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function normalizar(a: Vec3): Vec3 {
  const n = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
}

// Normal de un triángulo por producto vectorial y brillo de Lambert (producto
// punto con la dirección de la luz).
export function sombreado(a: Vec3, b: Vec3, c: Vec3, luz: Vec3) {
  const normal = normalizar(cruz(sub(b, a), sub(c, a)));
  const n = normal[2] < 0 ? ([-normal[0], -normal[1], -normal[2]] as Vec3) : normal;
  return { normal: n, brillo: Math.max(0, punto(n, normalizar(luz))) };
}
