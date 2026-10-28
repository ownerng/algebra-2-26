// Homografía: la matriz 3×3 que lleva los píxeles de la foto (tomada en
// perspectiva) al plano de la cartulina en milímetros, como si se viera
// perfectamente desde arriba.
//
//   [x, y, 1]ᵀ ~ H · [u, v, 1]ᵀ,   x = (h11 u + h12 v + h13) / (h31 u + h32 v + 1)
//
// Con 4 puntos conocidos (las esquinas de la cuadrícula) salen 8 ecuaciones
// lineales en las 8 incógnitas de H (h33 = 1): es el método DLT.

import type { Matriz, Vector } from './matematica.ts';

export type Punto2 = [number, number];

// Eliminación de Gauss con pivoteo parcial para A·x = b (n × n)
export function resolverLineal(A: Matriz, b: Vector): Vector | null {
  const n = b.length;
  const M = A.map((fila, i) => [...fila, b[i]!]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let f = col + 1; f < n; f++) if (Math.abs(M[f]![col]!) > Math.abs(M[piv]![col]!)) piv = f;
    if (Math.abs(M[piv]![col]!) < 1e-12) return null;
    [M[col], M[piv]] = [M[piv]!, M[col]!];
    for (let f = 0; f < n; f++) {
      if (f === col) continue;
      const k = M[f]![col]! / M[col]![col]!;
      for (let c = col; c <= n; c++) M[f]![c]! -= k * M[col]![c]!;
    }
  }
  return M.map((fila, i) => fila[n]! / fila[i]!);
}

export interface Homografia { H: Matriz; A: Matriz; b: Vector }

// Sistema DLT: por cada par (u, v) → (x, y) dos filas
//   [u v 1 0 0 0 −ux −vx] h = x
//   [0 0 0 u v 1 −uy −vy] h = y
export function homografiaDesde4(origen: Punto2[], destino: Punto2[]): Homografia | null {
  const A: Matriz = [];
  const b: Vector = [];
  for (let k = 0; k < 4; k++) {
    const [u, v] = origen[k]!, [x, y] = destino[k]!;
    A.push([u, v, 1, 0, 0, 0, -u * x, -v * x]);
    b.push(x);
    A.push([0, 0, 0, u, v, 1, -u * y, -v * y]);
    b.push(y);
  }
  const h = resolverLineal(A, b);
  if (!h) return null;
  return { H: [[h[0]!, h[1]!, h[2]!], [h[3]!, h[4]!, h[5]!], [h[6]!, h[7]!, 1]], A, b };
}

export function aplicarH(H: Matriz, [u, v]: Punto2): Punto2 {
  const w = H[2]![0]! * u + H[2]![1]! * v + H[2]![2]!;
  return [(H[0]![0]! * u + H[0]![1]! * v + H[0]![2]!) / w, (H[1]![0]! * u + H[1]![1]! * v + H[1]![2]!) / w];
}

// Inversa de una matriz 3×3 por la adjunta: H⁻¹ = adj(H) / det(H)
export function inversa3(M: Matriz): Matriz | null {
  const [[a, b, c], [d, e, f], [g, h, i]] = M as [[number, number, number], [number, number, number], [number, number, number]];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-15) return null;
  return [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
}

// Umbral de Otsu: el valor que mejor separa dos grupos en un histograma
// (maximiza la varianza entre clases).
export function umbralOtsu(valores: Uint8Array | number[]): number {
  const hist = new Array(256).fill(0);
  for (const v of valores) hist[Math.min(255, Math.max(0, Math.round(v)))]++;
  const total = valores.length;
  let suma = 0;
  for (let t = 0; t < 256; t++) suma += t * hist[t];
  let sumaB = 0, pesoB = 0, mejor = 0, umbral = 0;
  for (let t = 0; t < 256; t++) {
    pesoB += hist[t];
    if (!pesoB) continue;
    const pesoF = total - pesoB;
    if (!pesoF) break;
    sumaB += t * hist[t];
    const mB = sumaB / pesoB, mF = (suma - sumaB) / pesoF;
    const entre = pesoB * pesoF * (mB - mF) ** 2;
    if (entre > mejor) { mejor = entre; umbral = t; }
  }
  return umbral;
}
