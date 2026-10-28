// Visión por computador del escáner: de una foto a la silueta del objeto y su
// tamaño real. La foto es una matriz de alto × ancho píxeles con 3 valores
// (R, G, B) cada uno, es decir, un vector de R³ por píxel.

import type { Matriz } from './matematica.ts';

// Umbral de Otsu: el valor que mejor separa dos grupos en un histograma
// (maximiza la varianza entre clases).
export function umbralOtsu(valores: ArrayLike<number>): number {
  const hist = new Array(256).fill(0);
  for (let k = 0; k < valores.length; k++) hist[Math.min(255, Math.max(0, Math.round(valores[k]!)))]++;
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

// Color del fondo: la mediana de cada canal en el marco de la foto (el objeto
// está en el centro y no toca los bordes).
export function colorFondo(rgb: Uint8ClampedArray, ancho: number, alto: number, marco = 0.06): [number, number, number] {
  const mx = Math.max(1, Math.round(ancho * marco)), my = Math.max(1, Math.round(alto * marco));
  const canales: number[][] = [[], [], []];
  for (let y = 0; y < alto; y++) for (let x = 0; x < ancho; x++) {
    if (x >= mx && x < ancho - mx && y >= my && y < alto - my) continue;
    const k = (y * ancho + x) * 4;
    canales[0]!.push(rgb[k]!); canales[1]!.push(rgb[k + 1]!); canales[2]!.push(rgb[k + 2]!);
  }
  return canales.map((c) => { c.sort((a, b) => a - b); return c[Math.floor(c.length / 2)] ?? 0; }) as [number, number, number];
}

// Distancia de color de cada píxel al fondo: la norma euclidiana en R³
export function distanciaColor(rgb: Uint8ClampedArray, fondo: [number, number, number]): Uint8Array {
  const n = rgb.length / 4;
  const d = new Uint8Array(n);
  for (let k = 0; k < n; k++) {
    const dr = rgb[k * 4]! - fondo[0], dg = rgb[k * 4 + 1]! - fondo[1], db = rgb[k * 4 + 2]! - fondo[2];
    d[k] = Math.min(255, Math.sqrt(dr * dr + dg * dg + db * db));
  }
  return d;
}

// Se queda con la región conectada más grande (el objeto) y rellena sus huecos
export function componenteMayor(mascara: Uint8Array, ancho: number, alto: number): Uint8Array {
  const etiqueta = new Int32Array(mascara.length).fill(-1);
  let mejor = -1, tamMejor = 0, actual = 0;
  const pila: number[] = [];
  for (let s = 0; s < mascara.length; s++) {
    if (!mascara[s] || etiqueta[s] !== -1) continue;
    let tam = 0;
    pila.push(s); etiqueta[s] = actual;
    while (pila.length) {
      const k = pila.pop()!; tam++;
      const x = k % ancho, y = (k - x) / ancho;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= ancho || ny >= alto) continue;
        const q = ny * ancho + nx;
        if (mascara[q] && etiqueta[q] === -1) { etiqueta[q] = actual; pila.push(q); }
      }
    }
    if (tam > tamMejor) { tamMejor = tam; mejor = actual; }
    actual++;
  }
  const res = new Uint8Array(mascara.length);
  for (let k = 0; k < res.length; k++) res[k] = etiqueta[k] === mejor ? 1 : 0;
  // Huecos: todo lo que no esté conectado con el borde por fuera del objeto
  const fuera = new Uint8Array(res.length);
  for (let x = 0; x < ancho; x++) for (const y of [0, alto - 1]) { const k = y * ancho + x; if (!res[k] && !fuera[k]) { fuera[k] = 1; pila.push(k); } }
  for (let y = 0; y < alto; y++) for (const x of [0, ancho - 1]) { const k = y * ancho + x; if (!res[k] && !fuera[k]) { fuera[k] = 1; pila.push(k); } }
  while (pila.length) {
    const k = pila.pop()!;
    const x = k % ancho, y = (k - x) / ancho;
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]] as const) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || ny < 0 || nx >= ancho || ny >= alto) continue;
      const q = ny * ancho + nx;
      if (!res[q] && !fuera[q]) { fuera[q] = 1; pila.push(q); }
    }
  }
  for (let k = 0; k < res.length; k++) if (!fuera[k]) res[k] = 1;
  return res;
}

// Distancia (en píxeles) de cada píxel del objeto a su borde, con dos pasadas
// (aproximación de chaflán 3-4).
export function distanciaAlBorde(mascara: Uint8Array, ancho: number, alto: number): Float32Array {
  const INF = 1e9;
  const d = new Float32Array(mascara.length);
  for (let k = 0; k < d.length; k++) d[k] = mascara[k] ? INF : 0;
  const v = (x: number, y: number) => (x < 0 || y < 0 || x >= ancho || y >= alto ? 0 : d[y * ancho + x]!);
  for (let y = 0; y < alto; y++) for (let x = 0; x < ancho; x++) {
    const k = y * ancho + x;
    if (!d[k]) continue;
    d[k] = Math.min(d[k]!, v(x - 1, y) + 3, v(x, y - 1) + 3, v(x - 1, y - 1) + 4, v(x + 1, y - 1) + 4);
  }
  for (let y = alto - 1; y >= 0; y--) for (let x = ancho - 1; x >= 0; x--) {
    const k = y * ancho + x;
    if (!d[k]) continue;
    d[k] = Math.min(d[k]!, v(x + 1, y) + 3, v(x, y + 1) + 3, v(x + 1, y + 1) + 4, v(x - 1, y + 1) + 4);
  }
  for (let k = 0; k < d.length; k++) d[k] = d[k]! / 3;
  return d;
}

// Modelo de cámara estenopeica: un punto (X, Y, Z) del espacio se ve en el
// píxel (u, v) con  [u, v, 1]ᵀ ~ K·[X, Y, Z]ᵀ,  K = [f 0 cx ; 0 f cy ; 0 0 1].
// La distancia focal en píxeles sale del ángulo de visión horizontal.
export function matrizK(ancho: number, alto: number, anguloHorizontal: number): Matriz {
  const f = (ancho / 2) / Math.tan((anguloHorizontal * Math.PI) / 360);
  return [[f, 0, ancho / 2], [0, f, alto / 2], [0, 0, 1]];
}

// Del píxel al espacio, conociendo la distancia Z (la mide el sensor):
//   X = (u − cx)·Z / f,   Y = (v − cy)·Z / f.  Un píxel mide Z / f milímetros.
export function aEspacio(K: Matriz, u: number, v: number, Z: number): [number, number, number] {
  const f = K[0]![0]!, cx = K[0]![2]!, cy = K[1]![2]!;
  return [((u - cx) * Z) / f, ((v - cy) * Z) / f, Z];
}
