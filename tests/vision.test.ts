import { expect, test } from 'bun:test';
import { aEspacio, colorFondo, componenteMayor, distanciaAlBorde, distanciaColor, matrizK, umbralOtsu } from '../web/src/lib/vision.ts';
import { multiplicarMatVec } from '../web/src/lib/matematica.ts';

// Foto sintética 40 × 30: fondo claro y un rectángulo oscuro de 20 × 10 en el centro
function foto() {
  const w = 40, h = 30, rgb = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const k = (y * w + x) * 4, dentro = x >= 10 && x < 30 && y >= 10 && y < 20;
    rgb[k] = dentro ? 60 : 230; rgb[k + 1] = dentro ? 70 : 228; rgb[k + 2] = dentro ? 80 : 220; rgb[k + 3] = 255;
  }
  return { w, h, rgb };
}

test('el fondo se toma del borde y Otsu separa el objeto', () => {
  const { w, h, rgb } = foto();
  const fondo = colorFondo(rgb, w, h);
  expect(fondo).toEqual([230, 228, 220]);
  const d = distanciaColor(rgb, fondo);
  const t = umbralOtsu(d);
  const mascara = new Uint8Array(d.length);
  d.forEach((v, k) => { mascara[k] = v > t ? 1 : 0; });
  expect(mascara.reduce((a, b) => a + b, 0)).toBe(200);
});

test('componente mayor descarta ruido y rellena huecos', () => {
  const w = 10, h = 10, m = new Uint8Array(100);
  for (let y = 2; y < 8; y++) for (let x = 2; x < 8; x++) m[y * w + x] = 1;
  m[5 * w + 5] = 0; // hueco
  m[0] = 1;         // ruido suelto
  const r = componenteMayor(m, w, h);
  expect(r[0]).toBe(0);
  expect(r[5 * w + 5]).toBe(1);
  expect(r.reduce((a, b) => a + b, 0)).toBe(36);
});

test('la distancia al borde crece hacia el centro', () => {
  const w = 9, h = 9, m = new Uint8Array(81).fill(1);
  const d = distanciaAlBorde(m, w, h);
  expect(d[0]).toBeCloseTo(1, 5);
  expect(d[4 * w + 4]!).toBeGreaterThan(d[1 * w + 1]!);
});

test('modelo estenopeico: K proyecta y aEspacio deshace la proyección', () => {
  const K = matrizK(1280, 720, 65);
  const P = aEspacio(K, 900, 200, 300);
  const p = multiplicarMatVec(K, P);
  expect(p[0]! / p[2]!).toBeCloseTo(900, 9);
  expect(p[1]! / p[2]!).toBeCloseTo(200, 9);
});
