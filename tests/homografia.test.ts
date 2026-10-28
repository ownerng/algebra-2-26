import { expect, test } from 'bun:test';
import { aplicarH, homografiaDesde4, inversa3, resolverLineal, umbralOtsu } from '../web/src/lib/homografia.ts';
import { multiplicarMat } from '../web/src/lib/matematica.ts';

test('Gauss resuelve un sistema 3×3', () => {
  const x = resolverLineal([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], [8, -11, -3])!;
  [2, 3, -1].forEach((v, i) => expect(x[i]!).toBeCloseTo(v, 9));
});

test('la homografía lleva las 4 esquinas exactamente a su destino', () => {
  // Foto en perspectiva: la cartulina se ve como un trapecio
  const foto: [number, number][] = [[102, 410], [538, 395], [470, 120], [160, 135]];
  const mm: [number, number][] = [[0, 0], [330, 0], [330, 240], [0, 240]];
  const { H } = homografiaDesde4(foto, mm)!;
  foto.forEach((p, k) => {
    const q = aplicarH(H, p);
    expect(q[0]).toBeCloseTo(mm[k]![0], 6);
    expect(q[1]).toBeCloseTo(mm[k]![1], 6);
  });
});

test('una homografía afín conserva el punto medio; la inversa deshace H', () => {
  const foto: [number, number][] = [[0, 0], [200, 0], [200, 100], [0, 100]];
  const mm: [number, number][] = [[0, 0], [400, 0], [400, 200], [0, 200]];
  const { H } = homografiaDesde4(foto, mm)!;
  const [x, y] = aplicarH(H, [100, 50]);
  expect(x).toBeCloseTo(200, 9); expect(y).toBeCloseTo(100, 9);
  const I = multiplicarMat(inversa3(H)!, H);
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((f, i) => f.forEach((v, j) => expect(I[i]![j]!).toBeCloseTo(v, 9)));
});

test('Otsu separa dos grupos', () => {
  const v = [...Array(100).fill(20), ...Array(100).fill(200)];
  const t = umbralOtsu(v);
  expect(t).toBeGreaterThanOrEqual(20);
  expect(t).toBeLessThan(200);
});
