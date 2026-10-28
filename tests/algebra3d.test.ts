import { expect, test } from 'bun:test';
import { ajustarPlano, cruz, descomposicionCasilla, matrizCasilla, matrizVista, puntoDeCasilla, rotX, rotZ, sombreado, type Vec3 } from '../web/src/lib/algebra3d.ts';
import { det2, multiplicarMat } from '../web/src/lib/matematica.ts';

const cerca = (a: number, b: number, d = 9) => expect(a).toBeCloseTo(b, d);

test('A · [i, j, r, 1] da la posición en mm y la altura sobre la mesa', () => {
  const [x, y, z] = puntoDeCasilla(3, 2, 120, 20, 150);
  cerca(x, 60); cerca(y, 40); cerca(z, 30); // 150 − 120 = 30 mm de alto
});

test('A = T · F · E exactamente', () => {
  const { E, F, T } = descomposicionCasilla(20, 150);
  expect(multiplicarMat(T, multiplicarMat(F, E))).toEqual(matrizCasilla(20, 150));
});

test('mínimos cuadrados recupera un plano inclinado', () => {
  const puntos: Vec3[] = [];
  for (let x = 0; x <= 200; x += 20) for (let y = 0; y <= 160; y += 20) puntos.push([x, y, 0.05 * x - 0.02 * y + 3]);
  const p = ajustarPlano(puntos)!;
  cerca(p.a, 0.05); cerca(p.b, -0.02); cerca(p.c, 3);
});

test('las rotaciones son ortogonales: det = 1', () => {
  for (const R of [rotX(37), rotZ(-52), matrizVista(30, 60)]) {
    const det = R[0]![0]! * (R[1]![1]! * R[2]![2]! - R[1]![2]! * R[2]![1]!)
      - R[0]![1]! * (R[1]![0]! * R[2]![2]! - R[1]![2]! * R[2]![0]!)
      + R[0]![2]! * (R[1]![0]! * R[2]![1]! - R[1]![1]! * R[2]![0]!);
    cerca(det, 1);
  }
  cerca(det2([[1, 0], [0, 1]]), 1);
});

test('producto vectorial: normal de un triángulo horizontal apunta hacia arriba', () => {
  const n = cruz([1, 0, 0], [0, 1, 0]);
  expect(n).toEqual([0, 0, 1]);
  const s = sombreado([0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 1]);
  cerca(s.brillo, 1);
});
