import { expect, test } from 'bun:test';
import { rotacion, homogenea, inversaHomogenea, polarALocal, localAGlobal, procesarLectura, det2, multiplicarMat } from '../web/src/lib/matematica.ts';
import { distanciaSimulada, ALTURA_REAL_MM, alturaObjeto } from '../web/src/lib/simulador3d.ts';

const cerca = (a: number, b: number, digitos = 9) => expect(a).toBeCloseTo(b, digitos);

test('R(90°) lleva (r, 0) al eje y', () => {
  const [x, y] = polarALocal(500, 90);
  cerca(x!, 0); cerca(y!, 500);
});

test('el desfase d se suma antes de rotar', () => {
  const [x, y] = polarALocal(400, 60, 20);
  cerca(x!, 210); cerca(y!, 420 * Math.sin(Math.PI / 3));
});

test('det R = 1 y Rᵀ R = I', () => {
  const R = rotacion(37);
  cerca(det2(R), 1);
  const Rt = [[R[0]![0]!, R[1]![0]!], [R[0]![1]!, R[1]![1]!]];
  const I = multiplicarMat(Rt, R);
  cerca(I[0]![0]!, 1); cerca(I[0]![1]!, 0); cerca(I[1]![1]!, 1);
});

test('T con pose cero es la identidad', () => {
  expect(homogenea(0, 0, 0).map((f) => f.map((v) => Math.round(v) + 0))).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
});

test('T⁻¹ · T = I', () => {
  const P = multiplicarMat(inversaHomogenea(33, 120, -45), homogenea(33, 120, -45));
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((fila, i) => fila.forEach((v, j) => cerca(P[i]![j]!, v, 12)));
});

test('local -> global rota y luego traslada', () => {
  const [x, y] = localAGlobal([100, 0], { dx: 50, dy: 20, alpha: 90 });
  cerca(x!, 50); cerca(y!, 120);
});

test('el simulador 3D: sobre la mesa vacía mide la altura H y sobre la taza, menos', () => {
  const mesa = distanciaSimulada('taza', 0, 0, false);
  const taza = distanciaSimulada('taza', 90, 70, false);
  expect(Math.abs(mesa - ALTURA_REAL_MM)).toBeLessThan(2);
  expect(taza).toBeLessThan(ALTURA_REAL_MM - 0.8 * alturaObjeto('taza', 90, 70));
});
