import test from 'node:test';
import assert from 'node:assert/strict';
import { rotacion, homogenea, inversaHomogenea, polarALocal, localAGlobal, procesarLectura, det2, multiplicarMat } from '../web/js/matematica.js';
import { medirSimulado, POSES_REALES } from '../web/js/simulador.js';

const cerca = (a, b, tol = 1e-9) => assert.ok(Math.abs(a - b) < tol, `${a} ≉ ${b}`);

test('R(90°) lleva (r, 0) al eje y', () => {
  const [x, y] = polarALocal(500, 90);
  cerca(x, 0); cerca(y, 500);
});

test('el desfase d se suma antes de rotar', () => {
  const [x, y] = polarALocal(400, 60, 20);
  cerca(x, 210); cerca(y, 420 * Math.sin(Math.PI / 3));
});

test('det R = 1 y Rᵀ R = I', () => {
  const R = rotacion(37);
  cerca(det2(R), 1);
  const Rt = [[R[0][0], R[1][0]], [R[0][1], R[1][1]]];
  const I = multiplicarMat(Rt, R);
  cerca(I[0][0], 1); cerca(I[0][1], 0); cerca(I[1][1], 1);
});

test('T con pose cero es la identidad', () => {
  assert.deepEqual(homogenea(0, 0, 0).map((f) => f.map((v) => Math.round(v) + 0)), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
});

test('T⁻¹ · T = I', () => {
  const P = multiplicarMat(inversaHomogenea(33, 120, -45), homogenea(33, 120, -45));
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((fila, i) => fila.forEach((v, j) => cerca(P[i][j], v, 1e-12)));
});

test('local -> global rota y luego traslada', () => {
  const [x, y] = localAGlobal([100, 0], { dx: 50, dy: 20, alpha: 90 });
  cerca(x, 50); cerca(y, 120);
});

test('el simulador y la matemática coinciden: un punto medido desde P2 cae en la misma pared', () => {
  const pose = POSES_REALES[2];
  // desde P2 (250, 1100) mirando α = 180: θ = 0 apunta a -x, hacia la pared x = -900
  let r = -1;
  for (let i = 0; i < 20 && r < 0; i++) r = medirSimulado(pose, 0);
  const c = procesarLectura({ r, thetaServo: 0, d: 15, invertir: false, pose });
  assert.ok(Math.abs(c.global[0] - -900) < 40, `x global ${c.global[0]}`);
  cerca(c.global[1], 1100, 1e-6);
});
