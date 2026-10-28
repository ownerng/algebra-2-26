// Núcleo matemático del escáner: rotación 2D y matriz homogénea 3x3.
// Todas las distancias en milímetros; los ángulos públicos en grados.

export type Vector = number[];
export type Matriz = number[][];
export interface Pose {
  dx: number;
  dy: number;
  alpha: number;
}

export const aRad = (grados: number) => (grados * Math.PI) / 180;

// R(θ) = [cos θ  -sin θ ; sin θ  cos θ]
export function rotacion(thetaGrados: number): Matriz {
  const t = aRad(thetaGrados);
  const c = Math.cos(t);
  const s = Math.sin(t);
  return [
    [c, -s],
    [s, c],
  ];
}

// T(α, dx, dy) = [cos α  -sin α  dx ; sin α  cos α  dy ; 0  0  1]
export function homogenea(alphaGrados: number, dx: number, dy: number): Matriz {
  const [[c, ms], [s]] = rotacion(alphaGrados);
  return [
    [c, ms, dx],
    [s, c, dy],
    [0, 0, 1],
  ];
}

// T⁻¹ = [Rᵀ  -Rᵀt ; 0 0 1]: lleva un punto del mapa global al sistema local
export function inversaHomogenea(alphaGrados: number, dx: number, dy: number): Matriz {
  const [[c, ms], [s]] = rotacion(alphaGrados);
  return [
    [c, s, -(c * dx + s * dy)],
    [ms, c, -(ms * dx + c * dy)],
    [0, 0, 1],
  ];
}

export function multiplicarMatVec(M: Matriz, v: Vector): Vector {
  return M.map((fila) => fila.reduce((suma, m, j) => suma + m * v[j]!, 0));
}

export function multiplicarMat(A: Matriz, B: Matriz): Matriz {
  return A.map((fila) => B[0]!.map((_, j) => fila.reduce((suma, a, k) => suma + a * B[k]![j]!, 0)));
}

export function det2(M: Matriz): number {
  return M[0]![0]! * M[1]![1]! - M[0]![1]! * M[1]![0]!;
}

// El servo reporta su ángulo; la calibración puede invertir el sentido.
export function anguloEfectivo(thetaServo: number, invertir: boolean): number {
  return invertir ? 180 - thetaServo : thetaServo;
}

// Paso 1: (r, θ) -> (x, y) local.  d = distancia del eje de giro al lente.
export function polarALocal(r: number, thetaGrados: number, d = 0): Vector {
  return multiplicarMatVec(rotacion(thetaGrados), [r + d, 0]);
}

// Paso 2: (x, y) local -> (x, y) global con la pose del dispositivo.
export function localAGlobal([x, y]: Vector, pose: Pose): Vector {
  const [xg, yg] = multiplicarMatVec(homogenea(pose.alpha, pose.dx, pose.dy), [x!, y!, 1]);
  return [xg!, yg!];
}

export interface Calculo {
  theta: number;
  R: Matriz;
  vectorPolar: Vector;
  local: Vector;
  T: Matriz;
  homLocal: Vector;
  global: Vector;
}

// Cálculo completo de una lectura, con todos los intermedios para el panel.
export function procesarLectura({ r, thetaServo, d, invertir, pose }:
  { r: number; thetaServo: number; d: number; invertir: boolean; pose: Pose }): Calculo {
  const theta = anguloEfectivo(thetaServo, invertir);
  const R = rotacion(theta);
  const vectorPolar = [r + d, 0];
  const local = multiplicarMatVec(R, vectorPolar);
  const T = homogenea(pose.alpha, pose.dx, pose.dy);
  const homLocal = [local[0]!, local[1]!, 1];
  const global = multiplicarMatVec(T, homLocal);
  return { theta, R, vectorPolar, local, T, homLocal, global };
}
