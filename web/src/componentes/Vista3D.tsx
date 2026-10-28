// Vista 3D: el mapa 2D levantado como paredes de luz (planta en 3D).
// El sensor mide en un solo plano; la altura es solo visual.

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { COLORES, escaner, type Posicion } from '../lib/escaner.ts';

const MM = 0.001;
const ALTURA = 0.32;
const a3D = (x: number, y: number, h = 0) => new THREE.Vector3(x * MM, h, -y * MM);

function limpiar(grupo: THREE.Group) {
  for (const hijo of [...grupo.children]) {
    grupo.remove(hijo);
    const malla = hijo as THREE.Mesh;
    malla.geometry?.dispose();
    (malla.material as THREE.Material | undefined)?.dispose?.();
  }
}

function reconstruir(grupoEscaneos: THREE.Group, grupoPoses: THREE.Group) {
  limpiar(grupoEscaneos);
  const puntos = escaner.puntos();
  const aditivo = { transparent: true, blending: THREE.AdditiveBlending, depthWrite: false } as const;
  for (const id of [1, 2] as Posicion[]) {
    const lista = puntos[id];
    if (!lista.length) continue;
    const color = new THREE.Color(COLORES[id]);
    grupoEscaneos.add(new THREE.Points(
      new THREE.BufferGeometry().setFromPoints(lista.map((p) => a3D(p.x, p.y, ALTURA * 0.5))),
      new THREE.PointsMaterial({ color, size: 0.035, opacity: 0.95, ...aditivo }),
    ));
    grupoEscaneos.add(new THREE.LineSegments(
      new THREE.BufferGeometry().setFromPoints(lista.flatMap((p) => [a3D(p.x, p.y, 0), a3D(p.x, p.y, ALTURA)])),
      new THREE.LineBasicMaterial({ color, opacity: 0.3, ...aditivo }),
    ));
    const tri: THREE.Vector3[] = [];
    const bordes: THREE.Vector3[] = [];
    for (let i = 1; i < lista.length; i++) {
      const p = lista[i - 1]!, q = lista[i]!;
      if (q.theta - p.theta > 4 || Math.hypot(q.x - p.x, q.y - p.y) > 110) continue;
      const a = a3D(p.x, p.y, 0), b = a3D(q.x, q.y, 0), c = a3D(q.x, q.y, ALTURA), d = a3D(p.x, p.y, ALTURA);
      tri.push(a, b, c, a, c, d);
      bordes.push(d, c);
    }
    if (tri.length) {
      grupoEscaneos.add(new THREE.Mesh(
        new THREE.BufferGeometry().setFromPoints(tri),
        new THREE.MeshBasicMaterial({ color, opacity: 0.15, side: THREE.DoubleSide, ...aditivo }),
      ));
      grupoEscaneos.add(new THREE.LineSegments(
        new THREE.BufferGeometry().setFromPoints(bordes),
        new THREE.LineBasicMaterial({ color, opacity: 0.9, ...aditivo }),
      ));
    }
  }

  limpiar(grupoPoses);
  for (const id of [1, 2] as Posicion[]) {
    const pose = escaner.poses[id];
    const base = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.045, 0.05, 24), new THREE.MeshBasicMaterial({ color: COLORES[id] }));
    base.position.copy(a3D(pose.dx, pose.dy, 0.025));
    grupoPoses.add(base);
    const a = (pose.alpha * Math.PI) / 180;
    const o = a3D(pose.dx, pose.dy, 0.06);
    const ejeX = a3D(pose.dx + 200 * Math.cos(a), pose.dy + 200 * Math.sin(a), 0.06).sub(o).normalize();
    const ejeY = a3D(pose.dx - 200 * Math.sin(a), pose.dy + 200 * Math.cos(a), 0.06).sub(o).normalize();
    grupoPoses.add(new THREE.ArrowHelper(ejeX, o, 0.2, '#ff6f59', 0.04, 0.025));
    grupoPoses.add(new THREE.ArrowHelper(ejeY, o, 0.2, '#a6e36b', 0.04, 0.025));
  }
}

export function Vista3D({ girar }: { girar: boolean }) {
  const contenedor = useRef<HTMLDivElement>(null);
  const controlesRef = useRef<OrbitControls | null>(null);

  useEffect(() => {
    const div = contenedor.current!;
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    div.appendChild(renderer.domElement);

    const escena = new THREE.Scene();
    escena.background = new THREE.Color('#0d0905');
    escena.fog = new THREE.Fog('#0d0905', 3, 7);
    const camara = new THREE.PerspectiveCamera(50, 1, 0.01, 50);
    camara.position.set(0.4, 1.9, 1.6);
    const controles = new OrbitControls(camara, renderer.domElement);
    controles.target.set(0, 0, -0.6);
    controles.enableDamping = true;
    controles.autoRotateSpeed = 0.6;
    controlesRef.current = controles;

    const grilla = new THREE.GridHelper(6, 60, '#6b4a1a', '#2a1d0c');
    escena.add(grilla);
    const grupoEscaneos = new THREE.Group();
    const grupoPoses = new THREE.Group();
    escena.add(grupoEscaneos, grupoPoses);

    const haz = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 0.012),
      new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, side: THREE.DoubleSide }),
    );
    haz.geometry.translate(0.5, 0, 0);
    escena.add(haz);

    const redimensionar = () => {
      const { width, height } = div.getBoundingClientRect();
      if (!width || !height) return;
      renderer.setSize(width, height);
      camara.aspect = width / height;
      camara.updateProjectionMatrix();
    };
    const obs = new ResizeObserver(redimensionar);
    obs.observe(div);
    redimensionar();

    let version = -1;
    let cuadro = 0;
    const bucle = () => {
      if (escaner.version !== version) {
        reconstruir(grupoEscaneos, grupoPoses);
        version = escaner.version;
      }
      const u = escaner.ultima;
      if (u && performance.now() - u.t < 1500) {
        const pose = escaner.poses[u.posicion];
        const theta = escaner.invertir ? 180 - u.thetaServo : u.thetaServo;
        haz.visible = true;
        (haz.material as THREE.MeshBasicMaterial).color.set(COLORES[u.posicion]);
        haz.position.copy(a3D(pose.dx, pose.dy, ALTURA * 0.5));
        haz.rotation.set(0, ((pose.alpha + theta) * Math.PI) / 180, 0);
        haz.scale.x = (u.r > 0 ? Math.min(u.r + escaner.d, 1200) : 1200) * MM;
      } else {
        haz.visible = false;
      }
      controles.update();
      renderer.render(escena, camara);
      cuadro = requestAnimationFrame(bucle);
    };
    cuadro = requestAnimationFrame(bucle);

    return () => {
      cancelAnimationFrame(cuadro);
      obs.disconnect();
      controles.dispose();
      limpiar(grupoEscaneos);
      limpiar(grupoPoses);
      renderer.dispose();
      div.removeChild(renderer.domElement);
    };
  }, []);

  useEffect(() => { if (controlesRef.current) controlesRef.current.autoRotate = girar; }, [girar]);

  return <div ref={contenedor} className="vista3d" />;
}
