// Vista 3D con WebGL (three.js): la malla del objeto con la foto pegada encima,
// luces y sombreado suave. La cámara virtual se orienta con la misma matriz
// V = Rx(φ)·Rz(θ) del panel de cálculos: se ubica en Vᵀ·[0, −D, 0]ᵀ.

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { matrizVista, type Vec3 } from '../lib/algebra3d.ts';
import { escaneo } from '../lib/escaneo.ts';
import { camara, type ModeloCamara } from '../lib/camara.ts';
import { construirMalla } from '../lib/malla.ts';

// Dirección de la luz principal (ejes del objeto), la usa el panel de cálculos
export const LUZ: Vec3 = [-0.35, 0.55, 1];

function textura(m: ModeloCamara): THREE.Texture {
  const c = document.createElement('canvas');
  c.width = m.ancho; c.height = m.alto;
  c.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(m.rgb), m.ancho, m.alto), 0, 0);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

function objeto3D(m: ModeloCamara): { grupo: THREE.Group; tam: number; zMin: number } {
  const malla = construirMalla(m);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(malla.posiciones, 3));
  geo.setAttribute('uv', new THREE.BufferAttribute(malla.uvs, 2));
  geo.setIndex([...malla.frente, ...malla.atras, ...malla.paredes]);
  geo.addGroup(0, malla.frente.length, 0);
  geo.addGroup(malla.frente.length, malla.atras.length, 1);
  geo.addGroup(malla.frente.length + malla.atras.length, malla.paredes.length, 2);
  geo.computeVertexNormals();
  const mapa = textura(m);
  const materiales = [
    new THREE.MeshStandardMaterial({ map: mapa, roughness: 0.75, metalness: 0.05 }),
    new THREE.MeshStandardMaterial({ map: mapa, roughness: 0.9, color: 0x8a8580 }),
    new THREE.MeshStandardMaterial({ map: mapa, roughness: 0.9, color: 0xa8a29a, side: THREE.DoubleSide }),
  ];
  const grupo = new THREE.Group();
  grupo.add(new THREE.Mesh(geo, materiales));
  geo.computeBoundingBox();
  const b = geo.boundingBox!;
  const tam = Math.max(b.max.x - b.min.x, b.max.y - b.min.y, b.max.z - b.min.z);
  // Centra el objeto en el origen (x, y) para girarlo alrededor de su centro
  grupo.position.set(-(b.min.x + b.max.x) / 2, -(b.min.y + b.max.y) / 2, 0);
  return { grupo, tam, zMin: b.min.z };
}

export function Vista3D() {
  const contenedor = useRef<HTMLDivElement>(null);
  const escala = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const div = contenedor.current!;
    const render = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    render.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    render.toneMapping = THREE.ACESFilmicToneMapping;
    render.domElement.className = 'lienzo3d';
    render.domElement.setAttribute('aria-label', 'Objeto en 3D');
    div.appendChild(render.domElement);

    const escena = new THREE.Scene();
    const cam = new THREE.PerspectiveCamera(32, 1, 1, 20000);
    escena.add(new THREE.HemisphereLight(0xfff4e6, 0x2a2016, 1.1));
    const luz = new THREE.DirectionalLight(0xffffff, 2.2);
    const contra = new THREE.DirectionalLight(0xffb238, 0.8);
    escena.add(luz, contra);

    let actual: ModeloCamara | null = null;
    let obj: { grupo: THREE.Group; tam: number; zMin: number } | null = null;
    let rejilla: THREE.GridHelper | null = null;
    const quitar = () => {
      if (obj) {
        escena.remove(obj.grupo);
        obj.grupo.traverse((o) => {
          if (o instanceof THREE.Mesh) {
            o.geometry.dispose();
            for (const mat of o.material as THREE.MeshStandardMaterial[]) { mat.map?.dispose(); mat.dispose(); }
          }
        });
      }
      if (rejilla) { escena.remove(rejilla); rejilla.dispose(); }
      obj = null; rejilla = null;
    };

    const tamano = () => {
      const r = div.getBoundingClientRect();
      render.setSize(r.width, r.height, false);
      cam.aspect = r.width / Math.max(1, r.height);
      // El panel de abajo tapa parte de la escena: el centro de la imagen sube un poco
      cam.setViewOffset(r.width, r.height, 0, r.height * 0.14, r.width, r.height);
      cam.updateProjectionMatrix();
    };
    tamano();
    const obs = new ResizeObserver(tamano);
    obs.observe(div);

    // Arrastrar para girar, rueda para acercar
    let arrastre: { x: number; y: number } | null = null;
    const lienzo = render.domElement;
    const abajo = (e: PointerEvent) => { arrastre = { x: e.clientX, y: e.clientY }; lienzo.setPointerCapture(e.pointerId); };
    const mover = (e: PointerEvent) => {
      if (!arrastre) return;
      escaneo.vista.theta += (e.clientX - arrastre.x) * 0.4;
      escaneo.vista.phi = Math.min(90, Math.max(-60, escaneo.vista.phi + (e.clientY - arrastre.y) * 0.4));
      arrastre = { x: e.clientX, y: e.clientY };
    };
    const arriba = () => { arrastre = null; };
    const rueda = (e: WheelEvent) => { e.preventDefault(); escaneo.vista.zoom = Math.min(5, Math.max(0.3, escaneo.vista.zoom * Math.exp(-e.deltaY * 0.0015))); };
    lienzo.addEventListener('pointerdown', abajo);
    lienzo.addEventListener('pointermove', mover);
    lienzo.addEventListener('pointerup', arriba);
    lienzo.addEventListener('wheel', rueda, { passive: false });

    let t0 = performance.now(), id = 0;
    const cuadro = (t: number) => {
      id = requestAnimationFrame(cuadro);
      const dt = Math.min(0.1, (t - t0) / 1000); t0 = t;
      if (camara.modelo !== actual) {
        quitar();
        actual = camara.modelo;
        if (actual) {
          obj = objeto3D(actual);
          escena.add(obj.grupo);
          const celda = obj.tam > 250 ? 50 : obj.tam > 60 ? 10 : 5;
          const n = Math.ceil((obj.tam * 1.6) / celda);
          rejilla = new THREE.GridHelper(n * celda, n, 0x6b4d1f, 0x3a2c18);
          rejilla.rotation.x = Math.PI / 2;
          rejilla.position.z = obj.zMin - 0.5;
          escena.add(rejilla);
          if (escala.current) escala.current.textContent = `cuadros de ${celda >= 10 ? `${celda / 10} cm` : `${celda} mm`}`;
        }
      }
      if (!obj) { render.clear(); return; }
      if (escaneo.vista.girar && !arrastre) escaneo.vista.theta += 12 * dt;
      // La cámara virtual mira al origen desde Vᵀ·[0, −D, 0]ᵀ con "arriba" Vᵀ·[0, 0, 1]ᵀ
      const V = matrizVista(escaneo.vista.theta, escaneo.vista.phi);
      const D = (obj.tam * 3.1) / escaneo.vista.zoom;
      cam.position.set(-V[1]![0]! * D, -V[1]![1]! * D, -V[1]![2]! * D);
      cam.up.set(V[2]![0]!, V[2]![1]!, V[2]![2]!);
      cam.lookAt(0, 0, (obj.zMin + obj.tam * 0.15) / 2);
      // Luz principal desde arriba a la izquierda de la cámara; contraluz ámbar
      luz.position.copy(cam.position).add(new THREE.Vector3(-V[0]![0]! * D, -V[0]![1]! * D, -V[0]![2]! * D).multiplyScalar(0.6)).add(cam.up.clone().multiplyScalar(D * 0.8));
      contra.position.copy(cam.position).multiplyScalar(-1).add(new THREE.Vector3(0, 0, D));
      render.render(escena, cam);
    };
    id = requestAnimationFrame(cuadro);

    return () => {
      cancelAnimationFrame(id);
      obs.disconnect();
      lienzo.removeEventListener('pointerdown', abajo);
      lienzo.removeEventListener('pointermove', mover);
      lienzo.removeEventListener('pointerup', arriba);
      lienzo.removeEventListener('wheel', rueda);
      quitar();
      render.dispose();
      div.removeChild(lienzo);
    };
  }, []);

  return (
    <div ref={contenedor} className="vista3d-gl">
      <span ref={escala} className="escala-3d" />
      <span className="ayuda-3d">arrastra para girar · rueda para acercar</span>
    </div>
  );
}
