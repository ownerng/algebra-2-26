// Lienzo 3D (three.js): una cuadrícula de piso y, cuando hay modelo, el sólido
// (V, F) con la foto como textura y/o la nube de puntos. OrbitControls para
// girar (arrastrar), acercar (rueda) y desplazar (clic derecho).
// Una habitación se ve sin luces (la foto ya las trae) desde donde se tomó la
// foto, y la cámara se mece de lado a lado para que se note el 3D.

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { Modelo } from '../lib/api.ts';

export type Mostrar = 'malla' | 'puntos' | 'ambos';

const MECER = 0.2;    // radianes a cada lado al mecer la vista de una habitación (botón Girar)
const LIMITE = 0.45;  // y hasta dónde se puede girar a mano

interface Escena {
  render: THREE.WebGLRenderer;
  escena: THREE.Scene;
  camara: THREE.PerspectiveCamera;
  controles: OrbitControls;
  piso: THREE.GridHelper;
  objeto: { malla: THREE.Object3D; puntos: THREE.Points; tam: number } | null;
}

function cargarImagen(src: string): Promise<HTMLImageElement> {
  return new Promise((ok, mal) => {
    const img = new Image();
    img.onload = () => ok(img);
    img.onerror = mal;
    img.src = src;
  });
}

// Modelo generativo (GLB): sin escala real, se lleva a 200 mm de lado mayor y
// se centra en x, z. La nube de puntos son sus vértices.
async function crearGlb(url: string) {
  const escenaGlb = (await new GLTFLoader().loadAsync(url)).scene;
  const caja0 = new THREE.Box3().setFromObject(escenaGlb);
  const s = 200 / Math.max(...caja0.getSize(new THREE.Vector3()).toArray());
  const centro = caja0.getCenter(new THREE.Vector3()).multiplyScalar(s);
  escenaGlb.scale.setScalar(s);
  escenaGlb.position.set(-centro.x, 0, -centro.z);
  const malla = new THREE.Group().add(escenaGlb);
  malla.updateMatrixWorld(true);

  const pos: number[] = [], v = new THREE.Vector3();
  malla.traverse((o) => {
    if (!(o instanceof THREE.Mesh)) return;
    const p = o.geometry.getAttribute('position');
    for (let i = 0; i < p.count; i++) { v.fromBufferAttribute(p, i).applyMatrix4(o.matrixWorld); pos.push(v.x, v.y, v.z); }
  });
  const geoPuntos = new THREE.BufferGeometry();
  geoPuntos.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  const b = new THREE.Box3().setFromObject(malla);
  const tam = b.getSize(new THREE.Vector3()).length();
  const puntos = new THREE.Points(geoPuntos, new THREE.PointsMaterial({ size: tam / 400, color: 0xd8dde4 }));
  return { malla, puntos, tam, base: b.min.y, alto: b.max.y };
}

async function crearObjeto(m: Modelo) {
  if (m.glb) return crearGlb(m.glb);
  const r = await crearMalla(m);
  // Habitación (alto): cada mueble generado, en su lugar dentro de la maqueta
  const cargador = new GLTFLoader();
  for (const { glb, matriz } of m.muebles) {
    const pieza = (await cargador.loadAsync(glb)).scene;
    pieza.matrixAutoUpdate = false;
    pieza.matrix.fromArray(matriz);
    r.malla.add(pieza);
  }
  return r;
}

async function crearMalla(m: Modelo) {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(m.V, 3));
  geo.setIndex(new THREE.BufferAttribute(m.F, 1));

  // Color de cada vértice (sRGB 0–1): el que manda el servidor (varias vistas)
  // o el de la foto en su (u, v) (una sola foto)
  const colores = new Float32Array(m.N * 3), c = new THREE.Color();
  let malla: THREE.Mesh;
  if (m.colores) {
    for (let i = 0; i < m.N; i++) {
      c.setRGB(m.colores[i * 3]! / 255, m.colores[i * 3 + 1]! / 255, m.colores[i * 3 + 2]! / 255, THREE.SRGBColorSpace);
      c.toArray(colores, i * 3);
    }
    geo.setAttribute('color', new THREE.BufferAttribute(colores, 3));
    geo.computeVertexNormals();
    malla = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.75, metalness: 0.05 }));
  } else {
    const foto = await cargarImagen(m.foto);
    geo.setAttribute('uv', new THREE.BufferAttribute(m.uv!, 2));
    // F viene ordenada: frente, atrás (extrusión) y paredes
    const { frente = 0, atras = 0, paredes = 0 } = m.caras;
    geo.addGroup(0, frente * 3, 0);
    geo.addGroup(frente * 3, atras * 3, 1);
    geo.addGroup((frente + atras) * 3, paredes * 3, 2);
    geo.computeVertexNormals();
    const textura = new THREE.Texture(foto);
    textura.colorSpace = THREE.SRGBColorSpace;
    textura.anisotropy = 4;
    textura.needsUpdate = true;
    malla = new THREE.Mesh(geo, [
      // Habitación: sin luces y por las dos caras (al girar, lo que se vio muy de lado queda de espaldas)
      m.vista ? new THREE.MeshBasicMaterial({ map: textura, side: THREE.DoubleSide }) :new THREE.MeshStandardMaterial({ map: textura, roughness: 0.7, metalness: 0.05 }),
      new THREE.MeshStandardMaterial({ map: textura, roughness: 0.9, color: 0x9a948c }),
      new THREE.MeshStandardMaterial({ map: textura, roughness: 0.9, color: 0x7d7872 }),
    ]);
    const lienzo = document.createElement('canvas');
    lienzo.width = foto.naturalWidth; lienzo.height = foto.naturalHeight;
    const g = lienzo.getContext('2d', { willReadFrequently: true })!;
    g.drawImage(foto, 0, 0);
    const px = g.getImageData(0, 0, lienzo.width, lienzo.height).data;
    for (let i = 0; i < m.N; i++) {
      const x = Math.min(lienzo.width - 1, Math.floor(m.uv![i * 2]! * lienzo.width));
      const y = Math.min(lienzo.height - 1, Math.floor((1 - m.uv![i * 2 + 1]!) * lienzo.height));
      const k = (y * lienzo.width + x) * 4, sombra = i < m.nFrente ? 1 : 0.6;
      c.setRGB((px[k]! / 255) * sombra, (px[k + 1]! / 255) * sombra, (px[k + 2]! / 255) * sombra, THREE.SRGBColorSpace);
      c.toArray(colores, i * 3);
    }
  }
  geo.computeBoundingBox();

  // Nube de puntos: los mismos vértices con su color
  const geoPuntos = new THREE.BufferGeometry();
  geoPuntos.setAttribute('position', geo.getAttribute('position'));
  geoPuntos.setAttribute('color', new THREE.BufferAttribute(colores, 3));
  const b = geo.boundingBox!;
  const tam = b.getSize(new THREE.Vector3()).length();
  const puntos = new THREE.Points(geoPuntos, new THREE.PointsMaterial({ size: tam / 400, vertexColors: true }));
  return { malla, puntos, tam, base: b.min.y, alto: b.max.y };
}

function liberar(o: THREE.Object3D) {
  o.traverse((x) => {
    if (x instanceof THREE.Mesh || x instanceof THREE.Points) {
      x.geometry.dispose();
      for (const mat of [x.material].flat() as THREE.Material[]) {
        (mat as THREE.MeshStandardMaterial).map?.dispose();
        mat.dispose();
      }
    }
  });
}

function nuevoPiso(lado: number, divisiones: number) {
  const piso = new THREE.GridHelper(lado, divisiones, 0x5b6573, 0x2c323b);
  (piso.material as THREE.Material).transparent = true;
  (piso.material as THREE.Material).opacity = 0.8;
  return piso;
}

export function Escena3D({ modelo, mostrar, girar }: { modelo: Modelo | null; mostrar: Mostrar; girar: boolean }) {
  const contenedor = useRef<HTMLDivElement>(null);
  const ref = useRef<Escena | null>(null);

  useEffect(() => {
    const div = contenedor.current!;
    const render = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    render.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    render.toneMapping = THREE.ACESFilmicToneMapping;
    div.appendChild(render.domElement);

    const escena = new THREE.Scene();
    const camara = new THREE.PerspectiveCamera(40, 1, 1, 50000);
    camara.position.set(260, 220, 520);
    escena.add(new THREE.HemisphereLight(0xf2f5ff, 0x30281f, 1.2));
    const luz = new THREE.DirectionalLight(0xffffff, 2);
    luz.position.set(-1, 1.4, 1.6);
    const contra = new THREE.DirectionalLight(0x9fc4ff, 0.7);
    contra.position.set(1.2, 0.5, -1.5);
    escena.add(luz, contra);

    const piso = nuevoPiso(600, 24);
    escena.add(piso);

    const controles = new OrbitControls(camara, render.domElement);
    controles.enableDamping = true;
    controles.target.set(0, 60, 0);

    const tamano = () => {
      const r = div.getBoundingClientRect();
      render.setSize(r.width, r.height, false);
      camara.aspect = r.width / Math.max(1, r.height);
      camara.updateProjectionMatrix();
    };
    tamano();
    const obs = new ResizeObserver(tamano);
    obs.observe(div);

    ref.current = { render, escena, camara, controles, piso, objeto: null };
    render.setAnimationLoop(() => {
      // Mecer: al llegar a un extremo, la rotación automática cambia de sentido
      // (una velocidad positiva hace bajar el ángulo)
      const az = controles.getAzimuthalAngle();
      if (controles.autoRotate && Math.abs(az) > MECER - 0.01 && Math.sign(az) === -Math.sign(controles.autoRotateSpeed)) {
        controles.autoRotateSpeed *= -1;
      }
      controles.update();
      render.render(escena, camara);
    });

    return () => {
      render.setAnimationLoop(null);
      obs.disconnect();
      controles.dispose();
      escena.traverse((o) => { if (o !== escena) liberar(o); });
      render.dispose();
      div.removeChild(render.domElement);
      ref.current = null;
    };
  }, []);

  // Modelo nuevo: se reemplaza el objeto, el piso se ajusta a su tamaño y la
  // cámara lo encuadra de frente y un poco desde arriba
  useEffect(() => {
    const e = ref.current;
    if (!e) return;
    let vigente = true;
    if (e.objeto) { e.escena.remove(e.objeto.malla, e.objeto.puntos); liberar(e.objeto.malla); liberar(e.objeto.puntos); e.objeto = null; }
    if (!modelo) return;
    void crearObjeto(modelo).then(({ malla, puntos, tam, base, alto }) => {
      if (!vigente || ref.current !== e) { liberar(malla); liberar(puntos); return; }
      // El objeto se apoya sobre el piso: su y mínima queda en 0
      malla.position.y = puntos.position.y = -base;
      e.escena.add(malla, puntos);
      e.objeto = { malla, puntos, tam };
      e.escena.remove(e.piso); e.piso.dispose();
      const celda = tam > 400 ? 50 : tam > 80 ? 10 : 5;
      const n = Math.ceil((tam * 2) / celda);
      e.piso = nuevoPiso(n * celda, n);
      e.piso.visible = !modelo.vista;   // el piso de la habitación ya está en la malla
      e.escena.add(e.piso);
      const c = e.controles;
      if (modelo.vista) {
        // La cámara de la foto: el primer cuadro es la foto misma
        const [ox, oy, oz] = modelo.vista.origen as [number, number, number];
        const [mx, my, mz] = modelo.vista.mira as [number, number, number];
        e.camara.position.set(ox, oy - base, oz);
        c.target.set(mx, my - base, mz);
        e.camara.fov = modelo.vista.fov_v;
        // Solo se puede mirar cerca del punto de la foto: más allá, lo que no se
        // vio ni se rellenó deja huecos
        const desde = e.camara.position.clone().sub(c.target);
        const polar = Math.acos(desde.y / desde.length());
        c.minAzimuthAngle = -LIMITE; c.maxAzimuthAngle = LIMITE;
        c.minPolarAngle = polar - LIMITE / 2; c.maxPolarAngle = polar + LIMITE / 2;
        c.maxDistance = desde.length() * 1.3;
        c.enablePan = false;   // desplazarse destaparía lo que la foto no ve
        c.autoRotateSpeed = 1.5;
      } else {
        const altura = (alto - base) / 2;
        c.target.set(0, altura, 0);
        e.camara.position.set(tam * 0.35, altura + tam * 0.45, tam * 1.35);
        e.camara.fov = 40;
        c.minAzimuthAngle = -Infinity; c.maxAzimuthAngle = Infinity;
        c.minPolarAngle = 0; c.maxPolarAngle = Math.PI; c.maxDistance = Infinity; c.enablePan = true;
      }
      c.autoRotate = !!modelo.vista && girarRef.current;
      e.camara.near = tam / 100; e.camara.far = tam * 100;
      e.camara.updateProjectionMatrix();
      aplicar(e, mostrar);
    });
    return () => { vigente = false; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelo]);

  useEffect(() => { if (ref.current) aplicar(ref.current, mostrar); }, [mostrar]);

  const girarRef = useRef(girar);
  girarRef.current = girar;
  useEffect(() => { if (ref.current) ref.current.controles.autoRotate = girar && !!modelo?.vista; }, [girar, modelo]);

  return <div ref={contenedor} className="lienzo" aria-label="Escena 3D" />;
}

function aplicar(e: Escena, mostrar: Mostrar) {
  if (!e.objeto) return;
  e.objeto.malla.visible = mostrar !== 'puntos';
  e.objeto.puntos.visible = mostrar !== 'malla';
  // Con malla y puntos a la vez, los puntos se ven como "piel" de la malla
  (e.objeto.puntos.material as THREE.PointsMaterial).size = e.objeto.tam / (mostrar === 'ambos' ? 600 : 400);
}
