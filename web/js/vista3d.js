// Vista 3D: el mapa 2D extruido como paredes de luz (planta en 3D).
// El sensor mide en un solo plano; la altura es solo visual.

import * as THREE from 'three';
import { OrbitControls } from '../vendor/OrbitControls.js';

const MM = 0.001;       // 1 mm = 0.001 unidades (metros)
const ALTURA = 0.32;    // altura de las paredes dibujadas

// Plano del mapa (x, y) -> espacio 3D (x, altura, -y)
const a3D = (x, y, h = 0) => new THREE.Vector3(x * MM, h, -y * MM);

export class Vista3D {
  constructor(contenedor) {
    this.contenedor = contenedor;
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    contenedor.appendChild(this.renderer.domElement);

    this.escena = new THREE.Scene();
    this.escena.background = new THREE.Color('#03080b');
    this.escena.fog = new THREE.Fog('#03080b', 3, 7);

    this.camara = new THREE.PerspectiveCamera(50, 1, 0.01, 50);
    this.camara.position.set(0.4, 1.9, 1.6);
    this.controles = new OrbitControls(this.camara, this.renderer.domElement);
    this.controles.target.set(0, 0, -0.6);
    this.controles.enableDamping = true;
    this.controles.autoRotate = true;
    this.controles.autoRotateSpeed = 0.6;

    const grilla = new THREE.GridHelper(6, 60, '#1c6f7a', '#0d2a30');
    grilla.material.transparent = true;
    grilla.material.opacity = 0.55;
    this.escena.add(grilla);

    this.grupoEscaneos = new THREE.Group();
    this.grupoPoses = new THREE.Group();
    this.escena.add(this.grupoEscaneos, this.grupoPoses);

    this.haz = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 0.012),
      new THREE.MeshBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, side: THREE.DoubleSide })
    );
    this.haz.geometry.translate(0.5, 0, 0);
    this.escena.add(this.haz);

    this.version = -1;
    this.visible = false;
    new ResizeObserver(() => this.redimensionar()).observe(contenedor);
  }

  redimensionar() {
    const { width, height } = this.contenedor.getBoundingClientRect();
    if (!width || !height) return;
    this.renderer.setSize(width, height);
    this.camara.aspect = width / height;
    this.camara.updateProjectionMatrix();
  }

  limpiar(grupo) {
    for (const hijo of [...grupo.children]) {
      grupo.remove(hijo);
      hijo.geometry?.dispose();
      hijo.material?.dispose();
    }
  }

  reconstruir(vista) {
    this.limpiar(this.grupoEscaneos);
    for (const { color, puntos } of vista.escaneos) {
      if (!puntos.length) continue;
      const c = new THREE.Color(color);

      // Nube de puntos a la altura del sensor
      const geoPuntos = new THREE.BufferGeometry().setFromPoints(puntos.map((p) => a3D(p.x, p.y, ALTURA * 0.5)));
      this.grupoEscaneos.add(new THREE.Points(geoPuntos, new THREE.PointsMaterial({
        color: c, size: 0.035, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false,
      })));

      // Postes verticales en cada punto
      const postes = puntos.flatMap((p) => [a3D(p.x, p.y, 0), a3D(p.x, p.y, ALTURA)]);
      this.grupoEscaneos.add(new THREE.LineSegments(
        new THREE.BufferGeometry().setFromPoints(postes),
        new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.35, blending: THREE.AdditiveBlending, depthWrite: false })
      ));

      // Paredes translúcidas entre lecturas vecinas del mismo objeto
      const tri = [];
      const bordes = [];
      for (let i = 1; i < puntos.length; i++) {
        const p = puntos[i - 1], q = puntos[i];
        if (q.theta - p.theta > 4 || Math.hypot(q.x - p.x, q.y - p.y) > 110) continue;
        const a = a3D(p.x, p.y, 0), b = a3D(q.x, q.y, 0), c2 = a3D(q.x, q.y, ALTURA), d = a3D(p.x, p.y, ALTURA);
        tri.push(a, b, c2, a, c2, d);
        bordes.push(d, c2);
      }
      if (tri.length) {
        this.grupoEscaneos.add(new THREE.Mesh(
          new THREE.BufferGeometry().setFromPoints(tri),
          new THREE.MeshBasicMaterial({ color: c, transparent: true, opacity: 0.16, side: THREE.DoubleSide, blending: THREE.AdditiveBlending, depthWrite: false })
        ));
        this.grupoEscaneos.add(new THREE.LineSegments(
          new THREE.BufferGeometry().setFromPoints(bordes),
          new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending })
        ));
      }
    }

    this.limpiar(this.grupoPoses);
    for (const { pose, color } of vista.poses) {
      const base = new THREE.Mesh(
        new THREE.CylinderGeometry(0.035, 0.045, 0.05, 24),
        new THREE.MeshBasicMaterial({ color })
      );
      base.position.copy(a3D(pose.dx, pose.dy, 0.025));
      this.grupoPoses.add(base);
      // Ejes locales (columnas de R(α))
      const a = (pose.alpha * Math.PI) / 180;
      const o = a3D(pose.dx, pose.dy, 0.06);
      const ejeX = a3D(pose.dx + 200 * Math.cos(a), pose.dy + 200 * Math.sin(a), 0.06);
      const ejeY = a3D(pose.dx - 200 * Math.sin(a), pose.dy + 200 * Math.cos(a), 0.06);
      this.grupoPoses.add(new THREE.ArrowHelper(ejeX.clone().sub(o).normalize(), o, 0.2, '#ff6b6b', 0.04, 0.025));
      this.grupoPoses.add(new THREE.ArrowHelper(ejeY.clone().sub(o).normalize(), o, 0.2, '#7dff9a', 0.04, 0.025));
    }
  }

  dibujar(vista) {
    if (!this.visible) return;
    if (vista.version !== this.version) {
      this.reconstruir(vista);
      this.version = vista.version;
    }
    if (vista.haz) {
      const { pose, theta, r, color } = vista.haz;
      this.haz.visible = true;
      this.haz.material.color.set(color);
      this.haz.position.copy(a3D(pose.dx, pose.dy, ALTURA * 0.5));
      this.haz.rotation.set(0, ((pose.alpha + theta) * Math.PI) / 180, 0);
      this.haz.scale.x = (r > 0 ? Math.min(r, 1200) : 1200) * MM;
    } else {
      this.haz.visible = false;
    }
    this.controles.update();
    this.renderer.render(this.escena, this.camara);
  }

  set rotarSolo(valor) {
    this.controles.autoRotate = valor;
  }
}
