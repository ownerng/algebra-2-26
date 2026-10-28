// Vista 2D estilo radar: mapa global en milímetros, eje y hacia arriba.

import { aRad } from './matematica.js';

const ALCANCE = 1200;

function sprite(color, tam = 32) {
  const c = document.createElement('canvas');
  c.width = c.height = tam;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(tam / 2, tam / 2, 0, tam / 2, tam / 2, tam / 2);
  grad.addColorStop(0, '#ffffff');
  grad.addColorStop(0.18, color);
  grad.addColorStop(0.45, color + '55');
  grad.addColorStop(1, color + '00');
  g.fillStyle = grad;
  g.fillRect(0, 0, tam, tam);
  return c;
}

export class Radar2D {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.escala = 0.3;       // px por mm
    this.origen = [0, 0];    // px donde cae el (0,0) global
    this.ajustado = false;
    this.sprites = {};
    this.instalarInteraccion();
    new ResizeObserver(() => this.redimensionar()).observe(canvas);
    this.redimensionar();
  }

  redimensionar() {
    const dpr = window.devicePixelRatio || 1;
    const { width, height } = this.canvas.getBoundingClientRect();
    if (!width || !height) return;
    this.ancho = width;
    this.alto = height;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!this.ajustado) this.encuadrar(this.ultimasPoses);
  }

  encuadrar(poses = [{ dx: 0, dy: 0 }]) {
    this.ultimasPoses = poses;
    if (!this.ancho) return;
    const xs = poses.flatMap((p) => [p.dx - ALCANCE, p.dx + ALCANCE]);
    const ys = poses.flatMap((p) => [p.dy - ALCANCE * 0.25, p.dy + ALCANCE]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    this.escala = Math.min(this.ancho / (maxX - minX), this.alto / (maxY - minY)) * 0.92;
    this.origen = [
      this.ancho / 2 - ((minX + maxX) / 2) * this.escala,
      this.alto / 2 + ((minY + maxY) / 2) * this.escala,
    ];
    this.ajustado = true;
  }

  instalarInteraccion() {
    let arrastre = null;
    this.canvas.addEventListener('pointerdown', (e) => {
      arrastre = { x: e.clientX, y: e.clientY, o: [...this.origen] };
      this.canvas.setPointerCapture(e.pointerId);
    });
    this.canvas.addEventListener('pointermove', (e) => {
      if (!arrastre) return;
      this.origen = [arrastre.o[0] + e.clientX - arrastre.x, arrastre.o[1] + e.clientY - arrastre.y];
    });
    this.canvas.addEventListener('pointerup', () => { arrastre = null; });
    this.canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = this.canvas.getBoundingClientRect();
      const px = e.clientX - r.left, py = e.clientY - r.top;
      const f = Math.exp(-e.deltaY * 0.0015);
      this.origen = [px - (px - this.origen[0]) * f, py - (py - this.origen[1]) * f];
      this.escala *= f;
    }, { passive: false });
  }

  aPantalla(x, y) {
    return [this.origen[0] + x * this.escala, this.origen[1] - y * this.escala];
  }

  spriteDe(color) {
    if (!this.sprites[color]) this.sprites[color] = sprite(color);
    return this.sprites[color];
  }

  // vista = { poses: [{id, pose, color, activa, nombre}], escaneos: [{color, puntos:[{x,y,theta,t}]}],
  //           haz: {pose, theta, direccion, r, color} | null, ahora }
  dibujar(vista) {
    const g = this.ctx;
    if (!this.ancho) return;
    g.save();
    g.fillStyle = '#03080b';
    g.fillRect(0, 0, this.ancho, this.alto);
    this.dibujarCuadricula(g);

    const activa = vista.poses.find((p) => p.activa);
    if (activa) this.dibujarAnillos(g, activa);
    for (const p of vista.poses) this.dibujarPose(g, p);
    if (vista.haz) this.dibujarHaz(g, vista.haz);
    for (const esc of vista.escaneos) this.dibujarPuntos(g, esc, vista.ahora);
    this.dibujarEscalaGrafica(g);
    g.restore();
  }

  dibujarCuadricula(g) {
    const paso = this.escala * 100 < 8 ? 500 : 100;
    const [x0, y1] = this.aMundo(0, 0);
    const [x1, y0] = this.aMundo(this.ancho, this.alto);
    g.lineWidth = 1;
    for (let x = Math.floor(x0 / paso) * paso; x <= x1; x += paso) {
      const [sx] = this.aPantalla(x, 0);
      g.strokeStyle = x % 500 === 0 ? 'rgba(60,200,220,0.13)' : 'rgba(60,200,220,0.05)';
      g.beginPath(); g.moveTo(sx, 0); g.lineTo(sx, this.alto); g.stroke();
    }
    for (let y = Math.floor(y0 / paso) * paso; y <= y1; y += paso) {
      const [, sy] = this.aPantalla(0, y);
      g.strokeStyle = y % 500 === 0 ? 'rgba(60,200,220,0.13)' : 'rgba(60,200,220,0.05)';
      g.beginPath(); g.moveTo(0, sy); g.lineTo(this.ancho, sy); g.stroke();
    }
  }

  aMundo(px, py) {
    return [(px - this.origen[0]) / this.escala, (this.origen[1] - py) / this.escala];
  }

  dibujarAnillos(g, { pose, color }) {
    const [cx, cy] = this.aPantalla(pose.dx, pose.dy);
    const a0 = -aRad(pose.alpha);
    g.strokeStyle = color + '40';
    g.fillStyle = color + '99';
    g.font = '11px "JetBrains Mono", ui-monospace, monospace';
    g.lineWidth = 1;
    for (let r = 300; r <= ALCANCE; r += 300) {
      g.beginPath();
      g.arc(cx, cy, r * this.escala, a0 - Math.PI, a0, false);
      g.stroke();
      // Etiqueta sobre el eje x local (0°), justo debajo de la línea
      const [lx, ly] = [cx + Math.cos(a0) * r * this.escala, cy + Math.sin(a0) * r * this.escala];
      g.fillText(r / 10 + ' cm', lx - 18, ly + 14);
    }
    for (let t = 0; t <= 180; t += 30) {
      const a = a0 - aRad(t);
      g.strokeStyle = color + '22';
      g.beginPath();
      g.moveTo(cx, cy);
      g.lineTo(cx + Math.cos(a) * ALCANCE * this.escala, cy + Math.sin(a) * ALCANCE * this.escala);
      g.stroke();
      g.fillStyle = color + '88';
      g.fillText(t + '°', cx + Math.cos(a) * (ALCANCE + 60) * this.escala - 10, cy + Math.sin(a) * (ALCANCE + 60) * this.escala + 4);
    }
  }

  flecha(g, x0, y0, x1, y1, color, texto) {
    const ang = Math.atan2(y1 - y0, x1 - x0);
    g.strokeStyle = color;
    g.fillStyle = color;
    g.lineWidth = 2;
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    g.beginPath();
    g.moveTo(x1, y1);
    g.lineTo(x1 - 9 * Math.cos(ang - 0.4), y1 - 9 * Math.sin(ang - 0.4));
    g.lineTo(x1 - 9 * Math.cos(ang + 0.4), y1 - 9 * Math.sin(ang + 0.4));
    g.closePath(); g.fill();
    if (texto) g.fillText(texto, x1 + 6 * Math.cos(ang) - 6, y1 + 12 * Math.sin(ang) + 4);
  }

  dibujarPose(g, { id, pose, color, activa, nombre }) {
    const [cx, cy] = this.aPantalla(pose.dx, pose.dy);
    const L = 220;
    const a = aRad(pose.alpha);
    g.font = '12px "JetBrains Mono", ui-monospace, monospace';
    if (id !== 1) {
      const [ox, oy] = this.aPantalla(0, 0);
      g.setLineDash([5, 5]);
      g.strokeStyle = 'rgba(255,255,255,0.35)';
      g.lineWidth = 1.2;
      g.beginPath(); g.moveTo(ox, oy); g.lineTo(cx, cy); g.stroke();
      g.setLineDash([]);
      g.fillStyle = 'rgba(255,255,255,0.6)';
      g.fillText('t = (dx, dy)', (ox + cx) / 2 + 8, (oy + cy) / 2);
    }
    // Columnas de R(α): los ejes locales vistos desde el sistema global
    const ex = this.aPantalla(pose.dx + L * Math.cos(a), pose.dy + L * Math.sin(a));
    const ey = this.aPantalla(pose.dx - L * Math.sin(a), pose.dy + L * Math.cos(a));
    g.globalAlpha = activa ? 1 : 0.55;
    this.flecha(g, cx, cy, ex[0], ex[1], '#ff6b6b', 'x' + subindice(id));
    this.flecha(g, cx, cy, ey[0], ey[1], '#7dff9a', 'y' + subindice(id));
    g.shadowColor = color;
    g.shadowBlur = activa ? 18 : 6;
    g.fillStyle = color;
    g.beginPath(); g.arc(cx, cy, activa ? 7 : 5, 0, Math.PI * 2); g.fill();
    g.shadowBlur = 0;
    g.fillStyle = color;
    g.font = '600 12px "Chakra Petch", system-ui, sans-serif';
    g.fillText(nombre, cx + 10, cy + 18);
    g.globalAlpha = 1;
  }

  dibujarHaz(g, { pose, theta, direccion, r, color }) {
    const [cx, cy] = this.aPantalla(pose.dx, pose.dy);
    const R = ALCANCE * this.escala;
    const base = pose.alpha + theta;
    for (let i = 18; i >= 1; i--) {
      const a = -aRad(base - direccion * i);
      g.strokeStyle = color + Math.round((1 - i / 18) * 70).toString(16).padStart(2, '0');
      g.lineWidth = 3;
      g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + Math.cos(a) * R, cy + Math.sin(a) * R); g.stroke();
    }
    const a = -aRad(base);
    g.strokeStyle = color;
    g.lineWidth = 2;
    g.shadowColor = color;
    g.shadowBlur = 12;
    const largo = r > 0 ? Math.min(r, ALCANCE) * this.escala : R;
    g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + Math.cos(a) * largo, cy + Math.sin(a) * largo); g.stroke();
    g.shadowBlur = 0;
  }

  dibujarPuntos(g, { color, puntos }, ahora) {
    if (!puntos.length) return;
    // Contorno: une lecturas vecinas que pertenecen al mismo objeto
    g.strokeStyle = color + '88';
    g.lineWidth = 1.6;
    g.beginPath();
    for (let i = 1; i < puntos.length; i++) {
      const p = puntos[i - 1], q = puntos[i];
      if (q.theta - p.theta <= 4 && Math.hypot(q.x - p.x, q.y - p.y) < 110) {
        const [ax, ay] = this.aPantalla(p.x, p.y);
        const [bx, by] = this.aPantalla(q.x, q.y);
        g.moveTo(ax, ay); g.lineTo(bx, by);
      }
    }
    g.stroke();

    g.globalCompositeOperation = 'lighter';
    const spr = this.spriteDe(color);
    const blanco = this.spriteDe('#ffffff');
    const tam = Math.max(9, Math.min(18, 40 * this.escala));
    for (const p of puntos) {
      const [sx, sy] = this.aPantalla(p.x, p.y);
      const edad = ahora - p.t;
      if (edad >= 0 && edad < 600) {
        const k = 1 + (1 - edad / 600) * 0.8;
        g.globalAlpha = 0.5 * (1 - edad / 600);
        g.drawImage(blanco, sx - (tam * k) / 2, sy - (tam * k) / 2, tam * k, tam * k);
      }
      g.globalAlpha = 0.85;
      g.drawImage(spr, sx - tam / 2, sy - tam / 2, tam, tam);
    }
    g.globalAlpha = 1;
    g.globalCompositeOperation = 'source-over';
  }

  dibujarEscalaGrafica(g) {
    const mm = this.escala * 100 < 30 ? 500 : 100;
    const px = mm * this.escala;
    const x = 18, y = this.alto - 20;
    g.strokeStyle = 'rgba(200,240,255,0.7)';
    g.fillStyle = 'rgba(200,240,255,0.7)';
    g.lineWidth = 2;
    g.beginPath(); g.moveTo(x, y - 5); g.lineTo(x, y); g.lineTo(x + px, y); g.lineTo(x + px, y - 5); g.stroke();
    g.font = '11px "JetBrains Mono", ui-monospace, monospace';
    g.fillText(mm / 10 + ' cm', x + px + 8, y + 1);
  }
}

function subindice(n) {
  return String(n).replace(/\d/g, (d) => '₀₁₂₃₄₅₆₇₈₉'[d]);
}
