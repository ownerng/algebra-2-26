# Escáner 3D de una foto — Álgebra Lineal

Prototipo que convierte una foto tomada con la cámara del celular (DroidCam) en
un sólido 3D cerrado. Todo el proceso es álgebra lineal sobre la imagen: la
segmentación da la silueta, la silueta da una profundidad Z, la inversa de la
matriz intrínseca K lleva cada píxel a R³ y con esos puntos se arma la malla
V ∈ R^(N×3), F ∈ Z^(M×3). Tarda alrededor de un segundo por foto.

## Carpetas

| Ruta | Qué es |
|---|---|
| `servidor/` | Python (FastAPI): cámara DroidCam, segmentación, profundidad, K⁻¹ y malla |
| `servidor/escaner/` | Módulos: `camara`, `segmentacion`, `profundidad`, `geometria`, `malla`, `proceso`, `multivista` |
| `web/` | Página (React + Bun + three.js): encuadre, foto y vista 3D con OrbitControls |
| `tests/`, `servidor/tests/` | Pruebas de la matemática (`make test`) |
| `firmware/`, `experimentos/`, `docs/` | Versión anterior con el sensor VL53L0X y el informe en LaTeX |

## Uso

1. Instala DroidCam en el celular y ábrelo; anota la **WiFi IP** que muestra.
2. Ejecuta `./iniciar-web.sh`. La primera vez crea `servidor/.venv`, instala
   las dependencias y descarga las redes de segmentación (~170 MB) y de
   profundidad (Depth Anything V2, ~100 MB, en `~/.cache/escaner`).
3. En `http://localhost:3000`, pulsa **Cámara**, escribe la IP y **Conectar**.
   Cuando llega la imagen, el panel se cierra solo.
4. Pon el objeto dentro del recuadro de la cuadrícula y pulsa **Tomar foto**.
5. Gira con el mouse, acerca con la rueda y elige **Malla**, **Puntos** o
   **Ambos**. El botón **.obj** descarga el modelo.

### iPhone por cable (sin WiFi)

DroidCam deja de enviar video por WiFi si el celular comparte datos (punto de
acceso) en vez de estar conectado a una red. Por cable funciona siempre y sin
instalar nada: el servidor pide el video al iPhone a través de `usbmuxd`
(`servidor/escaner/usbmux.py`), que ya viene en Fedora.

1. Conecta el iPhone por cable, desbloquéalo y abre DroidCam (pantalla encendida).
2. En la página: **Cámara → USB**, elige **iPhone por cable** y **Conectar**.

La pestaña USB también lista las cámaras del computador (/dev/video*), por
ejemplo la webcam, para probar.

### Con 5 vistas

Elige **5 vistas** arriba de la cámara y toma las fotos dando una vuelta al
objeto: **frontal**, **lateral izquierda** (muévete a tu izquierda),
**trasera**, **lateral derecha** y al final la **superior** (desde arriba,
con la parte de arriba del celular hacia el fondo).
Cada foto se segmenta al tomarla y aparece en su casilla; toca una casilla para
repetirla. **Reconstruir 3D** talla el objeto con las siluetas (también sirve
con la frontal y al menos otra vista). Intenta mantener la misma distancia en
todas las fotos y el objeto centrado en el recuadro.

### Con el tablero de marcadores (más preciso)

Imprime el tablero (enlace **tablero impreso** en el modo de vistas, o
`http://localhost:8000/api/tablero.svg`) **al 100 %** y comprueba con una regla
que cada marcador mida 30 mm (si no, `ESCANER_MARCADOR_MM=<lo que mide>`). Pon
el objeto en el centro de la hoja, con el lado FRENTE hacia la foto frontal.

Si el tablero se ve en una foto, su casilla muestra ▦: esa foto trae su pose
exacta (solvePnP), así que ya no hace falta mantener la distancia ni los
ángulos, el modelo sale en milímetros reales y, con el tablero en todas las
fotos, **+ Vista extra** añade más vistas desde cualquier ángulo (8–12 fotos
dan un tallado bastante más fino). La hoja se quita de la silueta comparando
cada píxel con lo que se vería del tablero (homografía del plano); un objeto
blanco o negro puede confundirse con la parte de la hoja de su mismo color.

No hace falta un fondo neutro: la segmentación usa rembg (U²-Net), y si no
está disponible cae a GrabCut de OpenCV, inicializado con el recuadro de
encuadre. Para otra red: `ESCANER_MODELO=isnet-general-use ./iniciar-web.sh`
(más precisa y más lenta).

## Matemática

```
Segmentación:     máscara M(u,v) ∈ {0,1}  (rembg → apertura/cierre → región mayor → huecos rellenos)
Profundidad:      d = distancia al borde (|∇d| = 1),  s = d / d_max
                  D = disparidad de Depth Anything V2,  r = (D − D_borde) / (D_alto − D_borde) ∈ [0, 1]
                  h = R·0,75·(0,7·r + 0,3·√(1 − (1 − s)²))      (sin la red: + 0,06·detalle(∇L))
                  Z_frente = Z₀ − g − h,     Z_atrás = Z₀ + g + 0,6·h
Cámara:           K = [f 0 cx; 0 f cy; 0 0 1],   K⁻¹ = [1/f 0 −cx/f; 0 1/f −cy/f; 0 0 1]
Retroproyección:  P = Z·K⁻¹·[u, v, 1]ᵀ          (N píxeles a la vez: matriz 3×N)
Extrusión:        P_atrás = Z₀·K⁻¹·[u, v, 1]ᵀ + (Z_atrás − Z₀)·e₃
Malla:            F = [frente; frente invertido + N; paredes del contorno]
Borde fino:       x ← x − (α(x) − ½)·∇α / ‖∇α‖²   (Newton hacia α = ½ en cada nodo del contorno)
Ejes del objeto:  p = R·(P − c),   R = diag(1, −1, −1)
Volumen:          V = Σ det[p₀ p₁ p₂] / 6

Varias vistas (envolvente visual):
Cámaras:          λ·[u, v, 1]ᵀ = K·(R_v·X + t_v),   R_v = ejes de cada vista (frontal, ±X, trasera, arriba)
Escala:           d_lateral = f·alto / alto_px,   d_superior = f·ancho / ancho_px,
                  d_trasera = f·(ancho / ancho_px + alto / alto_px) / 2
Centro:           t_v = d_v·K⁻¹·[u_c, v_c, 1]ᵀ
Talla:            ocupación(X) = min_v silueta_v(proyección de X)   → marching cubes en 0,5
Color:            Σ_v w_v·foto_v(p) / Σ_v w_v,   w_v = max(0, n̂·(C_v − p)/‖C_v − p‖)⁴,   C_v = −R_vᵀ·t_v

Con el tablero (16 marcadores ArUco, 4 esquinas Xᵢ conocidas cada uno, en el plano Y = 0):
Pose:             λᵢ·[uᵢ, vᵢ, 1]ᵀ = K·(R·Xᵢ + t)   → solvePnP (IPPE + Levenberg-Marquardt)
Hoja en la foto:  λ·[u, v, 1]ᵀ = H·[X, Z, 1]ᵀ,   H = K·[r₁ r₃ t]   → se quita de la silueta
Talla:            gruesa en todo el espacio sobre la hoja → fina en la caja del objeto
```

Z₀ (400 mm) y el ángulo de visión (65°) son supuestos, así que el tamaño en
milímetros es aproximado; la forma no depende de Z₀.
