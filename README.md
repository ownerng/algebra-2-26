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

### ESP32-CAM por cable (sin WiFi)

Una ESP32-CAM (AI Thinker) conectada por un adaptador USB-serial (FTDI o la
base MB) manda el video por el mismo cable. Se graba una sola vez:

```
cd firmware/camara_esp32 && pio run -t upload      # PlatformIO (pip install platformio)
```

En la página: **Cámara → USB**, elige **ESP32-CAM por USB** (`ttyUSB0`) y
**Conectar**. Resolución por defecto SVGA (800×600, ~3 cuadros/s). Su ángulo de
visión es ~65°, el mismo `fov` que usa el escáner. El lente es de enfoque
manual: gíralo hasta ver nítido el objeto a la distancia de trabajo.

### Calibrar la cámara (una vez por cámara y resolución)

Imprime el tablero (`http://localhost:8000/api/tablero.svg`) **al 100 %** y
comprueba con una regla que cada marcador mida 30 mm (si no,
`ESCANER_MARCADOR_MM=<lo que mide>`); no recortes el borde blanco de la hoja.
Elige **Calibrar**, muestra el tablero a la cámara desde ángulos distintos
(inclinado, cerca, lejos y que caiga en cada esquina de la imagen) y pulsa
**Capturar** en cada uno: 10–15 fotos. **Calcular** halla la matriz K real y la
distorsión del lente y la guarda en `~/.cache/escaner/calibracion.json`; desde
ahí los dos modos la usan (la ficha del modelo dice «K calibrada») en lugar del
ángulo de visión supuesto de 65°. Un error por debajo de ~1 px es bueno.

### Procesamiento alto (IA generativa, solo objetos)

Arriba, **Procesamiento → Alto**: en lugar del cálculo local, la foto del objeto va a Hunyuan3D v3.1 Pro en
fal.ai, que devuelve el objeto completo con textura (.glb), parte de atrás
incluida. Tarda 1–4 min y cuesta ~US$0,38 por modelo. Necesita una clave de
fal.ai en el servidor (nunca en el repo):

```bash
FAL_KEY=tu_clave servidor/.venv/Scripts/uvicorn app:app --app-dir servidor --port 8000
```

### Habitaciones

Arriba, elige **Habitaciones**: toma la foto con la cámara o súbela desde el
equipo. Funciona mejor desde una esquina o la puerta, con el celular horizontal
a la altura del pecho y el piso a la vista. Usa tres redes (se descargan solas
la primera vez, ~230 MB): Depth Anything V2 Metric Indoor (profundidad en
metros), SegFormer-B2 (qué es cada píxel: pared, piso, sofá…) y LaMa (inventa
lo que hay detrás de los muebles). El visor arranca desde el punto de la foto;
arrastra para mirar alrededor o pulsa **Girar** para que se meza solo.

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
Suavizado:        Taubin, L = D⁻¹·A − I:  V ← V + 0,5·L·V,  V ← V − 0,53·L·V   (×10, sin encoger)
Color:            Σ_v w_v·foto_v(p) / Σ_v w_v,   w_v = max(0, n̂·(C_v − p)/‖C_v − p‖)⁴,   C_v = −R_vᵀ·t_v

Con el tablero (16 marcadores ArUco, 4 esquinas Xᵢ conocidas cada uno, en el plano Y = 0):
Pose:             λᵢ·[uᵢ, vᵢ, 1]ᵀ = K·(R·Xᵢ + t)   → solvePnP (IPPE + Levenberg-Marquardt)
Hoja en la foto:  λ·[u, v, 1]ᵀ = H·[X, Z, 1]ᵀ,   H = K·[r₁ r₃ t]   → se quita de la silueta
Talla:            gruesa en todo el espacio sobre la hoja → fina en la caja del objeto

Calibración (Zhang): cada foto del tablero da H = K·[r₁ r₂ t]; como r₁ ⊥ r₂ y ‖r₁‖ = ‖r₂‖,
                  h₁ᵀ·B·h₂ = 0,  h₁ᵀ·B·h₁ = h₂ᵀ·B·h₂,  B = K⁻ᵀK⁻¹   → B por mínimos cuadrados → K
                  y se refina con la distorsión (cv2.calibrateCamera). Al reducir la foto por s: K' = diag(s, s, 1)·K

Habitación (una foto):
Profundidad:      Z(u, v) en metros (Depth Anything V2 Metric Indoor),   P = Z·K⁻¹·[u, v, 1]ᵀ
Planos:           RANSAC (3 puntos → n = (b − a) × (c − a)) por clase de SegFormer (piso, techo, paredes) y
                  mínimos cuadrados: n = último vector singular de los puntos centrados (SVD)
Maqueta:          rayo r = K⁻¹·[u, v, 1]ᵀ ∩ plano n·P = d  →  Z = d / (n·r)
Cuarto vacío:     Z_vacío = min { d_k / (n_k·r) > 0 }   (cuarto convexo: el corte más cercano)
Bordes:           en el salto, Z = Σ w·Z·[misma clase] / Σ w·[misma clase]   (convolución normalizada)
Cortes:           de canto |n̂·r̂| < sen 5°  y  Z_max / Z_min > 1,12
Capa de atrás:    malla con Z_vacío detrás de los muebles, textura de LaMa
Enderezar:        R = I + [v]ₓ + [v]ₓ²/(1 + c),   v = n × arriba,  c = n·arriba   (Rodrigues: R·n = arriba)
Altura de cámara: −d, con el plano n·P = d
```

En el modo de 1 foto, Z₀ (400 mm) es supuesto, así que el tamaño en milímetros
es aproximado; la forma no depende de Z₀. Sin calibrar, el ángulo de visión
(65°) también es supuesto. Con el tablero, las medidas salen en mm reales.
