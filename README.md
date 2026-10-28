# Escáner LIDAR 2D — Álgebra Lineal

Prototipo con ESP32 + VL53L0X + SG90 que barre 0–180°, convierte cada lectura
`(r, θ)` a coordenadas cartesianas con una matriz de rotación y fusiona dos
escaneos en un mapa global con una matriz homogénea 3×3.

## Carpetas

| Ruta | Qué es |
|---|---|
| `docs/montaje-lidar.html` | Guía de armado: protoboard, esquemático, montaje mecánico |
| `docs/informe.tex` | Informe en LaTeX (APA 7). Compila en Overleaf o con `make informe` |
| `firmware/escaner_lidar/` | Programa del escáner para el ESP32 |
| `firmware/prueba_servo/` | Prueba solo del servo (0°, 90°, 180° y barrido) |
| `firmware/servo_reposo/` | Deja el servo quieto en 90° |
| `experimentos/generar.mjs` | Genera las cifras, tablas y gráficas del informe (`make datos`) |
| `web/` | Interfaz en React + Bun: radar 2D, mapa 3D, prototipo simulado y matrices en vivo |
| `tests/` | Pruebas de la matemática (`bun test tests/`) |
| `iniciar-web.sh` | Abre la interfaz en `http://localhost:3000` |

## 1. Subir el firmware

1. Abre **Arduino IDE** y carga `firmware/escaner_lidar/escaner_lidar.ino`.
2. Placa: **ESP32 Dev Module**. Puerto: `/dev/ttyUSB0` o `/dev/ttyACM0`.
3. Sube el programa. Si aparece “Failed to connect”, mantén presionado BOOT mientras dice “Connecting…”.
4. En el monitor serial (115200 baudios) deben salir líneas como `{"r":523,"theta":84}`.
5. **Cierra el monitor serial** antes de usar la web: el puerto solo lo puede usar un programa a la vez.

## 2. Abrir la interfaz

```bash
./iniciar-web.sh
```

Necesita [Bun](https://bun.sh). Usa **Chrome, Chromium o Edge**: Firefox no
tiene Web Serial. Pulsa **Conectar ESP32** y elige el puerto. Sin el sensor,
pulsa **Simular prototipo**: el panel del prototipo muestra lo que debería
hacer el aparato y el radar superpone la habitación real para comparar.
`localhost:3000/#demo` arranca la simulación directamente.

## 3. Hacer el mapa con dos posiciones

1. Pon el aparato en la posición 1. Elige **Posición 1** (tecla `1`) y deja que barra unas 2 veces.
2. Mueve el aparato. Mide con cinta cuánto se desplazó (`dx`, `dy` en mm) y cuánto giró (`α`).
3. Elige **Posición 2** (tecla `2`), escribe `dx`, `dy`, `α` y deja que barra.
4. Si los dos mapas no encajan, ajusta `α`, `dx` y `dy` hasta que las paredes coincidan.
   Esto también sirve como demostración en vivo: con una pose mal medida, el mapa se desarma.

`d` es la distancia del eje del servo al lente del sensor. Si al pedir 0° el
sensor apunta a la izquierda, activa **invertir ángulo**.

## Matemática

```
Paso 1   [xL]   [cos θ  −sin θ] [r + d]
         [yL] = [sin θ   cos θ] [  0  ]

Paso 2   [xG]   [cos α  −sin α  dx] [xL]
         [yG] = [sin α   cos α  dy] [yL]
         [1 ]   [  0       0     1] [1 ]
```

Código: `web/js/matematica.js`.
