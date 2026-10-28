# Escáner 3D por cuadrícula — Álgebra Lineal

Prototipo con un ESP32 y un sensor láser VL53L0X dentro de una caja. La caja se
mueve casilla por casilla sobre una cuadrícula dibujada en la mesa; en cada
casilla mide la distancia hacia abajo, y la página web convierte las medidas en
la superficie 3D del objeto. La versión anterior, con servo, está en la rama
`version-servo`.

## Carpetas

| Ruta | Qué es |
|---|---|
| `firmware/escaner_3d/` | Programa del ESP32: medición promediada con el botón BOOT o el comando `M` |
| `firmware/diagnostico_i2c/` | Revisa la conexión del sensor (SDA, SCL y dirección 0x29) |
| `web/` | Interfaz en React + Bun: cuadrícula guiada, figura 3D y cálculo paso a paso |
| `web/puente.ts` | Puente USB → WebSocket: la página funciona en cualquier navegador |
| `experimentos/` | Simulación y mediciones reales que generan las cifras del informe |
| `docs/informe.tex` | Informe en LaTeX (APA 7). Compila en Overleaf o con `make informe` |
| `tests/` | Pruebas de la matemática (`bun test tests/`) |

## Uso

1. Sube `firmware/escaner_3d` al ESP32 con Arduino IDE (placa **ESP32 Dev Module**).
2. Ejecuta `./iniciar-web.sh`: abre `http://localhost:3000` y el puente USB.
3. Pulsa **Conectar ESP32**. Sin hardware, pulsa **Simular prototipo**
   (`#demo-auto` al final de la dirección hace un escaneo automático).
4. Con la mesa vacía, pulsa **Medir la mesa** para obtener la altura H.
5. Pon el objeto en el centro de la cuadrícula y mide casilla por casilla
   siguiendo la página (botón **Medir**, barra espaciadora o BOOT).
6. Guarda o exporta la figura (`.obj`, `.csv` o `.json`).

## Matemática

```
[x, y, z, 1]ᵀ = A · [i, j, r, 1]ᵀ,   A = T·F·E = [Δ 0 0 0; 0 Δ 0 0; 0 0 −1 H; 0 0 0 1]
Plano de la mesa:  (MᵀM)·[a b c]ᵀ = Mᵀz,   altura h = z − (a·x + b·y + c)
Vista:  s = Π·Rx(φ)·Rz(θ)·(p − c)
Sombreado:  n = (B − A) × (C − A),   brillo = n̂ · ℓ̂
```

Código: `web/src/lib/algebra3d.ts`.
