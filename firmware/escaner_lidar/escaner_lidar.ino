/*
  Escáner LIDAR 2D — ESP32 + VL53L0X + SG90
  Proyecto de Álgebra Lineal

  Conexiones (ver docs/montaje-lidar.html):
    Servo SG90 : señal -> GPIO 13, rojo -> VIN (5 V), marrón -> GND
    VL53L0X    : SDA -> GPIO 21, SCL -> GPIO 22, VIN -> 3V3, GND -> GND

  Salida por Serial (115200 baudios), una línea JSON por lectura:
    {"r":523,"theta":84}     r en mm, theta en grados del servo
    {"r":-1,"theta":86}      sin objeto dentro del alcance (~1,2 m)
  Mensajes de estado:
    {"estado":"listo"} {"estado":"pausado"} {"estado":"escaneando"} {"error":"..."}

  Comandos que acepta (un carácter, desde la interfaz web o el monitor serial):
    S = iniciar / continuar el barrido
    P = pausar
    C = pausar y centrar el servo en 90°
*/

#include <Wire.h>
#include <ESP32Servo.h>
#include <Adafruit_VL53L0X.h>

const int SERVO_PIN  = 13;
const int SDA_PIN    = 21;
const int SCL_PIN    = 22;
const int ANG_MIN    = 0;
const int ANG_MAX    = 180;
const int PASO       = 2;     // grados entre lecturas
const int ESPERA_MS  = 35;    // tiempo para que el servo llegue al ángulo
const int PULSO_MIN  = 500;   // µs para 0°  (ajusta si el servo zumba en el extremo)
const int PULSO_MAX  = 2400;  // µs para 180°

Servo servo;
Adafruit_VL53L0X lidar;

bool escaneando = true;
int angulo = ANG_MIN;
int direccion = 1;

void estado(const char *texto) {
  Serial.printf("{\"estado\":\"%s\"}\n", texto);
}

void leerComandos() {
  while (Serial.available()) {
    char c = toupper(Serial.read());
    if (c == 'S') {
      escaneando = true;
      estado("escaneando");
    } else if (c == 'P') {
      escaneando = false;
      estado("pausado");
    } else if (c == 'C') {
      escaneando = false;
      angulo = 90;
      servo.write(angulo);
      estado("pausado");
    }
  }
}

void medir() {
  servo.write(angulo);
  delay(ESPERA_MS);

  VL53L0X_RangingMeasurementData_t m;
  lidar.rangingTest(&m, false);

  int r = (m.RangeStatus == 4) ? -1 : m.RangeMilliMeter;  // 4 = fuera de rango
  Serial.printf("{\"r\":%d,\"theta\":%d}\n", r, angulo);
}

void setup() {
  Serial.begin(115200);
  delay(300);

  Wire.begin(SDA_PIN, SCL_PIN);
  if (!lidar.begin()) {
    Serial.println("{\"error\":\"VL53L0X no encontrado: revisa SDA, SCL y 3.3 V\"}");
    while (true) delay(1000);
  }

  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, PULSO_MIN, PULSO_MAX);
  servo.write(angulo);
  delay(400);

  estado("listo");
}

void loop() {
  leerComandos();
  if (!escaneando) {
    delay(20);
    return;
  }

  medir();

  angulo += direccion * PASO;
  if (angulo >= ANG_MAX) { angulo = ANG_MAX; direccion = -1; }
  if (angulo <= ANG_MIN) { angulo = ANG_MIN; direccion = 1; }
}
