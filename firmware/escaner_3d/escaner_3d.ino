/*
  Escáner 3D por cuadrícula — ESP32 + VL53L0X
  Proyecto de Álgebra Lineal

  El sensor va dentro de una caja, mirando hacia abajo. La persona mueve la
  caja casilla por casilla sobre una cuadrícula dibujada en la mesa y en cada
  casilla pide una medición (botón BOOT de la placa o botón "Medir" de la
  página). La página convierte cada medición en un punto 3D.

  Conexiones:
    VL53L0X : VIN -> 3V3, GND -> GND, SDA -> GPIO 21, SCL -> GPIO 22
    Botón   : BOOT de la placa (GPIO 0), no necesita cables

  Salida por Serial (115200 baudios), una línea JSON por mensaje:
    {"r":143}                                  lectura en vivo, ~8 por segundo (-1 = sin eco)
    {"medida":143.2,"desv":1.1,"n":10,"origen":"boton"}   medición promediada
    {"estado":"listo"} {"estado":"sensor listo"} {"error":"..."}

  Comandos (un carácter):
    M = tomar una medición promediada (igual que el botón BOOT)
*/

#include <Wire.h>
#include <Adafruit_VL53L0X.h>

const int SDA_PIN    = 21;
const int SCL_PIN    = 22;
const int BOTON_PIN  = 0;     // botón BOOT, activo en bajo
const int MUESTRAS   = 10;    // lecturas que se promedian en cada medición
const int VIVO_MS    = 120;   // cada cuánto se envía la lectura en vivo

Adafruit_VL53L0X lidar;
bool sensorListo = false;
unsigned long ultimoVivo = 0;
unsigned long ultimoIntento = 0;
bool botonAntes = HIGH;

void estado(const char *texto) {
  Serial.printf("{\"estado\":\"%s\"}\n", texto);
}

void iniciarSensor() {
  sensorListo = lidar.begin();
  if (sensorListo) {
    // Modo de alta precisión: más lento (~200 ms) pero con menos ruido
    lidar.configSensor(Adafruit_VL53L0X::VL53L0X_SENSE_HIGH_ACCURACY);
    estado("sensor listo");
  } else {
    Serial.println("{\"error\":\"VL53L0X no encontrado: revisa SDA (GPIO 21), SCL (GPIO 22), VIN (3V3) y GND\"}");
  }
}

// Devuelve la distancia en mm o -1 si la lectura no es válida. El sensor
// marca cada lectura con un estado: solo el 0 es una medición confiable
// (los demás son fuera de rango, señal débil o fallo de fase).
// En este modo el sensor suele alternar una lectura válida y una fallida,
// así que se reintenta hasta 3 veces.
int leer() {
  VL53L0X_RangingMeasurementData_t m;
  for (int intento = 0; intento < 3; intento++) {
    lidar.rangingTest(&m, false);
    if (m.RangeStatus == 0 && m.RangeMilliMeter < 2000) return m.RangeMilliMeter;
  }
  return -1;
}

void medir(const char *origen) {
  if (!sensorListo) {
    Serial.println("{\"error\":\"No se puede medir: el sensor no responde\"}");
    return;
  }
  float suma = 0, suma2 = 0;
  int n = 0;
  for (int i = 0; i < MUESTRAS; i++) {
    int r = leer();
    if (r > 0) { suma += r; suma2 += (float)r * r; n++; }
  }
  if (n == 0) {
    Serial.printf("{\"medida\":-1,\"desv\":0,\"n\":0,\"origen\":\"%s\"}\n", origen);
    return;
  }
  float media = suma / n;
  float desv = n > 1 ? sqrt(max(0.0f, (suma2 - n * media * media) / (n - 1))) : 0;
  Serial.printf("{\"medida\":%.1f,\"desv\":%.1f,\"n\":%d,\"origen\":\"%s\"}\n", media, desv, n, origen);
}

void setup() {
  Serial.begin(115200);
  delay(300);
  pinMode(BOTON_PIN, INPUT_PULLUP);
  Wire.begin(SDA_PIN, SCL_PIN);
  iniciarSensor();
  estado("listo");
}

void loop() {
  // Comandos desde la página
  while (Serial.available()) {
    char c = toupper(Serial.read());
    if (c == 'M') medir("web");
  }

  // Botón BOOT: medir al soltarlo (flanco de subida), con antirrebote simple
  bool boton = digitalRead(BOTON_PIN);
  if (botonAntes == LOW && boton == HIGH) {
    delay(30);
    medir("boton");
  }
  botonAntes = boton;

  // Si el sensor no responde, se vuelve a buscar cada 2 s
  if (!sensorListo) {
    if (millis() - ultimoIntento > 2000) { ultimoIntento = millis(); iniciarSensor(); }
    return;
  }

  // Lectura en vivo para la página
  if (millis() - ultimoVivo > VIVO_MS) {
    ultimoVivo = millis();
    Serial.printf("{\"r\":%d}\n", leer());
  }
}
