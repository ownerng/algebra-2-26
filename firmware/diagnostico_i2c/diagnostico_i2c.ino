/*
  Diagnóstico del sensor VL53L0X.
  Cada segundo informa por Serial (115200):
    - si SDA (GPIO 21) y SCL (GPIO 22) tienen tensión: el módulo trae
      resistencias pull-up, así que con el sensor alimentado y conectado
      ambas líneas deben leer 1 aunque el pin interno tire hacia abajo;
    - qué direcciones responden en el bus I2C (el VL53L0X es la 0x29).
*/

#include <Wire.h>

const int SDA_PIN = 21;
const int SCL_PIN = 22;

void setup() {
  Serial.begin(115200);
  delay(300);
}

void loop() {
  pinMode(SDA_PIN, INPUT_PULLDOWN);
  pinMode(SCL_PIN, INPUT_PULLDOWN);
  delay(5);
  int sda = digitalRead(SDA_PIN);
  int scl = digitalRead(SCL_PIN);

  Wire.begin(SDA_PIN, SCL_PIN);
  Serial.printf("{\"sda\":%d,\"scl\":%d,\"dispositivos\":[", sda, scl);
  bool primero = true;
  for (uint8_t dir = 1; dir < 127; dir++) {
    Wire.beginTransmission(dir);
    if (Wire.endTransmission() == 0) {
      Serial.printf("%s%u", primero ? "" : ",", dir);
      primero = false;
    }
  }
  Serial.println("]}");
  Wire.end();
  delay(1000);
}
