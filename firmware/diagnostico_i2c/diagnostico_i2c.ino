/*
  Diagnóstico del sensor VL53L0X.
  Cada segundo informa por Serial (115200):
    - si SDA (GPIO 21) y SCL (GPIO 22) tienen tensión: el módulo trae
      resistencias pull-up, así que con el sensor alimentado y conectado
      ambas líneas deben leer 1 aunque el pin interno tire hacia abajo;
    - si alguna línea está en corto con GND (lee 0 aun con pull-up interno);
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

  // Con la resistencia interna hacia arriba: si aun así lee 0, la línea está
  // en corto con GND; si lee 1, está al aire (cable suelto o sin soldar).
  pinMode(SDA_PIN, INPUT_PULLUP);
  pinMode(SCL_PIN, INPUT_PULLUP);
  delay(5);
  int sdaArriba = digitalRead(SDA_PIN);
  int sclArriba = digitalRead(SCL_PIN);

  Wire.begin(SDA_PIN, SCL_PIN);
  Serial.printf("{\"sda\":%d,\"scl\":%d,\"sda_pullup\":%d,\"scl_pullup\":%d,\"dispositivos\":[", sda, scl, sdaArriba, sclArriba);
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
