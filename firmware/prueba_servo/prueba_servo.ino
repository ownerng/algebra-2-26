/*
  Prueba solo del servo (sin sensor).
  Mueve el SG90 a 0°, 90° y 180°, y luego hace un barrido lento.
  El LED azul de la placa (GPIO 2) parpadea en cada movimiento.
  Monitor serial: 115200 baudios.
*/

#include <ESP32Servo.h>

const int SERVO_PIN = 13;
const int LED_PIN   = 2;

Servo servo;

void ir(int angulo, int espera) {
  servo.write(angulo);
  digitalWrite(LED_PIN, !digitalRead(LED_PIN));
  Serial.printf("Servo en %d°\n", angulo);
  delay(espera);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, 500, 2400);
  delay(500);
  Serial.println("Prueba del servo: debe ir a 0°, 90° y 180° y luego barrer.");
}

void loop() {
  ir(0, 1000);
  ir(90, 1000);
  ir(180, 1000);
  ir(90, 1000);
  for (int a = 0; a <= 180; a += 2) { servo.write(a); delay(25); }
  for (int a = 180; a >= 0; a -= 2) { servo.write(a); delay(25); }
  Serial.println("Barrido completo. Repitiendo...");
}
