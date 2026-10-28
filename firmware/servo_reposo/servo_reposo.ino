/*
  Deja el servo quieto: lo lleva a 90° y luego suelta la señal.
  Úsalo para detener cualquier prueba.
*/

#include <ESP32Servo.h>

const int SERVO_PIN = 13;
Servo servo;

void setup() {
  Serial.begin(115200);
  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, 500, 2400);
  servo.write(90);
  delay(800);
  servo.detach();
  Serial.println("Servo detenido en 90°.");
}

void loop() {}
