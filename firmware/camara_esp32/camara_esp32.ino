/*
  Cámara USB para el escáner 3D — ESP32-CAM (AI Thinker, sensor OV2640)
  Proyecto de Álgebra Lineal

  Envía video MJPEG por el cable USB (puerto serial a 1 Mbaud): cada cuadro es
  un JPEG completo (empieza con FF D8 y termina con FF D9), uno detrás de otro.
  El servidor (servidor/escaner/camara.py) los separa por esos marcadores, igual
  que con el video de DroidCam, así que no hace falta WiFi ni contraseñas.

  Comandos (un carácter, desde el computador):
    + / -   sube o baja la resolución (QVGA … UXGA)
    q / Q   más / menos compresión (calidad JPEG)
    h / H   espejo horizontal activado / desactivado
    v / V   volteo vertical (de cabeza) activado / desactivado

  Compilar y grabar (Arduino IDE o arduino-cli):
    placa  : "AI Thinker ESP32-CAM"  (esp32:esp32:esp32cam)
    PSRAM  : habilitado

  El LED rojo de la placa (GPIO 33) parpadea mientras envía cuadros; si
  parpadea rápido, la cámara OV2640 no arrancó (revisa la cinta flexible).
*/

#include "esp_camera.h"

// Pines de la AI Thinker ESP32-CAM
#define PWDN_GPIO_NUM   32
#define RESET_GPIO_NUM  -1
#define XCLK_GPIO_NUM    0
#define SIOD_GPIO_NUM   26
#define SIOC_GPIO_NUM   27
#define Y9_GPIO_NUM     35
#define Y8_GPIO_NUM     34
#define Y7_GPIO_NUM     39
#define Y6_GPIO_NUM     36
#define Y5_GPIO_NUM     21
#define Y4_GPIO_NUM     19
#define Y3_GPIO_NUM     18
#define Y2_GPIO_NUM      5
#define VSYNC_GPIO_NUM  25
#define HREF_GPIO_NUM   23
#define PCLK_GPIO_NUM   22
#define LED_ROJO        33   // activo en bajo

const long BAUDIOS = 1000000;

// Resoluciones disponibles, de menor a mayor. 5 = SVGA (800x600) por defecto:
// ~40 KB por cuadro, unos 3 cuadros por segundo por el cable.
const framesize_t TAMANOS[] = {FRAMESIZE_QVGA, FRAMESIZE_VGA, FRAMESIZE_SVGA,
                               FRAMESIZE_XGA, FRAMESIZE_SXGA, FRAMESIZE_UXGA};
const int N_TAMANOS = sizeof(TAMANOS) / sizeof(TAMANOS[0]);
int tamano = 2;
int calidad = 10;            // 4 (mejor) … 63 (más comprimida)
bool espejoH = false, espejoV = false;
unsigned long ultimoLed = 0;
bool ledEncendido = false;

bool iniciarCamara() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM;  c.pin_d1 = Y3_GPIO_NUM;
  c.pin_d2 = Y4_GPIO_NUM;  c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM;  c.pin_d5 = Y7_GPIO_NUM;
  c.pin_d6 = Y8_GPIO_NUM;  c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM;
  c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM;
  c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM;
  c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM;
  c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  c.grab_mode = CAMERA_GRAB_LATEST;      // siempre el cuadro más reciente
  if (psramFound()) {
    c.frame_size = FRAMESIZE_UXGA;       // reserva memoria para el mayor tamaño
    c.jpeg_quality = calidad;
    c.fb_count = 2;
    c.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    c.frame_size = FRAMESIZE_SVGA;
    c.jpeg_quality = 12;
    c.fb_count = 1;
    c.fb_location = CAMERA_FB_IN_DRAM;
    tamano = 2;
  }
  if (esp_camera_init(&c) != ESP_OK) return false;
  sensor_t *s = esp_camera_sensor_get();
  s->set_framesize(s, TAMANOS[tamano]);
  s->set_quality(s, calidad);
  // Exposición y balance de blancos automáticos: la cámara se adapta sola
  s->set_exposure_ctrl(s, 1);
  s->set_gain_ctrl(s, 1);
  s->set_whitebal(s, 1);
  s->set_awb_gain(s, 1);
  s->set_lenc(s, 1);                     // corrige el oscurecimiento en las esquinas
  return true;
}

void aplicar() {
  sensor_t *s = esp_camera_sensor_get();
  s->set_framesize(s, TAMANOS[tamano]);
  s->set_quality(s, calidad);
  s->set_hmirror(s, espejoH);
  s->set_vflip(s, espejoV);
}

void leerComandos() {
  while (Serial.available()) {
    char ch = Serial.read();
    switch (ch) {
      case '+': if (tamano < N_TAMANOS - 1) tamano++; break;
      case '-': if (tamano > 0) tamano--; break;
      case 'q': if (calidad > 4) calidad--; break;
      case 'Q': if (calidad < 63) calidad++; break;
      case 'h': espejoH = true; break;
      case 'H': espejoH = false; break;
      case 'v': espejoV = true; break;
      case 'V': espejoV = false; break;
      default: return;
    }
    aplicar();
  }
}

void setup() {
  Serial.begin(BAUDIOS);
  pinMode(LED_ROJO, OUTPUT);
  digitalWrite(LED_ROJO, HIGH);
  if (!iniciarCamara()) {
    // Sin cámara: parpadeo rápido del LED y un aviso de texto cada segundo
    while (true) {
      Serial.println("{\"error\":\"no se pudo iniciar la camara OV2640: revisa la cinta flexible\"}");
      digitalWrite(LED_ROJO, LOW);  delay(150);
      digitalWrite(LED_ROJO, HIGH); delay(850);
    }
  }
  // Descarta los primeros cuadros: salen oscuros mientras se ajusta la exposición
  for (int i = 0; i < 4; i++) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (fb) esp_camera_fb_return(fb);
  }
}

void loop() {
  leerComandos();
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) { delay(10); return; }
  // El JPEG va tal cual: FF D8 … FF D9
  Serial.write(fb->buf, fb->len);
  esp_camera_fb_return(fb);

  if (millis() - ultimoLed > 500) {
    ultimoLed = millis();
    ledEncendido = !ledEncendido;
    digitalWrite(LED_ROJO, ledEncendido ? LOW : HIGH);
  }
}
