/*
  =============================================================================
  PROYECTO: Control de Motores por Gestos de Manos con IA (Piedra, Papel, Tijera)
  HARDWARE: ESP32 (NodeMCU-32S / ESP32 DevKit V1)
  COMUNICACIÓN: Serial UART por USB a 115200 baudios
  =============================================================================

  Este firmware recibe los comandos clasificados por la red neuronal desde la PC
  y controla físicamente los actuadores del robot:
  
  Mapeo de Gestos a Movimiento:
  - PIEDRA   ("ROCK" / "STOP")     -> Detener motor / Servo a 0 grados
  - PAPEL    ("PAPER" / "FORWARD")  -> Girar motor adelante / Servo a 180 grados
  - TIJERA   ("SCISSORS" / "RIGHT") -> Girar motor inverso / Servo a 90 grados
  
  Mecanismo de Seguridad Fail-Safe (Watchdog por Software):
  Si el cable se desconecta o la PC no envía comandos durante más de 800 ms,
  el ESP32 apaga o regresa automáticamente el actuador a posición de reposo.
*/

// =============================================================================
// SELECCIÓN DE TIPO DE ACTUADOR
// Descomenta la opción que vayas a utilizar:
// =============================================================================
#define MODO_SERVOMOTOR      // Opción A: Servomotor (SG90, MG90S, MG996R)
// #define MODO_MOTOR_DC     // Opción B: Motor DC con Driver L298N / L293D

// =============================================================================
// CONFIGURACIÓN DE PINES (ESP32)
// =============================================================================
const int PIN_LED_ESTADO = 2; // LED azul integrado en la mayoría de placas ESP32

#if defined(MODO_SERVOMOTOR)
  #include <ESP32Servo.h>
  Servo miServo;
  const int PIN_SERVO = 18;  // GPIO 18 conectado al cable de señal (naranja/blanco) del servo
  
  // Posiciones angulares para cada gesto
  const int ANGULO_PIEDRA = 0;     // Puño cerrado -> 0° (o pinza cerrada)
  const int ANGULO_TIJERA = 90;    // Dos dedos    -> 90° (posición intermedia)
  const int ANGULO_PAPEL  = 180;   // Mano abierta -> 180° (o pinza abierta)

#elif defined(MODO_MOTOR_DC)
  // Pines para puente H L298N o L293D
  const int PIN_IN1 = 19;     // GPIO 19 -> Dirección 1
  const int PIN_IN2 = 21;     // GPIO 21 -> Dirección 2
  const int PIN_ENA = 22;     // GPIO 22 -> PWM Velocidad
  
  // Canales PWM nativos del ESP32
  const int CANAL_PWM = 0;
  const int FREC_PWM = 1000;  // 1 kHz
  const int RESOLUCION_PWM = 8; // 8 bits (0 a 255)
#endif

// =============================================================================
// TEMPORIZADOR DE SEGURIDAD (FAIL-SAFE WATCHDOG)
// =============================================================================
unsigned long ultimoComandoMillis = 0;
const unsigned long TIMEOUT_SEGURIDAD_MS = 800; // 800 ms sin señal = STOP preventivo
bool estadoSeguroActivo = true;

// =============================================================================
// SETUP: Inicialización de periféricos
// =============================================================================
void setup() {
  Serial.begin(115200);
  pinMode(PIN_LED_ESTADO, OUTPUT);
  digitalWrite(PIN_LED_ESTADO, LOW);

#if defined(MODO_SERVOMOTOR)
  // Configuración de frecuencia para el servo (50 Hz estándar)
  ESP32PWM::allocateTimer(0);
  miServo.setPeriodHertz(50);
  miServo.attach(PIN_SERVO, 500, 2400); // Rango de pulso típico para SG90
  miServo.write(ANGULO_PIEDRA);         // Iniciar en posición de reposo
#elif defined(MODO_MOTOR_DC)
  pinMode(PIN_IN1, OUTPUT);
  pinMode(PIN_IN2, OUTPUT);
  ledcSetup(CANAL_PWM, FREC_PWM, RESOLUCION_PWM);
  ledcAttachPin(PIN_ENA, CANAL_PWM);
  detenerMotorDC();
#endif

  Serial.println("==================================================");
  Serial.println("[ESP32_OK] Firmware de Control Gestual Listo.");
  Serial.println("Esperando comandos desde la red neuronal de Python...");
  Serial.println("==================================================");

  // Parpadeo triple de confirmación
  for (int i = 0; i < 3; i++) {
    digitalWrite(PIN_LED_ESTADO, HIGH);
    delay(100);
    digitalWrite(PIN_LED_ESTADO, LOW);
    delay(100);
  }
}

// =============================================================================
// LOOP: Escucha Serial y Vigilancia de Seguridad
// =============================================================================
void loop() {
  // 1. Recepción y decodificación de comandos desde Python
  if (Serial.available() > 0) {
    String linea = Serial.readStringUntil('\n');
    linea.trim();
    linea.toUpperCase();

    if (linea.length() > 0) {
      ejecutarComando(linea);
      ultimoComandoMillis = millis(); // Renovar tiempo de actividad
      estadoSeguroActivo = false;
    }
  }

  // 2. Watchdog de Seguridad Fail-Safe
  // Si transcurre el tiempo límite sin datos, forzar estado seguro
  if (!estadoSeguroActivo && (millis() - ultimoComandoMillis > TIMEOUT_SEGURIDAD_MS)) {
    aplicarParadaSegura();
    estadoSeguroActivo = true;
  }
}

// =============================================================================
// EJECUCIÓN DE COMANDOS SEGÚN EL GESTO DETECTADO
// =============================================================================
void ejecutarComando(String cmd) {
  // GESTO: PIEDRA (Reposo / Stop)
  if (cmd == "ROCK" || cmd == "STOP" || cmd == "R" || cmd == "S") {
#if defined(MODO_SERVOMOTOR)
    miServo.write(ANGULO_PIEDRA);
#elif defined(MODO_MOTOR_DC)
    detenerMotorDC();
#endif
    digitalWrite(PIN_LED_ESTADO, LOW);
    Serial.println("[ACK] PIEDRA -> Reposo / Stop");
  }
  // GESTO: PAPEL (Adelante / Apertura)
  else if (cmd == "PAPER" || cmd == "FORWARD" || cmd == "P" || cmd == "F") {
#if defined(MODO_SERVOMOTOR)
    miServo.write(ANGULO_PAPEL);
#elif defined(MODO_MOTOR_DC)
    moverMotorDC(true, 220); // Giro horario a ~85% potencia
#endif
    digitalWrite(PIN_LED_ESTADO, HIGH);
    Serial.println("[ACK] PAPEL -> Adelante / Activo");
  }
  // GESTO: TIJERA (Giro / Media posición)
  else if (cmd == "SCISSORS" || cmd == "RIGHT" || cmd == "T" || cmd == "L") {
#if defined(MODO_SERVOMOTOR)
    miServo.write(ANGULO_TIJERA);
#elif defined(MODO_MOTOR_DC)
    moverMotorDC(false, 150); // Giro antihorario / Reversa
#endif
    // Parpadeo rápido del LED
    digitalWrite(PIN_LED_ESTADO, !digitalRead(PIN_LED_ESTADO));
    Serial.println("[ACK] TIJERA -> Giro / Posición Intermedia");
  }
}

// =============================================================================
// PARADA SEGURA PREVENTIVA
// =============================================================================
void aplicarParadaSegura() {
#if defined(MODO_SERVOMOTOR)
  miServo.write(ANGULO_PIEDRA);
#elif defined(MODO_MOTOR_DC)
  detenerMotorDC();
#endif
  digitalWrite(PIN_LED_ESTADO, LOW);
  Serial.println("[FAIL-SAFE] Timeout de comunicacion: Motor en STOP preventivo.");
}

#if defined(MODO_MOTOR_DC)
void moverMotorDC(bool adelante, int velocidad) {
  if (adelante) {
    digitalWrite(PIN_IN1, HIGH);
    digitalWrite(PIN_IN2, LOW);
  } else {
    digitalWrite(PIN_IN1, LOW);
    digitalWrite(PIN_IN2, HIGH);
  }
  ledcWrite(CANAL_PWM, constrain(velocidad, 0, 255));
}

void detenerMotorDC() {
  digitalWrite(PIN_IN1, LOW);
  digitalWrite(PIN_IN2, LOW);
  ledcWrite(CANAL_PWM, 0);
}
#endif
