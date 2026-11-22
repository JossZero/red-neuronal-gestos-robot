/*
  Firmware para Carrito Robótico Controlado por Gestos (Fase 10).
  Compatible con Arduino UNO, Nano, Mega y ESP32.
  Controlador de motores: Puente H L298N o L293D.

  Protocolo de Comunicación Serial (Baudrate: 115200):
  - 'S' o "STOP"     -> Detener motores.
  - 'F' o "FORWARD"  -> Ambos motores avanzan.
  - 'L' o "LEFT"     -> Giro sobre el eje a la izquierda.
  - 'R' o "RIGHT"    -> Giro sobre el eje a la derecha.
  - 'B' o "BACKWARD" -> Ambos motores retroceden.

  Mecanismo de Seguridad Fail-Safe (Timeout):
  Si no se recibe un comando durante más de 800 ms, los motores
  se apagan automáticamente para evitar choques si se pierde la conexión.
*/

// ==========================================
// Configuración de Pines del Driver L298N
// ==========================================
// Motor Izquierdo (Motor A)
const int IN1 = 4;
const int IN2 = 5;
const int ENA = 6;  // Pin PWM para velocidad Motor A

// Motor Derecho (Motor B)
const int IN3 = 7;
const int IN4 = 8;
const int ENB = 9;  // Pin PWM para velocidad Motor B

// Velocidad estándar de los motores (0 a 255)
const int VELOCIDAD_BASE = 200;

// Variables de seguridad y temporización
unsigned long ultimoComandoMillis = 0;
const unsigned long TIMEOUT_SEGURIDAD_MS = 800; // 800 ms sin comando = STOP

void setup() {
  // Configuración de pines como salidas
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);

  // Inicializar motores apagados
  detenerse();

  // Iniciar puerto serial a 115200 baudios
  Serial.begin(115200);
  while (!Serial) {
    ; // Esperar conexión serial (para placas con USB nativo)
  }

  Serial.println("[ROBOT_LISTO] Carrito receptor de comandos iniciado.");
}

void loop() {
  // 1. Lectura de comandos desde la PC por Serial
  if (Serial.available() > 0) {
    String comando = Serial.readStringUntil('\n');
    comando.trim();
    comando.toUpperCase();

    if (comando.length() > 0) {
      procesarComando(comando);
      ultimoComandoMillis = millis(); // Renovar temporizador de seguridad
    }
  }

  // 2. Temporizador de Seguridad (Watchdog por software)
  // Si transcurre el tiempo límite sin recibir señal, detenerse por seguridad
  if (millis() - ultimoComandoMillis > TIMEOUT_SEGURIDAD_MS) {
    detenerse();
  }
}

void procesarComando(String cmd) {
  if (cmd == "F" || cmd == "FORWARD") {
    avanzar();
    Serial.println("[ACK] FORWARD");
  } else if (cmd == "S" || cmd == "STOP") {
    detenerse();
    Serial.println("[ACK] STOP");
  } else if (cmd == "L" || cmd == "LEFT") {
    girarIzquierda();
    Serial.println("[ACK] LEFT");
  } else if (cmd == "R" || cmd == "RIGHT") {
    girarDerecha();
    Serial.println("[ACK] RIGHT");
  } else if (cmd == "B" || cmd == "BACKWARD") {
    retroceder();
    Serial.println("[ACK] BACKWARD");
  }
}

// ==========================================
// Funciones de Control de Movimiento
// ==========================================
void avanzar() {
  analogWrite(ENA, VELOCIDAD_BASE);
  analogWrite(ENB, VELOCIDAD_BASE);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void retroceder() {
  analogWrite(ENA, VELOCIDAD_BASE);
  analogWrite(ENB, VELOCIDAD_BASE);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void girarIzquierda() {
  analogWrite(ENA, VELOCIDAD_BASE);
  analogWrite(ENB, VELOCIDAD_BASE);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
}

void girarDerecha() {
  analogWrite(ENA, VELOCIDAD_BASE);
  analogWrite(ENB, VELOCIDAD_BASE);
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, HIGH);
}

void detenerse() {
  analogWrite(ENA, 0);
  analogWrite(ENB, 0);
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}
