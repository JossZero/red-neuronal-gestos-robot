"""
Módulo de Control Seguro e Integración Robótica (Fase 10).

Este módulo define la capa de control robótico desacoplada de la visión artificial.
Aplica mecanismos de seguridad industrial y filtrado temporal para evitar movimientos
erráticos, espurios o no deseados del robot físico ante predicciones inestables.

Características Clave:
1. Filtro de Seguridad por Umbral de Confianza (Confidence Threshold):
   Si la probabilidad predicha es menor a un umbral (ej: 70%), el sistema fuerza
   automáticamente el estado de reposo ('STOP').
2. Suavizado Temporal por Consenso (Majority Voting / N-Frame Confirmation):
   Para cambiar a un nuevo comando de movimiento (ej: 'FORWARD'), el modelo debe
   predecir consistentemente dicho comando durante N fotogramas consecutivos
   (por defecto 3 fotogramas).
3. Botón de Parada de Emergencia (E-Stop):
   Prioridad absoluta que anula cualquier predicción y detiene los motores.
4. Abstracción del Hardware:
   Permite operar en modo simulación (SimuladorRobot) o mediante comunicación
   física Serial (ControladorSerialRobot para Arduino, ESP32, etc.).
"""

from abc import ABC, abstractmethod
from collections import deque
import logging
from typing import Deque, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
logger = logging.getLogger("RobotController")

# Lista oficial de comandos de movimiento
COMANDOS_VALIDOS = {"STOP", "FORWARD", "LEFT", "RIGHT", "BACKWARD"}


class FiltroSeguridad:
    """
    Filtro temporal y de confianza para proteger al robot contra decisiones erráticas.
    """

    def __init__(
        self,
        umbral_confianza: float = 0.70,
        fotogramas_confirmacion: int = 3,
    ) -> None:
        """
        Args:
            umbral_confianza: Probabilidad mínima para aceptar una predicción (por defecto 0.70).
            fotogramas_confirmacion: Fotogramas consecutivos necesarios para conmutar estado.
        """
        self.umbral_confianza = umbral_confianza
        self.fotogramas_confirmacion = fotogramas_confirmacion
        self.comando_actual = "STOP"
        self.historial_predicciones: Deque[str] = deque(maxlen=fotogramas_confirmacion)

    def filtrar(self, comando_candidato: str, confianza: float) -> Tuple[str, bool]:
        """
        Evalúa la predicción candidata frente a las reglas de seguridad.

        Returns:
            Tuple:
                - comando_seguro: Comando aprobado para enviar al robot.
                - cambio_estado: True si el comando activo cambió con respecto al anterior.
        """
        comando_candidato = comando_candidato.upper()
        if comando_candidato not in COMANDOS_VALIDOS:
            comando_candidato = "STOP"

        # 1. Regla de confianza mínima
        if confianza < self.umbral_confianza and comando_candidato != "STOP":
            # Si no hay suficiente confianza, forzamos STOP preventivo en el buffer
            self.historial_predicciones.append("STOP")
        else:
            self.historial_predicciones.append(comando_candidato)

        # 2. Regla de consenso temporal (todos los últimos N fotogramas deben coincidir)
        if (
            len(self.historial_predicciones) == self.fotogramas_confirmacion
            and len(set(self.historial_predicciones)) == 1
        ):
            nuevo_comando = self.historial_predicciones[0]
            cambio = nuevo_comando != self.comando_actual
            self.comando_actual = nuevo_comando
            return self.comando_actual, cambio

        # Mantener el comando previo seguro si aún no hay consenso
        return self.comando_actual, False

    def parada_inmediata(self) -> None:
        """Restablece inmediatamente el estado a STOP y limpia el historial."""
        self.comando_actual = "STOP"
        self.historial_predicciones.clear()


class ControladorRobotBase(ABC):
    """Interfaz abstracta para desacoplar el protocolo físico de la lógica de decisión."""

    @abstractmethod
    def conectar(self) -> bool:
        """Establece la conexión con el hardware del robot."""
        pass

    @abstractmethod
    def enviar_comando(self, comando: str) -> bool:
        """Transmite un comando validado al robot."""
        pass

    @abstractmethod
    def parada_emergencia(self) -> None:
        """Detiene de inmediato todos los actuadores del robot."""
        pass

    @abstractmethod
    def desconectar(self) -> None:
        """Cierra el puerto o comunicación de forma segura."""
        pass


class SimuladorRobot(ControladorRobotBase):
    """
    Implementación virtual para pruebas en consola, simulaciones y sustentaciones
    cuando no se dispone del robot físico conectado.
    """

    def __init__(self) -> None:
        self.conectado = False
        self.ultimo_comando = "STOP"

    def conectar(self) -> bool:
        self.conectado = True
        logger.info("[SIMULADOR] Robot virtual conectado exitosamente.")
        return True

    def enviar_comando(self, comando: str) -> bool:
        if not self.conectado:
            logger.warning("[SIMULADOR] Robot no conectado. Comando rechazado.")
            return False

        comando = comando.upper()
        if comando not in COMANDOS_VALIDOS:
            comando = "STOP"

        if comando != self.ultimo_comando:
            logger.info(f"[SIMULADOR] Transicion de movimiento: {self.ultimo_comando} -> {comando}")
            self.ultimo_comando = comando
        return True

    def parada_emergencia(self) -> None:
        self.ultimo_comando = "STOP"
        logger.warning("[SIMULADOR] *** PARADA DE EMERGENCIA ACTIVADA *** Motores detenidos.")

    def desconectar(self) -> None:
        self.parada_emergencia()
        self.conectado = False
        logger.info("[SIMULADOR] Desconectado.")


class ControladorSerialRobot(ControladorRobotBase):
    """
    Controlador para transmisión serie (UART / USB) hacia microcontroladores
    como Arduino, ESP32 o Raspberry Pi Pico.
    """

    def __init__(self, puerto: str = "COM3", baudios: int = 115200) -> None:
        self.puerto = puerto
        self.baudios = baudios
        self.conexion_serial = None

    def conectar(self) -> bool:
        try:
            import serial
            self.conexion_serial = serial.Serial(self.puerto, self.baudios, timeout=1)
            logger.info(f"[SERIAL] Conectado a {self.puerto} a {self.baudios} baudios.")
            return True
        except ImportError:
            logger.error("[SERIAL] Paquete 'pyserial' no instalado. Usa: pip install pyserial")
            return False
        except Exception as e:
            logger.error(f"[SERIAL] Error al abrir puerto {self.puerto}: {e}")
            return False

    def enviar_comando(self, comando: str) -> bool:
        if self.conexion_serial is None or not self.conexion_serial.is_open:
            return False

        comando_str = f"{comando.upper()}\n"
        try:
            self.conexion_serial.write(comando_str.encode("ascii"))
            return True
        except Exception as e:
            logger.error(f"[SERIAL] Error de envio: {e}")
            return False

    def parada_emergencia(self) -> None:
        self.enviar_comando("STOP")
        logger.warning("[SERIAL] Parada de emergencia enviada al microcontrolador.")

    def desconectar(self) -> None:
        self.parada_emergencia()
        if self.conexion_serial and self.conexion_serial.is_open:
            self.conexion_serial.close()
        logger.info("[SERIAL] Puerto cerrado.")
