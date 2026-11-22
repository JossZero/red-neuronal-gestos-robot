"""
Módulo de Demostración en Tiempo Real con Cámara Web (Fase 9).

Este script permite interactuar con la red neuronal entrenada desde cero
utilizando el flujo en vivo de una cámara web conectada al equipo.

OPTIMIZACIONES PARA LOGITECH C920 / C922 Y CÁMARAS USB:
1. Backend DirectShow (cv2.CAP_DSHOW) en Windows para evitar desbordamientos de MSMF.
2. Compresión por hardware MJPG para no saturar el bus USB.
3. Modo de resolución ligera (424x240 o 640x360 en formato nativo 16:9):
   Elimina por completo el parpadeo, congelamiento y pantallas negras típicas
   de la C920 en resoluciones 4:3 altas.
4. Ventana escalable (848x480): La cámara captura ligero y fluido mientras la
   pantalla muestra una ventana cómoda y nítida.
5. Inferencia 100% matemática desde cero: La región de interés (ROI) se redimensiona
   a 28x28 píxeles en escala de grises y se evalúa con nuestra clase RedNeuronal (NumPy).
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.model.neural_network import RedNeuronal
from src.robot_controller import FiltroSeguridad, SimuladorRobot, ControladorSerialRobot

# Mapeo oficial de clases y traducción académica
CLASES = ["rock", "paper", "scissors"]
NOMBRES_ESPANOL = {
    "rock": "Piedra",
    "paper": "Papel",
    "scissors": "Tijera",
}
COLORES_BGR = {
    "rock": (60, 180, 75),       # Verde
    "paper": (230, 190, 50),     # Amarillo / Ocre
    "scissors": (70, 70, 240),   # Rojo / Naranja
}

# Mapeo de gestos a comandos del carrito robótico
MAPEO_RPS_ROBOT = {
    "rock": "STOP",         # Puño cerrado -> Detenerse
    "paper": "FORWARD",     # Palma abierta -> Avanzar
    "scissors": "RIGHT",    # Dos dedos -> Girar
}

RESOLUCIONES = {
    "baja": (424, 240),      # Ultra-fluido, recomendado para Logitech C920/C922
    "media": (640, 360),     # Formato 16:9 intermedio
    "estandar": (640, 480),  # Formato 4:3 clásico
}


def preprocesar_cuadro_roi(roi_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocesa la región de interés (ROI) extraída del fotograma:
    1. Conversión de BGR a escala de grises.
    2. Redimensionamiento bilineal a 28x28 píxeles.
    3. Normalización al rango [0.0, 1.0] con float32.
    4. Aplanamiento a vector de 784 características.

    Args:
        roi_bgr: Imagen BGR de la zona de la mano de forma (H, W, 3).

    Returns:
        Tuple:
            - vector_784: Vector aplanado (1, 784) listo para la red neuronal.
            - img_28x28_gris: Arreglo (28, 28) uint8 para visualización en pantalla.
    """
    gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    img_28x28_gris = cv2.resize(gris, (28, 28), interpolation=cv2.INTER_LINEAR)
    vector_784 = (img_28x28_gris.astype(np.float32) / 255.0).reshape(1, 784)
    return vector_784, img_28x28_gris


def dibujar_interfaz(
    fotograma: np.ndarray,
    roi_coords: Tuple[int, int, int, int],
    clase_predicha: str,
    confianza: float,
    probabilidades: Dict[str, float],
    img_28x28_gris: np.ndarray,
    fps: float,
    estado_robot: Optional[str] = None,
) -> np.ndarray:
    """
    Dibuja la interfaz visual HUD adaptada dinámicamente a cualquier resolución.
    """
    alto_f, ancho_f = fotograma.shape[:2]
    x1, y1, x2, y2 = roi_coords
    color_clase = COLORES_BGR.get(clase_predicha, (0, 255, 0))

    # 1. Dibujar recuadro ROI donde colocar la mano
    cv2.rectangle(fotograma, (x1, y1), (x2, y2), color_clase, 2)
    cv2.putText(
        fotograma,
        "Mano aqui",
        (x1, max(15, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42 if alto_f < 300 else 0.55,
        color_clase,
        1,
        cv2.LINE_AA,
    )

    # 2. Panel superior semi-transparente para métricas (adaptado al tamaño)
    w_panel = min(320, int(ancho_f * 0.50))
    h_panel = min(180, int(alto_f * 0.88))
    overlay = fotograma.copy()
    cv2.rectangle(overlay, (6, 6), (w_panel, h_panel), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, fotograma, 0.25, 0, fotograma)

    f_title = 0.48 if alto_f < 300 else 0.65
    f_body = 0.38 if alto_f < 300 else 0.46
    line_sp = 20 if alto_f < 300 else 24

    # 3. Textos informativos de predicción
    nombre_es = NOMBRES_ESPANOL.get(clase_predicha, clase_predicha)
    cv2.putText(
        fotograma,
        f"{clase_predicha.upper()} ({nombre_es})",
        (12, 28 if alto_f < 300 else 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        f_title,
        color_clase,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        fotograma,
        f"Conf: {confianza * 100:.1f}% | FPS: {fps:.1f}",
        (12, 48 if alto_f < 300 else 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        f_body,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # 4. Barras de probabilidad por cada clase
    y_offset = 68 if alto_f < 300 else 88
    for clase in CLASES:
        prob = probabilidades.get(clase, 0.0)
        texto_prob = f"{clase.capitalize():<8}: {prob * 100:4.1f}%"
        cv2.putText(
            fotograma,
            texto_prob,
            (12, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            f_body,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Barra visual
        x_bar = int(w_panel * 0.48)
        ancho_max = int(w_panel * 0.45)
        ancho_actual = int(prob * ancho_max)
        h_bar = 8 if alto_f < 300 else 10
        cv2.rectangle(fotograma, (x_bar, y_offset - h_bar), (x_bar + ancho_max, y_offset), (70, 70, 70), 1)
        cv2.rectangle(
            fotograma,
            (x_bar, y_offset - h_bar),
            (x_bar + ancho_actual, y_offset),
            COLORES_BGR[clase],
            -1,
        )
        y_offset += line_sp

    # 5. Instrucciones de salida
    cv2.putText(
        fotograma,
        "'q': salir | 's': foto",
        (12, h_panel - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        f_body,
        (160, 160, 160),
        1,
        cv2.LINE_AA,
    )

    # 6. Miniatura 28x28 ampliada en esquina superior derecha
    tam_mini = 54 if alto_f < 300 else 74
    miniatura_color = cv2.cvtColor(
        cv2.resize(img_28x28_gris, (tam_mini, tam_mini), interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )
    x_mini = ancho_f - tam_mini - 10
    y_mini = 10
    fotograma[y_mini:y_mini + tam_mini, x_mini:x_mini + tam_mini] = miniatura_color
    cv2.rectangle(fotograma, (x_mini, y_mini), (x_mini + tam_mini, y_mini + tam_mini), (255, 255, 255), 1)

    # 7. Indicador del Robot y Filtro de Seguridad
    if estado_robot:
        color_cmd = (0, 255, 0) if estado_robot == "FORWARD" else (0, 0, 255) if estado_robot == "STOP" else (255, 165, 0)
        h_tag = 26 if alto_f < 300 else 32
        w_tag = 200 if alto_f < 300 else 240
        y_tag = alto_f - h_tag - 8
        cv2.rectangle(fotograma, (8, y_tag), (8 + w_tag, y_tag + h_tag), (25, 25, 25), -1)
        cv2.rectangle(fotograma, (8, y_tag), (8 + w_tag, y_tag + h_tag), color_cmd, 1)
        cv2.putText(
            fotograma,
            f"ROBOT: {estado_robot}",
            (16, y_tag + int(h_tag * 0.72)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45 if alto_f < 300 else 0.58,
            color_cmd,
            2,
            cv2.LINE_AA,
        )

    return fotograma


def detectar_camaras_disponibles() -> List[Tuple[int, str]]:
    """
    Detecta los nombres amigables de las cámaras sin abrir/cerrar dispositivos repetidamente.
    """
    camaras = []
    if sys.platform.startswith("win"):
        try:
            res = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "Get-PnpDevice -Class Camera,Image | Select-Object -ExpandProperty FriendlyName"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            nombres = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
            for idx, nom in enumerate(nombres):
                camaras.append((idx, nom))
            if camaras:
                return camaras
        except Exception:
            pass

    # Fallback básico si PowerShell no responde
    return [(0, "Camara #0"), (1, "Camara #1")]


def seleccionar_camara_optima(id_camara_solicitado: Optional[int] = None) -> Tuple[int, str]:
    """
    Selecciona la cámara física (priorizando Logitech / USB física sobre Iriun o virtuales).
    """
    camaras = detectar_camaras_disponibles()
    if not camaras:
        return (0 if id_camara_solicitado is None else id_camara_solicitado, "Camara por defecto")

    if id_camara_solicitado is not None:
        nombre_sel = next((nom for idx, nom in camaras if idx == id_camara_solicitado), f"Camara #{id_camara_solicitado}")
        return id_camara_solicitado, nombre_sel

    # Priorizar cámara física no virtual
    virtuales = ["iriun", "obs", "droidcam", "virtual", "snap camera"]
    for idx, nombre in camaras:
        nombre_lower = nombre.lower()
        if not any(v in nombre_lower for v in virtuales):
            return idx, nombre

    return camaras[0]


def ejecutar_demostracion_webcam(
    id_camara: int = 1,
    ruta_modelo: Path = Path("models/rps_model.npz"),
    carpeta_capturas: Path = Path("outputs/captures"),
    resolucion: str = "baja",
    puerto_robot: Optional[str] = None,
    baudios: int = 115200,
    simular_robot: bool = False,
    umbral_confianza: float = 0.70,
    consenso_frames: int = 3,
) -> None:
    """
    Inicia el bucle principal de adquisición y clasificación en tiempo real,
    con integración opcional de control seguro para robot / microcontrolador ESP32.
    """
    if not ruta_modelo.exists():
        print(f"[ERROR] No se encontro el modelo entrenado en: {ruta_modelo}")
        print("Por favor, ejecuta primero: python -m src.train")
        return

    # Cargar red neuronal implementada desde cero con NumPy
    print(f"[INFO] Cargando modelo neuronal desde {ruta_modelo}...")
    red = RedNeuronal()
    red.cargar_modelo(ruta_modelo)

    # Configuración del Filtro de Seguridad y Control Robótico
    filtro = FiltroSeguridad(
        umbral_confianza=umbral_confianza,
        fotogramas_confirmacion=consenso_frames,
    )

    controlador = None
    if puerto_robot:
        print(f"[INFO] Iniciando conexion serial con microcontrolador en {puerto_robot} ({baudios} baudios)...")
        controlador = ControladorSerialRobot(puerto=puerto_robot, baudios=baudios)
        if not controlador.conectar():
            print(f"[ADVERTENCIA] No fue posible abrir el puerto serial {puerto_robot}.")
            print("El sistema continuara en modo visual sin envio fisico.")
            controlador = None
        else:
            print("[INFO] ESP32 / Microcontrolador conectado y listo.")
    elif simular_robot:
        print("[INFO] Modo simulador de robot activado.")
        controlador = SimuladorRobot()
        controlador.conectar()

    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    print(f"[INFO] Conectando a camara {id_camara} con backend DirectShow...")
    captura = cv2.VideoCapture(id_camara, backend)

    if not captura.isOpened() and backend == cv2.CAP_DSHOW:
        print("[INFO] Reintentando con backend estandar...")
        captura = cv2.VideoCapture(id_camara)

    if not captura.isOpened():
        print("=" * 70)
        print("[ADVERTENCIA] No fue posible acceder a la camara web.")
        print("Verifica que no este siendo utilizada por otra aplicacion.")
        print("=" * 70)
        if controlador:
            controlador.desconectar()
        return

    # Configuración de resolución óptima (424x240 por defecto para máxima fluidez y estabilidad)
    ancho_deseado, alto_deseado = RESOLUCIONES.get(resolucion, (424, 240))
    captura.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, ancho_deseado)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, alto_deseado)
    captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Desactivar autoenfoque continuo para evitar oscurecimientos o recalibración
    try:
        captura.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    except Exception:
        pass

    ancho_marco = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto_marco = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if ancho_marco <= 0 or alto_marco <= 0:
        ancho_marco, alto_marco = ancho_deseado, alto_deseado

    print(f"[INFO] Resolucion activa de la camara: {ancho_marco}x{alto_marco} (Modo: {resolucion.upper()})")

    # Coordenadas proporcionales del recuadro ROI
    tam_caja = int(min(alto_marco * 0.72, ancho_marco * 0.40))
    x1 = int(ancho_marco * 0.55)
    y1 = int((alto_marco - tam_caja) / 2)
    x2 = x1 + tam_caja
    y2 = y1 + tam_caja
    roi_coords = (x1, y1, x2, y2)

    carpeta_capturas.mkdir(parents=True, exist_ok=True)

    # Crear ventana escalable para verla en pantalla en tamaño cómodo (848x480)
    nombre_ventana = "Gesture Robot AI - Demostracion Webcam"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, 848, 480)

    print("=" * 70)
    print(">>> DEMOSTRACION EN TIEMPO REAL INICIADA")
    print("Controles:")
    print("  - Presiona 'q' o 'ESC' para cerrar la ventana.")
    print("  - Presiona 's' para guardar una captura del fotograma actual.")
    print("  - Presiona 'e' o 'ESPACIO' para parada de emergencia.")
    if controlador:
        print("  - Transmitiendo comandos validados al actuador/robot.")
    print("=" * 70)

    tiempo_anterior = time.time()
    ultimo_envio_serial = 0.0
    contador_capturas = 0
    fallos_consecutivos = 0
    estado_robot_activo = "STOP"

    # Descartar los primeros 3 fotogramas de calentamiento del sensor
    for _ in range(3):
        captura.read()

    try:
        while True:
            ret, fotograma = captura.read()
            if not ret or fotograma is None:
                fallos_consecutivos += 1
                if fallos_consecutivos > 30:
                    print("[ERROR] Se perdio la senal de la camara.")
                    break
                time.sleep(0.01)
                continue

            fallos_consecutivos = 0

            # Espejo horizontal natural
            fotograma = cv2.flip(fotograma, 1)

            # Extraer ROI
            roi = fotograma[y1:y2, x1:x2]
            if roi.shape[0] == 0 or roi.shape[1] == 0:
                continue

            # 1. Preprocesar exactamente igual que en el entrenamiento (28x28)
            vector_784, img_28x28_gris = preprocesar_cuadro_roi(roi)

            # 2. Inferencia con la red neuronal NumPy
            probs = red.predict_proba(vector_784)[0]
            idx_max = int(np.argmax(probs))
            clase_pred = CLASES[idx_max]
            conf = float(probs[idx_max])

            probabilidades = {CLASES[i]: float(probs[i]) for i in range(len(CLASES))}

            # 3. Filtrado de seguridad y consenso temporal
            comando_candidato = MAPEO_RPS_ROBOT.get(clase_pred, "STOP")
            comando_seguro, cambio_estado = filtro.filtrar(comando_candidato, conf)
            estado_robot_activo = comando_seguro

            # 4. Transmisión al ESP32 / Robot
            # Se envía inmediatamente si cambia el comando o periódicamente (latido cada 250ms)
            # para no disparar el watchdog timeout del microcontrolador
            tiempo_actual = time.time()
            if controlador is not None:
                debe_enviar = cambio_estado or (tiempo_actual - ultimo_envio_serial >= 0.25)
                if debe_enviar:
                    controlador.enviar_comando(comando_seguro)
                    ultimo_envio_serial = tiempo_actual

            # 5. Medición de FPS
            fps = 1.0 / max(tiempo_actual - tiempo_anterior, 1e-5)
            tiempo_anterior = tiempo_actual

            # 6. Dibujar interfaz HUD
            fotograma_salida = dibujar_interfaz(
                fotograma=fotograma,
                roi_coords=roi_coords,
                clase_predicha=clase_pred,
                confianza=conf,
                probabilidades=probabilidades,
                img_28x28_gris=img_28x28_gris,
                fps=fps,
                estado_robot=estado_robot_activo,
            )

            cv2.imshow(nombre_ventana, fotograma_salida)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla in [ord("q"), ord("Q"), 27]:
                print("[INFO] Saliendo de la demostracion...")
                break
            elif tecla in [ord("s"), ord("S")]:
                contador_capturas += 1
                ruta_guardado = carpeta_capturas / f"captura_webcam_{contador_capturas}.png"
                cv2.imwrite(str(ruta_guardado), fotograma_salida)
                print(f"[INFO] Fotograma guardado en: {ruta_guardado}")
            elif tecla in [ord("e"), ord("E"), ord(" ")]:
                filtro.parada_inmediata()
                if controlador:
                    controlador.parada_emergencia()
                print("[SEGURIDAD] Parada de emergencia manual activada por el usuario.")

    finally:
        if controlador:
            controlador.desconectar()
        captura.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demostración de Clasificación de Gestos en Tiempo Real con Webcam y Control Robótico."
    )
    parser.add_argument(
        "--camara",
        type=int,
        default=None,
        help="Índice de la cámara web (detecta automáticamente la Logitech o cámara física).",
    )
    parser.add_argument(
        "--resolucion",
        type=str,
        default="baja",
        choices=["baja", "media", "estandar"],
        help="Resolución de captura: 'baja' (424x240, recomendada para Logitech C920), 'media' (640x360), 'estandar' (640x480).",
    )
    parser.add_argument(
        "--listar-camaras",
        action="store_true",
        help="Lista todas las cámaras detectadas en el sistema y finaliza.",
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default="models/rps_model.npz",
        help="Ruta al modelo entrenado (.npz).",
    )
    parser.add_argument(
        "--capturas",
        type=str,
        default="outputs/captures",
        help="Carpeta donde guardar capturas con tecla 's'.",
    )
    parser.add_argument(
        "--puerto-robot",
        type=str,
        default=None,
        help="Puerto serial COM del ESP32 / Arduino (ej: COM3, COM4). Si se omite, opera sin hardware físico.",
    )
    parser.add_argument(
        "--baudios",
        type=int,
        default=115200,
        help="Velocidad en baudios para la comunicación serial (por defecto: 115200).",
    )
    parser.add_argument(
        "--simular-robot",
        action="store_true",
        help="Activa el simulador de robot por consola si no hay microcontrolador físico conectado.",
    )
    parser.add_argument(
        "--umbral-confianza",
        type=float,
        default=0.70,
        help="Confianza mínima requerida (0.0 a 1.0) para validar un comando gestual (por defecto: 0.70).",
    )
    parser.add_argument(
        "--consenso-frames",
        type=int,
        default=3,
        help="Cantidad de fotogramas consecutivos requeridos para confirmar una transición de comando (por defecto: 3).",
    )

    args = parser.parse_args()

    if args.listar_camaras:
        print("=" * 65)
        print(">>> CAMARAS DETECTADAS EN EL SISTEMA")
        print("=" * 65)
        camaras = detectar_camaras_disponibles()
        for idx, nom in camaras:
            print(f"  [{idx}] {nom}")
        print("=" * 65)
        return

    idx_cam, nombre_cam = seleccionar_camara_optima(args.camara)
    print(f"[INFO] Cámara seleccionada: Dispositivo {idx_cam} -> '{nombre_cam}'")

    ejecutar_demostracion_webcam(
        id_camara=idx_cam,
        ruta_modelo=Path(args.modelo),
        carpeta_capturas=Path(args.capturas),
        resolucion=args.resolucion,
        puerto_robot=args.puerto_robot,
        baudios=args.baudios,
        simular_robot=args.simular_robot,
        umbral_confianza=args.umbral_confianza,
        consenso_frames=args.consenso_frames,
    )


if __name__ == "__main__":
    main()
