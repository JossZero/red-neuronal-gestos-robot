"""
Herramienta de Captura y Creación de Dataset Propio para Comandos del Robot (Fase 10).

Permite recopilar de manera guiada y organizada las imágenes para los 5 comandos de movimiento:
1. STOP (Detenerse - ej: palma abierta frontal fija)
2. FORWARD (Avanzar - ej: dedo índice apuntando arriba o pulgar arriba)
3. LEFT (Girar izquierda - ej: mano señalando a la izquierda)
4. RIGHT (Girar derecha - ej: mano señalando a la derecha)
5. BACKWARD (Retroceder - ej: puño cerrado o pulgar abajo)

Flujo del script:
- Crea automáticamente la estructura en data/robot_raw/<clase>/.
- Muestra el visor de la cámara con la zona de captura (ROI).
- Permite cambiar de comando en vivo con las teclas 1 a 5.
- Permite capturar imágenes individuales (Barra espaciadora) o modo ráfaga temporizada ('c').
"""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.webcam_demo import seleccionar_camara_optima

COMANDOS_ROBOT = {
    "1": "stop",
    "2": "forward",
    "3": "left",
    "4": "right",
    "5": "backward",
}

DESCRIPCIONES = {
    "stop": "Palma abierta de frente",
    "forward": "Indice apuntando arriba",
    "left": "Mano senalando izquierda",
    "right": "Mano senalando derecha",
    "backward": "Puno o pulgar abajo",
}


def capturar_dataset(
    id_camara: int = 0,
    directorio_base: Path = Path("data/robot_raw"),
) -> None:
    """Ejecuta la interfaz de captura para recopilar el dataset de comandos."""
    for clase in COMANDOS_ROBOT.values():
        (directorio_base / clase).mkdir(parents=True, exist_ok=True)

    import sys
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    captura = cv2.VideoCapture(id_camara, backend)
    if not captura.isOpened() and backend == cv2.CAP_DSHOW:
        captura = cv2.VideoCapture(id_camara)

    if not captura.isOpened():
        print(f"[ERROR] No se pudo abrir la camara {id_camara}.")
        return

    captura.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    captura.set(cv2.CAP_PROP_FRAME_WIDTH, 424)
    captura.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if ancho <= 0 or alto <= 0:
        ancho, alto = 424, 240

    tam_caja = int(min(alto * 0.72, ancho * 0.40))
    x1 = int(ancho * 0.55)
    y1 = int((alto - tam_caja) / 2)
    x2 = x1 + tam_caja
    y2 = y1 + tam_caja

    comando_actual = "stop"
    modo_rafaga = False
    ultimo_tiempo_rafaga = 0.0

    print("=" * 70)
    print(">>> ASISTENTE DE CAPTURA DE DATASET PARA CONTROL DEL ROBOT")
    print("Comandos disponibles:")
    for k, v in COMANDOS_ROBOT.items():
        print(f"  [{k}] -> {v.upper()} ({DESCRIPCIONES[v]})")
    print("\nControles de captura:")
    print("  - [ESPACIO]: Capturar 1 imagen.")
    print("  - [c]: Activar / Desactivar modo rafaga (1 imagen cada 250 ms).")
    print("  - [q] o [ESC]: Finalizar captura.")
    print("=" * 70)

    nombre_ventana = "Captura de Gestos para Robot"
    cv2.namedWindow(nombre_ventana, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(nombre_ventana, 848, 480)

    try:
        while True:
            ret, fotograma = captura.read()
            if not ret:
                break

            fotograma = cv2.flip(fotograma, 1)
            roi = fotograma[y1:y2, x1:x2]

            conteo_actual = len(list((directorio_base / comando_actual).glob("*.png")))

            # Dibujar recuadro de captura
            cv2.rectangle(fotograma, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                fotograma,
                "Zona de Captura",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
            )

            # Panel informativo
            cv2.putText(
                fotograma,
                f"Comando Activo: {comando_actual.upper()}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                fotograma,
                f"Gesto sugerido: {DESCRIPCIONES[comando_actual]}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
            )
            cv2.putText(
                fotograma,
                f"Imagenes guardadas de '{comando_actual}': {conteo_actual}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (100, 255, 100),
                2,
            )

            estado_rafaga = "ACTIVO" if modo_rafaga else "INACTIVO"
            cv2.putText(
                fotograma,
                f"Modo Rafaga [c]: {estado_rafaga}",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255) if modo_rafaga else (180, 180, 180),
                1,
            )
            cv2.putText(
                fotograma,
                "Teclas: 1=STOP, 2=FWD, 3=LEFT, 4=RIGHT, 5=BACK | ESPACIO=Foto | q=Salir",
                (20, alto - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (180, 180, 180),
                1,
            )

            # Lógica de ráfaga
            ahora = time.time()
            if modo_rafaga and (ahora - ultimo_tiempo_rafaga) >= 0.25:
                nombre_archivo = f"{comando_actual}_{int(ahora * 1000)}.png"
                ruta_salida = directorio_base / comando_actual / nombre_archivo
                cv2.imwrite(str(ruta_salida), roi)
                ultimo_tiempo_rafaga = ahora

            cv2.imshow("Captura de Gestos para Robot", fotograma)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla in [ord("q"), ord("Q"), 27]:
                break
            elif chr(tecla) in COMANDOS_ROBOT:
                comando_actual = COMANDOS_ROBOT[chr(tecla)]
                print(f"[CAMBIO] Comando activo cambiado a: {comando_actual.upper()}")
            elif tecla == 32:  # Barra espaciadora
                nombre_archivo = f"{comando_actual}_{int(time.time() * 1000)}.png"
                ruta_salida = directorio_base / comando_actual / nombre_archivo
                cv2.imwrite(str(ruta_salida), roi)
                print(f"[GUARDADO] Imagen {nombre_archivo} en {comando_actual}/")
            elif tecla in [ord("c"), ord("C")]:
                modo_rafaga = not modo_rafaga
                print(f"[RAFAGA] Modo rafaga: {'ACTIVADO' if modo_rafaga else 'DESACTIVADO'}")

    finally:
        captura.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Asistente de Captura de Dataset para Robot.")
    parser.add_argument("--camara", type=int, default=None, help="ID de la cámara web (detecta automáticamente la cámara física).")
    parser.add_argument("--salida", type=str, default="data/robot_raw", help="Directorio destino.")
    args = parser.parse_args()

    idx_cam, nombre_cam = seleccionar_camara_optima(args.camara)
    print(f"[INFO] Cámara seleccionada: Dispositivo {idx_cam} -> '{nombre_cam}'")

    capturar_dataset(id_camara=idx_cam, directorio_base=Path(args.salida))


if __name__ == "__main__":
    main()
