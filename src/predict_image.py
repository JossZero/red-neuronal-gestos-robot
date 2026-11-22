"""
Módulo de Inferencia sobre Imágenes Individuales (Fase 8).

Permite clasificar una imagen nueva o arbitraria usando el modelo entrenado desde cero:
python -m src.predict_image ruta/a/imagen.png

El flujo de ejecución:
1. Valida la existencia del archivo de imagen.
2. Aplica exactamente el mismo preprocesamiento que en el entrenamiento:
   - Lectura con Pillow.
   - Conversión a escala de grises ('L').
   - Redimensionamiento bilineal a 28x28 píxeles.
   - Normalización de píxeles a [0.0, 1.0].
   - Aplanado a vector 1D de 784 características.
3. Carga los pesos del modelo (models/rps_model.npz).
4. Ejecuta la propagación hacia adelante (Forward).
5. Desglosa las probabilidades Softmax de cada clase.
6. Emite advertencias técnicas sobre las limitaciones en condiciones reales.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.data.preprocess import procesar_imagen
from src.model.neural_network import RedNeuronal

# Mapeo de nombres en español
NOMBRES_ESPANOL = {
    "rock": "Piedra",
    "paper": "Papel",
    "scissors": "Tijera",
}

ETIQUETAS_CLASE = ["rock", "paper", "scissors"]


def predecir_imagen(
    ruta_imagen: Path,
    ruta_modelo: Path = Path("models/rps_model.npz"),
) -> Tuple[str, float, Dict[str, float], np.ndarray]:
    """
    Realiza la inferencia sobre un archivo de imagen.

    Args:
        ruta_imagen: Ruta al archivo JPG o PNG a clasificar.
        ruta_modelo: Ruta a los pesos guardados (.npz).

    Returns:
        Tuple:
            - clase_predicha: Nombre de la clase ganadora ('rock', 'paper', 'scissors').
            - confianza: Probabilidad más alta [0.0, 1.0].
            - probabilidades: Diccionario {clase: probabilidad}.
            - vector_784: Vector preprocesado de 784 elementos.
    """
    if not ruta_imagen.exists():
        raise FileNotFoundError(f"No se encontro el archivo de imagen: {ruta_imagen}")

    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No se encontro el modelo entrenado en: {ruta_modelo}")

    # 1. Preprocesar la imagen
    vector_784 = procesar_imagen(ruta_imagen)  # Forma: (784,)

    # 2. Cargar modelo e inferir
    red = RedNeuronal()
    red.cargar_modelo(ruta_modelo)

    probs = red.predict_proba(vector_784.reshape(1, -1))[0]  # Forma: (3,)
    idx_ganador = int(np.argmax(probs))
    clase_predicha = ETIQUETAS_CLASE[idx_ganador]
    confianza = float(probs[idx_ganador])

    probabilidades = {
        ETIQUETAS_CLASE[i]: float(probs[i]) for i in range(len(ETIQUETAS_CLASE))
    }

    return clase_predicha, confianza, probabilidades, vector_784


def generar_barra_ascii(probabilidad: float, longitud: int = 25) -> str:
    """Genera una barra visual de progreso ASCII segura para cualquier consola."""
    bloques = int(round(probabilidad * longitud))
    return "#" * bloques + "-" * (longitud - bloques)


def guardar_diagnostico_visual(
    ruta_imagen: Path,
    vector_784: np.ndarray,
    clase_predicha: str,
    confianza: float,
    probabilidades: Dict[str, float],
    ruta_salida: Path,
) -> None:
    """
    Guarda una figura comparativa con la imagen original, la versión 28x28 y el gráfico de barras.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    img_original = Image.open(ruta_imagen)
    img_28x28 = vector_784.reshape(28, 28)

    fig, (ax_orig, ax_proc, ax_bar) = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)

    # 1. Imagen original
    ax_orig.imshow(img_original)
    ax_orig.set_title(f"Original ({img_original.size[0]}x{img_original.size[1]})", fontsize=11, fontweight="bold")
    ax_orig.axis("off")

    # 2. Entrada preprocesada (28x28 en escala de grises)
    ax_proc.imshow(img_28x28, cmap="gray")
    ax_proc.set_title("Preprocesada (28x28 Gris)", fontsize=11, fontweight="bold")
    ax_proc.axis("off")

    # 3. Gráfico de barras de probabilidades
    nombres = [f"{c.capitalize()} ({NOMBRES_ESPANOL[c]})" for c in ETIQUETAS_CLASE]
    valores = [probabilidades[c] * 100 for c in ETIQUETAS_CLASE]
    colores = ["#2ca02c" if c == clase_predicha else "#1f77b4" for c in ETIQUETAS_CLASE]

    barras = ax_bar.barh(nombres, valores, color=colores, height=0.55)
    ax_bar.set_xlim(0, 105)
    ax_bar.set_xlabel("Probabilidad (%)", fontsize=10, fontweight="bold")
    ax_bar.set_title(f"Prediccion: {clase_predicha.upper()} ({confianza*100:.1f}%)", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="x", linestyle=":", alpha=0.6)

    # Etiquetas de porcentaje al final de cada barra
    for barra in barras:
        ancho = barra.get_width()
        ax_bar.text(
            ancho + 2,
            barra.get_y() + barra.get_height() / 2,
            f"{ancho:.1f}%",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=300)
    plt.close(fig)
    img_original.close()
    print(f"[PLOT] Diagnostico visual guardado en: {ruta_salida}")


def imprimir_resultado(
    ruta_imagen: Path,
    clase_predicha: str,
    confianza: float,
    probabilidades: Dict[str, float],
) -> None:
    """Imprime el reporte detallado en la terminal con formato académico."""
    nombre_es = NOMBRES_ESPANOL.get(clase_predicha, clase_predicha)
    print("\n" + "=" * 70)
    print(">>> RESULTADO DE INFERENCIA EN IMAGEN AISLADA")
    print("=" * 70)
    print(f"* Archivo analizado:    {ruta_imagen}")
    print(f"* Clase detectada:      {clase_predicha.upper()} ({nombre_es})")
    print(f"* Nivel de confianza:   {confianza * 100:.2f}%")
    print("-" * 70)
    print("DISTRIBUCION DE PROBABILIDADES SOFTMAX:")
    for clase in ETIQUETAS_CLASE:
        prob = probabilidades[clase]
        es = NOMBRES_ESPANOL[clase]
        marca = "[SELECCIONADA]" if clase == clase_predicha else "              "
        barra = generar_barra_ascii(prob, longitud=20)
        print(f"  {clase.capitalize():<8} ({es:<7}): {barra} {prob * 100:6.2f}%  {marca}")
    print("-" * 70)
    print("AVISO ACADEMICO Y TECNICO:")
    print("  Una prediccion correcta sobre una imagen estatica aislada valida el")
    print("  funcionamiento matematico del modelo. Sin embargo, no garantiza")
    print("  robustez ante fondos ruidosos, iluminacion cambiante o rotaciones")
    print("  en tiempo real sin segmentacion previa o tecnicas de aumento de datos.")
    print("=" * 70 + "\n")


def ejecutar_demo(ruta_modelo: Path) -> None:
    """Ejecuta una demostración automática tomando un ejemplo de cada clase del dataset crudo."""
    print("[INFO] No se especifico ninguna imagen. Ejecutando demostracion con muestras del dataset...")
    carpeta_raw = Path("data/raw")
    if not carpeta_raw.exists():
        print("[ERROR] Carpeta data/raw no encontrada para la demo.")
        return

    for clase in ETIQUETAS_CLASE:
        archivos = list((carpeta_raw / clase).glob("*.png"))
        if not archivos:
            continue
        muestra = archivos[0]
        clase_pred, conf, probs, vec = predecir_imagen(muestra, ruta_modelo)
        imprimir_resultado(muestra, clase_pred, conf, probs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inferencia de Reconocimiento de Gestos sobre una Imagen."
    )
    parser.add_argument(
        "imagen",
        type=str,
        nargs="?",
        default=None,
        help="Ruta al archivo de imagen JPG o PNG a clasificar.",
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default="models/rps_model.npz",
        help="Ruta al archivo de pesos del modelo.",
    )
    parser.add_argument(
        "--guardar-visualizacion",
        type=str,
        default="outputs/figures/prediction_demo.png",
        help="Ruta para guardar el diagnostico visual comparativo.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Ejecuta una clasificacion automatica sobre 3 ejemplos del dataset crudo.",
    )

    args = parser.parse_args()
    ruta_modelo = Path(args.modelo)

    if args.demo or args.imagen is None:
        ejecutar_demo(ruta_modelo)
        return

    ruta_img = Path(args.imagen)
    clase_pred, conf, probs, vec = predecir_imagen(ruta_img, ruta_modelo)
    imprimir_resultado(ruta_img, clase_pred, conf, probs)

    if args.guardar_visualizacion:
        guardar_diagnostico_visual(
            ruta_imagen=ruta_img,
            vector_784=vec,
            clase_predicha=clase_pred,
            confianza=conf,
            probabilidades=probs,
            ruta_salida=Path(args.guardar_visualizacion),
        )


if __name__ == "__main__":
    main()
