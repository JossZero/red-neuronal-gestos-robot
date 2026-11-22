"""
Módulo de Entrenamiento para la Red Neuronal de Reconocimiento de Gestos (Fase 6).

Este script implementa el bucle de optimización por Descenso de Gradiente
por Mini-Lotes (Mini-Batch Gradient Descent) implementado puramente con NumPy.

Flujo del proceso:
1. Carga de particiones preprocesadas (Train y Val) desde 'data/processed/rps_dataset_processed.npz'.
2. Inicialización del modelo 'RedNeuronal' (784 -> 32 -> 3).
3. En cada época:
   a. Mezcla pseudoaleatoria del conjunto de entrenamiento con semilla reproducible.
   b. Iteración por mini-lotes (Forward, Pérdida, Backward, Actualización).
   c. Evaluación de métricas globales (pérdida y exactitud) en Train y Val.
   d. Registro histórico de métricas.
   e. Punto de control (checkpoint): guardado del mejor modelo según menor pérdida de validación.
4. Exportación del historial a 'outputs/logs/training_history.csv'.
5. Generación de curvas de aprendizaje en 'outputs/figures/training_curves.png'.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
# Configurar backend no interactivo antes de importar pyplot para ejecución limpia
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model.neural_network import RedNeuronal


def cargar_datos(ruta_npz: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga las particiones de entrenamiento y validación desde el archivo NPZ.

    Args:
        ruta_npz: Ruta al archivo 'rps_dataset_processed.npz'.

    Returns:
        Tuple: (X_train, y_train, X_val, y_val)
    """
    if not ruta_npz.exists():
        raise FileNotFoundError(f"No se encontró el dataset preprocesado en {ruta_npz}")

    with np.load(ruta_npz) as datos:
        X_train = datos["X_train"].astype(np.float32)
        y_train = datos["y_train"].astype(np.int64)
        X_val = datos["X_val"].astype(np.float32)
        y_val = datos["y_val"].astype(np.int64)

    return X_train, y_train, X_val, y_val


def entrenar(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epocas: int = 50,
    batch_size: int = 32,
    lr: float = 0.03,
    semilla: int = 42,
    ruta_modelo_salida: Path = Path("models/rps_model.npz"),
) -> Tuple[RedNeuronal, List[Dict[str, float]], int]:
    """
    Ejecuta el ciclo de entrenamiento por mini-lotes y guarda el mejor modelo.

    Args:
        X_train: Características de entrenamiento (m, 784).
        y_train: Etiquetas de entrenamiento (m,).
        X_val: Características de validación (m_val, 784).
        y_val: Etiquetas de validación (m_val,).
        epocas: Cantidad total de iteraciones sobre todo el dataset.
        batch_size: Tamaño de cada lote.
        lr: Tasa de aprendizaje (learning rate).
        semilla: Semilla para reproducibilidad.
        ruta_modelo_salida: Destino donde se guardarán los mejores pesos.

    Returns:
        Tuple: (red_entrenada, historial_metricas, mejor_epoca)
    """
    red = RedNeuronal(n_entrada=784, n_oculta=32, n_salida=3, semilla=semilla)
    n_muestras = len(X_train)

    historial: List[Dict[str, float]] = []
    mejor_val_loss = float("inf")
    mejor_epoca = 1

    # Parámetros óptimos en memoria para restaurar al final
    mejores_pesos = {
        "W1": red.W1.copy(),
        "b1": red.b1.copy(),
        "W2": red.W2.copy(),
        "b2": red.b2.copy(),
    }

    print("=" * 75)
    print(">>> INICIANDO ENTRENAMIENTO DE LA RED NEURONAL DESDE CERO (NumPy)")
    print("=" * 75)
    print(f"* Epocas:             {epocas}")
    print(f"* Tamano de lote:     {batch_size}")
    print(f"* Tasa aprendizaje:   {lr}")
    print(f"* Semilla:            {semilla}")
    print(f"* Muestras Train:     {n_muestras}")
    print(f"* Muestras Val:       {len(X_val)}")
    print("-" * 75)
    print(f"{'Epoca':^7} | {'Train Loss':^12} | {'Train Acc':^11} | {'Val Loss':^12} | {'Val Acc':^11} | {'Estado':^10}")
    print("-" * 75)

    for epoca in range(1, epocas + 1):
        # 1. Barajar el conjunto de entrenamiento en cada época
        rng = np.random.RandomState(semilla + epoca)
        indices = rng.permutation(n_muestras)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        # 2. Iterar por mini-lotes
        for inicio in range(0, n_muestras, batch_size):
            fin = inicio + batch_size
            X_lote = X_shuffled[inicio:fin]
            y_lote = y_shuffled[inicio:fin]
            red.paso_entrenamiento(X_lote, y_lote, tasa_aprendizaje=lr)

        # 3. Métricas completas al cierre de la época
        probs_train, _ = red.forward(X_train)
        train_loss = red.calcular_perdida(probs_train, y_train)
        train_acc = red.calcular_accuracy(X_train, y_train)

        probs_val, _ = red.forward(X_val)
        val_loss = red.calcular_perdida(probs_val, y_val)
        val_acc = red.calcular_accuracy(X_val, y_val)

        # 4. Checkpoint del mejor modelo según val_loss
        es_mejor = val_loss < mejor_val_loss
        estado = "Mejorando" if es_mejor else ""
        if es_mejor:
            mejor_val_loss = val_loss
            mejor_epoca = epoca
            mejores_pesos["W1"] = red.W1.copy()
            mejores_pesos["b1"] = red.b1.copy()
            mejores_pesos["W2"] = red.W2.copy()
            mejores_pesos["b2"] = red.b2.copy()
            estado = "[OPTIMO]"

        registro = {
            "epoca": epoca,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        historial.append(registro)

        if epoca == 1 or epoca % 5 == 0 or es_mejor or epoca == epocas:
            print(
                f"{epoca:^7d} | {train_loss:^12.4f} | {train_acc * 100:^10.2f}% | "
                f"{val_loss:^12.4f} | {val_acc * 100:^10.2f}% | {estado:^10}"
            )

    print("-" * 75)
    print(f"[OK] Entrenamiento completado. Mejor epoca: {mejor_epoca} (Val Loss: {mejor_val_loss:.4f})")

    # Restaurar los mejores pesos en el objeto de la red y persistirlos en disco
    red.W1 = mejores_pesos["W1"]
    red.b1 = mejores_pesos["b1"]
    red.W2 = mejores_pesos["W2"]
    red.b2 = mejores_pesos["b2"]

    ruta_modelo_salida.parent.mkdir(parents=True, exist_ok=True)
    red.guardar_modelo(ruta_modelo_salida)
    print(f"[SAVED] Modelo optimo guardado en: {ruta_modelo_salida}")

    return red, historial, mejor_epoca


def guardar_historial_csv(historial: List[Dict[str, float]], ruta_csv: Path) -> None:
    """
    Exporta el historial de entrenamiento a un archivo CSV.

    Args:
        historial: Lista de diccionarios con métricas por época.
        ruta_csv: Destino del archivo CSV.
    """
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    campos = ["epoca", "train_loss", "train_acc", "val_loss", "val_acc"]
    with open(ruta_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(historial)
    print(f"[LOG] Historial de entrenamiento registrado en: {ruta_csv}")


def graficar_curvas_entrenamiento(
    historial: List[Dict[str, float]],
    mejor_epoca: int,
    ruta_figura: Path,
) -> None:
    """
    Genera y guarda una gráfica comparativa con las curvas de Pérdida y Exactitud.

    Args:
        historial: Lista de diccionarios con métricas.
        mejor_epoca: Número de época donde se logró el mejor val_loss.
        ruta_figura: Destino donde se guardará la imagen PNG.
    """
    ruta_figura.parent.mkdir(parents=True, exist_ok=True)

    epocas = [h["epoca"] for h in historial]
    train_loss = [h["train_loss"] for h in historial]
    val_loss = [h["val_loss"] for h in historial]
    train_acc = [h["train_acc"] * 100 for h in historial]
    val_acc = [h["val_acc"] * 100 for h in historial]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Gráfica de Pérdida
    ax_loss.plot(epocas, train_loss, label="Entrenamiento", color="#1f77b4", linewidth=2)
    ax_loss.plot(epocas, val_loss, label="Validación", color="#ff7f0e", linewidth=2)
    ax_loss.axvline(
        x=mejor_epoca,
        color="#2ca02c",
        linestyle="--",
        alpha=0.7,
        label=f"Mejor Época ({mejor_epoca})",
    )
    ax_loss.set_title("Curva de Pérdida (Cross-Entropy Loss)", fontsize=13, fontweight="bold")
    ax_loss.set_xlabel("Época", fontsize=11)
    ax_loss.set_ylabel("Pérdida", fontsize=11)
    ax_loss.grid(True, linestyle=":", alpha=0.6)
    ax_loss.legend(loc="upper right", frameon=True)

    # 2. Gráfica de Exactitud (Accuracy)
    ax_acc.plot(epocas, train_acc, label="Entrenamiento", color="#1f77b4", linewidth=2)
    ax_acc.plot(epocas, val_acc, label="Validación", color="#ff7f0e", linewidth=2)
    ax_acc.axvline(
        x=mejor_epoca,
        color="#2ca02c",
        linestyle="--",
        alpha=0.7,
        label=f"Mejor Época ({mejor_epoca})",
    )
    ax_acc.set_title("Curva de Exactitud (Accuracy %)", fontsize=13, fontweight="bold")
    ax_acc.set_xlabel("Época", fontsize=11)
    ax_acc.set_ylabel("Exactitud (%)", fontsize=11)
    ax_acc.grid(True, linestyle=":", alpha=0.6)
    ax_acc.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    fig.savefig(ruta_figura, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Grafica de curvas de aprendizaje guardada en: {ruta_figura}")


def main() -> None:
    """Función de entrada para ejecución desde línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Entrenamiento de la Red Neuronal desde Cero con NumPy."
    )
    parser.add_argument(
        "--datos",
        type=str,
        default="data/processed/rps_dataset_processed.npz",
        help="Ruta al archivo NPZ con datos preprocesados.",
    )
    parser.add_argument(
        "--epocas",
        type=int,
        default=50,
        help="Número total de épocas de entrenamiento (default: 50).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño del mini-lote (default: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.03,
        help="Tasa de aprendizaje (default: 0.03).",
    )
    parser.add_argument(
        "--semilla",
        type=int,
        default=42,
        help="Semilla aleatoria reproducible (default: 42).",
    )
    parser.add_argument(
        "--modelo-salida",
        type=str,
        default="models/rps_model.npz",
        help="Ruta donde se guardará el modelo entrenado.",
    )
    parser.add_argument(
        "--logs-salida",
        type=str,
        default="outputs/logs/training_history.csv",
        help="Ruta del archivo CSV para el historial.",
    )
    parser.add_argument(
        "--grafica-salida",
        type=str,
        default="outputs/figures/training_curves.png",
        help="Ruta del archivo PNG para la gráfica de curvas.",
    )

    args = parser.parse_args()

    ruta_datos = Path(args.datos)
    ruta_modelo = Path(args.modelo_salida)
    ruta_logs = Path(args.logs_salida)
    ruta_grafica = Path(args.grafica_salida)

    # 1. Cargar datos
    X_train, y_train, X_val, y_val = cargar_datos(ruta_datos)

    # 2. Entrenar red
    red, historial, mejor_epoca = entrenar(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epocas=args.epocas,
        batch_size=args.batch_size,
        lr=args.lr,
        semilla=args.semilla,
        ruta_modelo_salida=ruta_modelo,
    )

    # 3. Guardar historial y gráficas
    guardar_historial_csv(historial, ruta_logs)
    graficar_curvas_entrenamiento(historial, mejor_epoca, ruta_grafica)


if __name__ == "__main__":
    main()
