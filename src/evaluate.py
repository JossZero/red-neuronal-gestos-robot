"""
Módulo de Evaluación de la Red Neuronal en el Conjunto de Prueba (Fase 7).

Este script evalúa el modelo entrenado sobre las 329 muestras independientes de test,
las cuales NO fueron vistas durante el entrenamiento ni la validación.

Métricas implementadas desde cero con NumPy (sin scikit-learn):
- Exactitud Global (Accuracy).
- Pérdida de Entropía Cruzada en Test.
- Matriz de Confusión 3x3.
- Precisión (Precision), Sensibilidad (Recall) y Puntuación F1 por clase.
- Promedios Macro y Ponderado.
- Detección y visualización de aciertos vs confusiones típicas.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.model.neural_network import RedNeuronal

# Mapeo oficial de etiquetas
CLASES = ["rock", "paper", "scissors"]
NOMBRES_ESPANOL = ["Piedra", "Papel", "Tijera"]


def cargar_datos_prueba(ruta_npz: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Carga X_test e y_test desde el archivo NPZ."""
    if not ruta_npz.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {ruta_npz}")

    with np.load(ruta_npz) as datos:
        X_test = datos["X_test"].astype(np.float32)
        y_test = datos["y_test"].astype(np.int64)

    return X_test, y_test


def calcular_matriz_confusion(y_real: np.ndarray, y_pred: np.ndarray, n_clases: int = 3) -> np.ndarray:
    """
    Calcula la matriz de confusión C de tamaño (n_clases, n_clases) con NumPy.
    C[i, j] representa las muestras con clase real i que fueron clasificadas como j.
    """
    matriz = np.zeros((n_clases, n_clases), dtype=np.int64)
    for real, pred in zip(y_real, y_pred):
        matriz[real, pred] += 1
    return matriz


def calcular_metricas_por_clase(matriz_conf: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calcula Precision, Recall y F1-score por clase y promedios a partir de la matriz de confusión.
    """
    n_clases = matriz_conf.shape[0]
    total_muestras = np.sum(matriz_conf)
    soporte_por_clase = np.sum(matriz_conf, axis=1)

    metricas: Dict[str, Dict[str, float]] = {}

    precisiones = []
    recalls = []
    f1s = []

    for c in range(n_clases):
        tp = float(matriz_conf[c, c])
        fp = float(np.sum(matriz_conf[:, c]) - tp)
        fn = float(np.sum(matriz_conf[c, :]) - tp)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisiones.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        nombre_clase = f"{CLASES[c]} ({NOMBRES_ESPANOL[c]})"
        metricas[nombre_clase] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "soporte": int(soporte_por_clase[c]),
        }

    # Promedio Macro (no ponderado)
    metricas["Promedio Macro"] = {
        "precision": float(np.mean(precisiones)),
        "recall": float(np.mean(recalls)),
        "f1_score": float(np.mean(f1s)),
        "soporte": int(total_muestras),
    }

    # Promedio Ponderado por soporte
    pesos = soporte_por_clase / total_muestras
    metricas["Promedio Ponderado"] = {
        "precision": float(np.sum(np.array(precisiones) * pesos)),
        "recall": float(np.sum(np.array(recalls) * pesos)),
        "f1_score": float(np.sum(np.array(f1s) * pesos)),
        "soporte": int(total_muestras),
    }

    return metricas


def graficar_matriz_confusion(matriz_conf: np.ndarray, ruta_salida: Path) -> None:
    """Genera y guarda una gráfica de calor de la matriz de confusión con porcentajes y conteos."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cax = ax.imshow(matriz_conf, cmap="Blues", interpolation="nearest")

    cbar = fig.colorbar(cax, fraction=0.046, pad=0.04)
    cbar.set_label("Cantidad de Muestras", fontsize=11)

    etiquetas = [f"{c.capitalize()}\n({es})" for c, es in zip(CLASES, NOMBRES_ESPANOL)]
    ax.set_xticks(np.arange(len(etiquetas)))
    ax.set_yticks(np.arange(len(etiquetas)))
    ax.set_xticklabels(etiquetas, fontsize=10)
    ax.set_yticklabels(etiquetas, fontsize=10)

    # Imprimir conteos y porcentajes dentro de cada celda
    total_por_fila = np.sum(matriz_conf, axis=1, keepdims=True)
    umbral = matriz_conf.max() / 2.0

    for i in range(len(etiquetas)):
        for j in range(len(etiquetas)):
            conteo = matriz_conf[i, j]
            porcentaje = (conteo / total_por_fila[i, 0]) * 100
            color = "white" if conteo > umbral else "black"
            texto = f"{conteo}\n({porcentaje:.1f}%)"
            ax.text(j, i, texto, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    ax.set_title("Matriz de Confusión en Conjunto de Prueba (Test Set)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Predicción del Modelo", fontsize=11, fontweight="bold")
    ax.set_ylabel("Clase Verdadera", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Matriz de confusion guardada en: {ruta_salida}")


def graficar_ejemplos_prediccion(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    ruta_salida: Path,
) -> None:
    """
    Muestra una cuadrícula de 8 ejemplos (4 aciertos y 4 errores)
    reconstruyendo la imagen 28x28 para inspección visual cualitativa.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    indices_aciertos = np.where(y_test == y_pred)[0]
    indices_errores = np.where(y_test != y_pred)[0]

    rng = np.random.RandomState(42)
    muestras_aciertos = rng.choice(indices_aciertos, size=min(4, len(indices_aciertos)), replace=False)
    muestras_errores = rng.choice(indices_errores, size=min(4, len(indices_errores)), replace=False)

    indices_mostrar = list(muestras_aciertos) + list(muestras_errores)

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=300)
    axes = axes.flatten()

    for idx_panel, idx_muestra in enumerate(indices_mostrar):
        ax = axes[idx_panel]
        img = X_test[idx_muestra].reshape(28, 28)
        real_idx = y_test[idx_muestra]
        pred_idx = y_pred[idx_muestra]
        confianza = probs[idx_muestra, pred_idx] * 100

        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.axis("off")

        es_acierto = (real_idx == pred_idx)
        color_borde = "#2ca02c" if es_acierto else "#d62728"
        estado = "CORRECTO" if es_acierto else "ERROR"

        titulo = (
            f"[{estado}]\n"
            f"Real: {CLASES[real_idx]} ({NOMBRES_ESPANOL[real_idx]})\n"
            f"Pred: {CLASES[pred_idx]} ({confianza:.1f}%)"
        )
        ax.set_title(titulo, fontsize=10, color=color_borde, fontweight="bold")

    plt.suptitle("Muestras Cualitativas de Test: Aciertos (arriba) y Confusiones (abajo)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=300)
    plt.close(fig)
    print(f"[PLOT] Mosaico de ejemplos de test guardado en: {ruta_salida}")


def evaluar(
    ruta_datos: Path = Path("data/processed/rps_dataset_processed.npz"),
    ruta_modelo: Path = Path("models/rps_model.npz"),
    ruta_matriz_figura: Path = Path("outputs/figures/confusion_matrix.png"),
    ruta_ejemplos_figura: Path = Path("outputs/figures/evaluation_samples.png"),
    ruta_reporte: Path = Path("outputs/logs/evaluation_report.json"),
) -> Dict[str, float]:
    """
    Ejecuta el protocolo completo de evaluación sobre el conjunto de test.
    """
    print("=" * 75)
    print(">>> EVALUACION DEL MODELO EN CONJUNTO DE PRUEBA INDEPENDIENTE")
    print("=" * 75)

    # 1. Cargar datos y modelo
    X_test, y_test = cargar_datos_prueba(ruta_datos)
    red = RedNeuronal()
    red.cargar_modelo(ruta_modelo)

    n_test = len(y_test)
    probs = red.predict_proba(X_test)
    y_pred = red.predict(X_test)

    # 2. Métricas globales
    loss_test = red.calcular_perdida(probs, y_test)
    acc_test = red.calcular_accuracy(X_test, y_test)
    total_aciertos = int(np.sum(y_pred == y_test))

    print(f"* Total imagenes en Test: {n_test}")
    print(f"* Aciertos totales:       {total_aciertos} / {n_test}")
    print(f"* Exactitud Global (Acc): {acc_test * 100:.2f}%")
    print(f"* Perdida Cross-Entropy:  {loss_test:.4f}")
    print("-" * 75)

    # 3. Matriz de Confusión
    matriz_conf = calcular_matriz_confusion(y_test, y_pred, n_clases=3)
    print("MATRIZ DE CONFUSION (Filas: Real, Columnas: Prediccion):")
    print(f"{'':>12} | {'rock':^10} | {'paper':^10} | {'scissors':^10} | {'Total':^8}")
    print("-" * 62)
    for i, nombre in enumerate(CLASES):
        fila = matriz_conf[i]
        total_fila = np.sum(fila)
        print(f"{nombre:>12} | {fila[0]:^10} | {fila[1]:^10} | {fila[2]:^10} | {total_fila:^8}")
    print("-" * 62)

    # 4. Métricas por clase
    metricas = calcular_metricas_por_clase(matriz_conf)
    print("\nMETRICAS DETALLADAS POR CLASE:")
    print(f"{'Clase':^24} | {'Precision':^11} | {'Recall':^11} | {'F1-Score':^11} | {'Soporte':^8}")
    print("-" * 75)
    for nombre, valores in metricas.items():
        print(
            f"{nombre:<24} | {valores['precision'] * 100:^10.2f}% | "
            f"{valores['recall'] * 100:^10.2f}% | {valores['f1_score'] * 100:^10.2f}% | "
            f"{valores['soporte']:^8d}"
        )
    print("-" * 75)

    # 5. Generar figuras
    graficar_matriz_confusion(matriz_conf, ruta_matriz_figura)
    graficar_ejemplos_prediccion(X_test, y_test, y_pred, probs, ruta_ejemplos_figura)

    # 6. Guardar reporte JSON
    reporte = {
        "dataset_test_samples": n_test,
        "total_aciertos": total_aciertos,
        "test_accuracy": float(acc_test),
        "test_loss": float(loss_test),
        "matriz_confusion": matriz_conf.tolist(),
        "metricas_por_clase": metricas,
    }
    ruta_reporte.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_reporte, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=4, ensure_ascii=False)
    print(f"[REPORT] Reporte completo exportado a: {ruta_reporte}")

    return {
        "accuracy": float(acc_test),
        "loss": float(loss_test),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación de la Red Neuronal en Test.")
    parser.add_argument("--datos", type=str, default="data/processed/rps_dataset_processed.npz")
    parser.add_argument("--modelo", type=str, default="models/rps_model.npz")
    parser.add_argument("--matriz-figura", type=str, default="outputs/figures/confusion_matrix.png")
    parser.add_argument("--ejemplos-figura", type=str, default="outputs/figures/evaluation_samples.png")
    parser.add_argument("--reporte", type=str, default="outputs/logs/evaluation_report.json")
    args = parser.parse_args()

    evaluar(
        ruta_datos=Path(args.datos),
        ruta_modelo=Path(args.modelo),
        ruta_matriz_figura=Path(args.matriz_figura),
        ruta_ejemplos_figura=Path(args.ejemplos_figura),
        ruta_reporte=Path(args.reporte),
    )


if __name__ == "__main__":
    main()
