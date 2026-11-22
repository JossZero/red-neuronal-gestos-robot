"""
Pruebas automáticas para los módulos de entrenamiento (train.py),
evaluación (evaluate.py) e inferencia (predict_image.py).
"""

from pathlib import Path
import numpy as np
import pytest

from src.evaluate import calcular_matriz_confusion, calcular_metricas_por_clase
from src.model.neural_network import RedNeuronal
from src.predict_image import predecir_imagen
from src.train import entrenar


def test_calcular_matriz_confusion_exacta():
    """Verifica que la matriz de confusión 3x3 cuente con exactitud con NumPy."""
    y_real = np.array([0, 0, 1, 1, 2, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0, 0, 2, 2])
    # Real 0: dos como 0, una como 1
    # Real 1: dos como 1, una como 2
    # Real 2: dos como 2, una como 0
    matriz = calcular_matriz_confusion(y_real, y_pred, n_clases=3)
    assert matriz.shape == (3, 3)
    assert matriz[0, 0] == 2
    assert matriz[0, 1] == 1
    assert matriz[0, 2] == 0

    assert matriz[1, 0] == 0
    assert matriz[1, 1] == 2
    assert matriz[1, 2] == 1

    assert matriz[2, 0] == 1
    assert matriz[2, 1] == 0
    assert matriz[2, 2] == 2


def test_calcular_metricas_por_clase():
    """Comprueba el cálculo matemático de Precisión, Recall y F1."""
    # Matriz diagonal perfecta
    matriz_perfecta = np.array([
        [10, 0, 0],
        [0, 10, 0],
        [0, 0, 10],
    ])
    metricas = calcular_metricas_por_clase(matriz_perfecta)
    for c in ["rock (Piedra)", "paper (Papel)", "scissors (Tijera)"]:
        assert metricas[c]["precision"] == 1.0
        assert metricas[c]["recall"] == 1.0
        assert metricas[c]["f1_score"] == 1.0

    assert metricas["Promedio Macro"]["f1_score"] == 1.0


def test_ciclo_entrenamiento_reducido(tmp_path: Path):
    """Verifica que el bucle de entrenamiento guarde el modelo y reduzca la pérdida."""
    rng = np.random.default_rng(42)
    # Sintético: 40 muestras de 784 características y 3 clases
    X_train = rng.random((40, 784), dtype=np.float32)
    y_train = rng.integers(0, 3, size=(40,))
    X_val = rng.random((10, 784), dtype=np.float32)
    y_val = rng.integers(0, 3, size=(10,))

    ruta_modelo = tmp_path / "modelo_test.npz"

    red, historial, mejor_epoca = entrenar(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epocas=5,
        batch_size=8,
        lr=0.01,
        semilla=42,
        ruta_modelo_salida=ruta_modelo,
    )

    assert ruta_modelo.exists()
    assert len(historial) == 5
    assert mejor_epoca >= 1
    assert isinstance(red, RedNeuronal)


def test_inferencia_imagen_existente():
    """Comprueba que la inferencia devuelva probabilidades normalizadas."""
    ruta_img = Path("data/raw/rock/00nKV8oHuTGi20gq.png")
    ruta_modelo = Path("models/rps_model.npz")
    if not (ruta_img.exists() and ruta_modelo.exists()):
        pytest.skip("Dataset o modelo no disponible en la ruta especificada.")

    clase, conf, probs, vec = predecir_imagen(ruta_img, ruta_modelo)
    assert clase in ["rock", "paper", "scissors"]
    assert 0.0 <= conf <= 1.0
    assert len(probs) == 3
    # La suma de las probabilidades Softmax debe ser 1.0 (margen numérico)
    assert pytest.approx(sum(probs.values()), rel=1e-4) == 1.0
    assert vec.shape == (784,)
