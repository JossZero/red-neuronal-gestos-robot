"""
Pruebas unitarias para la red neuronal implementada desde cero con NumPy (src/model/neural_network.py).
Incluye verificación dimensional, estabilidad de softmax, conservación de probabilidades,
actualización de pesos, persistencia y gradiente numérico (Gradient Checking).
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

from src.model.neural_network import RedNeuronal


def test_dimensiones_inicializacion():
    """Verifica las formas de las matrices de pesos y vectores de sesgo."""
    red = RedNeuronal(n_entrada=784, n_oculta=32, n_salida=3, semilla=42)
    
    assert red.W1.shape == (784, 32), f"W1 debe ser (784, 32), fue {red.W1.shape}"
    assert red.b1.shape == (1, 32), f"b1 debe ser (1, 32), fue {red.b1.shape}"
    assert red.W2.shape == (32, 3), f"W2 debe ser (32, 3), fue {red.W2.shape}"
    assert red.b2.shape == (1, 3), f"b2 debe ser (1, 3), fue {red.b2.shape}"
    assert red.W1.dtype == np.float32
    assert red.W2.dtype == np.float32


def test_estabilidad_y_suma_softmax():
    """Verifica que Softmax no desborde con valores extremos y que las filas sumen 1.0."""
    Z_extremo = np.array([
        [1000.0, 1001.0, 999.0],      # Valores muy altos (provocarían overflow sin truco del max)
        [-500.0, -500.0, -500.0],     # Valores negativos
        [0.0, 0.0, 0.0],              # Ceros
    ], dtype=np.float32)

    A = RedNeuronal.softmax(Z_extremo)

    assert not np.isnan(A).any(), "Softmax no debe producir NaN"
    assert not np.isinf(A).any(), "Softmax no debe producir Inf"
    sumas = np.sum(A, axis=1)
    assert np.allclose(sumas, 1.0, atol=1e-5), f"Las probabilidades deben sumar 1.0, suman {sumas}"


def test_forward_y_perdida():
    """Verifica que el paso hacia adelante produzca probabilidades válidas y pérdida finita."""
    red = RedNeuronal(n_entrada=784, n_oculta=32, n_salida=3, semilla=42)
    m = 10
    X_mock = np.random.rand(m, 784).astype(np.float32)
    y_mock = np.random.randint(0, 3, size=(m,), dtype=np.int64)

    A2, cache = red.forward(X_mock)

    assert A2.shape == (m, 3)
    assert np.all(A2 >= 0.0) and np.all(A2 <= 1.0)
    
    perdida = red.calcular_perdida(A2, y_mock)
    assert np.isfinite(perdida), "La pérdida debe ser un número real finito"
    assert perdida > 0.0, "La pérdida debe ser positiva"


def test_paso_entrenamiento_modifica_pesos():
    """Verifica que un paso de descenso de gradiente altere los pesos en la dirección de aprendizaje."""
    red = RedNeuronal(n_entrada=784, n_oculta=32, n_salida=3, semilla=42)
    X_lote = np.random.rand(8, 784).astype(np.float32)
    y_lote = np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64)

    W1_inicial = red.W1.copy()
    W2_inicial = red.W2.copy()
    b1_inicial = red.b1.copy()
    b2_inicial = red.b2.copy()

    perdida = red.paso_entrenamiento(X_lote, y_lote, tasa_aprendizaje=0.1)

    assert not np.allclose(red.W1, W1_inicial), "W1 debe modificarse tras el entrenamiento"
    assert not np.allclose(red.W2, W2_inicial), "W2 debe modificarse tras el entrenamiento"
    assert not np.allclose(red.b1, b1_inicial), "b1 debe modificarse tras el entrenamiento"
    assert not np.allclose(red.b2, b2_inicial), "b2 debe modificarse tras el entrenamiento"
    assert np.isfinite(perdida)


def test_guardar_y_cargar_modelo():
    """Verifica que guardar y recargar el modelo preserve exactamente los parámetros y predicciones."""
    red = RedNeuronal(n_entrada=784, n_oculta=32, n_salida=3, semilla=42)
    X_prueba = np.random.rand(5, 784).astype(np.float32)
    probs_originales = red.predict_proba(X_prueba)

    with tempfile.TemporaryDirectory() as tmpdir:
        ruta_guardado = Path(tmpdir) / "modelo_prueba.npz"
        red.guardar_modelo(ruta_guardado)
        assert ruta_guardado.exists()

        red_recargada = RedNeuronal(n_entrada=784, n_oculta=32, n_salida=3)
        red_recargada.cargar_modelo(ruta_guardado)

        assert np.allclose(red.W1, red_recargada.W1)
        assert np.allclose(red.b1, red_recargada.b1)
        assert np.allclose(red.W2, red_recargada.W2)
        assert np.allclose(red.b2, red_recargada.b2)

        probs_recargadas = red_recargada.predict_proba(X_prueba)
        assert np.allclose(probs_originales, probs_recargadas, atol=1e-6)


def test_gradiente_numerico_gradient_checking():
    """
    Verificación de Gradiente (Gradient Checking) mediante diferencias finitas:
    Compara el gradiente analítico dW2 con la aproximación numérica:
    (L(W + eps) - L(W - eps)) / (2 * eps)
    Demuestra matemáticamente que backpropagation está implementado correctamente.
    """
    # Usar red pequeña y float64 para gradient checking de alta precisión
    n_in = 8
    n_hid = 4
    n_out = 3
    red = RedNeuronal(n_entrada=n_in, n_oculta=n_hid, n_salida=n_out, semilla=123)

    # Convertir a float64 temporalmente para eliminar el error por cancelación de float32
    red.W1 = red.W1.astype(np.float64)
    red.b1 = red.b1.astype(np.float64)
    red.W2 = red.W2.astype(np.float64)
    red.b2 = red.b2.astype(np.float64)

    X = np.random.randn(5, n_in).astype(np.float64)
    y = np.array([0, 1, 2, 1, 0], dtype=np.int64)

    # 1. Gradiente analítico
    A2, cache = red.forward(X)
    grads = red.backward(y, cache)
    dW2_analitico = grads["dW2"]
    dW1_analitico = grads["dW1"]

    # 2. Gradiente numérico para W2
    epsilon = 1e-7
    dW2_numerico = np.zeros_like(red.W2)
    for i in range(red.W2.shape[0]):
        for j in range(red.W2.shape[1]):
            red.W2[i, j] += epsilon
            A2_mas, _ = red.forward(X)
            loss_mas = red.calcular_perdida(A2_mas, y)

            red.W2[i, j] -= 2 * epsilon
            A2_menos, _ = red.forward(X)
            loss_menos = red.calcular_perdida(A2_menos, y)

            red.W2[i, j] += epsilon
            dW2_numerico[i, j] = (loss_mas - loss_menos) / (2.0 * epsilon)

    # 3. Gradiente numérico para W1
    dW1_numerico = np.zeros_like(red.W1)
    for i in range(red.W1.shape[0]):
        for j in range(red.W1.shape[1]):
            red.W1[i, j] += epsilon
            A2_mas, _ = red.forward(X)
            loss_mas = red.calcular_perdida(A2_mas, y)

            red.W1[i, j] -= 2 * epsilon
            A2_menos, _ = red.forward(X)
            loss_menos = red.calcular_perdida(A2_menos, y)

            red.W1[i, j] += epsilon
            dW1_numerico[i, j] = (loss_mas - loss_menos) / (2.0 * epsilon)

    # Error relativo: ||d_analitico - d_numerico|| / (||d_analitico|| + ||d_numerico||)
    err_W2 = np.linalg.norm(dW2_analitico - dW2_numerico) / (np.linalg.norm(dW2_analitico) + np.linalg.norm(dW2_numerico))
    err_W1 = np.linalg.norm(dW1_analitico - dW1_numerico) / (np.linalg.norm(dW1_analitico) + np.linalg.norm(dW1_numerico))

    assert err_W2 < 1e-6, f"Error relativo en dW2: {err_W2:.6e}"
    assert err_W1 < 1e-6, f"Error relativo en dW1: {err_W1:.6e}"
