"""
Pruebas unitarias para el módulo de preprocesamiento de imágenes (src/data/preprocess.py).
Verifica dimensiones, normalización, codificación de etiquetas, partición estratificada
y manejo de errores.
"""

import numpy as np
import pytest
from PIL import Image
from pathlib import Path

from src.data.preprocess import (
    LABEL_MAP,
    INV_LABEL_MAP,
    IMG_FEATURES,
    procesar_imagen,
    division_estratificada,
)


def test_procesar_imagen_dimensiones_y_rango():
    """Verifica que procesar_imagen devuelva un vector de 784 floats en [0.0, 1.0]."""
    # Crear una imagen sintética en memoria de 300x200 RGB
    img_sintetica = Image.new("RGB", (300, 200), color=(128, 200, 50))
    
    vector = procesar_imagen(img_sintetica)
    
    assert isinstance(vector, np.ndarray), "La salida debe ser un arreglo NumPy"
    assert vector.shape == (IMG_FEATURES,), f"La forma debe ser ({IMG_FEATURES},), pero fue {vector.shape}"
    assert vector.dtype == np.float32, f"El tipo debe ser float32, pero fue {vector.dtype}"
    assert vector.min() >= 0.0, "Los valores no deben ser menores a 0.0"
    assert vector.max() <= 1.0, "Los valores no deben ser mayores a 1.0"


def test_procesar_imagen_valores_extremos():
    """Verifica la normalización con imágenes totalmente negras y totalmente blancas."""
    negra = Image.new("L", (100, 100), color=0)
    blanca = Image.new("L", (100, 100), color=255)

    vec_negra = procesar_imagen(negra)
    vec_blanca = procesar_imagen(blanca)

    assert np.allclose(vec_negra, 0.0), "La imagen negra debe ser un vector de ceros"
    assert np.allclose(vec_blanca, 1.0), "La imagen blanca debe ser un vector de unos"


def test_procesar_imagen_tipo_invalido():
    """Verifica que procesar_imagen lance TypeError si se le pasa un objeto no soportado."""
    with pytest.raises(TypeError):
        procesar_imagen(12345)


def test_mapeo_de_etiquetas():
    """Verifica la consistencia del mapeo de clases y su inversa."""
    clases_esperadas = {"rock", "paper", "scissors"}
    assert set(LABEL_MAP.keys()) == clases_esperadas
    assert set(LABEL_MAP.values()) == {0, 1, 2}
    
    for k, v in LABEL_MAP.items():
        assert INV_LABEL_MAP[v] == k


def test_division_estratificada_sin_solapamiento():
    """Verifica que la división estratificada preserve proporciones y no tenga solapamientos."""
    # Simular 300 muestras: 100 por cada clase
    n_muestras = 300
    X_mock = np.random.rand(n_muestras, IMG_FEATURES).astype(np.float32)
    y_mock = np.array([0] * 100 + [1] * 100 + [2] * 100, dtype=np.int64)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = division_estratificada(
        X_mock, y_mock, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    # 1. Dimensiones de X
    assert X_train.shape[1] == IMG_FEATURES
    assert X_val.shape[1] == IMG_FEATURES
    assert X_test.shape[1] == IMG_FEATURES

    # 2. Suma de muestras
    assert len(X_train) + len(X_val) + len(X_test) == n_muestras
    assert len(y_train) + len(y_val) + len(y_test) == n_muestras

    # 3. Tamaños esperados por partición: 70% de 300 = 210, 15% = 45, 15% = 45
    assert len(y_train) == 210
    assert len(y_val) == 45
    assert len(y_test) == 45

    # 4. Proporción exacta por clase dentro de cada partición:
    # Train: 70 de cada clase
    assert np.sum(y_train == 0) == 70
    assert np.sum(y_train == 1) == 70
    assert np.sum(y_train == 2) == 70

    # Val: 15 de cada clase
    assert np.sum(y_val == 0) == 15
    assert np.sum(y_val == 1) == 15
    assert np.sum(y_val == 2) == 15

    # Test: 15 de cada clase
    assert np.sum(y_test == 0) == 15
    assert np.sum(y_test == 1) == 15
    assert np.sum(y_test == 2) == 15


def test_division_estratificada_reproducibilidad():
    """Verifica que con la misma semilla se obtengan exactamente las mismas particiones."""
    X_mock = np.random.rand(90, IMG_FEATURES).astype(np.float32)
    y_mock = np.array([0] * 30 + [1] * 30 + [2] * 30, dtype=np.int64)

    res1 = division_estratificada(X_mock, y_mock, seed=123)
    res2 = division_estratificada(X_mock, y_mock, seed=123)

    assert np.allclose(res1[0][0], res2[0][0]), "X_train debe ser idéntico con la misma semilla"
    assert np.array_equal(res1[0][1], res2[0][1]), "y_train debe ser idéntico con la misma semilla"
