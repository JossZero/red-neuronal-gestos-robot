"""
Pruebas automáticas para el módulo de demostración por webcam (webcam_demo.py).
Verifica el procesamiento de cuadros ROI y la generación de la interfaz sin requerir hardware de cámara físico.
"""

from pathlib import Path
import sys

# Permitir ejecución directa del archivo sin requerir PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.webcam_demo import preprocesar_cuadro_roi, dibujar_interfaz, CLASES


def test_preprocesar_cuadro_roi_dimensiones():
    """Verifica que cualquier ROI arbitrario BGR se convierta en vector (1, 784) en [0.0, 1.0]."""
    # Imagen de prueba sintética BGR de 200x200
    roi_bgr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

    vector_784, img_28x28_gris = preprocesar_cuadro_roi(roi_bgr)

    assert vector_784.shape == (1, 784)
    assert vector_784.dtype == np.float32
    assert 0.0 <= vector_784.min() and vector_784.max() <= 1.0
    assert img_28x28_gris.shape == (28, 28)
    assert img_28x28_gris.dtype == np.uint8


def test_dibujar_interfaz_no_modifica_dimensiones():
    """Verifica que el dibujo del HUD mantenga las dimensiones originales del marco."""
    fotograma = np.zeros((480, 640, 3), dtype=np.uint8)
    roi_coords = (300, 100, 520, 320)
    img_28x28_gris = np.zeros((28, 28), dtype=np.uint8)
    probabilidades = {"rock": 0.8, "paper": 0.1, "scissors": 0.1}

    resultado = dibujar_interfaz(
        fotograma=fotograma,
        roi_coords=roi_coords,
        clase_predicha="rock",
        confianza=0.8,
        probabilidades=probabilidades,
        img_28x28_gris=img_28x28_gris,
        fps=30.0,
    )

    assert resultado.shape == (480, 640, 3)
    assert resultado.dtype == np.uint8


def test_dibujar_interfaz_con_estado_robot():
    """Verifica que el dibujo del HUD funcione correctamente al incluir la etiqueta del robot."""
    fotograma = np.zeros((480, 640, 3), dtype=np.uint8)
    roi_coords = (300, 100, 520, 320)
    img_28x28_gris = np.zeros((28, 28), dtype=np.uint8)
    probabilidades = {"rock": 0.1, "paper": 0.85, "scissors": 0.05}

    resultado = dibujar_interfaz(
        fotograma=fotograma,
        roi_coords=roi_coords,
        clase_predicha="paper",
        confianza=0.85,
        probabilidades=probabilidades,
        img_28x28_gris=img_28x28_gris,
        fps=28.5,
        estado_robot="FORWARD",
    )

    assert resultado.shape == (480, 640, 3)
    assert resultado.dtype == np.uint8



if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))

