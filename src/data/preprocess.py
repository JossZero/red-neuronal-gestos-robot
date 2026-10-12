"""
Módulo de preprocesamiento de imágenes para el proyecto Gesture Robot AI.
Implementa conversión a escala de grises, redimensionamiento a 28x28, normalización,
aplanado y partición estratificada sin bibliotecas de alto nivel.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
from PIL import Image

# Mapeo oficial de clases a etiquetas numéricas
LABEL_MAP: Dict[str, int] = {
    "rock": 0,
    "paper": 1,
    "scissors": 2,
}

# Mapeo inverso de etiquetas numéricas a nombres de clase
INV_LABEL_MAP: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

# Dimensiones estándar del proyecto
IMG_WIDTH = 28
IMG_HEIGHT = 28
IMG_FEATURES = IMG_WIDTH * IMG_HEIGHT  # 784


def procesar_imagen(fuente: Union[str, Path, Image.Image]) -> np.ndarray:
    """
    Preprocesa una imagen individual:
    1. Carga la imagen (si se proporciona una ruta) o la toma en memoria.
    2. Convierte al modo escala de grises ('L').
    3. Redimensiona a (28, 28) usando interpolación bilineal.
    4. Convierte a arreglo de NumPy con tipo float32.
    5. Normaliza los píxeles al rango [0.0, 1.0] dividiendo por 255.0.
    6. Aplana a un vector unidimensional de longitud 784.

    Args:
        fuente: Ruta al archivo de imagen o instancia de PIL.Image.

    Returns:
        np.ndarray: Vector 1D de forma (784,) con valores float32 en [0.0, 1.0].
    """
    cerrar_al_terminar = False
    if isinstance(fuente, (str, Path)):
        img = Image.open(fuente)
        cerrar_al_terminar = True
    elif isinstance(fuente, Image.Image):
        img = fuente
    else:
        raise TypeError(f"Tipo de fuente no soportado: {type(fuente)}")

    try:
        # Conversión a escala de grises
        img_gris = img.convert("L")
        
        # Redimensionamiento a 28x28 píxeles
        img_redimensionada = img_gris.resize((IMG_WIDTH, IMG_HEIGHT), Image.Resampling.BILINEAR)
        
        # Conversión a NumPy y normalización a [0.0, 1.0]
        arreglo = np.asarray(img_redimensionada, dtype=np.float32) / 255.0
        
        # Aplanado a vector 1D de 784 elementos
        vector_aplanado = arreglo.reshape(-1)
        
        return vector_aplanado
    finally:
        if cerrar_al_terminar:
            img.close()


def cargar_dataset_completo(directorio_raw: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Recorre los subdirectorios de clases en data/raw/, procesa cada imagen válida y
    retorna matrices completas de características y etiquetas.

    Args:
        directorio_raw: Ruta a la carpeta que contiene las carpetas 'rock', 'paper', 'scissors'.

    Returns:
        Tuple:
            - X: Matriz de características de forma (N, 784) tipo float32.
            - y: Vector de etiquetas enteras de forma (N,) tipo int64.
            - rutas_archivos: Lista de rutas de los archivos procesados.
    """
    directorio_raw = Path(directorio_raw)
    caracteristicas = []
    etiquetas = []
    rutas_exitosas = []
    archivos_fallidos = []

    for clase, etiqueta in LABEL_MAP.items():
        dir_clase = directorio_raw / clase
        if not dir_clase.is_dir():
            raise FileNotFoundError(f"No se encontró el directorio para la clase '{clase}' en {dir_clase}")

        archivos_imagenes = sorted([
            f for f in dir_clase.iterdir()
            if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ])

        for ruta_img in archivos_imagenes:
            try:
                vec = procesar_imagen(ruta_img)
                caracteristicas.append(vec)
                etiquetas.append(etiqueta)
                rutas_exitosas.append(str(ruta_img))
            except Exception as e:
                archivos_fallidos.append((ruta_img, str(e)))

    if archivos_fallidos:
        print(f"[ADVERTENCIA] Se ignoraron {len(archivos_fallidos)} imágenes que no pudieron ser leídas.")

    X = np.array(caracteristicas, dtype=np.float32)
    y = np.array(etiquetas, dtype=np.int64)

    return X, y, rutas_exitosas


def division_estratificada(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Divide el dataset en conjuntos de Entrenamiento, Validación y Prueba manteniendo
    la proporción de cada clase (estratificación) sin dependencias externas.

    Args:
        X: Matriz de datos de forma (N, 784).
        y: Vector de etiquetas de forma (N,).
        train_ratio: Proporción para entrenamiento (default: 0.70).
        val_ratio: Proporción para validación (default: 0.15).
        test_ratio: Proporción para prueba (default: 0.15).
        seed: Semilla para reproducibilidad.

    Returns:
        ((X_train, y_train), (X_val, y_val), (X_test, y_test))
    """
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Las proporciones deben sumar 1.0"

    rng = np.random.default_rng(seed)
    clases_unicas = np.unique(y)

    indices_train = []
    indices_val = []
    indices_test = []

    for clase in clases_unicas:
        indices_clase = np.where(y == clase)[0]
        rng.shuffle(indices_clase)

        n_total = len(indices_clase)
        n_train = int(np.round(n_total * train_ratio))
        n_val = int(np.round(n_total * val_ratio))
        # El resto va a test para asegurar que no se pierdan muestras por redondeo
        n_test = n_total - (n_train + n_val)

        idx_train_c = indices_clase[:n_train]
        idx_val_c = indices_clase[n_train:n_train + n_val]
        idx_test_c = indices_clase[n_train + n_val:]

        indices_train.extend(idx_train_c)
        indices_val.extend(idx_val_c)
        indices_test.extend(idx_test_c)

    # Convertir a arreglos NumPy
    idx_train = np.array(indices_train, dtype=int)
    idx_val = np.array(indices_val, dtype=int)
    idx_test = np.array(indices_test, dtype=int)

    # Validar que no exista ningún solapamiento entre particiones
    set_train = set(idx_train)
    set_val = set(idx_val)
    set_test = set(idx_test)
    assert len(set_train & set_val) == 0, "Solapamiento detectado entre Train y Val"
    assert len(set_train & set_test) == 0, "Solapamiento detectado entre Train y Test"
    assert len(set_val & set_test) == 0, "Solapamiento detectado entre Val y Test"
    assert len(set_train) + len(set_val) + len(set_test) == len(y), "La suma de particiones no coincide con el total"

    # Barajar internamente cada partición para que las clases queden mezcladas
    rng.shuffle(idx_train)
    rng.shuffle(idx_val)
    rng.shuffle(idx_test)

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def guardar_datos_procesados(
    ruta_archivo: Union[str, Path],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Guarda los arreglos procesados de forma segura en formato comprimido .npz."""
    ruta_archivo = Path(ruta_archivo)
    ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ruta_archivo,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )


def cargar_datos_procesados(ruta_archivo: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Carga los arreglos procesados desde un archivo .npz."""
    ruta_archivo = Path(ruta_archivo)
    if not ruta_archivo.exists():
        raise FileNotFoundError(f"No existe el archivo de datos procesados en {ruta_archivo}")
    with np.load(ruta_archivo) as datos:
        return {clave: datos[clave] for clave in datos.files}
