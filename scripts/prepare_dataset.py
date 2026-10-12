"""
Script ejecutable para preparar, preprocesar y particionar el dataset RPS (Fase 3).
Genera data/processed/rps_dataset_processed.npz de forma reproducible.
"""

import sys
from pathlib import Path
from collections import Counter
import numpy as np

# Configurar salida estándar UTF-8 para Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Añadir raíz del proyecto al path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data.preprocess import (
    LABEL_MAP,
    INV_LABEL_MAP,
    cargar_dataset_completo,
    division_estratificada,
    guardar_datos_procesados,
    cargar_datos_procesados,
)

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_NPZ = PROCESSED_DIR / "rps_dataset_processed.npz"


def main():
    print("=" * 65)
    print("FASE 3: PREPROCESAMIENTO Y PARTICIÓN DEL DATASET")
    print("=" * 65)

    if not RAW_DIR.exists():
        print(f"Error: El directorio de datos crudos no existe en {RAW_DIR}")
        return 1

    print(f"1. Cargando y preprocesando imágenes desde: {RAW_DIR}")
    print("   - Conversión a escala de grises ('L')")
    print("   - Redimensionamiento a 28x28 píxeles (784 características)")
    print("   - Normalización a flotantes en [0.0, 1.0]")
    print(f"   - Mapeo de clases: {LABEL_MAP}")

    X, y, rutas = cargar_dataset_completo(RAW_DIR)
    print(f"\n[OK] Imágenes preprocesadas con éxito:")
    print(f"   - Matriz X total: forma {X.shape}, tipo {X.dtype}, min={X.min():.4f}, max={X.max():.4f}")
    print(f"   - Vector y total: forma {y.shape}, tipo {y.dtype}")
    
    conteo_total = Counter(y)
    for c_id, cant in sorted(conteo_total.items()):
        print(f"     * Clase {c_id} ({INV_LABEL_MAP[c_id]}): {cant} muestras ({cant / len(y) * 100:.2f}%)")

    print("\n2. Partición estratificada (70% Train, 15% Val, 15% Test) con semilla 42...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = division_estratificada(
        X, y, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    print("\n[OK] Resumen de particiones generadas:")
    print(f"   - Entrenamiento (Train): X={X_train.shape}, y={y_train.shape}")
    conteo_train = Counter(y_train)
    for c_id, cant in sorted(conteo_train.items()):
        print(f"       {INV_LABEL_MAP[c_id]}: {cant} ({cant / len(y_train) * 100:.2f}%)")

    print(f"   - Validación (Val):     X={X_val.shape}, y={y_val.shape}")
    conteo_val = Counter(y_val)
    for c_id, cant in sorted(conteo_val.items()):
        print(f"       {INV_LABEL_MAP[c_id]}: {cant} ({cant / len(y_val) * 100:.2f}%)")

    print(f"   - Prueba (Test):        X={X_test.shape}, y={y_test.shape}")
    conteo_test = Counter(y_test)
    for c_id, cant in sorted(conteo_test.items()):
        print(f"       {INV_LABEL_MAP[c_id]}: {cant} ({cant / len(y_test) * 100:.2f}%)")

    print(f"\n3. Guardando arreglos procesados en: {OUTPUT_NPZ}")
    guardar_datos_procesados(OUTPUT_NPZ, X_train, y_train, X_val, y_val, X_test, y_test)

    # Verificación de recarga
    print("4. Verificando lectura y recarga segura del archivo guardado...")
    datos_recargados = cargar_datos_procesados(OUTPUT_NPZ)
    assert datos_recargados["X_train"].shape == X_train.shape
    assert datos_recargados["y_train"].shape == y_train.shape
    assert datos_recargados["X_val"].shape == X_val.shape
    assert datos_recargados["y_val"].shape == y_val.shape
    assert datos_recargados["X_test"].shape == X_test.shape
    assert datos_recargados["y_test"].shape == y_test.shape
    assert np.allclose(datos_recargados["X_train"], X_train)
    assert np.array_equal(datos_recargados["y_train"], y_train)

    peso_mb = OUTPUT_NPZ.stat().st_size / (1024 * 1024)
    print(f"[OK] Archivo guardado e íntegro: {OUTPUT_NPZ.name} ({peso_mb:.2f} MB)")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
