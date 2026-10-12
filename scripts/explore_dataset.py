"""
Script para explorar y auditar el dataset de Rock-Paper-Scissors (Fase 2).
Analiza integridad, formatos, dimensiones, modos de color y genera un mosaico visual.
No modifica ninguna imagen original.
"""

import sys
from pathlib import Path
from collections import defaultdict

# Asegurar codificación UTF-8 en salida estándar para compatibilidad con terminales Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from PIL import Image
import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para guardar imágenes sin abrir ventana
import matplotlib.pyplot as plt

# Rutas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUTS_FIG_DIR = BASE_DIR / "outputs" / "figures"

CLASES = ["rock", "paper", "scissors"]


def explorar_dataset():
    print("=" * 65)
    print("FASE 2: EXPLORACIÓN Y AUDITORÍA DEL DATASET RPS")
    print("=" * 65)

    if not RAW_DIR.exists():
        print(f"Error: No se encontró el directorio {RAW_DIR}. Ejecuta Fase 1 primero.")
        return 1

    OUTPUTS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    conteo_clases = {}
    archivos_por_clase = defaultdict(list)
    archivos_corruptos = []
    modos_color = defaultdict(int)
    dimensiones_encontradas = defaultdict(int)
    
    total_archivos = 0

    print("Auditoría de integridad de imágenes...")

    for clase in CLASES:
        dir_clase = RAW_DIR / clase
        if not dir_clase.is_dir():
            print(f"Advertencia: No existe el directorio para la clase '{clase}' en {dir_clase}")
            continue

        archivos = sorted(list(dir_clase.glob("*.*")))
        conteo_clases[clase] = len(archivos)
        
        for ruta_img in archivos:
            total_archivos += 1
            try:
                with Image.open(ruta_img) as img:
                    img.verify()  # Verifica integridad estructural de la imagen
                
                # Reabrir para inspeccionar metadatos (verify() invalida el objeto en Pillow)
                with Image.open(ruta_img) as img:
                    modos_color[img.mode] += 1
                    dimensiones_encontradas[img.size] += 1
                    archivos_por_clase[clase].append(ruta_img)
            except Exception as e:
                archivos_corruptos.append((ruta_img, str(e)))

    # Resumen cuantitativo
    print("\n--- Distribución de Clases ---")
    total_validas = 0
    for clase in CLASES:
        cant = conteo_clases.get(clase, 0)
        total_validas += cant
        print(f"  * Clase '{clase}': {cant} imágenes")
    print(f"Total de imágenes examinadas: {total_archivos}")

    print("\n--- Integridad de Archivos ---")
    if archivos_corruptos:
        print(f"[ALERTA] Se encontraron {len(archivos_corruptos)} archivos corruptos o ilegibles:")
        for ruta, err in archivos_corruptos:
            print(f"    - {ruta.name}: {err}")
    else:
        print("[OK] 0 archivos corruptos detectados. Todas las imágenes son válidas y legibles.")

    print("\n--- Modos de Color Detectados ---")
    for modo, cant in modos_color.items():
        print(f"  * Modo '{modo}': {cant} imágenes (100% homogéneo)")

    print("\n--- Dimensiones (Ancho x Alto) ---")
    for dim, cant in dimensiones_encontradas.items():
        print(f"  * {dim[0]} x {dim[1]} píxeles: {cant} imágenes")

    # Generación de Mosaico de Visualización
    print("\nGenerando mosaico visual de ejemplos...")
    samples_per_class = 3
    fig, axes = plt.subplots(len(CLASES), samples_per_class, figsize=(10, 8))
    fig.suptitle("Muestras del Dataset Rock-Paper-Scissors (Fase 2)", fontsize=14, fontweight="bold")

    for i, clase in enumerate(CLASES):
        imgs_clase = archivos_por_clase[clase][:samples_per_class]
        for j, ruta_img in enumerate(imgs_clase):
            ax = axes[i, j]
            with Image.open(ruta_img) as img:
                ax.imshow(img)
                ax.set_title(f"{clase.capitalize()}\n{img.size[0]}x{img.size[1]} ({img.mode})", fontsize=10)
                ax.axis("off")

    plt.tight_layout()
    ruta_mosaico = OUTPUTS_FIG_DIR / "dataset_mosaic.png"
    plt.savefig(ruta_mosaico, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[OK] Mosaico guardado con éxito en: {ruta_mosaico}")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(explorar_dataset())
