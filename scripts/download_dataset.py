"""
Script para descargar, validar y extraer el dataset de Rock-Paper-Scissors (RPS).
Cumple con las directrices de la Fase 1 del proyecto.
"""

import os
import sys
import zipfile
import urllib.request
import urllib.error
from pathlib import Path

# URLs de descarga recomendadas
# La URL de Dropbox suele necesitar ?dl=1 para descarga directa
DROPBOX_URL_RAW = "https://www.dropbox.com/s/sr1iui3zfdk0o53/rps-cv-image.zip"
DROPBOX_URL_DIRECT = "https://www.dropbox.com/s/sr1iui3zfdk0o53/rps-cv-image.zip?dl=1"

# Carpetas del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = BASE_DIR / "data" / "downloads"
RAW_DIR = BASE_DIR / "data" / "raw"
ZIP_FILE_PATH = DOWNLOADS_DIR / "rps-cv-image.zip"

CLASES_ESPERADAS = ["rock", "paper", "scissors"]


def reportar_progreso(bloque_num, tamano_bloque, tamano_total):
    """Muestra el progreso de la descarga en porcentaje."""
    descargado = bloque_num * tamano_bloque
    if tamano_total > 0:
        porcentaje = min(100.0, (descargado / tamano_total) * 100)
        mb_descargados = descargado / (1024 * 1024)
        mb_totales = tamano_total / (1024 * 1024)
        sys.stdout.write(f"\rProgreso: {porcentaje:.1f}% ({mb_descargados:.1f} MB / {mb_totales:.1f} MB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDescargados {descargado / (1024 * 1024):.1f} MB...")
        sys.stdout.flush()


def verificar_dataset_extraido(ruta_raw: Path):
    """
    Comprueba si las carpetas de las clases ya existen y contienen imágenes válidas.
    Retorna (True, conteo_dict, carpeta_base_clases) si está completo.
    """
    if not ruta_raw.exists():
        return False, {}, None

    # Buscar dónde están las carpetas rock, paper, scissors
    # Pueden estar directamente en data/raw o dentro de un subdirectorio como data/raw/rps-cv-images
    posibles_bases = [ruta_raw] + [d for d in ruta_raw.iterdir() if d.is_dir()]
    
    for base in posibles_bases:
        conteos = {}
        todas_existen = True
        for clase in CLASES_ESPERADAS:
            dir_clase = base / clase
            if not dir_clase.is_dir():
                todas_existen = False
                break
            imagenes = [f for f in dir_clase.iterdir() if f.is_file() and f.suffix.lower() in [".png", ".jpg", ".jpeg"]]
            conteos[clase] = len(imagenes)
        
        if todas_existen and all(conteos[c] > 0 for c in CLASES_ESPERADAS):
            return True, conteos, base

    return False, {}, None


def descargar_zip(url: str, destino: Path) -> bool:
    """Descarga el archivo ZIP siguiendo redirecciones y mostrando progreso."""
    print(f"Descargando dataset desde: {url}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    
    # Manejar User-Agent para evitar bloqueos 403
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req) as respuesta, open(destino, "wb") as archivo_out:
            tamano_total = int(respuesta.info().get("Content-Length", -1))
            tamano_bloque = 1024 * 64  # 64 KB
            bloque_num = 0
            
            while True:
                buffer = respuesta.read(tamano_bloque)
                if not buffer:
                    break
                archivo_out.write(buffer)
                bloque_num += 1
                reportar_progreso(bloque_num, tamano_bloque, tamano_total)
                
        print("\nDescarga finalizada.")
        return True
    except Exception as e:
        print(f"\nError durante la descarga: {e}")
        if destino.exists():
            destino.unlink()  # Eliminar archivo incompleto
        return False


def validar_y_extraer_zip(zip_path: Path, destino_raw: Path) -> bool:
    """Verifica la integridad del ZIP y lo extrae en data/raw."""
    print(f"Validando integridad del archivo ZIP: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        print(f"Error: {zip_path} no es un archivo ZIP válido.")
        return False

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            corrupto = zf.testzip()
            if corrupto:
                print(f"Error: archivo ZIP corrupto en el elemento: {corrupto}")
                return False
            
            print("Archivo ZIP íntegro y verificado. Procediendo a extraer...")
            destino_raw.mkdir(parents=True, exist_ok=True)
            zf.extractall(destino_raw)
            print(f"Extracción completada en: {destino_raw}")
            return True
    except Exception as e:
        print(f"Error al extraer archivo ZIP: {e}")
        return False


def main():
    print("=" * 60)
    print("FASE 1: DESCARGA Y VERIFICACIÓN DEL DATASET ROCK-PAPER-SCISSORS")
    print("=" * 60)

    # 1. Verificar si ya está descargado y extraído
    ya_extraido, conteos, base_clases = verificar_dataset_extraido(RAW_DIR)
    if ya_extraido:
        print("El dataset ya se encuentra extraído y listo.")
        print(f"Ruta base detectada: {base_clases}")
        print("Conteo de imágenes por clase:")
        total = 0
        for clase, cantidad in conteos.items():
            print(f"  - {clase}: {cantidad} imágenes")
            total += cantidad
        print(f"Total general de imágenes: {total}")
        return 0

    # 2. Verificar si el ZIP ya está descargado
    if ZIP_FILE_PATH.exists():
        print(f"Se encontró un archivo ZIP descargado previamente en {ZIP_FILE_PATH}")
        if not validar_y_extraer_zip(ZIP_FILE_PATH, RAW_DIR):
            print("El ZIP existente está corrupto. Reintentando descarga...")
            ZIP_FILE_PATH.unlink()
    
    # 3. Descargar si no está el ZIP o si no se ha extraído
    if not ZIP_FILE_PATH.exists():
        # Intentar enlace directo con ?dl=1 primero
        exito = descargar_zip(DROPBOX_URL_DIRECT, ZIP_FILE_PATH)
        if not exito:
            print("Reintentando con URL alternativa...")
            exito = descargar_zip(DROPBOX_URL_RAW, ZIP_FILE_PATH)
        
        if not exito:
            print("\n" + "!" * 60)
            print("FALLO LA DESCARGA AUTOMÁTICA DIRECTA.")
            print("Alternativa oficial usando Kaggle CLI:")
            print("  1. Asegúrate de tener credenciales en ~/.kaggle/kaggle.json")
            print("  2. Ejecuta el comando:")
            print("     kaggle datasets download -d drgfreeman/rockpaperscissors -p data/downloads")
            print("  3. Vuelve a ejecutar este script para validar y extraer.")
            print("!" * 60)
            return 1

    # 4. Extraer ZIP
    exito_extraccion = validar_y_extraer_zip(ZIP_FILE_PATH, RAW_DIR)
    if not exito_extraccion:
        return 1

    # 5. Detección y verificación de carpetas extraídas
    ya_extraido, conteos, base_clases = verificar_dataset_extraido(RAW_DIR)
    if not ya_extraido:
        print("\nAdvertencia: No se encontraron las carpetas completas 'rock', 'paper', 'scissors'.")
        print("Contenido encontrado en data/raw:")
        for elemento in RAW_DIR.rglob("*"):
            if elemento.is_dir():
                print(f"  [DIR] {elemento.relative_to(RAW_DIR)}")
        return 1

    print("\n" + "=" * 60)
    print("VERIFICACIÓN EXITOSA DEL DATASET")
    print(f"Directorio de clases: {base_clases}")
    total = 0
    for clase in CLASES_ESPERADAS:
        cant = conteos.get(clase, 0)
        print(f"  - Clase '{clase}': {cant} imágenes")
        total += cant
    print(f"Total de imágenes extraídas: {total}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
