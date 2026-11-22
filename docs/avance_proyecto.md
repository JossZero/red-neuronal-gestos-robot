# Bitácora de Avance del Proyecto

## Resumen General
- **Proyecto:** Clasificador de Gestos con Red Neuronal desde Cero (NumPy) para Control de Robot.
- **Entorno:** Windows, Python 3.13.2 en `.venv`.
- **Hardware:** CPU (sin dependencias aceleradoras ni bibliotecas de caja negra).

---

## Registro por Fases

### Fase 0: Inspección del Entorno y Planificación
- **Estado:** ✅ Completada.
- **Acciones Realizadas:**
  - Inspección del sistema: Python 3.13.2 disponible en la máquina del usuario.
  - Creación del entorno virtual `.venv` para aislar paquetes.
  - Definición de dependencias mínimas en `requirements.txt` (`numpy`, `Pillow`, `matplotlib`, `pytest`).
  - Creación de `.gitignore`, `AGENTS.md` y `README.md`.
  - Definición de la estructura de directorios.
- **Pruebas ejecutadas:** Verificación de ejecución de Python y Pip dentro de `.venv`.

---

### Fase 1: Descarga y Verificación del Dataset
- **Estado:** ✅ Completada y verificada.
- **Acciones Realizadas:**
  - Creación del script `scripts/download_dataset.py` con manejo de redirecciones HTTP, validación de integridad zipfile y conteo de clases.
  - Descarga del archivo `rps-cv-image.zip` (153.1 MB) desde el enlace de Dropbox con fallback a Kaggle documentado.
  - Extracción en `data/raw/`.
  - Verificación de la estructura de clases y lectura del archivo de autoría y licencia (`README_rpc-cv-images.txt`, CC-BY-SA 4.0).
- **Resultados Reales Obtenidos:**
  - `rock`: 726 imágenes (.png, RGB 300x200)
  - `paper`: 712 imágenes (.png, RGB 300x200)
  - `scissors`: 750 imágenes (.png, RGB 300x200)
  - **Total:** 2,188 imágenes extraídas correctamente sin archivos corruptos.
- **Siguiente Fase:** Fase 2 (Exploración del dataset, visualización y estadísticas).

---

### Fase 2: Exploración del Dataset
- **Estado:** ✅ Completada y verificada.
- **Acciones Realizadas:**
  - Creación del script `scripts/explore_dataset.py` con verificación estructural de integridad usando Pillow (`img.verify()`), análisis de dimensiones, modos de color y conteos por clase.
  - Generación de un mosaico visual de ejemplos representativos de cada clase (3x3) guardado en `outputs/figures/dataset_mosaic.png`.
- **Resultados Reales Obtenidos:**
  - Total de imágenes analizadas: 2,188.
  - Archivos corruptos o ilegibles: **0** (100% íntegros).
  - Distribución por clase:
    - `rock`: 726 imágenes (33.18%)
    - `paper`: 712 imágenes (32.54%)
    - `scissors`: 750 imágenes (34.28%)
    - *Dataset perfectamente balanceado*.
  - Modos de color: 100% **RGB** (3 canales).
  - Dimensiones: 100% homogéneas de **300 × 200 píxeles**.
  - Fondo: Color verde uniforme de chroma-key con iluminación consistente.
- **Siguiente Fase:** Fase 3 (Preprocesamiento: conversión a escala de grises, redimensionamiento a 28×28, normalización a [0, 1], aplanado a 784 y partición estratificada).

### Fase 3: Preprocesamiento de Imágenes
- **Estado:** ✅ Completada y verificada.
- **Acciones Realizadas:**
  - Implementación de `src/data/preprocess.py` con preprocesamiento funcional (Pillow -> Grises 'L' -> 28x28 Bilineal -> normalización float32 en [0.0, 1.0] -> vector 1D de 784 características).
  - Algoritmo de partición estratificada sin dependencias externas (`division_estratificada`) con semilla fija (`seed=42`) y verificación matemática de no solapamiento ($Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, $Val \cap Test = \emptyset$).
  - Guardado de arreglos comprimidos en `data/processed/rps_dataset_processed.npz`.
  - Creación y ejecución de suite de pruebas unitarias automatizadas en `tests/test_preprocessing.py`.
- **Resultados Reales Obtenidos:**
  - Pruebas unitarias: **6 de 6 pruebas superadas (100% pass)** en 0.38 segundos.
  - Dimensiones de la matriz de entrada procesada:
    - `X_train`: `(1531, 784)` | `y_train`: `(1531,)` (70%)
    - `X_val`: `(328, 784)` | `y_val`: `(328,)` (15%)
    - `X_test`: `(329, 784)` | `y_test`: `(329,)` (15%)
  - Rango de valores en píxeles: Min = 0.0118, Max = 0.9725.
  - Mapeo de etiquetas confirmado: `rock: 0`, `paper: 1`, `scissors: 2`.
  - Peso del archivo procesado: `1.58 MB`.
- **Siguiente Fase:** Fase 4 (Implementación de la Red Neuronal desde cero con NumPy: arquitectura 784 -> 32 -> 3, ReLU, Softmax numéricamente estable, Categorical Cross-Entropy, Backpropagation y descenso de gradiente).

### Fase 4: Implementación de la Red Neuronal con NumPy
- **Estado:** ✅ Completada y verificada.
- **Acciones Realizadas:**
  - Creación de `src/model/neural_network.py` con la clase `RedNeuronal`:
    - Arquitectura densa feedforward: $784 \to 32 \to 3$.
    - Inicialización de He para capa oculta y Xavier para capa de salida.
    - Activación ReLU vectorial y su derivada analítica.
    - Softmax numéricamente estable (restando $\max(Z_2)$ por fila).
    - Pérdida de entropía cruzada categórica con $\epsilon = 10^{-15}$.
    - Retropropagación analítica exacta ($dZ_2 = (A_2 - Y)/m$, $dW_2$, $db_2$, $dA_1$, $dZ_1$, $dW_1$, $db_1$).
    - Descenso por gradiente, cálculo de probabilidades, predicción discreta y cálculo de exactitud.
    - Métodos `guardar_modelo` y `cargar_modelo` para persistencia en `.npz`.
  - Creación de documentación académica completa en `docs/explicacion_tecnica.md`.
  - Creación de pruebas unitarias en `tests/test_neural_network.py` incluyendo verificación analítica de gradientes (Gradient Checking).
- **Resultados Reales Obtenidos:**
  - Las 12 pruebas unitarias automatizadas (`test_preprocessing.py` y `test_neural_network.py`) pasaron al 100%.
  - Error relativo de Gradient Checking en $W_2$: $9.99 \times 10^{-9}$ (confirmación analítica de la derivada).
  - Error relativo de Gradient Checking en $W_1$: $1.34 \times 10^{-8}$ (confirmación analítica a través de ReLU y regla de la cadena).
- **Siguiente Fase:** Fase 5 / 6 (Entrenamiento y registro de métricas reales con mini-batches y guardado del mejor modelo).

### Fase 5: Pruebas Unitarias y Validación Matemática
- **Estado:** ✅ Completada y verificada.
- **Acciones Realizadas:**
  - Ejecución de la suite completa de 12 pruebas unitarias con `pytest`:
    - `test_dimensiones_inicializacion`: Formas exactas de $W_1$ (784, 32), $b_1$ (1, 32), $W_2$ (32, 3), $b_2$ (1, 3).
    - `test_estabilidad_y_suma_softmax`: Verificación con logits extremos ($z = \pm 1000$) y comprobación de suma unitaria $\sum p = 1.0$.
    - `test_forward_y_perdida`: Probabilidades en $[0, 1]$ y pérdida finita y positiva.
    - `test_paso_entrenamiento_modifica_pesos`: Descenso de gradiente altera los parámetros.
    - `test_guardar_y_cargar_modelo`: Persistencia exacta de pesos y logits en formato `.npz`.
    - `test_gradiente_numerico_gradient_checking`: Comparación analítica vs aproximación por diferencias finitas centrales en float64.
- **Resultados Reales Obtenidos:**
  - **12 de 12 pruebas pasadas exitosamente (100% pass)** en 0.70 segundos.
  - Error relativo en $W_2$: $9.99 \times 10^{-9} < 10^{-6}$.
  - Error relativo en $W_1$: $1.34 \times 10^{-8} < 10^{-6}$.
- **Siguiente Fase:** Fase 6 completada.

### Fase 6: Entrenamiento y Registro de Curvas
- **Estado:** ✅ Completado.
- **Acciones y Resultados:**
  - Implementación de `src/train.py` con descenso de gradiente por mini-lotes (batch size: 32, $\eta = 0.03$, 50 épocas).
  - Mejor época alcanzada: **Época 48** (Pérdida Val: $0.5963$, Exactitud Val: $77.44\%$, Exactitud Train: $82.30\%$).
  - Pesos óptimos guardados en `models/rps_model.npz`.
  - Historial completo por época exportado a `outputs/logs/training_history.csv`.
  - Curvas de pérdida y exactitud generadas en `outputs/figures/training_curves.png`.

### Fase 7: Evaluación Final y Matriz de Confusión
- **Estado:** ✅ Completado.
- **Acciones y Resultados:**
  - Implementación de `src/evaluate.py` para evaluación sobre las 329 muestras independientes de test.
  - **Exactitud Global en Test:** **$75.38\%$** ($248 / 329$ aciertos).
  - **Pérdida Cross-Entropy en Test:** $0.6269$.
  - **Desglose de métricas:**
    - Piedra (`rock`): Precisión $76.11\%$, Recall $78.90\%$, F1 $77.48\%$.
    - Papel (`paper`): Precisión $72.87\%$, Recall $87.85\%$, F1 $79.66\%$.
    - Tijera (`scissors`): Precisión $78.16\%$, Recall $60.18\%$, F1 $68.00\%$.
    - Macro F1: $75.05\%$. F1 Ponderado: $74.93\%$.
  - Matriz de confusión visual guardada en `outputs/figures/confusion_matrix.png`.
  - Análisis cualitativo de aciertos y errores en `outputs/figures/evaluation_samples.png`.
  - Reporte estructurado guardado en `outputs/logs/evaluation_report.json`.

### Fase 8: Inferencia sobre Imágenes Nuevas
- **Estado:** ✅ Completado.
- **Acciones y Resultados:**
  - Implementación de `src/predict_image.py`.
  - Soporta inferencia por CLI sobre cualquier imagen JPG/PNG o en modo demo (`--demo`).
  - Muestra distribución de probabilidades Softmax y emite advertencias técnicas sobre condiciones de iluminación y fondo.
  - Genera diagnóstico visual comparativo en `outputs/figures/prediction_demo.png`.

### Fase 9: Demostración Interactiva con Webcam en Tiempo Real
- **Estado:** ✅ Completado.
- **Acciones y Resultados:**
  - Implementación de `src/webcam_demo.py` utilizando OpenCV únicamente para captura de video y HUD visual.
  - La inferencia matemática se ejecuta directamente mediante el forward propagation de `RedNeuronal` (NumPy).
  - Preprocesamiento de la región de interés (ROI) a $28 \times 28$ píxeles en escala de grises y normalización $[0.0, 1.0]$.
  - Visualización en vivo con barras de probabilidad, FPS, traducción de clases al español y ventana de depuración de la entrada procesada.

### Fase 10: Adaptación a Comandos Robóticos y Seguridad
- **Estado:** ✅ Completado.
- **Acciones y Resultados:**
  - Implementación del script de recolección de datos guiado en `scripts/capture_robot_dataset.py` para almacenar imágenes de los 5 comandos (`STOP`, `FORWARD`, `LEFT`, `RIGHT`, `BACKWARD`) en `data/robot_raw/`.
  - Capa de seguridad y control en `src/robot_controller.py`:
    - **Filtro de Umbral de Confianza:** Predicciones con confianza $< 70\%$ fuerzan el estado seguro `STOP`.
    - **Consenso Temporal:** Se requiere la confirmación idéntica durante $3$ fotogramas seguidos antes de conmutar comandos de movimiento, evitando sacudidas en los motores.
    - **Controlador Robótico Desacoplado:** Soporte para simulación virtual (`SimuladorRobot`) y enlace serie (`ControladorSerialRobot`) con parada de emergencia.
  - Validación con pruebas unitarias en `tests/test_robot_controller.py` y `tests/test_webcam_demo.py` (22 pruebas totales pasando al 100%).

