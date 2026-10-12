# Estado Actual del Proyecto y Punto de Pausa

> **Nota de Control:** El proyecto ha sido pausado al inicio de la **Fase 6** para continuar en la próxima sesión de trabajo.

---

## 📌 Resumen Ejecutivo de lo Completado

1. **Fase 0: Entorno y Estructura Base** ✅
   - Entorno virtual local `.venv` configurado con Python 3.13 en Windows (CPU).
   - Dependencias mínimas instaladas: `numpy`, `Pillow`, `matplotlib`, `pytest` (sin TensorFlow, PyTorch ni scikit-learn).
   - Estructura limpia y organizada en carpetas `data/`, `docs/`, `models/`, `outputs/`, `scripts/`, `src/` y `tests/`.

2. **Fase 1: Descarga y Verificación del Dataset** ✅
   - Descarga directa de `rps-cv-image.zip` (153.1 MB) y extracción en `data/raw/`.
   - 2,188 imágenes verificadas bajo licencia CC-BY-SA 4.0 (`rock`: 726, `paper`: 712, `scissors`: 750).

3. **Fase 2: Exploración del Dataset** ✅
   - Auditoría de integridad: 0 archivos corruptos, 100% legibles por Pillow.
   - Resolución y formato: 100% homogéneo en RGB de $300 \times 200$ píxeles.
   - Generación del mosaico visual de muestras en `outputs/figures/dataset_mosaic.png`.

4. **Fase 3: Preprocesamiento y Partición de Datos** ✅
   - Conversión a escala de grises, redimensionamiento a $28 \times 28$ píxeles, normalización a $[0.0, 1.0]$ y aplanado a vector de $784$ elementos.
   - División estratificada (70% Train, 15% Val, 15% Test) sin solapamiento guardada en `data/processed/rps_dataset_processed.npz`:
     - $X_{\text{train}}$: `(1531, 784)` | $y_{\text{train}}$: `(1531,)`
     - $X_{\text{val}}$: `(328, 784)` | $y_{\text{val}}$: `(328,)`
     - $X_{\text{test}}$: `(329, 784)` | $y_{\text{test}}$: `(329,)`

5. **Fase 4: Implementación de la Red Neuronal desde Cero (NumPy)** ✅
   - Perceptrón Multicapa denso en `src/model/neural_network.py` ($784 \to 32 \to 3$).
   - Inicializaciones He y Xavier, activación ReLU, Softmax numéricamente estable y Entropía Cruzada Categórica.
   - Retropropagación analítica exacta implementada vectorialmente.
   - Persistencia de parámetros en formato `.npz`.
   - Fundamentación teórica completa documentada en `docs/explicacion_tecnica.md`.

6. **Fase 5: Validación Matemática y Gradient Checking** ✅
   - 12 pruebas unitarias ejecutadas con `pytest` pasando al 100%.
   - **Gradient Checking:** Comparación analítica vs numérica por diferencias finitas centrales con error relativo $< 1.35 \times 10^{-8}$ tanto en $W_1$ como en $W_2$, comprobando empíricamente que las derivadas son exactas.

---

## ⏸️ Punto de Pausa: Fase 6

- **Fase actual pausada:** **Fase 6: Entrenamiento y Registro de Curvas**.
- **Qué se hará al reanudar:**
  1. Crear `src/train.py`.
  2. Entrenar la red por mini-lotes (35 épocas, batch size de 32, lr de 0.05).
  3. Registrar métricas de Train y Val por época en `outputs/logs/training_history.csv`.
  4. Guardar los mejores pesos en `models/rps_model.npz`.
  5. Graficar las curvas de pérdida y exactitud en `outputs/figures/training_curves.png`.
  6. Continuar con la Fase 7 (Evaluación en Test con matriz de confusión) y Fase 8 (Inferencia con imágenes individuales).

---

## 🛠️ Comandos Rápidos de Verificación para la Presentación

Si necesitas comprobar o mostrar el avance rápidamente a los profesores:

- **Ejecutar todas las pruebas unitarias y verificación de gradiente:**
  ```powershell
  .\.venv\Scripts\python.exe -m pytest -v
  ```
- **Verificar la preparación del dataset:**
  ```powershell
  .\.venv\Scripts\python.exe scripts/prepare_dataset.py
  ```
- **Auditoría visual del dataset:**
  Abrir la imagen generada en `outputs/figures/dataset_mosaic.png`.
