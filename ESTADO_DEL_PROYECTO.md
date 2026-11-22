# Estado Actual del Proyecto y Registro de Avances

> **Última Actualización:** 2026-10-25
> **Estado:** Fases 0 a 8 completadas con éxito y verificadas empíricamente.

---

## 📌 Resumen Ejecutivo de Fases Completadas

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

6. **Fase 5: Validación Matemática y Gradient Checking** ✅
   - Pruebas unitarias ejecutadas con `pytest` pasando al 100%.
   - **Gradient Checking:** Comparación analítica vs numérica por diferencias finitas centrales con error relativo $< 1.35 \times 10^{-8}$ tanto en $W_1$ como en $W_2$.

7. **Fase 6: Entrenamiento y Registro de Curvas** ✅
   - Módulo implementado en `src/train.py`.
   - Bucle por mini-lotes (batch size 32, $\eta = 0.03$, 50 épocas, barajado reproducible).
   - Mejor modelo alcanzado en la **Época 48** con pérdida de validación $0.5963$ y exactitud de validación del $77.44\%$.
   - Registro de métricas en `outputs/logs/training_history.csv`.
   - Pesos óptimos guardados en `models/rps_model.npz`.
   - Curvas de pérdida y exactitud guardadas en `outputs/figures/training_curves.png`.

8. **Fase 7: Evaluación en Test y Matriz de Confusión** ✅
   - Módulo implementado en `src/evaluate.py`.
   - Evaluación sobre 329 muestras independientes de prueba:
     - **Exactitud Global (Accuracy):** **$75.38\%$** ($248 / 329$ aciertos).
     - **Pérdida en Test:** $0.6269$.
     - **Precision por clase:** Piedra: $76.11\%$, Papel: $72.87\%$, Tijera: $78.16\%$.
     - **Recall por clase:** Piedra: $78.90\%$, Papel: $87.85\%$, Tijera: $60.18\%$.
     - **F1-Score por clase:** Piedra: $77.48\%$, Papel: $79.66\%$, Tijera: $68.00\%$.
   - Matriz de confusión guardada en `outputs/figures/confusion_matrix.png`.
   - Muestras cualitativas (aciertos vs confusiones) en `outputs/figures/evaluation_samples.png`.
   - Reporte JSON detallado en `outputs/logs/evaluation_report.json`.

9. **Fase 8: Inferencia sobre Imágenes Individuales** ✅
   - Script implementado en `src/predict_image.py`.
   - Aplica el preprocesamiento exacto (escala de grises, $28 \times 28$, $[0, 1]$, $784$), carga el modelo y retorna la predicción con desglose de probabilidades y diagnóstico visual.
   - Gráfica diagnóstica guardada en `outputs/figures/prediction_demo.png`.

10. **Fase 9: Demostración Interactiva con Webcam** ✅
   - Script implementado en `src/webcam_demo.py`.
   - Captura de fotogramas en vivo con OpenCV delimitando una zona de interés (ROI).
   - Inferencia en tiempo real utilizando puramente la red NumPy ($784 \to 32 \to 3$).
   - Interfaz HUD con barras de probabilidad, etiqueta en español y visualización en miniatura de la entrada procesada ($28 \times 28$).
   - Opciones para capturar instantáneas ('s') y salir limpiamente ('q' o ESC).

11. **Fase 10: Arquitectura de Control Robótico y Seguridad** ✅
   - Módulo de control seguro implementado en `src/robot_controller.py`:
     - **Filtro de Seguridad por Umbral:** Si la confianza es $< 70\%$, conmuta a parada preventiva (`STOP`).
     - **Consenso Temporal:** Exige $3$ fotogramas consecutivos consistentes antes de autorizar cualquier movimiento (`FORWARD`, `LEFT`, `RIGHT`, `BACKWARD`).
     - **Capa de Abstracción:** `SimuladorRobot` para sustentaciones y `ControladorSerialRobot` para microcontroladores (Arduino / ESP32).
   - Script de captura guiada de dataset en `scripts/capture_robot_dataset.py` para recopilar imágenes de los 5 comandos gestuales en `data/robot_raw/`.

---

## 🛠️ Comandos Rápidos de Verificación para la Presentación

1. **Ejecutar la suite completa de pruebas automáticas (22 pruebas al 100%):**
   ```powershell
   .\.venv\Scripts\python.exe -m pytest -v
   ```

2. **Re-entrenar la red neuronal desde cero:**
   ```powershell
   .\.venv\Scripts\python.exe -m src.train --epocas 50 --batch-size 32 --lr 0.03
   ```

3. **Evaluar el modelo en el conjunto de prueba y generar reportes:**
   ```powershell
   .\.venv\Scripts\python.exe -m src.evaluate
   ```

4. **Clasificar una imagen individual (o ejecutar demo con 3 muestras):**
   ```powershell
   # Modo demostración automática:
   .\.venv\Scripts\python.exe -m src.predict_image --demo

   # Modo imagen específica:
   .\.venv\Scripts\python.exe -m src.predict_image "data/raw/rock/00nKV8oHuTGi20gq.png"
   ```

5. **Iniciar la demostración en tiempo real con Webcam:**
   ```powershell
   .\.venv\Scripts\python.exe -m src.webcam_demo
   ```

6. **Iniciar el asistente para capturar dataset propio de comandos robóticos:**
   ```powershell
   .\.venv\Scripts\python.exe scripts/capture_robot_dataset.py
   ```

