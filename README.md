# Red Neuronal desde Cero con NumPy: Reconocimiento de Gestos para Control de Robot

Este proyecto implementa una red neuronal artificial multicapa completamente desde cero, utilizando únicamente **Python** y **NumPy** para todas las operaciones matemáticas, propagación hacia adelante, retropropagación y descenso por gradiente.

El objetivo principal es académico: comprender a fondo la formulación matemática y el flujo de aprendizaje de una red neuronal para defender su funcionamiento ante un panel evaluador, sin depender de bibliotecas de alto nivel como TensorFlow, PyTorch o scikit-learn.

---

## 📋 Arquitectura de la Red (Fase RPS)

- **Entrada:** Imagen en escala de grises de 28×28 píxeles normalizada a $[0, 1]$.
- **Aplanamiento:** Vector de $784$ características ($28 \times 28 = 784$).
- **Capa Oculta:** $32$ neuronas con función de activación ReLU.
- **Capa de Salida:** $3$ neuronas (Piedra: 0, Papel: 1, Tijera: 2) con función de activación Softmax numéricamente estable.
- **Función de Pérdida:** Entropía cruzada categórica (Categorical Cross-Entropy).
- **Optimizador:** Descenso por gradiente en mini-lotes (Mini-Batch Gradient Descent).

---

## 📁 Estructura del Proyecto

```text
├── AGENTS.md                   # Directrices y reglas de desarrollo
├── README.md                   # Documentación principal
├── requirements.txt            # Dependencias mínimas
├── .gitignore                  # Exclusiones de control de versiones
├── data/
│   ├── downloads/              # Archivos ZIP descargados
│   ├── raw/                    # Imágenes crudas extraídas
│   └── processed/              # Arreglos NumPy preprocesados (X_train, y_train, etc.)
├── docs/
│   ├── avance_proyecto.md      # Registro del estado y avance por fases
│   └── explicacion_tecnica.md  # Fundamento matemático y teórico
├── models/                     # Pesos guardados del modelo (.npz)
├── outputs/
│   ├── figures/                # Gráficas de entrenamiento y matrices de confusión
│   └── logs/                   # Historial de métricas (CSV)
├── scripts/
│   ├── download_dataset.py       # Script para descargar y verificar el dataset
│   ├── explore_dataset.py        # Script de exploración visual y estadística
│   ├── prepare_dataset.py        # Script para preprocesar y particionar los datos
│   └── capture_robot_dataset.py  # Asistente de captura de gestos propios para el robot
├── src/
│   ├── data/
│   │   └── preprocess.py         # Lógica de preprocesamiento y extracción de características
│   ├── model/
│   │   └── neural_network.py     # Implementación matemática de la red con NumPy
│   ├── train.py                  # Bucle de entrenamiento por mini-lotes
│   ├── evaluate.py               # Evaluación en test y matriz de confusión
│   ├── predict_image.py          # Inferencia con imágenes individuales
│   ├── webcam_demo.py            # Demostración interactiva en vivo con cámara
│   └── robot_controller.py       # Capa de control robótico seguro y consenso temporal
└── tests/
    ├── test_preprocessing.py     # Pruebas de preprocesamiento
    ├── test_neural_network.py    # Pruebas de la arquitectura y gradient checking
    ├── test_training_and_evaluation.py # Pruebas de entrenamiento e inferencia
    ├── test_webcam_demo.py       # Pruebas del preprocesamiento y renderizado de webcam
    └── test_robot_controller.py  # Pruebas del filtro de seguridad y simulador robótico
```

---

## 🚀 Instalación y Entorno

1. **Crear y activar el entorno virtual:**
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Instalar dependencias mínimas:**
   ```powershell
   python -m pip install -r requirements.txt
   ```

---

## 🔄 Fases del Proyecto

- [x] **Fase 0:** Inspección del entorno y planificación inicial.
- [x] **Fase 1:** Descarga y verificación del dataset (Rock-Paper-Scissors).
- [x] **Fase 2:** Exploración del dataset y visualización.
- [x] **Fase 3:** Preprocesamiento y división estratificada (Train / Val / Test).
- [x] **Fase 4:** Implementación de la Red Neuronal desde cero con NumPy.
- [x] **Fase 5:** Pruebas unitarias matemáticas y gradiente.
- [x] **Fase 6:** Entrenamiento por mini-lotes y registro de curvas reales.
- [x] **Fase 7:** Evaluación en conjunto de prueba y matriz de confusión.
- [x] **Fase 8:** Inferencia en imágenes individuales con diagnóstico visual.
- [x] **Fase 9:** Demostración interactiva con Webcam y HUD en tiempo real.
- [x] **Fase 10:** Adaptación a comandos robóticos (Filtro de seguridad, consenso temporal y asistente de captura).

---

## ⚡ Guía de Ejecución

### 1. Entrenar el modelo
```powershell
python -m src.train --epocas 50 --batch-size 32 --lr 0.03
```
Genera:
- Pesos del modelo: `models/rps_model.npz`
- Historial en CSV: `outputs/logs/training_history.csv`
- Curvas de aprendizaje: `outputs/figures/training_curves.png`

### 2. Evaluar el modelo en el conjunto de prueba
```powershell
python -m src.evaluate
```
Genera:
- Reporte detallado de métricas (Precision, Recall, F1, Accuracy): `outputs/logs/evaluation_report.json`
- Gráfica de Matriz de Confusión: `outputs/figures/confusion_matrix.png`
- Ejemplos de aciertos y errores: `outputs/figures/evaluation_samples.png`

### 3. Clasificar una imagen aislada
```powershell
# Ejecutar demostración automática:
python -m src.predict_image --demo

# Clasificar una imagen específica:
python -m src.predict_image ruta/a/imagen.png
```

### 4. Demostración en vivo con Webcam y Control de Robot (HUD en tiempo real)
El modelo entrenado ya viene preinstalado en `models/rps_model.npz`, por lo que puedes ejecutar la demostración directamente sin necesidad de re-entrenar:

```powershell
# Modo simulador de robot (sin hardware conectado):
python -m src.webcam_demo --simular-robot

# Conectado físicamente al ESP32 / Arduino (ejemplo puerto COM3):
python -m src.webcam_demo --puerto-robot COM3 --umbral-confianza 0.75 --consenso-frames 3
```
*Controles: Presiona 'q' o ESC para salir; 's' para capturar foto; 'e' o ESPACIO para parada de emergencia (E-Stop).*

### 5. Capturar dataset personalizado para los 5 comandos del robot
```powershell
python scripts/capture_robot_dataset.py
```
*Permite capturar imágenes de `STOP`, `FORWARD`, `LEFT`, `RIGHT` y `BACKWARD` en `data/robot_raw/`.*

### 6. Ejecutar la suite completa de pruebas automáticas (23 pruebas)
```powershell
python -m pytest -v
```
