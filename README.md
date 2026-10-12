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
│   ├── download_dataset.py     # Script para descargar y verificar el dataset
│   ├── explore_dataset.py      # Script de exploración visual y estadística
│   └── prepare_dataset.py      # Script para preprocesar y particionar los datos
├── src/
│   ├── data/
│   │   └── preprocess.py       # Lógica de preprocesamiento y aumento de datos
│   ├── model/
│   │   └── neural_network.py   # Implementación matemática de la red con NumPy
│   ├── train.py                # Bucle de entrenamiento y validación
│   ├── evaluate.py             # Evaluación en test y cálculo de métricas
│   └── predict_image.py        # Inferencia con imágenes individuales
└── tests/
    ├── test_preprocessing.py   # Pruebas de preprocesamiento
    └── test_neural_network.py  # Pruebas unitarias de las operaciones de la red
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
- [ ] **Fase 6:** Entrenamiento y registro de métricas reales ⏸️ *(Punto de pausa actual)*.
- [ ] **Fase 7:** Evaluación en conjunto de prueba y matriz de confusión.
- [ ] **Fase 8:** Inferencia en imágenes individuales.
- [ ] **Fase 9:** Demostración webcam (opcional y posterior).
- [ ] **Fase 10:** Adaptación a comandos robóticos (STOP, FORWARD, etc.).
