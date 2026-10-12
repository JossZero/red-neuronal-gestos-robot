# Reglas y Guía para Agentes (AGENTS.md)

Este repositorio alberga el proyecto de desarrollo de una red neuronal artificial construida y entrenada **desde cero** con Python y NumPy para clasificación de gestos y futuro control robótico.

## Reglas Inviolables
1. **Sin bibliotecas de alto nivel para IA/ML:** Queda estrictamente prohibido el uso de TensorFlow, Keras, PyTorch, scikit-learn, OpenCV DNN o servicios en la nube para entrenar o inferir.
2. **Entrenamiento desde cero:** El cálculo hacia adelante (forward propagation), la función de pérdida (cross-entropy), la retropropagación (backpropagation) y el descenso de gradiente (mini-batch gradient descent) deben implementarse con NumPy.
3. **Bibliotecas auxiliares permitidas:**
   - `Pillow` (PIL) únicamente para cargar, convertir a escala de grises y redimensionar imágenes.
   - `matplotlib` únicamente para graficar curvas de aprendizaje y mostrar imágenes de prueba.
   - `pytest` para pruebas automáticas.
4. **Reproducibilidad:** Fijar siempre semillas aleatorias (`seed`) en NumPy y particiones de datos.
5. **Comprobación empírica:** No fingir métricas ni resultados. Toda afirmación debe estar respaldada por ejecuciones reales.
6. **Código y documentación en español:** Todo el código, comentarios y documentación técnica deben estar explicados en español con claridad académica orientada a sustentación ante profesores.
7. **Desarrollo por fases:** Seguir el orden estricto de las fases y validar cada una antes de avanzar.
