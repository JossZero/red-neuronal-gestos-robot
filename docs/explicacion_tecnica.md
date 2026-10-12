# Fundamento Teórico y Explicación Académica de la Red Neuronal

Este documento contiene la fundamentación matemática y conceptual de la red neuronal desarrollada para la clasificación de gestos de mano (Piedra, Papel, Tijera) y control robótico. Está redactado con rigor académico para servir de apoyo en la sustentación ante el comité evaluador.

---

## 1. Conceptos Fundamentales

### ¿Qué es una neurona artificial?
Una neurona artificial es una unidad computacional inspirada en la neurona biológica. Recibe un vector de entradas $x = [x_1, x_2, \dots, x_n]$, calcula una combinación lineal ponderada de estas entradas mediante pesos sinápticos $w$ y un sesgo $b$, y finalmente aplica una función no lineal de activación $f$:

$$z = \sum_{j=1}^{n} x_j w_j + b = x \cdot w + b$$
$$a = f(z)$$

### ¿Qué representan los 784 valores de entrada?
Cada imagen original de $300 \times 200$ píxeles en RGB es reducida a una cuadrícula en escala de grises de $28 \times 28$ píxeles. Al aplanar esta matriz bidimensional fila por fila, se obtiene un vector unidimensional de $28 \times 28 = 784$ elementos.
Cada componente $x_j$ representa la intensidad luminosa de un píxel específico, escalada al rango continuo $[0.0, 1.0]$, donde $0.0$ es negro puro y $1.0$ es blanco puro.

### ¿Qué son los pesos ($W$) y los sesgos ($b$)?
- **Pesos ($W$):** Representan la fuerza y polaridad de la conexión entre pares de neuronas. Un peso positivo indica que la característica estimula la activación, mientras que un peso negativo la inhibe. La magnitud determina la importancia relativa de esa característica.
- **Sesgos ($b$):** Permiten desplazar horizontalmente la función de activación, otorgando a la neurona la libertad de dispararse (activarse) incluso si la entrada ponderada es cero.

### ¿Qué hace la capa oculta de 32 neuronas?
Actúa como un extractor automático de características de nivel intermedio. Cada una de las 32 neuronas aprende a identificar patrones espaciales recurrentes en la mano (e.g., bordes diagonales, regiones compactas como el puño en piedra, extensiones alargadas como los dedos en tijera o superficies abiertas en papel). Transforma el espacio original de 784 dimensiones en una representación latente de 32 dimensiones más fácil de separar linealmente.

---

## 2. Funciones Matemáticas de Activación y Pérdida

### Función ReLU (Rectified Linear Unit)
Se utiliza en la capa oculta para introducir no linealidad, permitiendo que la red aproxime funciones complejas:

$$\text{ReLU}(z) = \max(0, z)$$

**Su derivada para backpropagation:**
$$\text{ReLU}'(z) = \begin{cases} 1 & \text{si } z > 0 \\ 0 & \text{si } z \le 0 \end{cases}$$

*Ventaja académica:* Mitiga el problema del desvanecimiento del gradiente (vanishing gradient) presente en funciones sigmoide o tanh, y su cómputo matricial es extremadamente eficiente.

### Función Softmax (Capa de Salida)
Convierte el vector de logits no acotados $Z_2 \in \mathbb{R}^3$ en una distribución de probabilidad $\hat{y} \in [0, 1]^3$ tal que $\sum_{k=0}^{2} \hat{y}_k = 1$:

$$\text{Softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=0}^{2} e^{z_j}}$$

**Estabilidad Numérica (Truco de resta del máximo):**
Si algún $z_k$ es muy grande (e.g., $> 700$), $e^{z_k}$ produce desbordamiento numérico (`overflow` / `inf` / `NaN`). Matemáticamente, restar una constante $C = \max_j(z_j)$ preserva exactamente la probabilidad:

$$\frac{e^{z_k - C}}{\sum_j e^{z_j - C}} = \frac{e^{z_k} e^{-C}}{\sum_j e^{z_j} e^{-C}} = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

### Función de Pérdida: Entropía Cruzada Categórica (Cross-Entropy)
Mide la discrepancia entre la distribución real $y$ (codificada en one-hot) y la predicha $\hat{y}$:

$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=0}^{C-1} y_{ik} \log(\hat{y}_{ik} + \epsilon)$$

Donde $\epsilon = 10^{-15}$ previene el error matemático de $\log(0)$. Cuando la red asigna alta probabilidad a la clase correcta, la pérdida tiende a 0; si asigna baja probabilidad, la penalización crece exponencialmente.

---

## 3. Propagación y Aprendizaje

### Propagación hacia adelante (Forward Propagation)
Es el cálculo secuencial de capa en capa para transformar la imagen de entrada en predicciones de salida:
1. $Z_1 = X \cdot W_1 + b_1$  *(Combinación lineal oculta, forma $m \times 32$)*
2. $A_1 = \text{ReLU}(Z_1)$    *(Activación oculta, forma $m \times 32$)*
3. $Z_2 = A_1 \cdot W_2 + b_2$ *(Combinación lineal de salida, forma $m \times 3$)*
4. $A_2 = \text{Softmax}(Z_2)$ *(Distribución de probabilidad, forma $m \times 3$)*

### Retropropagación (Backpropagation)
Es la aplicación sistemática de la **regla de la cadena** del cálculo diferencial multivariable para obtener la sensibilidad de la función de pérdida respecto a cada parámetro de la red.

La derivada combinada de la entropía cruzada y softmax con respecto a los logits $Z_2$ se simplifica elegantemente a:

$$dZ_2 = \frac{\partial \mathcal{L}}{\partial Z_2} = \frac{1}{m} (A_2 - Y)$$

A partir de este gradiente, se calculan analíticamente los gradientes de los parámetros:
- **Capa de Salida:**
  $$\frac{\partial \mathcal{L}}{\partial W_2} = A_1^T \cdot dZ_2$$
  $$\frac{\partial \mathcal{L}}{\partial b_2} = \sum_{\text{filas}} dZ_2$$
- **Propagación hacia Capa Oculta:**
  $$dA_1 = dZ_2 \cdot W_2^T$$
  $$dZ_1 = dA_1 \odot \text{ReLU}'(Z_1)$$
- **Capa Oculta:**
  $$\frac{\partial \mathcal{L}}{\partial W_1} = X^T \cdot dZ_1$$
  $$\frac{\partial \mathcal{L}}{\partial b_1} = \sum_{\text{filas}} dZ_1$$

### Descenso por Gradiente (Gradient Descent)
Los parámetros se actualizan en sentido opuesto al gradiente con una tasa de aprendizaje $\eta$:

$$W_1 \leftarrow W_1 - \eta \frac{\partial \mathcal{L}}{\partial W_1}$$
$$b_1 \leftarrow b_1 - \eta \frac{\partial \mathcal{L}}{\partial b_1}$$
$$W_2 \leftarrow W_2 - \eta \frac{\partial \mathcal{L}}{\partial W_2}$$
$$b_2 \leftarrow b_2 - \eta \frac{\partial \mathcal{L}}{\partial b_2}$$

---

## 4. Conceptos de Entrenamiento y Generalización

- **Época:** Una pasada completa a través de la totalidad de los datos de entrenamiento.
- **Mini-lote (Mini-batch):** Un subconjunto pequeño de muestras (e.g., 32 imágenes) utilizado para estimar el gradiente y actualizar los pesos con menor varianza que SGD puro y mayor eficiencia computacional que Batch Gradient Descent completo.
- **Diferencia entre conjuntos:**
  - *Entrenamiento (70%):* Usado exclusivamente para calcular gradientes y ajustar pesos.
  - *Validación (15%):* Usado para monitorear el aprendizaje al final de cada época y detectar sobreajuste sin contaminar los pesos.
  - *Prueba (15%):* Reservado y sellado; se evalúa una sola vez al final para medir el rendimiento en datos nunca antes vistos.
- **Sobreajuste (Overfitting):** Ocurre cuando la red memoriza los datos de entrenamiento (pérdida de entrenamiento sigue bajando pero la pérdida de validación empieza a subir). Se detecta comparando las curvas de ambas pérdidas.
- **Limitaciones de imágenes 28×28:** Reducir la imagen a 28×28 elimina detalles finos de la textura de la piel o uñas, pero preserva la silueta global de la mano, lo cual es ideal para discriminar puño, palma abierta y dedos en V con poco costo computacional.
- **Diferencia con los comandos del robot:** El dataset RPS clasifica exclusivamente tres gestos bajo un fondo verde estático de laboratorio. Un robot móvil opera en entornos dinámicos y requiere comandos de control (`STOP`, `FORWARD`, `LEFT`, etc.) que deben ser entrenados en su propio dataset específico.
- **Por qué NumPy no es una biblioteca de ML de alto nivel:** NumPy es únicamente una biblioteca de álgebra lineal y matrices multidimensionales (equivalente a operaciones con arreglos en C o Fortran). La arquitectura, la propagación, las derivadas simbólicas del cálculo tensorial, el bucle de optimización y la convergencia están implementados explícitamente en nuestro código en Python.
