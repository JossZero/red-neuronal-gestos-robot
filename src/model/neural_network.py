"""
Implementación desde cero de una Red Neuronal Artificial (Perceptrón Multicapa) con NumPy.
Diseñada para el reconocimiento de gestos de mano (Fase 4).

Arquitectura:
- Capa de Entrada: 784 características (imágenes 28x28 en escala de grises).
- Capa Oculta: 32 neuronas con activación ReLU.
- Capa de Salida: 3 neuronas con activación Softmax numéricamente estable.
- Función de Pérdida: Entropía Cruzada Categórica (Categorical Cross-Entropy).
- Optimización: Descenso de Gradiente (Mini-Batch Gradient Descent).
"""

from pathlib import Path
from typing import Dict, Tuple, Union, Optional
import numpy as np


class RedNeuronal:
    """
    Red neuronal de dos capas densas implementada puramente con NumPy.
    
    Ecuaciones hacia adelante (Forward):
        Z1 = X · W1 + b1
        A1 = ReLU(Z1) = max(0, Z1)
        Z2 = A1 · W2 + b2
        A2 = Softmax(Z2) = exp(Z2 - max(Z2)) / sum(exp(Z2 - max(Z2)))
    """

    def __init__(
        self,
        n_entrada: int = 784,
        n_oculta: int = 32,
        n_salida: int = 3,
        semilla: Optional[int] = 42,
    ):
        """
        Inicializa los pesos y sesgos de la red neuronal.

        Args:
            n_entrada: Dimensión de entrada (por defecto 784).
            n_oculta: Número de neuronas en la capa oculta (por defecto 32).
            n_salida: Número de clases en la salida (por defecto 3).
            semilla: Semilla para reproducibilidad de pesos iniciales.
        """
        self.n_entrada = n_entrada
        self.n_oculta = n_oculta
        self.n_salida = n_salida
        self.semilla = semilla

        rng = np.random.default_rng(semilla)

        # 1. Inicialización He (Kaiming) para la capa ReLU:
        # Desviación estándar = sqrt(2 / n_entrada)
        escala_he = np.sqrt(2.0 / n_entrada)
        self.W1 = rng.normal(loc=0.0, scale=escala_he, size=(n_entrada, n_oculta)).astype(np.float32)
        self.b1 = np.zeros((1, n_oculta), dtype=np.float32)

        # 2. Inicialización Xavier/Glorot para la capa de salida Softmax:
        # Desviación estándar = sqrt(2 / (n_oculta + n_salida))
        escala_xavier = np.sqrt(2.0 / (n_oculta + n_salida))
        self.W2 = rng.normal(loc=0.0, scale=escala_xavier, size=(n_oculta, n_salida)).astype(np.float32)
        self.b2 = np.zeros((1, n_salida), dtype=np.float32)

    @staticmethod
    def relu(Z: np.ndarray) -> np.ndarray:
        """
        Función de activación Rectified Linear Unit (ReLU).
        f(z) = max(0, z)
        """
        return np.maximum(0.0, Z)

    @staticmethod
    def relu_derivada(Z: np.ndarray) -> np.ndarray:
        """
        Derivada de la función ReLU con respecto a Z.
        f'(z) = 1 si z > 0, de lo contrario 0.
        """
        return (Z > 0.0).astype(np.float32)

    @staticmethod
    def softmax(Z: np.ndarray) -> np.ndarray:
        """
        Función Softmax numéricamente estable.
        Resta el valor máximo de cada fila antes de calcular la exponencial para evitar
        desbordamiento por arriba (overflow numérico).
        
        P(y = k | x) = exp(z_k - max(z)) / sum_j(exp(z_j - max(z)))
        """
        # Restar el máximo por fila (axis=1) para estabilidad
        Z_estabilizado = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_estabilizado)
        suma_exp = np.sum(exp_Z, axis=1, keepdims=True)
        return exp_Z / suma_exp

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Propagación hacia adelante (Forward Propagation).

        Args:
            X: Matriz de entrada de forma (m, n_entrada).

        Returns:
            Tuple:
                - A2: Probabilidades predichas de forma (m, n_salida).
                - cache: Diccionario con valores intermedios necesarios para backpropagation.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Capa oculta
        Z1 = np.dot(X, self.W1) + self.b1  # Forma: (m, 32)
        A1 = self.relu(Z1)                 # Forma: (m, 32)

        # Capa de salida
        Z2 = np.dot(A1, self.W2) + self.b2  # Forma: (m, 3)
        A2 = self.softmax(Z2)              # Forma: (m, 3)

        cache = {
            "X": X,
            "Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2,
        }
        return A2, cache

    @staticmethod
    def calcular_perdida(A2: np.ndarray, y: np.ndarray, epsilon: float = 1e-15) -> float:
        """
        Calcula la pérdida de Entropía Cruzada Categórica promedio del lote.
        
        L = - (1 / m) * sum_i( log(A2[i, y_i] + epsilon) )

        Args:
            A2: Matriz de probabilidades predichas de forma (m, n_salida).
            y: Vector de etiquetas verdaderas enteras de forma (m,).
            epsilon: Valor pequeño para evitar log(0).

        Returns:
            float: Valor escalar de la pérdida.
        """
        m = len(y)
        if m == 0:
            return 0.0

        # Limitar probabilidades para evitar log(0) o división por cero
        A2_recortado = np.clip(A2, epsilon, 1.0 - epsilon)
        
        # Extraer la probabilidad asignada a la clase correcta para cada muestra
        probabilidades_correctas = A2_recortado[np.arange(m), y]
        
        # Pérdida promedio
        perdida = -np.mean(np.log(probabilidades_correctas))
        return float(perdida)

    def backward(
        self,
        y: np.ndarray,
        cache: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Retropropagación del gradiente (Backpropagation).
        Aplica la regla de la cadena para calcular las derivadas parciales de la pérdida
        con respecto a los pesos y sesgos.

        Args:
            y: Vector de etiquetas reales enteras de forma (m,).
            cache: Valores intermedios calculados durante forward.

        Returns:
            Dict: Gradientes dW1, db1, dW2, db2.
        """
        m = len(y)
        X = cache["X"]
        Z1 = cache["Z1"]
        A1 = cache["A1"]
        A2 = cache["A2"]

        # 1. Gradiente de la salida con respecto a Z2:
        # La combinación de Softmax + Cross-Entropy se simplifica analíticamente a:
        # dZ2 = (A2 - Y_one_hot) / m
        dZ2 = A2.copy()
        dZ2[np.arange(m), y] -= 1.0
        dZ2 /= m  # Normalizar por el tamaño del lote

        # 2. Gradientes de los parámetros de la capa 2:
        # dW2 = A1^T · dZ2
        # db2 = sum_filas(dZ2)
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # 3. Propagación del error hacia la capa oculta:
        # dA1 = dZ2 · W2^T
        # dZ1 = dA1 ⊙ ReLU'(Z1)
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivada(Z1)

        # 4. Gradientes de los parámetros de la capa 1:
        # dW1 = X^T · dZ1
        # db1 = sum_filas(dZ1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        return {
            "dW1": dW1.astype(np.float32),
            "db1": db1.astype(np.float32),
            "dW2": dW2.astype(np.float32),
            "db2": db2.astype(np.float32),
        }

    def actualizar_parametros(self, gradientes: Dict[str, np.ndarray], tasa_aprendizaje: float) -> None:
        """
        Actualiza los parámetros mediante Descenso por Gradiente:
        theta = theta - eta * dTheta

        Args:
            gradientes: Diccionario con dW1, db1, dW2, db2.
            tasa_aprendizaje: Magnitud del paso (learning rate).
        """
        self.W1 -= tasa_aprendizaje * gradientes["dW1"]
        self.b1 -= tasa_aprendizaje * gradientes["db1"]
        self.W2 -= tasa_aprendizaje * gradientes["dW2"]
        self.b2 -= tasa_aprendizaje * gradientes["db2"]

    def paso_entrenamiento(
        self,
        X_lote: np.ndarray,
        y_lote: np.ndarray,
        tasa_aprendizaje: float,
    ) -> float:
        """
        Ejecuta un paso completo de entrenamiento sobre un lote:
        Forward -> Pérdida -> Backward -> Actualizar parámetros.

        Returns:
            float: Valor de pérdida en el lote antes de la actualización.
        """
        A2, cache = self.forward(X_lote)
        perdida = self.calcular_perdida(A2, y_lote)
        gradientes = self.backward(y_lote, cache)
        self.actualizar_parametros(gradientes, tasa_aprendizaje)
        return perdida

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna las probabilidades de clase predichas para cada muestra."""
        probabilidades, _ = self.forward(X)
        return probabilidades

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retorna la clase con mayor probabilidad predicha (índice 0, 1 o 2)."""
        probabilidades = self.predict_proba(X)
        return np.argmax(probabilidades, axis=1)

    def calcular_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calcula el porcentaje de aciertos (accuracy) en [0.0, 1.0]."""
        predicciones = self.predict(X)
        return float(np.mean(predicciones == y))

    def guardar_modelo(self, ruta_archivo: Union[str, Path]) -> None:
        """Guarda los pesos entrenados y arquitectura en un archivo .npz."""
        ruta_archivo = Path(ruta_archivo)
        ruta_archivo.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            ruta_archivo,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            n_entrada=self.n_entrada,
            n_oculta=self.n_oculta,
            n_salida=self.n_salida,
        )

    def cargar_modelo(self, ruta_archivo: Union[str, Path]) -> None:
        """Carga los pesos y arquitectura desde un archivo .npz previamente guardado."""
        ruta_archivo = Path(ruta_archivo)
        if not ruta_archivo.exists():
            raise FileNotFoundError(f"No se encontró el archivo de modelo en {ruta_archivo}")

        with np.load(ruta_archivo) as datos:
            self.W1 = datos["W1"].astype(np.float32)
            self.b1 = datos["b1"].astype(np.float32)
            self.W2 = datos["W2"].astype(np.float32)
            self.b2 = datos["b2"].astype(np.float32)
            self.n_entrada = int(datos["n_entrada"])
            self.n_oculta = int(datos["n_oculta"])
            self.n_salida = int(datos["n_salida"])
