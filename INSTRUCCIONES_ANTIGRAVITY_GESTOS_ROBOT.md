# Proyecto: Entrenamiento desde cero de una red neuronal para reconocer gestos de mano y controlar un robot

## 1. Instrucciones generales para Google Antigravity

Actúa como asistente de desarrollo para construir este proyecto por
etapas. Antes de escribir código, revisa estas instrucciones completas y
crea un plan de implementación. Después trabaja por fases, ejecutando
pruebas reales y explicando brevemente qué se hizo, qué archivos se
crearon y cómo comprobarlo.

**Objetivo académico principal:** entrenar una red neuronal pequeña
implementada por nosotros mismos, para poder explicar al profesor cómo
aprende. No se permite resolver la clasificación usando un modelo
preentrenado ni delegando el entrenamiento a TensorFlow, Keras, PyTorch,
scikit-learn u otra biblioteca de aprendizaje automático de alto nivel.

### Reglas obligatorias

1.  Implementar la red neuronal y su entrenamiento manualmente usando
    **Python + NumPy** para operaciones matriciales.
2.  No usar modelos preentrenados, transfer learning, AutoML ni
    servicios de IA externos.
3.  No usar TensorFlow, Keras, PyTorch, scikit-learn ni OpenCV DNN para
    entrenar o clasificar.
4.  Para cargar, redimensionar y convertir imágenes se puede usar
    **Pillow (PIL)**. Para gráficas se puede usar Matplotlib. Estas
    bibliotecas no deben entrenar el modelo.
5.  Mantener el código modular, comentado en español y apropiado para
    estudiantes que necesitan comprenderlo y defenderlo.
6.  No fingir resultados. Ejecutar las pruebas y reportar las métricas
    reales obtenidas.
7.  No descargar ni instalar dependencias sin comprobar primero si son
    necesarias.
8.  No solicitar al usuario contraseñas, cookies, claves privadas ni
    tokens. Nunca incluir credenciales en el repositorio.
9.  No descargar todo el dataset repetidamente. Comprobar primero si ya
    existe.
10. Si una decisión importante no está definida, escoger la alternativa
    más sencilla que respete estas instrucciones y documentarla.
11. Trabajar por fases. No saltar a webcam o control físico hasta que el
    entrenamiento y las pruebas básicas funcionen.
12. Crear un `README.md` con instrucciones para instalar, preparar los
    datos, entrenar, evaluar y ejecutar la demostración.
13. Crear un archivo `requirements.txt` con dependencias mínimas y, si
    procede, un `.gitignore`.
14. Guardar semillas aleatorias para que las pruebas sean reproducibles.
15. Evitar imprimir o guardar todas las matrices de pesos en la consola;
    guardar los pesos entrenados en un archivo.
16. No subir el dataset ni pesos grandes a Git por defecto.

## 2. Descripción y alcance del proyecto

El proyecto consiste en crear un sistema capaz de reconocer gestos de
una mano y, en una fase posterior, utilizar las predicciones para
controlar un robot.

Se trabajará en dos etapas distintas:

### Etapa A: prueba de concepto con Rock, Paper, Scissors

Usaremos un dataset público de imágenes de los gestos:

-   `rock` (piedra)
-   `paper` (papel)
-   `scissors` (tijeras)

Esta etapa sirve para verificar que nuestra red neuronal puede aprender
a clasificar imágenes. **No significa que piedra, papel y tijeras sean
todavía los comandos finales del robot.**

### Etapa B: gestos propios para el robot

Cuando la red funcione con el dataset de prueba, se diseñará una segunda
versión del dataset con gestos que representen comandos del robot, por
ejemplo:

-   `STOP`: detenerse.
-   `FORWARD`: avanzar.
-   `LEFT`: girar a la izquierda.
-   `RIGHT`: girar a la derecha.
-   `BACKWARD`: retroceder.

Los gestos concretos deberán definirse claramente antes de recopilar las
imágenes. No asumir que el modelo de piedra, papel y tijeras ya puede
reconocer esos cinco comandos: será necesario reunir imágenes
etiquetadas de esos gestos y volver a entrenar.

La integración con webcam y robot físico es una fase posterior. Primero
se demostrará la clasificación de imágenes estáticas.

## 3. Dataset inicial recomendado

### Dataset seleccionado

**Rock-Paper-Scissors Images**, de drgfreeman, publicado en Kaggle y
asociado al repositorio de datos de su proyecto de visión artificial.

-   Página de Kaggle:
    https://www.kaggle.com/drgfreeman/rockpaperscissors/data
-   Repositorio relacionado:
    https://github.com/DrGFreeman/rps-cv-data-science
-   Descarga ZIP indicada en el repositorio del proyecto:
    https://www.dropbox.com/s/sr1iui3zfdk0o53/rps-cv-image.zip
-   Licencia declarada: Creative Commons Attribution-ShareAlike 4.0 (CC
    BY-SA 4.0).
-   Tamaño reportado: 2,188 imágenes aproximadamente.
-   Clases y cantidades reportadas: `rock` 726, `paper` 710 y `scissors`
    752.
-   Las imágenes originales son RGB y aproximadamente de 200 × 300
    píxeles.
-   Las imágenes se encuentran organizadas en carpetas por clase
    (`rock`, `paper`, `scissors`) en la distribución habitual del
    dataset.

Consulta la página del dataset y la licencia antes de redistribuirlo o
publicar derivados. Si se utiliza públicamente, conservar atribución y
cumplir los términos de compartir bajo la misma licencia cuando
corresponda.

### Método de descarga recomendado: ZIP directo

Para este proyecto se recomienda intentar primero la descarga del ZIP
público enlazado en el repositorio relacionado. Es la ruta más simple
para un entorno local y evita depender inicialmente de una cuenta o API
token de Kaggle.

Implementa un pequeño procedimiento seguro de preparación que:

1.  Compruebe si el dataset ya está descargado y extraído.
2.  Cree la carpeta `data/raw/` si no existe.
3.  Descargue el ZIP a `data/downloads/` usando Python estándar o
    `curl`, si está disponible.
4.  Compruebe que la descarga terminó correctamente y que el archivo ZIP
    se puede abrir.
5.  Extraiga el archivo sin sobrescribir datos válidos innecesariamente.
6.  Detecte la estructura real de carpetas después de extraerlo.
7.  Localice las carpetas de las tres clases incluso si vienen dentro de
    una carpeta contenedora.
8.  Verifique cuántas imágenes hay por clase y muestre un resumen.
9.  No borre el ZIP automáticamente hasta comprobar que la extracción
    está completa.
10. Si la URL directa falla, documente el error y presente la
    alternativa de descarga con Kaggle CLI.

**No uses MCP ni microissant para descargar este dataset.** No son
necesarios para este flujo. No configures servicios externos o MCP solo
por usarlos.

### Alternativa: Kaggle CLI

Solo si la descarga ZIP directa falla, preparar instrucciones opcionales
para descargarlo con la CLI oficial de Kaggle.

-   Documentación de Kaggle CLI: https://github.com/Kaggle/kaggle-cli
-   Identificador de dataset: `drgfreeman/rockpaperscissors`

El comando habitual es:

``` bash
kaggle datasets download -d drgfreeman/rockpaperscissors -p data/downloads
```

Después, extraer el ZIP en `data/raw/` una vez validado. La opción
`--unzip` está disponible en la CLI, pero conviene conservar el ZIP
durante la primera comprobación.

La CLI puede requerir configurar credenciales de Kaggle localmente. Si
hace falta, muestra al usuario los pasos oficiales para configurarlas,
pero **nunca le pidas que pegue sus credenciales en el chat ni las
escribas en código**.

### Nota sobre nombres y estructura

No asumir que el ZIP contiene una única estructura exacta. Inspeccionar
el contenido real y adaptar el descubrimiento de carpetas. Si no se
encuentran inequívocamente las tres clases, detenerse y mostrar el árbol
de carpetas detectado en vez de continuar con etiquetas incorrectas.

## 4. Arquitectura de la red neuronal

Implementar una red neuronal pequeña desde cero con esta arquitectura
inicial:

-   Entrada: imagen en escala de grises de **28 × 28 píxeles**.
-   Aplanamiento: 28 × 28 = **784 características**.
-   Capa oculta: **32 neuronas** con activación ReLU.
-   Capa de salida: **3 neuronas**, una por clase.
-   Activación de salida: Softmax para obtener probabilidades.
-   Función de pérdida: entropía cruzada (cross-entropy).
-   Entrenamiento: retropropagación (backpropagation) y descenso por
    gradiente (gradient descent).
-   Inicialización: pesos aleatorios reproducibles, preferentemente He
    para la capa ReLU y una escala razonable para la salida.
-   Optimizador inicial: descenso por gradiente por mini-lotes
    (mini-batch gradient descent), implementado manualmente.

Dimensiones esperadas:

-   `X`: `(n_muestras, 784)`.
-   `W1`: `(784, 32)`.
-   `b1`: `(1, 32)`.
-   `W2`: `(32, 3)`.
-   `b2`: `(1, 3)`.
-   `Y`: etiquetas enteras de longitud `n_muestras` o codificación
    one-hot generada por nuestro código.

Usar `float32` cuando sea conveniente para ahorrar memoria, sin
sacrificar estabilidad numérica. En Softmax, restar el máximo de cada
fila antes de aplicar exponencial para reducir problemas de
desbordamiento. Añadir un epsilon pequeño al logaritmo de la pérdida.

### Prohibiciones y límites técnicos

-   No llamar a una API de entrenamiento de modelos.
-   No utilizar funciones de entrenamiento de una biblioteca de ML.
-   No copiar una red ya entrenada.
-   No usar los pesos de otro modelo.
-   Sí se permite usar NumPy para multiplicación matricial, sumas,
    exponenciales, máximos, índices y operaciones matemáticas
    elementales.
-   Sí se permite Pillow para abrir, convertir y redimensionar imágenes.
-   Sí se permite Matplotlib para visualizar ejemplos y curvas de
    aprendizaje.

## 5. Preprocesamiento de imágenes

Crear un módulo que prepare los datos de forma reproducible:

1.  Abrir cada imagen válida con Pillow.
2.  Convertirla a escala de grises (`L`).
3.  Redimensionarla a 28 × 28 píxeles usando un método de interpolación
    apropiado.
4.  Convertirla a un arreglo NumPy.
5.  Convertir los valores de píxel de `[0, 255]` a `[0, 1]`, dividiendo
    entre 255.
6.  Aplanar la imagen a un vector de 784 valores.
7.  Asignar una etiqueta consistente a cada clase:
    -   `rock` → 0
    -   `paper` → 1
    -   `scissors` → 2
8.  Ignorar archivos que no sean imágenes y reportar imágenes dañadas o
    ilegibles sin detener todo el proceso.
9.  Guardar la correspondencia entre número y clase en un archivo o en
    una constante bien documentada.
10. Mezclar los datos con una semilla fija.
11. Separar los datos en entrenamiento, validación y prueba,
    aproximadamente:
    -   70 % entrenamiento
    -   15 % validación
    -   15 % prueba
12. Hacer la separación de forma estratificada, si es posible, para
    mantener proporciones similares por clase en cada partición.
13. Evitar duplicados exactos entre particiones cuando sea práctico.
14. No usar el conjunto de prueba para elegir hiperparámetros ni para
    tomar decisiones durante el entrenamiento.

Si el formato de datos guardado incluye arreglos NumPy, utilizar
formatos seguros y documentar sus dimensiones. No cargar archivos pickle
de fuentes desconocidas.

## 6. Plan de desarrollo por fases

### Fase 0 --- Inspección del entorno y planificación

-   Revisar la carpeta del proyecto y el sistema operativo.
-   Comprobar la versión de Python disponible.
-   Crear un plan corto y mostrarlo al usuario.
-   No asumir que hay GPU disponible; el modelo debe funcionar en CPU.
-   Crear la estructura inicial del proyecto.
-   Crear un entorno virtual local (`.venv`) si es apropiado.
-   No ejecutar comandos destructivos ni borrar archivos existentes.

### Fase 1 --- Descargar y verificar el dataset

Crear un script, por ejemplo `scripts/download_dataset.py`, que
automatice la descarga ZIP recomendada, la validación, la extracción y
la detección de carpetas.

Resultados requeridos: - Archivo descargado o explicación clara de por
qué no se pudo descargar. - Dataset extraído. - Conteo de imágenes por
clase. - Resumen de errores. - Instrucciones de alternativa Kaggle CLI
si la descarga directa falla.

### Fase 2 --- Explorar el dataset

Crear `scripts/explore_dataset.py` que: - Muestre el total de imágenes
por clase. - Muestre ejemplos de imágenes con sus etiquetas. -
Identifique archivos inválidos. - Muestre el tamaño y modo de color de
algunas imágenes. - Guarde, si es posible, una imagen tipo mosaico para
inspección. - No modifique las imágenes originales.

### Fase 3 --- Preparar los datos

Crear `src/data/preprocess.py` y/o `scripts/prepare_dataset.py` para: -
Convertir a escala de grises. - Redimensionar a 28 × 28. - Normalizar. -
Aplanar. - Codificar etiquetas. - Separar entrenamiento, validación y
prueba. - Guardar los arreglos procesados en `data/processed/`. -
Imprimir las dimensiones de `X_train`, `X_val`, `X_test` y sus
etiquetas.

El script debe poder ejecutarse más de una vez sin crear datos
inconsistentes. Documentar si los arreglos procesados se regeneran o se
reutilizan.

### Fase 4 --- Implementar la red desde cero

Crear `src/model/neural_network.py`.

Implementar manualmente y comentar: - Inicialización de pesos y
sesgos. - Propagación hacia adelante (forward propagation). - ReLU. -
Softmax numéricamente estable. - Entropía cruzada. - Retropropagación de
gradientes. - Actualización de parámetros por descenso de gradiente. -
Predicción de una imagen o lote. - Cálculo de accuracy. - Guardado y
carga de parámetros entrenados.

Separar claramente el código de la red del código de lectura de
imágenes. Las operaciones deben funcionar con lotes de imágenes, no solo
con una imagen a la vez.

### Fase 5 --- Entrenar

Crear `src/train.py` o `scripts/train.py`.

Valores iniciales razonables: - 20 a 50 épocas como primer
experimento. - Tamaño de lote: 32. - Tasa de aprendizaje inicial: 0.01,
ajustable según estabilidad. - Semilla aleatoria fija. - Usar
entrenamiento por mini-lotes. - Calcular y registrar pérdida y accuracy
de entrenamiento y validación al final de cada época. - Barajar solo el
conjunto de entrenamiento en cada época. - Guardar el mejor modelo según
la pérdida de validación. - Evitar usar el conjunto de prueba para
elegir el mejor modelo. - No prometer una exactitud mínima; medir y
reportar el resultado real.

Si la pérdida se vuelve `NaN`, los gradientes explotan o el
entrenamiento no mejora, diagnosticar la causa y ajustar de forma
justificada la tasa de aprendizaje, inicialización o implementación
matemática. No ocultar el problema.

Guardar: - Parámetros del modelo entrenado, por ejemplo
`models/rps_model.npz`. - Historial de entrenamiento, por ejemplo
`outputs/training_history.csv`. - Gráfica de pérdida y accuracy, por
ejemplo `outputs/training_curves.png`.

### Fase 6 --- Evaluar

Crear `src/evaluate.py` o `scripts/evaluate.py`.

Evaluar una sola vez el mejor modelo con el conjunto de prueba separado.
Reportar: - Accuracy global. - Número de imágenes de prueba. - Accuracy
o aciertos por clase. - Matriz de confusión calculada con NumPy. -
Ejemplos de predicciones correctas e incorrectas. - Tiempo de
entrenamiento aproximado y parámetros principales usados.

No utilizar la exactitud como única medida. Mostrar dónde se confunde el
modelo.

No usar scikit-learn para generar métricas: implementar las métricas
necesarias con NumPy para mantener el enfoque desde cero.

### Fase 7 --- Probar una imagen nueva

Crear un script, por ejemplo `src/predict_image.py`, que reciba la ruta
de una imagen:

``` bash
python -m src.predict_image ruta/a/una_imagen.jpg
```

Debe: - Cargar los pesos guardados. - Aplicar exactamente el mismo
preprocesamiento que durante el entrenamiento. - Mostrar la clase
predicha. - Mostrar las probabilidades de las tres clases. - Aclarar que
una predicción de imagen aislada no garantiza que funcione en
condiciones reales.

### Fase 8 --- Demostración con webcam (opcional y posterior)

Solo cuando las fases anteriores estén completas: - Evaluar si instalar
OpenCV es necesario para capturar fotogramas de la webcam. - Si se usa
OpenCV, usarlo únicamente para captura de cámara y visualización; no
usar sus modelos DNN ni clasificadores preentrenados. - Preprocesar cada
fotograma exactamente igual que las imágenes de entrenamiento. - Mostrar
la predicción y las probabilidades. - No afirmar que la clasificación
será fiable con cualquier fondo, iluminación o distancia. - Añadir un
mecanismo sencillo para cerrar la aplicación y manejar la cámara no
disponible.

La red entrenada con Rock-Paper-Scissors solo reconocerá las clases para
las que fue entrenada. No interpretar automáticamente `rock`, `paper` y
`scissors` como comandos de movimiento del robot sin definirlo
explícitamente.

### Fase 9 --- Dataset de comandos del robot

Diseñar un procedimiento para crear un dataset nuevo con las clases
`STOP`, `FORWARD`, `LEFT`, `RIGHT` y `BACKWARD`.

-   Definir un gesto visual distinto para cada comando.
-   Recopilar imágenes con variaciones razonables de personas,
    iluminación, fondo, distancia y orientación.
-   Usar consentimiento de las personas fotografiadas.
-   Guardar imágenes en carpetas separadas por etiqueta.
-   Mantener conjuntos de entrenamiento, validación y prueba separados.
    Cuando sea posible, separar por persona o sesión para evaluar mejor
    la generalización.
-   No reutilizar las etiquetas RPS para representar gestos distintos.
-   Reentrenar la misma arquitectura o documentar cambios justificados.
-   Evaluar los cinco comandos antes de conectarlos al robot.

El objetivo inicial es aprender el procedimiento; no es obligatorio
tener una gran cantidad de imágenes propias para la primera demostración
RPS. Para la versión del robot se necesitará un dataset que represente
los gestos definidos.

### Fase 10 --- Integración con el robot (futura)

No implementar esta fase hasta que el usuario lo solicite y el
clasificador de gestos propios haya sido evaluado.

En una integración futura: - Separar la clasificación de imagen de la
comunicación con el robot. - Definir el protocolo de comunicación (por
ejemplo, serial o Wi-Fi) según el hardware disponible. - Añadir una zona
de seguridad: si la confianza es baja, no enviar un comando de
movimiento. - Exigir varias predicciones consecutivas consistentes antes
de ejecutar un comando, si procede. - Incluir un comando de parada
segura y un mecanismo físico de apagado. - No mover el robot durante
pruebas automáticas de entrenamiento. - No asumir qué microcontrolador,
motores o protocolo se usarán; pedir esos detalles antes de escribir la
integración específica.

## 7. Estructura de carpetas sugerida

Usar una estructura similar a esta y ajustarla solo si existe una razón
clara:

``` text
gesture-robot-ai/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── downloads/
│   ├── raw/
│   └── processed/
├── models/
├── outputs/
│   ├── figures/
│   └── logs/
├── scripts/
│   ├── download_dataset.py
│   ├── explore_dataset.py
│   └── prepare_dataset.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocess.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── neural_network.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict_image.py
└── tests/
    ├── test_preprocessing.py
    └── test_neural_network.py
```

Crear carpetas vacías solo cuando se necesiten. No versionar el
contenido descargado de `data/raw/`, datos procesados grandes, entornos
virtuales ni modelos generados, salvo que el usuario lo pida.

## 8. Dependencias iniciales

Mantener dependencias mínimas:

-   `numpy`: operaciones matemáticas y matriz de la red.
-   `Pillow`: lectura y preprocesamiento de imágenes.
-   `matplotlib`: gráficas de entrenamiento y visualización de ejemplos.

No instalar inicialmente TensorFlow, Keras, PyTorch, scikit-learn ni
Jupyter salvo petición explícita y justificada. `pytest` puede añadirse
para pruebas automáticas. OpenCV se considera opcional y solo se
incorpora en la fase de webcam.

Crear `requirements.txt` con las dependencias efectivamente utilizadas y
versiones compatibles. No fijar versiones arbitrarias sin comprobar el
entorno.

## 9. Pruebas y validaciones obligatorias

Crear pruebas que comprueben como mínimo:

1.  El preprocesamiento devuelve vectores de 784 valores.
2.  Los valores de píxel normalizados están entre 0 y 1.
3.  Las etiquetas solo corresponden a las tres clases esperadas.
4.  Las dimensiones de pesos y sesgos son correctas.
5.  Softmax devuelve valores finitos y las probabilidades de cada fila
    suman aproximadamente 1.
6.  La pérdida es finita para datos de prueba válidos.
7.  El entrenamiento modifica los pesos.
8.  Una pasada de entrenamiento sobre un conjunto pequeño no produce
    errores de dimensiones.
9.  Guardar y volver a cargar el modelo conserva las predicciones,
    dentro de una tolerancia numérica razonable.
10. Un archivo de imagen inválido se maneja de forma controlada.
11. Las particiones no se solapan por índices y mantienen proporciones
    razonables por clase.

Cuando sea viable, realizar una prueba de gradiente numérico en un
modelo diminuto para comprobar que la retropropagación está implementada
correctamente. Documentar sus límites.

## 10. Explicación académica que debe incluirse

En el `README.md` o en `docs/explicacion_tecnica.md`, explicar con
lenguaje claro:

-   Qué es una neurona artificial.
-   Qué representan los 784 valores de entrada.
-   Qué son los pesos y los sesgos.
-   Qué hace la capa oculta de 32 neuronas.
-   Para qué sirve ReLU.
-   Por qué Softmax produce probabilidades.
-   Qué mide la función de pérdida.
-   Qué significa propagación hacia adelante.
-   Cómo se calcula el error.
-   Qué hace la retropropagación.
-   Cómo el descenso por gradiente actualiza los pesos.
-   Qué son una época y un mini-lote.
-   Diferencia entre entrenamiento, validación y prueba.
-   Qué es sobreajuste (overfitting) y cómo se detecta.
-   Limitaciones de una imagen de 28 × 28 en escala de grises.
-   Por qué un buen resultado en RPS no demuestra por sí solo que
    funcione con los comandos del robot.
-   Por qué NumPy no es una biblioteca de entrenamiento de modelos de
    alto nivel: aquí se usa para cálculos numéricos, mientras que la
    lógica de aprendizaje se implementa en nuestro código.

Añadir ecuaciones legibles de ReLU, Softmax, entropía cruzada y
actualización de parámetros. Mantener la explicación consistente con el
código real.

## 11. Resultados y presentación del avance

Crear `docs/avance_proyecto.md` con una plantilla que se actualice al
terminar cada fase. Incluir: - Objetivo. - Dataset y licencia. -
Preprocesamiento aplicado. - Arquitectura implementada. -
Hiperparámetros usados. - Resultados reales de entrenamiento y
evaluación. - Gráficas y matriz de confusión. - Problemas encontrados y
cómo se resolvieron. - Limitaciones actuales. - Siguientes pasos. - Qué
parte del aprendizaje está implementada manualmente.

No inventar resultados ni escribir que una fase está terminada si no se
ejecutó. Marcar claramente lo pendiente.

## 12. Privacidad, licencias y buenas prácticas

-   Mantener una nota de atribución del dataset y su licencia CC BY-SA
    4.0.
-   Revisar la licencia antes de redistribuir imágenes o publicar un
    dataset derivado.
-   No subir credenciales ni archivos de configuración privados.
-   Añadir al `.gitignore` `/.venv/`, `__pycache__/`, datos
    descargados/procesados y archivos de modelo generados, según
    convenga.
-   No recopilar fotografías de personas sin su consentimiento.
-   Si se comparte el proyecto, explicar de dónde provienen los datos y
    qué restricciones de licencia aplican.

## 13. Orden exacto de trabajo solicitado a Antigravity

Proceder en este orden:

1.  Inspeccionar el entorno y presentar un plan.
2.  Crear la estructura de carpetas, `AGENTS.md`, `README.md`,
    `.gitignore` y dependencias mínimas.
3.  Implementar y ejecutar la descarga del ZIP; verificar la estructura
    y conteos.
4.  Explorar el dataset y mostrar ejemplos.
5.  Implementar el preprocesamiento y comprobar dimensiones.
6.  Implementar la red neuronal desde cero con NumPy.
7.  Añadir pruebas unitarias de las operaciones matemáticas.
8.  Ejecutar un entrenamiento real de prueba con un subconjunto pequeño
    para comprobar el flujo completo.
9.  Si la prueba funciona, entrenar con el dataset preparado.
10. Evaluar en el conjunto de prueba y guardar métricas/gráficas.
11. Probar predicciones sobre imágenes independientes.
12. Documentar los resultados reales y las limitaciones.
13. Detenerse y presentar un resumen antes de comenzar la webcam,
    dataset propio de comandos o integración física.

Después de cada fase, informar: - Qué se creó o cambió. - Qué comandos
se ejecutaron. - Qué pruebas pasaron o fallaron. - Qué resultados reales
se observaron. - Qué falta para la siguiente fase.

**Prioridad absoluta:** que el equipo comprenda y pueda explicar el
entrenamiento de una red neuronal pequeña implementada desde cero. Es
preferible una implementación sencilla, correcta y comprobable a una
solución compleja que oculte cómo aprende el modelo.
