# Análisis de Supervivencia del Titanic con Árboles de Decisión

Este proyecto emplea un modelo de árbol de decisión para analizar los factores que influyeron en la supervivencia de los pasajeros del Titanic. Desarrollado con fines educativos, sirve como una práctica introductoria a Machine Learning. El objetivo principal es demostrar cómo los árboles de decisión pueden ser utilizados para analizar datos y realizar predicciones.


## Descripción

El script realiza las siguientes acciones:

1.  **Importa bibliotecas necesarias:**
    *   `pandas`: Para manipulación de datos.
    *   `numpy`: Para operaciones numéricas.
    *   `matplotlib.pyplot`: Para visualizaciones gráficas.
    *   `seaborn`: Para visualizaciones estadísticas.
    *   `sklearn.tree`: Para el modelo de árbol de decisión y su visualización.
    *   `sklearn.metrics`: Para evaluar el modelo.

2.  **Carga el dataset:** Lee el archivo `DataSet_Titanic.csv` en un DataFrame de pandas.
3.  **Prepara los datos:**
    *   Define la variable objetivo (`y`, que indica la supervivencia) y las variables predictoras (`X`, que incluyen características como edad, sexo, clase, etc.).
4.  **Crea y entrena el modelo de Árbol de Decisión:**
    *   Crea un modelo de `DecisionTreeClassifier` con una profundidad máxima de 2 niveles y un estado aleatorio para la reproducibilidad.
    *   Entrena el modelo utilizando los datos de entrenamiento.
5.  **Evalúa el modelo:**
    *   Realiza predicciones con el modelo.
    *   Calcula y muestra la precisión del modelo utilizando `accuracy_score`.
6.  **Muestra la matriz de confusión:**
    *   Calcula y normaliza la matriz de confusión para evaluar el rendimiento del modelo por clases.
    *   Visualiza la matriz de confusión usando `ConfusionMatrixDisplay`.
7.  **Visualiza el árbol de decisión:**
    *   Grafica el árbol de decisión para entender cómo el modelo toma decisiones, utilizando `plot_tree`.
8.  **Grafica la importancia de las características:**
    *   Obtiene la importancia de cada característica del modelo.
    *   Crea un gráfico de barras para visualizar qué características son más importantes para el modelo.

## Requisitos

*   Python 3.6 o superior.
*   Las siguientes bibliotecas de Python:
    *   `pandas`
    *   `numpy`
    *   `matplotlib`
    *   `seaborn`
    *   `scikit-learn`
    *   Puedes instalarlas usando pip:
        ```bash
        pip3 install pandas numpy matplotlib seaborn scikit-learn
        ```

## Instalación y Uso

1.  Asegúrate de tener Python y pip instalados.
2.  Clona o descarga este repositorio.
3.  Instala las dependencias con `pip3 install -r requirements.txt` (opcional, si creas un archivo `requirements.txt`).
4.  Asegúrate de tener el archivo `DataSet_Titanic.csv` y cambia la ruta del script para que sea correcta.
5.  Ejecuta el script `titanic_predictor.py`:
    ```bash
    python3 titanic_predictor.py
    ```

## Salida del Script

El script produce las siguientes salidas:

*   **Precisión del modelo:** Un valor numérico que indica qué tan bien predice el modelo en el dataset de entrenamiento.
*   **Matriz de Confusión:** Una gráfica que muestra la distribución de verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.
*   **Visualización del Árbol de Decisión:** Un gráfico del árbol de decisión que muestra las decisiones tomadas por el modelo.
*   **Importancia de las características:** Un gráfico de barras que muestra qué variables fueron más importantes en la decisión del modelo.

## Consideraciones

*   **Dataset:** Asegúrate de que el archivo `DataSet_Titanic.csv` esté correctamente ubicado y tenga el formato esperado.
*   **Profundidad del árbol:** El modelo utiliza una profundidad máxima de 2 niveles. Ajusta este parámetro para explorar diferentes niveles de complejidad y cómo afecta el rendimiento del modelo.
*   **Interpretación:** El modelo permite una fácil interpretación de las decisiones, lo que es útil para entender el proceso de toma de decisión.
*   **Overfitting:** Un árbol de decisión demasiado profundo puede sobreajustar los datos de entrenamiento y generalizar mal en nuevos datos.
*   **Preprocesamiento de datos:** Este script asume que el dataset ya está preprocesado, así que es posible que antes debas limpiarlo y prepararlo.
