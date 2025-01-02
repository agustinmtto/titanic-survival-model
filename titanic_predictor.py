# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Cargar el dataset de Titanic
df = pd.read_csv("/Users/agustinmaretto/Proyectos/Python Total/Dia 15/DataSet_Titanic.csv")

# Separar variables predictoras (X) y objetivo (y)
X = df.drop("Sobreviviente", axis=1)
y = df["Sobreviviente"]

# Crear y entrenar el modelo de Árbol de Decisión
arbol = DecisionTreeClassifier(max_depth=2, random_state=42)
arbol.fit(X, y)

# Evaluar el modelo
y_pred = arbol.predict(X)
print("Precisión:", accuracy_score(y, y_pred))

# Mostrar matriz de confusión
cm = confusion_matrix(y, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=arbol.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Visualizar el árbol de decisión
plt.figure(figsize=(10, 8))
plot_tree(arbol, filled=True, feature_names=X.columns, class_names=["No Sobrevive", "Sobrevive"])
plt.show()

# Graficar la importancia de las características
importancias = arbol.feature_importances_
sns.barplot(x=importancias, y=X.columns)
plt.title("Importancia de las Características")
plt.show()
