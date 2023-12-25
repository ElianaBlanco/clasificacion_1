import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Eliminamos la columna categoria_edad
df_sin_categoria_edad = df.drop(columns=['categoria_edad'])

# Graficamos la distribución de clases
plt.hist(df_sin_categoria_edad['DEATH_EVENT'], bins=2)
plt.xlabel('Clases')
plt.ylabel('Frecuencia')
plt.title('Distribución de Clases')
plt.show()

# Definimos X sin la columna DEATH_EVENT y y como la columna DEATH_EVENT
X = df_sin_categoria_edad.drop(columns=['DEATH_EVENT'])
y = df_sin_categoria_edad['DEATH_EVENT']

# Partición estratificada del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Ajustamos un árbol de decisión
modelo_arbol = DecisionTreeClassifier(max_depth=5, min_samples_leaf=3)
modelo_arbol.fit(X_train, y_train)

# Realizamos las predicciones sobre el conjunto de test
predicciones = modelo_arbol.predict(X_test)

# Calculamos el accuracy sobre el conjunto de test
accuracy = accuracy_score(y_test, predicciones)
print(f'Accuracy del árbol de decisión: {accuracy}')
