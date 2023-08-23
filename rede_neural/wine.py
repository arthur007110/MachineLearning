import tensorflow as tf
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

wine = load_wine()
X = wine.data
y = wine.target

# pre-processamento para tornar todos os atributos entre 0 e 1
min_values = [9999999 for _ in range(len(X[0]))]
max_values = [0 for _ in range(len(X[0]))]

for i in range(len(X)):
    for j in range(len(X[i])):
        if X[i][j] < min_values[j]:
            min_values[j] = X[i][j]
        if X[i][j] > max_values[j]:
            max_values[j] = X[i][j]

for i in range(len(X)):
    for j in range(len(X[i])):
        X[i][j] = (X[i][j] - min_values[j]) / (max_values[j] - min_values[j])

print(X[:2])

acuracias = []
for k in range(30):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=k)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation="relu", input_shape=(13,)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=500, verbose=0)

    # Realize a predição
    predicoes = model.predict(X_test)

    # Acurácia
    acuracia = model.evaluate(X_test, y_test)[1]
    acuracias.append(acuracia)

print("Acurácia média:", sum(acuracias) / len(acuracias))
print("Desvio padrão:", np.std(acuracias) * 100)
