import random
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('rede_neural/spiral.csv')
df = pd.DataFrame(data)

print(df.head())
X = df.iloc[:, 0:2].values
y = df.iloc[:, 2].values
print(X[:2], y[:2])
acuracias = []

for k in range(30):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=k+random.randint(0, 100))

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation="sigmoid", input_shape=(2,)),
        tf.keras.layers.Dense(4, activation="sigmoid"),
        tf.keras.layers.Dense(0, activation="sigmoid"),
        tf.keras.layers.Dense(3, activation="sigmoid")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.3)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=500, verbose=0)

    # Realize a predição
    #predicoes = model.predict(X_test)

    # Acurácia
    acuracia = model.evaluate(X_test, y_test)[1]
    acuracias.append(acuracia)

print("Acurácia média:", sum(acuracias) / len(acuracias))
print("Desvio padrão:", np.std(acuracias))
