import random
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('rede_neural/spiral.csv')
df = pd.DataFrame(data)

print(df.head())
X = df.iloc[:, 0:2].values
y = df.iloc[:, 2].values
print(X[:2], y[:2])
acuracias = []

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random.randint(0, 100))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="sigmoid", input_shape=(2,)),
    tf.keras.layers.Dense(4, activation="sigmoid"),
    tf.keras.layers.Dense(4, activation="sigmoid"),
    tf.keras.layers.Dense(3, activation="sigmoid")
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.3)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

data = model.fit(X_train, y_train, epochs=5000, validation_data=(X_test, y_test), verbose=0)
train_loss = data.history['loss']
val_loss = data.history['val_loss']

epochs = range(0, len(train_loss), 100)
# Realize a predição
#predicoes = model.predict(X_test)

plt.plot(epochs, train_loss[::100], label='Loss do conjunto de treino')
plt.plot(epochs, val_loss[::100], label='Loss do conjunto de teste')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()