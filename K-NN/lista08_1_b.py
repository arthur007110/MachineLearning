import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
df = pd.read_csv(url, names=names)

def convert_to_categorical(value):
    if value <= 5.0:
        return 'low'
    elif value <= 6.0:
        return 'medium'
    else:
        return 'high'

df['sepal_length'] = df['sepal_length'].apply(convert_to_categorical)
df['sepal_width'] = df['sepal_width'].apply(convert_to_categorical)
df['petal_length'] = df['petal_length'].apply(convert_to_categorical)
df['petal_width'] = df['petal_width'].apply(convert_to_categorical)

def hamming_distance(x1, x2):
    return (x1 != x2).sum()

accuracies = []

# 100 repetições
for _ in range(100):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['class'], test_size=0.5, stratify=df['class'])

    X_train_temp = X_train.astype(str)
    X_test_temp = X_test.astype(str)

    dist_matrix = np.zeros((len(X_test), len(X_train)))
    for i, row_test in enumerate(X_test_temp.values):
        for j, row_train in enumerate(X_train_temp.values):
            dist_matrix[i, j] = hamming_distance(row_test, row_train)

    nearest_neighbors = np.argmin(dist_matrix, axis=1)
    
    y_pred = y_train.iloc[nearest_neighbors].values

    accuracy = accuracy_score(y_test, y_pred)

    accuracies.append(accuracy)

mean_accuracy = np.mean(accuracies)

confidence_interval = 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))

print("Acurácia Média:", mean_accuracy)
print("Intervalo de Confiança:", mean_accuracy - confidence_interval, "a", mean_accuracy + confidence_interval)
