from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

def convert_to_categorical(value):
    if value <= 5.0:
        return 'low'
    elif value <= 6.0:
        return 'medium'
    else:
        return 'high'

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['class'] = iris.target

df['sepal length (cm)'] = df['sepal length (cm)'].apply(convert_to_categorical)
df['sepal width (cm)'] = df['sepal width (cm)'].apply(convert_to_categorical)
df['petal length (cm)'] = df['petal length (cm)'].apply(convert_to_categorical)
df['petal width (cm)'] = df['petal width (cm)'].apply(convert_to_categorical)
def hamming_distance(x1, x2):
    return (x1 != x2).sum()

accuracies_hamming = []

# 100 repetições
for _ in range(100):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['class'], test_size=0.5, stratify=df['class'])

    dist_matrix = np.zeros((len(X_test), len(X_train)))
    for i, row_test in enumerate(X_test.values):
        for j, row_train in enumerate(X_train.values):
            dist_matrix[i, j] = hamming_distance(row_test, row_train)

    nearest_neighbors = np.argmin(dist_matrix, axis=1)
    
    y_pred = y_train.iloc[nearest_neighbors].values

    accuracy = accuracy_score(y_test, y_pred)

    accuracies_hamming.append(accuracy)

mean_accuracy_hamming = np.mean(accuracies_hamming)

confidence_interval_hamming = 1.96 * np.std(accuracies_hamming) / np.sqrt(len(accuracies_hamming))

print("Acurácia Média 1-NN (Distância de Hamming):", mean_accuracy_hamming)
print("Intervalo de Confiança 1-NN (Distância de Hamming):", mean_accuracy_hamming - confidence_interval_hamming, "a", mean_accuracy_hamming + confidence_interval_hamming)

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.5, stratify=iris.target)

classifier_euclidean = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
classifier_euclidean.fit(X_train, y_train)

y_pred_euclidean = classifier_euclidean.predict(X_test)
accuracy_euclidean = accuracy_score(y_test, y_pred_euclidean)

print("Acurácia 1-NN (Distância Euclidiana) na base Iris original:", accuracy_euclidean)