import numpy as np
import pandas as pd
from scipy.stats import t, tstd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

def run_experiment(X, y):
    accuracies = []
    for _ in range(100):
        X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(X, y, test_size=0.5, stratify=y)
    
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(X_train_exp, y_train_exp)
    
        y_pred_exp = classifier.predict(X_test_exp)
        accuracy = accuracy_score(y_test_exp, y_pred_exp)
    
        accuracies.append(accuracy)
    media = np.mean(accuracies)
    scale = tstd(accuracies)
    interval = [t.interval(.95, len(accuracies)-1, loc=media, scale=scale)]
    return (media, interval)

data = open("wine/wine.data", "r")
data = data.read()
data = data.split("\n")
data = data[:-1]
data = [i.split(",") for i in data]

X = []
y = []
for i in data:
    X.append([float(j) for j in i[1:]])
    y.append(i[0])

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

mean_accuracy_experiment, confidence_interval_experiment  = run_experiment(X, y)
print("Acurácia Média:", mean_accuracy_experiment)
print("Intervalo de Confiança:", confidence_interval_experiment)
