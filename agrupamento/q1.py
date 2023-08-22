import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

attributes_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=attributes_names)
iris = ['iris-setosa', 'iris-versicolor', 'iris-virginica','outros']
df['class'] = df['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

X = df.iloc[:, 1:]
y = df['class']

#print(y)

predicts = []
distancia = []
desvio = []
for i in range (1,5):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init=100, max_iter=300)
    kmeans.fit(X)
    predicts.append(kmeans.predict(X))
    
    for j in range(1,10):
        distances = kmeans.transform(X).min(axis=1)
        distancia.append(np.mean(distances))
        desvio.append(np.std(distances))
        
for i in predicts:
    classes = np.zeros(4)
    for j in range(len(i)):
        classes[i[j]] += 1
    print(classes)
    plt.bar(iris,classes, color='blue')
    #print(plt.)
   
    plt.show()

print(distancia)
print(desvio)
print(np.mean(distancia), np.mean(desvio))

