import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import pandas as pd

attributes_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=attributes_names)
iris = ['iris-setosa', 'iris-versicolor', 'iris-virginica','outros']
df['class'] = df['class'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

X = df.iloc[:, 1:]
y = df['class']

#stratified holdout 50/50
classes = np.zeros(3)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
for train_index, test_index in sss.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    #kmedoids in train
    kmedoids = KMedoids(n_clusters=9).fit(X_train)

    #get centroids
    centroids = kmedoids.cluster_centers_
    # removing elements from train that are not centroids
    X_train = np.delete(X_train, kmedoids.medoid_indices_, axis=0)
    y_train = np.delete(y_train, kmedoids.medoid_indices_, axis=0)
    X_test = np.delete(X_test, kmedoids.medoid_indices_, axis=0)
    y_test = np.delete(y_test, kmedoids.medoid_indices_, axis=0)
    # using k-nn to predict the labels of the centroids
    print(len(X_train), len(y_train), len(centroids))
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    
    predicts = knn.predict(X_test)
    print(len(predicts), len(y_test))
    for j in range(len(predicts)):
        if predicts[j] == y_test[j]:
            classes[predicts[j]] += 1
        

print(classes)



    

