from sklearn import tree
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import confusion_matrix

data = open("./datasets/wdbc/wdbc.data", "r")
data = data.read()
data = data.split("\n")
data = [i.split(",") for i in data]

parameter_amount = 32
index_of_class = 2
y = []  #y representa as classes
X = [] #X representa as caracteristicas

for i in data:
  y.append( [1 if j == 'M' else 0 for j in i[1:2]])
  X.append([float(j) for j in i[2:]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)
print(len(X_train), len(X_test), len(y_train), len(y_test))
n_random = random.sample(range(1, 10), 3)
for i in n_random:
  dt = tree.DecisionTreeClassifier(min_samples_leaf=i, )
  dt.fit(X_train, y_train)

  predictions = dt.predict(X_test)
  predictions_train = dt.predict(X_train)
  accuracy_train = accuracy_score(y_train, predictions_train)
  accuracy = accuracy_score(y_test, predictions)
  print(f' taxa de acerto do teste com folha {i}: {accuracy}')
  print(f' taxa de acerto do treino com folha {i}: {accuracy_train}')

print(confusion_matrix(y_test, predictions))
  

