from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import t, tstd
import math
import datasets
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def validatePredictions(predictions, correctValues): 
  correct = 0
  incorrect = 0
  for i in range(len(predictions)):
    if predictions[i] != correctValues[i]:
      incorrect += 1
    else:
      correct += 1
  #print("Correct: ", correct, "Incorrect: ", incorrect, "Accuracy: ", correct/(correct+incorrect))
  return correct/(correct+incorrect)

datasets = datasets.Datasets()
data, classes_data, numerical_classes, labels = datasets.read(dataset_name='forest_fires')

#print(data[:5])

# Carregando os dados em um DataFrame do pandas
data = pd.read_csv('./datasets/car_evaluation/car.data', sep=',')
data_t = pd.read_csv('./datasets/car_evaluation/cara.data', sep=',')
#print(data[:5])
categorical_boolean_columns = ['buying','maint','doors','persons','lug_boot','safety','class']

# Criando uma instância do LabelEncoder
label_encoder = LabelEncoder()

for column in categorical_boolean_columns:
    data_t[column] = label_encoder.fit_transform(data_t[column])

print(data_t[:5])
data = data.values.tolist()
data_t = data_t.values.tolist()

X = [[(j[:6])for j in i] for i in data]
y = [[(j[6])for j in i] for i in data]

y2 = y = data_t.columns.get_loc("class")
X2 = [[(j[:6])for j in i] for i in data_t]
#y2 = [[(j[6])for j in i] for i in data_t]
#y2 = [(i) for i in y]

print(y[:5])
accuracy = []
accuracy2 = []

rmse = []
neigh = KNeighborsClassifier(n_neighbors=1, weights="uniform")
for i in range(100):
    x_train,x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=5)
    x2_train,x2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.5, random_state=5)
    neigh.fit(x_train, y_train)
    predictions = neigh.predict(x_test)
    accuracy.append(validatePredictions(predictions, y_test))
    
    neigh.fit(x2_train, y2_train)
    predictions = neigh.predict(x2_test)
    accuracy2.append(validatePredictions(predictions, y2_test))

print(accuracy[0])

media = [sum(accuracy)/len(accuracy) ]

print(media)
scale = tstd(accuracy)
interval = [t.interval(.95, len(accuracy)-1, loc=media, scale=scale)]

print(interval)

media2 = [sum(accuracy2)/len(accuracy2) ]
scale2 = tstd(accuracy2)
interval2 = [t.interval(.95, len(accuracy2)-1, loc=media2, scale=scale2)]

print(interval2)
