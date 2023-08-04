from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, mean_squared_error
from sklearn import svm
from scipy.stats import t, tstd
import math, random
import pandas as pd
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

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
print(raw_df[:1])
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

parameter_amount = 13
index_of_class = 0
y = []  #y representa as classes
X = [] #X representa as caracteristicas

for i in data:
  X.append([float(j) for j in i])


intervals = list(range(int(min(target)),int(max(target)), 10))
print(intervals)

for i in target:
  if (i >= intervals[0] and i < intervals[1]):
    y.append(0)
  elif (i >= intervals[1] and i < intervals[2]):
    y.append(1)
  elif (i >= intervals[2] and i < intervals[3]):
    y.append(2)
  elif (i >= intervals[3] and i < intervals[4]):
    y.append(3)
  else:
    y.append(4)


rmse = []
rmse_lib = [] #RMSE com a biblioteca
x_train,x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)
for i in range(1,10):
    print(x_train[:5], y_train[:5])
    print(len(x_train), len(x_test), len(y_train), len(y_test))
    neigh = KNeighborsClassifier(n_neighbors=i, weights="uniform")
    neigh.fit(x_train, y_train)
    predictions = neigh.predict(x_test)
    print("predictions: ", predictions)
    rmse.append(np.sqrt(np.mean((y_test - predictions) ** 2)))
    rmse_lib.append(mean_squared_error(y_test, predictions)) #RMSE com a biblioteca


print("rsme: ", rmse)
print("rsme_lib: ", rmse_lib)
