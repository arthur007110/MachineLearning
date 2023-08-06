from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn import svm
from scipy.stats import t, tstd
import math, random


def split(data,y, splits):
    size = len(data)
    list = [i for i in range(size)]
    list.sort(key=lambda x: random.random())
    split_size = math.floor(size/splits)
    X = []
    for i in range(splits):
        X.append(list[i*split_size:(i+1)*split_size])
    
    size = len(y)
    list = [i for i in range(size)]
    list.sort(key=lambda x: random.random())
    split_size = math.floor(size/splits)
    y = []
    for i in range(splits):
        y.append(list[i*split_size:(i+1)*split_size])
    return X, y


# function to partition the data into train and test sets based on the train_size in percentage
def partition(X, y, train_size, ):
  x_train = []
  x_test = []
  y_train = []
  y_test = []
  size = len(X)
  list = [i for i in range(size)]
  list.sort(key=lambda x: random.random())

  train_size = math.floor(train_size * size)
  for i in list:
    if len(x_train) < train_size:
      x_train.append(X[i])
      y_train.append(y[i])
    else:
      x_test.append(X[i])
      y_test.append(y[i])
  return x_train, x_test, y_train, y_test

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

data = open("K-NN/wine/wine.data", "r")
data = data.read()
data = data.split("\n")
data = data[:-1]
data = [i.split(",") for i in data]

parameter_amount = 13
index_of_class = 0
y = []  #y representa as classes
X = [] #X representa as caracteristicas

for i in data:
  y.append(int(i[index_of_class]))
  X.append([float(j) for j in i[index_of_class+1:]])


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

neigh = KNeighborsClassifier(n_neighbors=1, weights="uniform")

accuracy = []
holdout_predictions = []
skf_predictions = []
f1 = []


x_train,x_test, y_train, y_test = partition(X, y, train_size=0.1)
for i in range(10):
    #print(x_train[:5], y_train[:5])
    #print(len(x_train), len(x_test), len(y_train), len(y_test))
    neigh.fit(x_train, y_train)
    predictions = neigh.predict(x_test)
    holdout_predictions.append(validatePredictions(predictions, y_test))

clf = svm.SVC(kernel='linear', C=10).fit(x_train, y_train)
#print("Media 10-CV: ", clf.score(x_test, y_test))

#print(holdout_predictions)
media = sum(holdout_predictions)/len(holdout_predictions)
#print("Media holdout: ", media)
train_index, test_index = split(X, y, 10)

for j in range(len(train_index)):
    #print( len(train_index),  len(test_index)/(len(train_index)+len(test_index))*100)
    x_train_fold, x_test_fold, y_train_fold, y_test_fold = [], [], [], []
    for i in train_index[j]:
      x_train_fold.append(X[i])
      y_train_fold.append(y[i])
    for i in test_index[j]:
      x_test_fold.append(X[i])
      y_test_fold.append(y[i])

    neigh.fit(x_train_fold, y_train_fold)
    predictions = neigh.predict(x_test_fold)
    skf_predictions.append(validatePredictions(predictions, y_test_fold))

print(skf_predictions)
media = sum(skf_predictions)/len(skf_predictions)
print("Media skf: ", media)

