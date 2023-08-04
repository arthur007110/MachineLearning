from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn import svm
from scipy.stats import t, tstd
import math, random

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
  X.append([float(j) for j in i[index_of_class+1:12]])


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
neigh = KNeighborsClassifier(n_neighbors=1, weights="uniform")

accuracy = []
holdout_predictions = []
skf_predictions = []
f1 = []
x_train,x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random.randint(1,10))
for i in range(10):
    print(x_train[:5], y_train[:5])
    print(len(x_train), len(x_test), len(y_train), len(y_test))
    neigh.fit(x_train, y_train)
    predictions = neigh.predict(x_test)
    holdout_predictions.append(validatePredictions(predictions, y_test))

clf = svm.SVC(kernel='linear', C=10).fit(x_train, y_train)
print("Media 10-CV: ", clf.score(x_test, y_test))

print(holdout_predictions)
media = sum(holdout_predictions)/len(holdout_predictions)
print("Media holdout: ", media)

for train_index, test_index in skf.split(X,y):
    print( len(train_index), len(test_index), len(test_index)/(len(train_index)+len(test_index))*100)
    x_train_fold, x_test_fold, y_train_fold, y_test_fold = [], [], [], []
    for i in train_index:
      x_train_fold.append(X[i])
      y_train_fold.append(y[i])
    for i in test_index:
      x_test_fold.append(X[i])
      y_test_fold.append(y[i])

    neigh.fit(x_train_fold, y_train_fold)
    predictions = neigh.predict(x_test_fold)
    skf_predictions.append(validatePredictions(predictions, y_test_fold))

print(skf_predictions)
media = sum(skf_predictions)/len(skf_predictions)
print("Media skf: ", media)

