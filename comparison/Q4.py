from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score
from scipy.stats import t, tstd
import math

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

parameter_amount = 14
index_of_class = 1
y = []  #y representa as classes
X = [] #X representa as caracteristicas

for i in data:
  y.append(float(i[index_of_class-1]))
  X.append([float(j) for j in i[index_of_class:]])


x_train = X[:math.ceil(len(X)/2)]
y_train = y[:math.ceil(len(y)/2)]


x_test = X[math.ceil(len(X)/2):]
y_test = y[math.ceil(len(y)/2):]


#print("tamanho do teste: ", x_test[:5], y_test[:5])
#print("tamanho do treino: ",x_train[:5], y_train[:5])

t_predictions = [[] for _ in range(15)]

for l in range(1, 16):
    neigh = KNeighborsClassifier(n_neighbors=l, weights="distance")
    for i in range(100):
        x_train,x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
        neigh.fit(x_train, y_train)
        predictions = neigh.predict(x_test)
        t_predictions[l-1].append(validatePredictions(predictions, y_test))
media = [sum(i)/len(i) for i in t_predictions]
scale = [tstd(i) for i in t_predictions]
interval = [t.interval(.95, len(t_p0redictions[i])-1, loc=media[i], scale=scale[i]) for i in range(len(t_predictions))]

print(interval)
print("")
print([i[1]-i[0] for i in interval])

