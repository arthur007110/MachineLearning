from sklearn.neighbors import KNeighborsClassifier
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
X2 = []

for i in data:
  y.append(float(i[index_of_class-1]))
  X.append([float(j) for j in i[index_of_class:]])
  X2.append([float(j) for j in i[index_of_class:parameter_amount-1]])

neigh = KNeighborsClassifier(n_neighbors=1, weights="uniform")

x_train = X[:math.ceil(len(X)/2)]
y_train = y[:math.ceil(len(y)/2)]

x2_train = X2[:math.ceil(len(X2)/2)]

x_test = X[math.ceil(len(X)/2):]
x2_test =  X2[math.ceil(len(X2)/2):]
y_test = y[math.ceil(len(y)/2):]


print("tamanho do teste: ", len(x_test), len(y_test))
print("tamanho do treino: ",len(x_train), len(y_train))

t_predictions = [[] for _ in range(2)]
f1 = [[] for _ in range(2)]

for i in range(100):
    neigh.fit(x_train, y_train)
    predictions = neigh.predict(x_test)
    f1[0].append(f1_score(y_test, predictions, average="micro"))
    t_predictions[0].append(validatePredictions(predictions, y_test))
    neigh.fit(x2_train, y_train)
    predictions = neigh.predict(x2_test)
    f1[1].append(f1_score(y_test, predictions, average="micro"))
    t_predictions[1].append(validatePredictions(predictions, y_test))
    

media = sum(t_predictions[0])/len(t_predictions[0])
media2 = sum(t_predictions[1])/len(t_predictions[1])
print('tamanho do teste: ', len(t_predictions[0]), len(t_predictions[1]))
print("Média da precisão: ", media, media2)
print("Maior precisão: ", max(t_predictions[0]), max(t_predictions[1]))
print("Minimo medida F:",min(f1[0]), min(f1[1]))
#print(t_predictions)

scale = tstd(t_predictions[0])
scale2 = tstd(t_predictions[1])
if scale == 0.0:
    scale = 0.000001

if scale2 == 0.0:
    scale2 = 0.000001

interval = t.interval(0.95, len(t_predictions[0])-1, loc=media, scale=scale)
interval2 = t.interval(0.95, len(t_predictions[1])-1, loc=media2, scale=scale2)
print("Intervalo de confiança: ", interval, interval2)

#print(t_predictions[0], t_predictions[1])

