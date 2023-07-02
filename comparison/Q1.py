from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy.stats import t 
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

data = open("comparison/skin/Skin_NonSkin.txt", "r")
data = data.read()
data = data.split("\n")
data = data[:-1]
data = [i.split("\t") for i in data]

parameter_amount = 4
index_of_class = 3
y = []  #y representa as classes
X = [] #X representa as caracteristicas

for i in data:
  y.append(int(i[index_of_class]))
  X.append([int(j) for j in i[:index_of_class]])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
neigh = KNeighborsClassifier(n_neighbors=1, weights="distance")

accuracy = []
t_predictions = []
#print(y[:1], X[:1])
f1 = []
for train_index, test_index in skf.split(X, y):
    print("iteration: ", len(t_predictions)+1)
    #print(train_index[:5], test_index[:5])
    x_train_fold, x_test_fold, y_train_fold, y_test_fold = [], [], [], []
    for i in train_index:
      x_test_fold.append(X[i])
      x_train_fold.append(X[i])
      y_train_fold.append(y[i])
      y_test_fold.append(y[i])
    
    #print(y_train_fold[:5], y_test_fold[:5],x_train_fold[:5], x_test_fold[:5])
    neigh.fit(x_train_fold, y_train_fold)
    predictions = neigh.predict(x_test_fold)
    f1.append(f1_score(y_test_fold, predictions, average="micro"))
    t_predictions.append(validatePredictions(predictions, y_test_fold))
    

print('tamanho do teste: ', len(t_predictions))
print("Média da precisão: ", sum(t_predictions)/len(t_predictions))
print("Maior precisão: ", max(t_predictions))
print("Minimo medida F:",min(f1))
print(t_predictions)

interval = t.interval(0.95, len(t_predictions)-1, loc=sum(t_predictions)/len(t_predictions), scale=math.sqrt(sum([(i-(sum(t_predictions)/len(t_predictions)))**2 for i in t_predictions])/(len(t_predictions)-1)))
plt.hist(f1)
#plt.show()

print("Intervalo de confiança: ", interval)