from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split

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

irisData = open("K-NN/iris.data", "r")
irisData = irisData.read()
irisData = irisData.split("\n")
irisData = irisData[:-1]

irisData = [i.split(",") for i in irisData]

numericalClasses = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

# Knowledge Base will be a list with half of the data for each class
knowldegeBase = irisData


y = [numericalClasses[i[4]] for i in knowldegeBase]
X = [[float(j) for j in k[:4]] for k in knowldegeBase]

neigh = KNeighborsClassifier(n_neighbors=1)
# for i in

for i in range(4):
    for l in range(4):
        fixa = [k[i:l] for k in X]
        x_train,x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=2+i)
        neigh.fit(x_train, y_train)
        predictions = neigh.predict(x_test)

        print("Precisoes sem as colunas:", validatePredictions(predictions, y_test))


