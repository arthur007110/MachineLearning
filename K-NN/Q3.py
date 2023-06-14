from sklearn.neighbors import KNeighborsClassifier

def validatePredictions(predictions, testSet, numericalClasses): 
  correct = 0
  incorrect = 0
  for i in range(len(predictions)):
    if predictions[i] != numericalClasses[testSet[i][4]]:
      incorrect += 1
      print("[X] :\t", testSet[i], "Predicted: ", predictions[i], "Actual: ", numericalClasses[testSet[i][4]])
    else:
      correct += 1
      print("[O] :\t", testSet[i], "Predicted: ", predictions[i], "Actual: ", numericalClasses[testSet[i][4]])
  
  print("Correct: ", correct, "Incorrect: ", incorrect)

irisData = open("iris.data", "r")
irisData = irisData.read()
irisData = irisData.split("\n")
irisData = irisData[:-1]

irisData = [i.split(",") for i in irisData]

irisSetosa = [i for i in irisData if i[4] == "Iris-setosa"]
irisVersicolor = [i for i in irisData if i[4] == "Iris-versicolor"]
irisVirginica = [i for i in irisData if i[4] == "Iris-virginica"]
numericalClasses = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}

# Knowledge Base will be a list with half of the data for each class
knowldegeBase = irisSetosa[:25] + irisVersicolor[:25] + irisVirginica[:25]
testSet = irisSetosa[25:] + irisVersicolor[25:] + irisVirginica[25:]

X = [[float(j) for j in i[:4]] for i in knowldegeBase]
y = [numericalClasses[i[4]] for i in knowldegeBase]

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)

testValues = [[float(j) for j in i[:4]] for i in testSet]
predictions = neigh.predict(testValues)

validatePredictions(predictions, testSet, numericalClasses)
