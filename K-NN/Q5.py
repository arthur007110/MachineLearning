from sklearn.neighbors import KNeighborsClassifier
import math

def validatePredictions(predictions, testSet, numericalClasses): 
  correct = 0
  incorrect = 0
  for i in range(len(predictions)):
    if predictions[i] != numericalClasses[testSet[i][0]]:
      incorrect += 1
    else:
      correct += 1
  print("Correct: ", correct, "Incorrect: ", incorrect)

data = open("wine/wine.data", "r")
data = data.read()
data = data.split("\n")
data = data[:-1]
data = [i.split(",") for i in data]

parameter_amount = 13

numericalClasses = {}
classesData = {}

for i in data:
  if i[0] not in numericalClasses:
    numericalClasses[i[0]] = len(numericalClasses)
    classesData[i[0]] = []
  classesData[i[0]].append(i)

knowldegeBase = []
testSet = []
for i in classesData:
  knowldegeBase += classesData[i][math.ceil(len(classesData[i])/2):]
  testSet += classesData[i][:math.ceil(len(classesData[i])/2)]

X = [[float(j) for j in i[parameter_amount:]] for i in knowldegeBase]
y = [numericalClasses[i[0]] for i in knowldegeBase]

neigh = KNeighborsClassifier(n_neighbors=7, weights="distance")
neigh.fit(X, y)

testValues = [[float(j) for j in i[parameter_amount:]] for i in testSet]
predictions = neigh.predict(testValues)

validatePredictions(predictions, testSet, numericalClasses)
