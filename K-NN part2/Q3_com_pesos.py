from algorithm.knn import * 
import math

KNN = KNN(k=7, weights='distance')

parameter_amount = 4
data = open("../K-NN/iris.data", "r")
data = data.read()
data = data.split("\n")
data = data[:-1]
data = [i.split(",") for i in data]

numericalClasses = {}
classesData = {}

for i in data:
  if i[-1] not in numericalClasses:
    numericalClasses[i[-1]] = len(numericalClasses)
    classesData[i[-1]] = []
  classesData[i[-1]].append(i)

knowldegeBase = []
testSet = []
for i in classesData:
  knowldegeBase += classesData[i][math.ceil(len(classesData[i])/2):]
  testSet += classesData[i][:math.ceil(len(classesData[i])/2)]

X = [[float(j) for j in i[:-1]] for i in knowldegeBase]
y = [numericalClasses[i[-1]] for i in knowldegeBase]
testValues = [[float(j) for j in i[:-1]] for i in testSet]

KNN.fit(X, y)

predictions = [KNN.predict(i) for i in testValues]
cont = 0
t = 0
for i in range(len(predictions)):
  if predictions[i] != numericalClasses[testSet[i][-1]]:
    t += 1
  else:
    cont += 1

print("Correct: ", cont, "Incorrect: ", t)