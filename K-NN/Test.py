from algorithm.knn import * 
import math

KNN = KNN()

parameter_amount = 12
data = open("K-NN/wine/wine.data", "r")
data = data.read()
data = data.split("\n")
data = data[:-1]
data = [i.split(",")[:(parameter_amount + 1)] for i in data]
print(data[0])

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

for i in knowldegeBase:
    #KNN.fit(i[:parameter_amount],i[parameter_amount:]) //iris
    KNN.fit(i[1:],i[0])

pred = []
for i in testSet:
    predictions = KNN.predict(i)
    pred.append(predictions)

cont = 0
t = 0
for i in pred:
    if i == True:
        cont += 1
    else:
        t += 1

print("Correct: ", cont, "Incorrect: ", t)




