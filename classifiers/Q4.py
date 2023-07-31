from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import math
import numpy as np


def printar(matrix):
    print(f"{'':<5} {'1':<8} {'2':<8} {'3':<8}")
    
    for k,v in enumerate(matrix):
        print(f"{k+1:<5} {v[0]:<8} {v[1]:<8} {v[2]:<8}")

        #print(f"{k}\t{matrix[numericalClasses[k]]}")
    return ""

def confusionMatrix(predictions, correctValues, numericalClasses):
    matrix = [[0 for i in range(len(numericalClasses))] for j in range(len(numericalClasses))]
    
    correctValues = []
    for i in realValues:
      correctValues.append(int(i[0])-1)
    
    mat = confusion_matrix(correctValues, predictions)
  
    for i in range(len(predictions)):

      matrix[predictions[i]][numericalClasses[realValues[i][0]]] += 1

    print(mat)
    printar(matrix)
    return mat,matrix

def validatePredictions(predictions, testSet, numericalClasses): 
  correct = 0
  incorrect = 0
  for i in range(len(predictions)):
    if predictions[i] != numericalClasses[testSet[i][0]]:
      incorrect += 1
    else:
      correct += 1
  print("Correct: ", correct, "Incorrect: ", incorrect)
  return correct, incorrect

data = open("K-NN/wine/wine.data", "r")
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
  testSet += classesData[i][:math.ceil(len(classesData[i])/3.5)]
  knowldegeBase += classesData[i][math.ceil(len(classesData[i])/3.5):math.ceil(len(classesData[i])/2)]


X = [[float(j) for j in i[1:]] for i in knowldegeBase]
y = [numericalClasses[i[0]] for i in knowldegeBase]
testValues = [[float(j) for j in i[1:]] for i in testSet]

neigh = KNeighborsClassifier(n_neighbors=1, weights="distance")
neigh.fit(X, y)

predictions = neigh.predict(testValues)

correct = 0
incorrect = 0
for i in range(len(predictions)):
  if predictions[i] != numericalClasses[testSet[i][0]]:
    incorrect += 1
  else:
    correct += 1

print("Correct: ", correct, "Incorrect: ", incorrect, "Accuracy: ", correct/(correct+incorrect))

correctValues = []
correctValues = []
for i in testSet:
  correctValues.append(int(i[0])-1)

confMatrix = confusionMatrix(predictions, testSet, numericalClasses)
matrix = confMatrix[1]
skMatrix = confMatrix[0]
recall = []
recallSk = recall_score(correctValues, predictions, average=None)
for i in range(len(matrix)):
    recall.append(matrix[i][i]/sum([j[i] for j in matrix]))

print("Recall: ", recall)
print("Recall Sk: ", recallSk)
precision = []
precisionSk = precision_score(correctValues, predictions, average=None)

for i in matrix:
    precision.append(i[i.index(max(i))]/sum(i))

print("Precision: ", precision)
print("Precision Sk: ", precisionSk)
f = []
fSk = f1_score(correctValues, predictions, average=None)
for i in range(len(precision)):
  f.append(2*(precision[i]*recall[i])/(precision[i]+recall[i]))


print("F: ", f)
print("F Sk: ", fSk)