'''
Construa sua própria implementação do classicador pelo vizinho mais próximo

(1 − NN) utilizando distância euclidiana. Avalie este classicador utilizando metade dos exem-
plos de cada classe da base Iris (archive.ics.uci.edu/ml/datasets/iris) como conjunto de

teste e o restante como conjunto de treinamento. Atenção: construa também a implementação
da distância euclidiana.
'''
import math
from decimal import Decimal 
class KNN:
  def __init__(self, k=1, distance='euclidean', p=3, weights='uniform'):
    self.k = k
    self.p = p
    self.distance = distance
    self.weights = weights
    self.X_train = []
    self.y_train = []
      
  def fit(self, X, y):
    self.X_train = X
    self.y_train = y

  def calculate_distance(self,x, y):
    if(self.distance == 'euclidean'):
      return self.euclidean_distance(x,y)
    elif(self.distance == 'minkowski'):
      return self.minkowski_distance(x,y,self.p)
    else:
      raise Exception('Distance not implemented')

  def euclidean_distance(self,x, y):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))

  def p_root(self, value, root):
    root_value = 1 / float(root)
    return round(Decimal(value) ** Decimal(root_value), 3)
  
  def minkowski_distance(self, x, y, p_value): 
    summation = sum(pow(abs(a-b), p_value) for a, b in zip(x, y))
    return self.p_root(summation, p_value)
  
  def init_neighborhood(self):
    neighborhood = []
    for _ in range(self.k):
      neighborhood.append([math.inf,'null'])
    return neighborhood
  
  def classify_by_neighborhood(self,neighborhood):
    classes = {}
    for i in neighborhood:
      if(i[1] == 'null'):
        continue
      if i[1] not in classes:
        classes[i[1]] = 0
      if(self.weights == 'uniform'):
        classes[i[1]] += 1
      elif(self.weights == 'distance'):
        if(i[0] == 0):
          classes[i[1]] += 1
        else:
          classes[i[1]] += 1/i[0]
      else:
        raise Exception('Weight type invalid')
    return max(classes, key=classes.get)
  
  def classify_by_neighborhood_weighted(self,neighborhood):
    classes = {}
    for i in neighborhood:
      if(i[1] == 'null'):
        continue
      if i[1] not in classes:
        classes[i[1]] = 0
      classes[i[1]] += 1/i[0]
    return max(classes, key=classes.get)

  def predict(self,x):
    neighborhood = self.init_neighborhood()

    for i in range(len(self.X_train)):
      distance = self.calculate_distance(x,self.X_train[i])
      
      for j in range(len(neighborhood)):
        if distance < neighborhood[j][0]:
          neighborhood[j][0] = distance
          neighborhood[j][1] = self.y_train[i]
          break
    return self.classify_by_neighborhood(neighborhood)
