'''
Construa sua própria implementação do classicador pelo vizinho mais próximo

(1 − NN) utilizando distância euclidiana. Avalie este classicador utilizando metade dos exem-
plos de cada classe da base Iris (archive.ics.uci.edu/ml/datasets/iris) como conjunto de

teste e o restante como conjunto de treinamento. Atenção: construa também a implementação
da distância euclidiana.
'''

class KNN:
    def __init__(self):
        self.X_train = []
        self.y_train = []
        
    def fit(self, X, y):
        self.X_train.append(X)
        self.y_train.append(y)
    
    def euclidean_distance(self,x, y):
        return abs(float(x) - float(y))

    def predict(self,x):
        menor = [1000,'a']
        
        for i in range(len(self.X_train)):
            sumDist = 0
            for j in range(len(x)-1):
                tmp = self.euclidean_distance(x[j],self.X_train[i][j])
                sumDist += float(format(tmp, ".2f"))
            
            if sumDist < menor[0]:
                menor[0] = sumDist
                menor[1] = self.y_train[i]

        if(menor[1] == x[0]):
            return True
        return False
        
    






