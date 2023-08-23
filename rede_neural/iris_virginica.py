from neuronio import Neuronio
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

# Pre-processamento para tornar todas as flores Virginica como 1 e as demais como 0
# y = [1 if i == 0 else 0 for i in y]
y = [1 if i == 2 else 0 for i in y]

acuracias = []
for j in range(100):
    # Separa os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Crie um neurônio com 4 entradas
    num_inputs = 4
    neuron = Neuronio(num_inputs, funcao_ativacao="step")

    # Treine o neurônio
    neuron.fit(X_train, y_train, taxa_aprendizado=0.1, epocas=100)

    # Realize a predição
    predicoes = neuron.predict(X_test)

    acuracia = np.mean([1 if predicoes[i] == y_test[i]
                        else 0 for i in range(len(predicoes))])
    acuracias.append(acuracia)
    print("Acurácia:", acuracia)

print("Acurácia média:", np.mean(acuracias))
