from neuronio import Neuronio
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
y_copy = y.copy()

taxas_aprendizado = [0.1, 1, 10]

for t in taxas_aprendizado:
    print("Taxa de aprendizado:", t)
    y = [1 if i == 0 else 0 for i in y_copy]

    acuracias = []
    for j in range(30):
        # Separa os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3)

        # Crie um neurônio com 4 entradas
        num_inputs = 4
        neuron = Neuronio(num_inputs, funcao_ativacao="step")

        # Treine o neurônio
        neuron.fit(X_train, y_train, taxa_aprendizado=t, epocas=1)

        # Realize a predição
        predicoes = neuron.predict(X_test)

        acuracia = np.mean([1 if predicoes[i] == y_test[i]
                            else 0 for i in range(len(predicoes))])
        acuracias.append(acuracia)

    print("Acurácia média[Iris Setosa]:", np.mean(acuracias))
    print("Desvio padrão[Iris Setosa]:", np.std(acuracias))

    y = [1 if i == 2 else 0 for i in y_copy]

    acuracias = []
    for j in range(30):
        # Separa os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3)

        # Crie um neurônio com 4 entradas
        num_inputs = 4
        neuron = Neuronio(num_inputs, funcao_ativacao="step")

        # Treine o neurônio
        neuron.fit(X_train, y_train, taxa_aprendizado=t, epocas=100)

        # Realize a predição
        predicoes = neuron.predict(X_test)

        acuracia = np.mean([1 if predicoes[i] == y_test[i]
                            else 0 for i in range(len(predicoes))])
        acuracias.append(acuracia)

    print("Acurácia média[Iris Virginica]:", np.mean(acuracias))
    print("Desvio padrão[Iris Virginica]:", np.std(acuracias))
    print()
