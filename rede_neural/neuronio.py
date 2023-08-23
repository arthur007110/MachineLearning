import numpy as np

class Neuronio:
    def __init__(self, qtd_inputs, funcao_ativacao="step", debug=False):
        self.pesos = np.random.rand(qtd_inputs)
        self.vies = np.random.rand()
        self.funcao_ativacao = funcao_ativacao
        self.debug = debug

    def fit(self, inputs, yd, taxa_aprendizado=0.1, epocas=1000):
        for epoca in range(epocas):
            for i in range(len(inputs)):
                soma_ponderada = np.dot(inputs[i], self.pesos) + self.vies

                saida = self.aplicar_funcao_ativacao(soma_ponderada)

                erro = yd[i] - saida

                # Atualiza os pesos
                for j in range(len(self.pesos)):
                    self.pesos[j] = self.pesos[j] + \
                        (taxa_aprendizado * erro * inputs[i][j])

                # Atualiza o viés
                self.vies = self.vies + (taxa_aprendizado * erro)
            if self.debug:
                print("Época:", epoca, " | Erro:", erro)
                print("Pesos:", self.pesos, " | Viés:", self.vies)

    def predict(self, inputs):
        predicoes = []
        for i in range(len(inputs)):
            soma_ponderada = np.dot(inputs[i], self.pesos) + self.vies
            predicoes.append(self.aplicar_funcao_ativacao(soma_ponderada))

        return predicoes

    def aplicar_funcao_ativacao(self, soma_ponderada):
        if self.funcao_ativacao == "step":
            return self.step_function(soma_ponderada)
        elif self.funcao_ativacao == "sigmoid":
            return self.sigmoid(soma_ponderada)

    def step_function(self, soma_ponderada):
        if soma_ponderada >= 1:
            return 1
        return 0

    def sigmoid(self, soma_ponderada):
        return 1 / (1 + np.exp(-soma_ponderada))
