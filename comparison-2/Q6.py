import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
import numpy as np

# Carregar a base de dados Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Filtrar apenas os atributos numéricos e a coluna Pclass
df_numeric = df.select_dtypes(include=[np.number, 'int']).drop(columns=['PassengerId', 'Survived'])

# Definir a variável alvo como 'Pclass'
target = df['Pclass']

# Definir a tabela de custo
table_cost = {
    (1, 1): 0, (1, 2): 500, (1, 3): 200,
    (2, 1): 100, (2, 2): 0, (2, 3): 40,
    (3, 1): 50, (3, 2): 10, (3, 3): 0
}

# Função para calcular o custo do classificador com base nas previsões e nos valores reais
def calculate_cost(y_true, y_pred):
    cost = 0
    for true_class, pred_class in zip(y_true, y_pred):
        cost += table_cost[(true_class, pred_class)]
    return cost

# Função para executar o experimento e retornar o custo médio
def run_experiment():
    costs = []
    for _ in range(10):
        # Dividir a base de dados em treino e teste usando Holdout 80/20
        X_train, X_test, y_train, y_test = train_test_split(df_numeric, target, test_size=0.2)

        # Preencher valores ausentes usando a média dos atributos
        imputer = SimpleImputer(strategy='mean')
        X_train_filled = imputer.fit_transform(X_train)
        X_test_filled = imputer.transform(X_test)

        # Treinar o classificador 1-NN
        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(X_train_filled, y_train)

        # Fazer previsões no conjunto de teste
        y_pred = classifier.predict(X_test_filled)

        # Calcular o custo do classificador
        cost = calculate_cost(y_test, y_pred)
        costs.append(cost)

    return np.mean(costs)

# Executar o experimento 10 vezes e calcular o intervalo de confiança
experiment_results = [run_experiment() for _ in range(10)]
mean_cost_experiment = np.mean(experiment_results)
confidence_interval_experiment = 1.96 * np.std(experiment_results) / np.sqrt(len(experiment_results))

print("Custo Médio do Classificador:", mean_cost_experiment)
print("Intervalo de Confiança do Custo:", mean_cost_experiment - confidence_interval_experiment, "a", mean_cost_experiment + confidence_interval_experiment)
