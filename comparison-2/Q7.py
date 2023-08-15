import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from scipy.stats import ttest_rel
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

# Função para executar o experimento com um valor de k e retornar o custo médio
def run_experiment_k(k):
    costs = []
    for _ in range(10):
        # Dividir a base de dados em treino e teste usando Holdout 80/20
        X_train, X_test, y_train, y_test = train_test_split(df_numeric, target, test_size=0.2)

        # Preencher valores ausentes usando a média dos atributos
        imputer = SimpleImputer(strategy='mean')
        X_train_filled = imputer.fit_transform(X_train)
        X_test_filled = imputer.transform(X_test)

        # Treinar o classificador k-NN com o valor de k atual
        classifier = KNeighborsClassifier(n_neighbors=k)
        classifier.fit(X_train_filled, y_train)

        # Fazer previsões no conjunto de teste
        y_pred = classifier.predict(X_test_filled)

        # Calcular o custo do classificador
        cost = calculate_cost(y_test, y_pred)
        costs.append(cost)

    return np.mean(costs)

# Executar o experimento para diferentes valores de k (k variando de 2 a 6)
k_values = [1, 2, 3, 4, 5, 6]
experiment_results = {k: run_experiment_k(k) for k in k_values}

# Executar o experimento para o 1-NN (k=1)
mean_cost_experiment_1nn = run_experiment_k(1)

# Realizar o teste de hipótese comparando cada valor de k com o 1-NN
p_values = [ttest_rel([mean_cost_experiment_1nn] * 10, [experiment_results[k]] * 10).pvalue for k in k_values]

# Verificar se há diferença significativa entre algum valor de k e o 1-NN
significance_level = 0.05
significant_k = [k_values[i] for i, p_value in enumerate(p_values) if p_value < significance_level]

print("Resultados do Experimento:")
for k, cost in experiment_results.items():
    print(f"k={k}, Custo Médio={cost}")

print("\nCusto Médio do 1-NN (k=1):", mean_cost_experiment_1nn)

if len(significant_k) > 0:
    print("\nValores de k com diferença significativa para o 1-NN:", significant_k)
else:
    print("\nNenhum valor de k apresenta diferença significativa para o 1-NN.")
