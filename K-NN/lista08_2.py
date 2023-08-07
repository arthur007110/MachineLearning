import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data"
names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'class']
df = pd.read_csv(url, names=names)

df.replace('?', np.nan, inplace=True)

df = df.astype(float)

threshold = len(df) * 0.5
df.dropna(thresh=threshold, axis=1, inplace=True)

# (a) Dividir a base em treino (90%) e teste (10%) de forma estratificada:
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

# (b) Preencher os valores omissos no conjunto de treino:
imputer = SimpleImputer(strategy='mean')
X_train_filled = imputer.fit_transform(X_train)
X_train = pd.DataFrame(X_train_filled, columns=X_train.columns)

# (c) Preencher os valores omissos no conjunto de teste utilizando o método e os valores definidos para o conjunto de treino:
X_test_filled = imputer.transform(X_test)
X_test = pd.DataFrame(X_test_filled, columns=X_test.columns)

# (d) Repetir 30 vezes o experimento para calcular o intervalo de confiança para a taxa de acerto do classificador utilizando 100 repetições deste experimento:
def run_experiment():
    accuracies = []
    for _ in range(100):
        X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(X, y, test_size=0.1, stratify=y)
    
        X_train_filled_exp = imputer.fit_transform(X_train_exp)
        X_train_exp = pd.DataFrame(X_train_filled_exp, columns=X_train_exp.columns)

        X_test_filled_exp = imputer.transform(X_test_exp)
        X_test_exp = pd.DataFrame(X_test_filled_exp, columns=X_test_exp.columns)

        classifier = KNeighborsClassifier(n_neighbors=1)
        classifier.fit(X_train_exp, y_train_exp)
    
        y_pred_exp = classifier.predict(X_test_exp)
        accuracy = accuracy_score(y_test_exp, y_pred_exp)
    
        accuracies.append(accuracy)
    
    return np.mean(accuracies)

experiment_results = [run_experiment() for _ in range(30)]
mean_accuracy_experiment = np.mean(experiment_results)
confidence_interval_experiment = 1.96 * np.std(experiment_results) / np.sqrt(len(experiment_results))

print("Acurácia Média do Experimento:", mean_accuracy_experiment)
print("Intervalo de Confiança do Experimento:", mean_accuracy_experiment - confidence_interval_experiment, "a", mean_accuracy_experiment + confidence_interval_experiment)

# (e) Realizar um teste de hipótese comparando o intervalo calculado anteriormente com um intervalo de confiança calculado de forma similar removendo todas as colunas que apresentavam valores omissos:

df_no_missing = df.dropna(axis=1)

X_no_missing = df_no_missing.drop('class', axis=1)
y_no_missing = df_no_missing['class']
X_train_no_missing, X_test_no_missing, y_train_no_missing, y_test_no_missing = train_test_split(X_no_missing, y_no_missing, test_size=0.1, stratify=y_no_missing, random_state=42)

experiment_results_no_missing = [run_experiment() for _ in range(30)]
mean_accuracy_experiment_no_missing = np.mean(experiment_results_no_missing)
confidence_interval_experiment_no_missing = 1.96 * np.std(experiment_results_no_missing) / np.sqrt(len(experiment_results_no_missing))

print("Acurácia Média do Experimento sem Valores Omissos:", mean_accuracy_experiment_no_missing)
print("Intervalo de Confiança do Experimento sem Valores Omissos:", mean_accuracy_experiment_no_missing - confidence_interval_experiment_no_missing, "a", mean_accuracy_experiment_no_missing + confidence_interval_experiment_no_missing)
