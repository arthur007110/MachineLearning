import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class']
df = pd.read_csv(url, names=names)

df.replace('?', np.nan, inplace=True)

categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

df_encoded.dropna(inplace=True)

X = df_encoded.drop('class', axis=1)
y = df_encoded['class']
print(len(X), len(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

imputer = SimpleImputer(strategy='mean')
X_train_filled = imputer.fit_transform(X_train)
X_train = pd.DataFrame(X_train_filled, columns=X_train.columns)

X_test_filled = imputer.transform(X_test)
X_test = pd.DataFrame(X_test_filled, columns=X_test.columns)

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
    
        data = X_test_exp.values.tolist()
        y_pred_exp = classifier.predict(data)
        accuracy = accuracy_score(y_test_exp, y_pred_exp)
    
        accuracies.append(accuracy)
    
    return np.mean(accuracies)

experiment_results = [run_experiment() for _ in range(30)]
mean_accuracy_experiment = np.mean(experiment_results)
confidence_interval_experiment = 1.96 * np.std(experiment_results) / np.sqrt(len(experiment_results))

print("Acurácia Média do Experimento:", mean_accuracy_experiment)
print("Intervalo de Confiança do Experimento:", mean_accuracy_experiment - confidence_interval_experiment, "a", mean_accuracy_experiment + confidence_interval_experiment)
