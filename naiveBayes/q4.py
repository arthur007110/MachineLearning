from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneOut
import pandas as pd
import numpy as np

def apply_parzen_window(data, window_width):
    kde = KernelDensity(kernel='gaussian', bandwidth=window_width).fit(data)
    return np.exp(kde.score_samples(data))

# Função para discretizar os atributos numéricos
def discretize_attributes(data, window_width):
    discretized_data = pd.DataFrame(0, index=data.index, columns=data.columns)
    #print(discretized_data,data.columns)
    for column in data.columns:
        # talvez exista uma forma mais facil de fazer isso, 
        # faz coluna por coluna, porem recebe uma parcela dos index por vez
       discretized_data[column] = apply_parzen_window(data[[column]].values.reshape(-1, 1), window_width)
    return discretized_data

def calc_parzen(X,y,window_widths):
    y_pred = []
    y_true = []
    for window_width in window_widths:
        for train_index, test_index in leave_one_out.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            #print(X_train[:5])
            # Discretização dos atributos numéricos no conjunto de treinamento
            X_train_discretized = discretize_attributes(X_train, window_width)

            # Discretização dos atributos numéricos no conjunto de teste
            X_test_discretized = discretize_attributes(X_test, window_width)

            # Treinamento do classificador Naive Bayes
            gnb.fit( X_train_discretized.values.tolist(), y_train)

            y_pred.append(gnb.predict(X_test_discretized.values.tolist())[0])
            y_true.append(y_test.values[0])

        accuracy = accuracy_score(y_true, y_pred)
        print("Naive Bayes para a largura de janela", window_width)
        print("Acurácia Média:", accuracy)
        print("Desvio Padrão:", np.std(y_pred))
        print("Intervalo de Confiança:", accuracy - 1.96 * np.std(y_pred) / len(y_true)
            ** 0.5, "a", accuracy + 1.96 * np.std(y_pred) / len(y_true) ** 0.5)
        print("Matriz de Confusão:")
        print(confusion_matrix(y_true, y_pred))

# Inicializando bases de dados
url_original = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
url_wdbc = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
url_wpbc = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data"
names_original = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
                  'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                  'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
                  'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst',
                  'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
                  'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']
names_wpbc = ['id', 'outcome', 'time', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
              'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
              'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
              'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst',
              'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
              'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'attr_1', 'attr_2']
df_original = pd.read_csv(url_original, names=names_original)
df_wdbc = pd.read_csv(url_wdbc, names=names_original)
df_wpbc = pd.read_csv(url_wpbc, names=names_wpbc, header=None)

# Pré-processamento das bases de dados
df_original.drop('id', axis=1, inplace=True)
df_original['diagnosis'] = df_original['diagnosis'].map({'M': 1, 'B': 0})

df_wdbc.drop('id', axis=1, inplace=True)
df_wdbc['diagnosis'] = df_wdbc['diagnosis'].map({'M': 1, 'B': 0})

df_wpbc.drop('id', axis=1, inplace=True)
df_wpbc['outcome'] = df_wpbc['outcome'].map({'R': 1, 'N': 0})
df_wpbc['attr_2'] = df_wpbc['attr_2'].map(
    lambda x: 0 if x == '?' else int(x))


# Leave-one-out para calcular a acurácia(média, desvio padrão e intervalo de confiança) e a matriz de confusão
leave_one_out = LeaveOneOut()

# Naive Bayes para a base de dados original
gnb = GaussianNB()

X = df_original.iloc[:, 1:]
y = df_original['diagnosis']

# Larguras de janela a serem testadas
window_widths = [0.001, 0.01, 0.1, 1, 10, 100, 1000]



# Naive Bayes para a base de dados original
calc_parzen(X,y,window_widths)

# Naive Bayes para a base de dados wdbc
gnb = GaussianNB()

X = df_wdbc.iloc[:, 1:]
y = df_wdbc['diagnosis']

calc_parzen(X,y,window_widths)

# Naive Bayes para a base de dados wpbc
gnb = GaussianNB()

X = df_wpbc.iloc[:, 1:]
y = df_wpbc['outcome']


calc_parzen(X,y,window_widths)


