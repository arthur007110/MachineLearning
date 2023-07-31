from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.stats import t, tstd
import math
import datasets
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def validatePredictions(predictions, correctValues): 
  correct = 0
  incorrect = 0
  for i in range(len(predictions)):
    if predictions[i] != correctValues[i]:
      incorrect += 1
    else:
      correct += 1
  #print("Correct: ", correct, "Incorrect: ", incorrect, "Accuracy: ", correct/(correct+incorrect))
  return correct/(correct+incorrect)

datasets = datasets.Datasets()
data, classes_data, numerical_classes, labels = datasets.read(dataset_name='forest_fires')

#print(data[:5])

# Carregando os dados em um DataFrame do pandas
data = pd.read_csv('./datasets/forest_fires/forestfires.csv', sep=',')
#print(data[:5])


data = data.iloc[1: , :]


categorical_columns = ['day','month']

# Criando uma instância do LabelEncoder
label_encoder = LabelEncoder()

data = pd.get_dummies(data, columns=categorical_columns)


y = data.columns.get_loc("area")

print(y)

data = data.values.tolist()

X = []
for i in range(len(data)):
    x2 = [] 
    for j in range(len(data[i])):
        if j != y:
           x2.append(float(data[i][j]))

    X.append(x2)
y = [int(i[y])+1 for i in data]
# transforma em log
y = np.log(y)
y = [int(i) for i in y]
print(y[:5])
accuracy = []

rmse = []
neigh = KNeighborsClassifier(n_neighbors=1, weights="uniform")
for i in range(100):
    x_train,x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
    neigh.fit(x_train, y_train)
    predictions = neigh.predict(x_test)
    
    y_test = np.exp(y_test)
    accuracy.append(validatePredictions(predictions, y_test))
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

print(accuracy[0])

media = [sum(accuracy)/len(accuracy) ]

print(media)
scale = tstd(accuracy)
interval = [t.interval(.95, len(accuracy)-1, loc=media, scale=scale)]

print(interval)

n = len(y_test)
alpha = 0.05  # 95% confidence interval
df = n - 1
t_critical = t.ppf(1 - alpha/2, df)

ci = (rmse - t_critical * (rmse / np.sqrt(n)), rmse + t_critical * (rmse / np.sqrt(n)))

print('intervalo do RMSE',ci)
