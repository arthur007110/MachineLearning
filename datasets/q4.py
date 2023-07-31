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
df = pd.DataFrame(data)

days_map = {
  'sun': 1,
  'mon': 2,
  'tue': 3,
  'wed': 4,
  'thu': 5,
  'fri': 6,
  'sat': 7,
}

months_map = {
  'jan': 1,
  'feb': 2,
  'mar': 3,
  'apr': 4,
  'may': 5,
  'jun': 6,
  'jul': 7,
  'aug': 8,
  'sep': 9,
  'oct': 10,
  'nov': 11,
  'dec': 12,
}

days_df = df.drop(['X','Y','month','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area'], axis=1, inplace=False)
months_df = df.drop(['X','Y','day','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area'], axis=1, inplace=False)
remaining_df = df.drop(['day','month'], axis=1, inplace=False)
remaining_df = remaining_df.values.tolist()

for index, day in enumerate(days_df['day']):
  day_number = days_map[day]
  remaining_df[index].append(math.sin(2 * math.pi * (day_number/7)))
  remaining_df[index].append(math.cos(2 * math.pi * (day_number/7)))

for index, month in enumerate(months_df['month']):
  month_number = months_map[month]
  remaining_df[index].append(math.sin(2 * math.pi * (month_number/12)))
  remaining_df[index].append(math.cos(2 * math.pi * (month_number/12)))

finnaly_df = pd.DataFrame(remaining_df, columns=['X','Y','FFMC','DMC','DC','ISI','temp','RH','wind','rain','area', 'day_sin', 'day_cos', 'month_sin', 'month_cos'])
print(finnaly_df)

finnaly_df = finnaly_df.iloc[1: , :]

# Iterando sobre as colunas categóricas e aplicando a transformação

y = finnaly_df.columns.get_loc("area")

print(y)

data = finnaly_df.values.tolist()

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
accuracy = []

rmse = []
neigh = KNeighborsClassifier(n_neighbors=1, weights="uniform")
for i in range(100):
    x_train,x_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=i)
    neigh.fit(x_train, y_train)
    predictions = neigh.predict(x_test)
    accuracy.append(validatePredictions(predictions, y_test))
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

print(accuracy[0])

media = [sum(accuracy)/len(accuracy) ]

print(media)
scale = tstd(accuracy)
interval = [t.interval(.95, len(accuracy)-1, loc=media, scale=scale)]

print(interval)

n = len(y_test)
alpha = 0.05 
df = n - 1
t_critical = t.ppf(1 - alpha/2, df)

ci = (rmse - t_critical * (rmse / np.sqrt(n)), rmse + t_critical * (rmse / np.sqrt(n)))

print('intervalo do RMSE',ci)
