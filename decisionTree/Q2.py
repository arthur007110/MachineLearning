from sklearn import tree
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.metrics import accuracy_score
import random
from sklearn.metrics import confusion_matrix
import graphviz
from matplotlib import pyplot as plt


data = pd.read_csv('./datasets/car_evaluation/car.data', sep=',')
categorical_boolean_columns = ['buying','maint','doors','persons','lug_boot','safety','class']

# Criando uma instância do LabelEncoder
label_encoder = LabelEncoder()

for column in categorical_boolean_columns:
    data[column] = label_encoder.fit_transform(data[column])

data = data.values.tolist()

parameter_amount = 6
index_of_class = 6
y = []  #y representa as classes
X = [] #X representa as caracteristicas

for i in data:
  y.append( [j for j in i[6:]])
  X.append([j for j in i[:parameter_amount]])

print(X[:5],y[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)
n_random = [1,3,5]
for i in n_random:
  dt = tree.DecisionTreeClassifier(min_samples_leaf=i, criterion='log_loss')
  dt.fit(X_train, y_train)

  predictions = dt.predict(X_test)
  predictions_train = dt.predict(X_train)
  #print(dt)
  #dot_data = tree.export_graphviz(dt, out_file=None)
  #graph = graphviz.Source(dot_data)
  accuracy_train = accuracy_score(y_train, predictions_train)
  accuracy = accuracy_score(y_test, predictions)
  print(f' taxa de acerto do teste com folha {i}: {accuracy}')
  print(f' taxa de acerto do treino com folha {i}: {accuracy_train}')

print(confusion_matrix(y_test, predictions))
print(predictions[0])
  
fig = plt.figure(figsize=(40,40), layout='constrained', dpi=300)
_ = tree.plot_tree(dt,filled=True, 
  feature_names=['buying','maint','doors','persons','lug_boot','safety'], 
  class_names=['unacc','acc','good','vgood'])
fig.savefig("decistion_tree.png")

#graph.render("stroke prediction",view=True)

