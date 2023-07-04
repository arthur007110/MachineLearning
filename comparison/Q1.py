from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy.stats import t, tstd
import math, random

def salvarArquivo(conteudo, nome="resultado.txt"):
  arquivo = open("comparison\\"+str(nome), "a")
  arquivo.write(conteudo)
  arquivo.close()

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

data = open("comparison/skin/Skin_NonSkin.txt", "r")
data = data.read()
data = data.split("\n")
data = data[:-1]
data = [i.split("\t") for i in data]

parameter_amount = 4
index_of_class = 3
y = []  #y representa as classes
X = [] #X representa as caracteristicas

for i in data:
  y.append(int(i[index_of_class]))
  X.append([int(j) for j in i[:index_of_class]])


skf = StratifiedKFold(n_splits=100, shuffle=True, random_state=1)
neigh = KNeighborsClassifier(n_neighbors=1, weights="uniform")

r1 = random.randint(1, 3)
r2 = random.randint(1, 3)

X_Random = []
y_random = []
for i in range(parameter_amount):
  X_Random += X[:math.ceil(len(X)/r1)]
  y_random += y[:math.ceil(len(y)/r1)]

accuracy = []
t_predictions = []
#print(y[:1], X[:1])
salvarArquivo("tamanho do teste: "+str(len(y_random))+"\n","precisoes.txt")
salvarArquivo("tamanho do treino: "+str(len(X_Random))+ "\n","precisoes.txt")
f1 = []
for train_index, test_index in skf.split(X_Random,y_random):
    x_train_fold, x_test_fold, y_train_fold, y_test_fold = [], [], [], []
    for i in train_index:
      x_train_fold.append(X_Random[i])
      y_train_fold.append(y_random[i])
     
    
    for i in test_index:
      x_test_fold.append(X_Random[i])
      y_test_fold.append(y_random[i])
    
    #print(y_train_fold[:5], y_test_fold[:5],x_train_fold[:5], x_test_fold[:5])
    neigh.fit(x_train_fold, y_train_fold)
    predictions = neigh.predict(x_test_fold)
    f1.append(f1_score(y_test_fold, predictions, average="micro"))
    t_predictions.append(validatePredictions(predictions, y_test_fold))
    

media = sum(f1)/len(f1)
print('testes feitos: ', len(t_predictions))
print("Média da Medida F: ", media)
print("Maior medida f: ", max(f1))
print("Minimo medida F:",min(f1))
#print(t_predictions)


scale = tstd(f1)
if scale == 0.0:
    scale = 0.000001

interval = t.interval(0.95, len(f1)-1, loc=media, scale=scale)
plt.hist(f1)
#plt.show()

print("Intervalo de confiança: ", interval)
salvarArquivo("Media da medida f: "+str(media)+"\n")
salvarArquivo("Maior medida f: "+str(max(t_predictions))+"\n")
salvarArquivo("Minimo medida F: "+str(min(f1))+"\n")
salvarArquivo("Intervalo de confianca: "+str(interval)+"\n")

for i, index in enumerate(t_predictions):
  salvarArquivo(str(i)+" -Precisao: "+str(index)+"\n", "precisoes.txt")
#salvarArquivo("histograma: "+plt.hist(f1)+"\n", "histograma.png")

plt.show()