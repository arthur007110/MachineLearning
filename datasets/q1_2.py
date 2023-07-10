# import sys
 
# # adding Folder_2/subfolder to the system path
# sys.path.insert(0, 'C:\\Users\\arthur\\Downloads\\cars\\datasets')
import datasets
import pandas as pd
from sklearn.preprocessing import LabelEncoder

datasets = datasets.Datasets()
data, classes_data, numerical_classes, labels = datasets.read(dataset_name='student_performance')

print(data[:5])

# Carregando os dados em um DataFrame do pandas
data = pd.read_csv('datasets/student_performance/data.csv', sep=';')
print(data[:5])

#categorical_classes = classes_data.pop('Mjob','Fjob','reason','guardian')

#categorical_numeric = classes_data.pop('age')

categorical_boolean_columns = ['sex','school','address','famsize','Pstatus','schoolsup','famsup','paid',
'activities','nursery','higher','internet','romantic']
# Selecionando apenas as colunas categóricas
categorical_columns = ['Mjob','Fjob','reason','guardian']

# Criando uma instância do LabelEncoder
label_encoder = LabelEncoder()

data = pd.get_dummies(data, columns=categorical_columns)
# Iterando sobre as colunas categóricas e aplicando a transformação
for column in categorical_boolean_columns:
    data[column] = label_encoder.fit_transform(data[column])


#print(data[:10])

# Exibindo o DataFrame com os valores numéricos]
print(data[:10])