# import sys
 
# # adding Folder_2/subfolder to the system path
# sys.path.insert(0, 'C:\\Users\\arthur\\Downloads\\cars\\datasets')
import datasets

datasets = datasets.Datasets()
data, classes_data, numerical_classes, labels = datasets.read(dataset_name='student_performance')

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Carregando os dados em um DataFrame do pandas
data = pd.read_csv('dados.csv')

# Selecionando apenas as colunas categóricas
categorical_columns = ['coluna1', 'coluna2', 'coluna3']

# Criando uma instância do LabelEncoder
label_encoder = LabelEncoder()

# Iterando sobre as colunas categóricas e aplicando a transformação
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Exibindo o DataFrame com os valores numéricos
print(data)