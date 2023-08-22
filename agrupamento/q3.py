import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

data = pd.read_csv('agrupamento/maisAssistidos.csv')
df = pd.DataFrame(data)
df.replace('?', 0, inplace=True)

notas = df.iloc[:, 1:].values
Z = linkage(notas, method='average', metric='euclidean')

plt.figure(figsize=(12, 6))
dendrogram(Z, labels=df['Nome do filme'].values, orientation='left')
plt.title('Dendograma Aglomerativo de Filmes')
plt.xlabel('Distância')
plt.ylabel('Filmes')
plt.show()

pca = PCA(n_components=2)
notas_reduzidas = pca.fit_transform(notas)

# Plotar o scatter plot das notas reduzidas
plt.figure(figsize=(8, 6))
plt.scatter(notas_reduzidas[:, 0], notas_reduzidas[:, 1])
plt.title('Distribuição das Notas Reduzidas')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.show()
