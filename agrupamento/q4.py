import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import math

imagem = Image.open('agrupamento/winxp.jpg')

pixels = np.array(imagem)
altura, largura, canais = pixels.shape
print(altura, largura, canais)

vetor_atributos = pixels.reshape(altura * largura, 3)
print(vetor_atributos)
list_k = [8,64,512]
for i in list_k:
    kmeans = KMeans(n_clusters=i, random_state=0, n_init=100, max_iter=10)
    kmeans.fit(vetor_atributos)

    imagem_reconstruida = np.zeros((altura, largura, canais), dtype=np.uint8)
    centroides = np.round(kmeans.cluster_centers_).astype(int)

    nova_imagem = centroides[kmeans.labels_]
    nova_imagem = nova_imagem.reshape(altura, largura, -1)
    nova_imagem = Image.fromarray(np.uint8(nova_imagem))
    nova_imagem.save(f'resultado_k{i}.jpg')
        
