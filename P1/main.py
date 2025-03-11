import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Vérification de la compatibilité avec Python 3.7
import sys

if sys.version_info < (3, 7):
    raise Exception("Ce script nécessite Python 3.7 ou une version ultérieure")

# Génération de données synthétiques
n_samples = 300
n_features = 2
n_clusters = 3
random_state = 42

data, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)

print(data)

# Visualisation des données initiales
plt.scatter(data[:, 0], data[:, 1], s=50, cmap='viridis')
plt.title("Données initiales")
plt.show()

# Application de l'algorithme K-Means
kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualisation des clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroides')
plt.title("Résultat du clustering K-Means")
plt.legend()
plt.show()
