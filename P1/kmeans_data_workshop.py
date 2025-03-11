import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import os
import tarfile

extract_path = 'C:/Users/hugol/OneDrive/Documents/4_CESI/1-Projet IA/1-prosit/workshop/'
csv_file_path = os.path.join(extract_path, 'housing.csv')
housing_data = pd.read_csv(csv_file_path)

# Les 2 premires clonnes
points = housing_data.iloc[:, :2].values.tolist()

nbres_centroides = 5

def dist_euclidienne(p1, p2):
    varl_x = (p1[0] - p2[0]) ** 2
    varl_y = (p1[1] - p2[1]) ** 2
    return math.sqrt(varl_x + varl_y)

def initialisation_centroides(points, nbres_centroides):
    return random.sample(points, nbres_centroides)

def assignation_points(points, centroids):
    clusters = [[] for _ in range(len(centroids))]
    for point in points:
        distances = [dist_euclidienne(point, centroid) for centroid in centroids]
        min_index = distances.index(min(distances))
        clusters[min_index].append(point)
    return clusters

def mise_a_jour_centroides(clusters):
    new_centroids = []
    for cluster in clusters:
        if cluster:
            mean_x = sum(p[0] for p in cluster) / len(cluster)
            mean_y = sum(p[1] for p in cluster) / len(cluster)
            new_centroids.append([mean_x, mean_y])
        else:
            new_centroids.append(random.choice(points))
    return new_centroids

def plot_clusters(points, centroids, clusters, title):
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    plt.figure()
    all_points = list(zip(*points))
    plt.scatter(all_points[0], all_points[1], c='gray', marker='o', label='Données')
    
    for i, cluster in enumerate(clusters):
        cluster_points = list(zip(*cluster))
        if cluster_points:
            plt.scatter(cluster_points[0], cluster_points[1], c=colors[i % len(colors)], label=f'Cluster {i+1}')
    
    centroid_points = list(zip(*centroids))
    plt.scatter(centroid_points[0], centroid_points[1], c='black', marker='x', s=100, label='Centroides')
    plt.title(title)
    plt.legend()
    plt.show()

def kmeans(points, nbres_centroides, max_iterations=100):
    centroids = initialisation_centroides(points, nbres_centroides)
    
    #print("Données d'entrée :")
    #print("Centroides initiaux:", centroids)
    
    plot_clusters(points, centroids, [[] for _ in centroids], "Données Initiales")
    
    for _ in range(max_iterations):
        clusters = assignation_points(points, centroids)
        new_centroids = mise_a_jour_centroides(clusters)
        
        if new_centroids == centroids:
            break
        centroids = new_centroids
    
    #print("\nDonnées finales :")
    print("Centroides finaux:", centroids)
    print("Clusters:", clusters)
    
    plot_clusters(points, centroids, clusters, "Données Finales")
    
    return centroids, clusters

centroids, clusters = kmeans(points, nbres_centroides)
