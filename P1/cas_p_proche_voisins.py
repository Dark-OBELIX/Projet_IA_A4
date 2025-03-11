import random
import math
import matplotlib.pyplot as plt

nbres_points = 20000
nbres_centroides = 5
range_min = 0
range_max = 180

def dist_euclidienne(p1, p2):
    varl_x = (p1[0] - p2[0]) ** 2
    varl_y = (p1[1] - p2[1]) ** 2
    return math.sqrt(varl_x + varl_y)

def generation_points(range_min, range_max, nbres_points):
    t = []
    for i in range(nbres_points):
        tt = [random.randrange(range_min, range_max), random.randrange(range_min, range_max)]
        t.append(tt)
    return t

def initialisation_centroides(range_min, range_max, nbres_centroides):
    centroids = []
    for i in range(nbres_centroides):
        centroids.append([random.randint(range_min, range_max), random.randint(range_min, range_max)])
    return centroids

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
            new_centroids.append([random.randint(range_min, range_max), random.randint(range_min, range_max)])
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

def kmeans(nbres_points, nbres_centroides, range_min, range_max, max_iterations=100):
    points = generation_points(range_min, range_max, nbres_points)
    centroids = initialisation_centroides(range_min, range_max, nbres_centroides)
    
    #print("Données d'entrée :")
    #print("Points:", points)
    #print("Centroides initiaux:", centroids)
    
    plot_clusters(points, centroids, [[] for _ in centroids], "Données Initiales")
    
    for _ in range(max_iterations):
        clusters = assignation_points(points, centroids)
        new_centroids = mise_a_jour_centroides(clusters)
        
        if new_centroids == centroids:
            break
        centroids = new_centroids
    
    #print("\nDonnées finales :")
    #print("Centroides finaux:", centroids)
    #print("Clusters:", clusters)
    
    plot_clusters(points, centroids, clusters, "Données Finales")
    
    return centroids, clusters

centroids, clusters = kmeans(nbres_points, nbres_centroides, range_min, range_max)
