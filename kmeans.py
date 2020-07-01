from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
import math
import matplotlib.pyplot as plt
from graphcut import similarity_distance
from kmeanspavan import spss

# creating graph
graph = nx.Graph()
graph.add_nodes_from([1,2,3,4,5,6])
graph.nodes[1]['info'] = {'dist': (0,0), 'color': 60}
graph.nodes[2]['info'] = {'dist': (1,0), 'color': 40}
graph.nodes[3]['info'] = {'dist': (0,1), 'color': 50}
graph.nodes[4]['info'] = {'dist': (1,1), 'color': 60}
graph.nodes[5]['info'] = {'dist': (0,2), 'color': 1}
graph.nodes[6]['info'] = {'dist': (1,2), 'color': 2}
graph.add_edges_from([(1,2), (1,3), (2, 4), (3, 4), (3, 5), (4, 6), (5, 6)], weight=0)
r = 2

for (i, j) in graph.edges:
    color1 = graph.nodes[i]['info']['color']
    color2 = graph.nodes[j]['info']['color']
    d1 = graph.nodes[i]['info']['dist']
    d2 = graph.nodes[j]['info']['dist']
    graph[i][j]['weight'] =  similarity_distance(color1, color2, d1, d2, 51, 102.4, 2)
    
A = np.array([
    [0., 1., 1., 0., 0., 1., 0., 0., 1., 1.],
    [1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.], 
    [0., 0., 0., 0., 1., 1., 0., 0., 0., 0.],
    [0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
    [1., 0., 0., 1., 1., 0., 1., 1., 0., 0.],
    [0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
    [0., 0., 0., 0., 0., 1., 1., 0., 0., 0.],
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
    [1., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
    ])

# our adjacency matrix
#print("Adjacency Matrix:")
#print(A)

# diagonal matrix
D = np.diag(A.sum(axis=1))

def graphcut_kmeans(graph, m, k, info=False):
    # graph laplacian
    L = nx.normalized_laplacian_matrix(graph).toarray()#D-A
    
    # eigenvalues and eigenvectors
    vals, vecs = np.linalg.eigh(L)

    # sort these based on the eigenvalues
    #vecs = vecs[:,np.argsort(vals)]
    #vals = vals[np.argsort(vals)]
    # kmeans on first k vectors with nonzero eigenvalues
    points = vecs[:,0:k]
    centers = spss(points, m, k)
    kmeans = KMeans(n_clusters=k,init=centers, n_init=1)
    kmeans.fit(points)
    colors = kmeans.labels_
    centers = kmeans.cluster_centers_
    if info:
        print("Adjacency matrix: ", nx.adjacency_matrix(graph))
        clusters = [list() for i in range(k)]
        for i in range(m):
            clusters[colors[i]].append(i)
        print("Clusters: ", clusters)
    return colors

# Clusters: [2 1 1 0 0 0 3 3 2 2]

if __name__ == "__main__":
    print(graphcut_kmeans(graph, 6, 2, info=True))