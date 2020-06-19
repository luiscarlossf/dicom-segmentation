from superpixel import return_superpixels
from graphcut import cut, similarity_distance
from kmeans import graphcut_kmeans
import cv2
import networkx as nx
import matplotlib.pyplot as plt 
import math
from skimage.morphology import opening

if __name__ == "__main__":
    for i in range(30):
        g = nx.Graph()
        image = cv2.imread("./outputs/lung.png", 0)
        """
        cv2.imwrite("./outputs/saida1.png", image)
        input()
        image = cv2.imread("./outputs/saida.png", 0)
        retval = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3), (0,0))
        s1 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, retval)
        cv2.imwrite("./outputs/saida1.png", s1)
        input()
        retval = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
        s2 = opening(s1, retval)#cv2.morphologyEx(s1, cv2.MORPH_OPEN, retval)
        cv2.imwrite("./outputs/saida1.png", s2)
        input()
        """
        
        sp, adjacency = return_superpixels(image, info=True)
        #g.add_nodes_from(list(sp.keys()))
        
        
        for key in sp.keys():
            g.add_node(key, info=sp[key], color='red')
        
        for i in adjacency:
            for j in adjacency[i]:
                try:
                    g[j][i]
                    continue #Não adiciona arestas repetidas.
                except Exception:
                    pass
                g.add_edge(i, j)
                color1 = g.nodes[i]['info']['color']
                color2 = g.nodes[j]['info']['color']
                d1 = g.nodes[i]['info']['centroid']
                d2 = g.nodes[j]['info']['centroid']
                g[i][j]['weight'] =  similarity_distance(color1, color2, d1, d2,di=51, dd=26000,r=80)
        """
        for (i, j) in g.edges:
            print("{0} => {1}".format((i,j), g[i][j]['weight']))
            res = input()
            if res =='s':
                break
        """
        labels = graphcut_kmeans(graph=g, m=len(g.nodes), k=2)
        clusters = [list() for i in range(2)]
        for i in range(len(g.nodes)):
            clusters[labels[i]].append(list(g.nodes)[i])
        print("Clusters: ", clusters)
    """
    clusters = [list() for i in range(2)]
    for i in range(len(g.nodes)):
        clusters[labels[i]].append(list(g.nodes)[i])
    #print("Clusters: ", clusters)
    sub1 = g.copy()
    sub1.remove_nodes_from(clusters[0])
    g = sub1
    labels = graphcut_kmeans(graph=g, m=len(g.nodes), k=3)
    clusters = [list() for i in range(3)]
    for i in range(len(g.nodes)):
        clusters[labels[i]].append(list(g.nodes)[i])
    #print("Clusters: ", clusters)
    lungs = list(map(len, clusters))
    maximum = max(lungs)
    index = lungs.index(maximum)
    clusters.pop(index)
    lungs = clusters
    #print("Lungs: ", lungs)
    #print(graphcut_kmeans(graph=g, m=len(g.nodes), k=4, info=True))
    #nx.draw(g, with_labels=True, font_weight='bold')
    #plt.show()
    """
