from superpixel import return_superpixels, pre_process, get_image_lut, show, get_datamark, load_datasets, get_superpixels
from graphcut import cut, similarity_distance
from kmeans import graphcut_kmeans
import cv2
import networkx as nx
import matplotlib.pyplot as plt 
import math
from skimage.morphology import opening
import pydicom
import numpy as np

if __name__ == "__main__":
    g = nx.Graph()
    """
    image = cv2.imread("./outputs/lung.png", 0)
    
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
    datasets = load_datasets('C:/Users/Luis Carlos/Documents/LCTSC/LCTSC-Test-S1-101')
    datamarked = get_datamark(datasets, 'LUNG')
    dataset = pydicom.dcmread('./outputs/000075.dcm')
    for dataset in datamarked:
        pixel_array = get_image_lut(dataset)
        image = pre_process(pixel_array)
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
                g[i][j]['weight'] =  similarity_distance(color1, color2, d1, d2,di=51, dd=26000,r=75)
                #print("({0}, {1}) -> {2}".format(i, j, g[i][j]['weight']))
        #input()        
        
        remove_whites = [ i for i in g.nodes if g.nodes[i]['info']['color'] >= 150]
        sub1 = g.copy()
        sub1.remove_nodes_from(remove_whites)
        g = sub1
    
        clusters = list()
        cont =0
        for i in nx.connected_components(g):
            cont += 1
            if len(i) < 350:
                clusters.append(i)
        print(cont)
        """
        
        labels = graphcut_kmeans(graph=g, m=len(g.nodes), k=7)
        clusters = [list() for i in range(6)]
        for i in range(len(g.nodes)):
            clusters[labels[i]].append(list(g.nodes)[i])
        
        print("Clusters: ", clusters)
        
        for i in range(1):
            removes = list(map(len, clusters))
            maximum = max(removes)
            index = removes.index(maximum)
            removed = clusters.pop(index)
            sub1 = g.copy()
            sub1.remove_nodes_from(removed)
            g = sub1
        """
        lungs = clusters
        #print("Lungs: ", lungs)
        image = cv2.imread("./outputs/saida-superpixels.png", 0)
        #show(image)
        #show(pixel_array)
        for i in lungs:
            for j in i:
                pixel_array[tuple(g.nodes[j]['info']['coordinates'])] = 255
        
        filename = './outputs/lungs/'+ str(round(dataset.ImagePositionPatient[2], 2)) + '0.png'
        
        cv2.imwrite(filename, pixel_array)
        print("Saved: ", filename)
        #show(pixel_array)
        
    #print(graphcut_kmeans(graph=g, m=len(g.nodes), k=4, info=True))
    #nx.draw(g, with_labels=True, font_weight='bold')
    #plt.show()
