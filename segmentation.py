from superpixel import return_superpixels
from graphcut import cut
import cv2
import networkx as nx
import matplotlib.pyplot as plt 
import math
from skimage.morphology import opening

if __name__ == "__main__":
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
            g.add_edge(i, j)
            color1 = g.nodes[i]['info']['color']
            color2 = g.nodes[j]['info']['color']
            mean = (color1 + color2)/2
            soma = ((color1 - mean)**2) + ((color2 - mean)**2)
            p1 = (color1 - color2) ** 2
            p2 = p1 / ((math.sqrt(soma/2)**2) + 1e-5)
            d1 = g.nodes[i]['info']['centroid']
            d2 = g.nodes[j]['info']['centroid']
            p3 = math.sqrt((((d1[0]-d2[0])**2)+((d1[1]-d2[1])**2)))
            g[i][j]['weight'] =  math.exp(-(p2)) * math.exp(-p3) #math.exp(-(abs(color1 - color2) * abs(color1 - color2))/(2*((math.sqrt(soma/2))**2))) 

    print(cut(g))
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()
