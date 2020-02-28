from superpixel import return_superpixels
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
    
    sp, adjacency = return_superpixels(image)
    #g.add_nodes_from(list(sp.keys()))
    
    nodes_r = list()
    nodes_b = list()
    labels = {}
    for key in sp.keys():
        if sp[key]['color'] < 15 :
            g.add_node(key, info=sp[key], flag=False, color='red')
            nodes_r.append(key)
            labels[key] = key
        else:
            #g.add_node(key, info=sp[key], flag=False)
            nodes_b.append(key)
        #labels[key] = key

    for t in adjacency:
        if (t[0] not in nodes_r) or (t[1] not in nodes_r):
            continue
        color1 = sp[t[0]]['color']
        color2 = sp[t[1]]['color']
        mean = (color1 + color2)/2
        soma = ((color1 - mean)**2) + ((color2 - mean)**2) + 0.1
        weight = math.exp(-(abs(color1 - color2) * abs(color1 - color2))/(math.sqrt(soma/2)))
        if weight <= 1.5:
            g.add_edge(t[0], t[1], weight=weight)
    #g.add_edges_from(list(adjacency))
    pos = nx.circular_layout(g)
    plt.subplot(121)
    nx.draw_networkx_nodes(g, pos=pos, nodelist=nodes_r, node_color='red')
   # nx.draw_networkx_nodes(g, pos=pos, nodelist=nodes_b, node_color='b')
    nx.draw_networkx_edges(g, pos=pos, edgelist=list(g.edges))
    nx.draw_networkx_labels(g, pos, labels)
    plt.show()
