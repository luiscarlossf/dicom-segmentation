import cv2
import numpy as np
from superpixel import get_coordinates, return_superpixels
import math
import networkx as nx

image = cv2.imread("./outputs/saida1.png", 0)
g = nx.Graph()
g.add_nodes_from([1,2,3,4,5,6])
g.nodes[1]['info'] = {'dist': (0,0), 'color': 40}
g.nodes[2]['info'] = {'dist': (1,0), 'color': 30}
g.nodes[3]['info'] = {'dist': (0,1), 'color': 1}
g.nodes[4]['info'] = {'dist': (1,1), 'color': 2}
g.nodes[5]['info'] = {'dist': (0,2), 'color': 50}
g.nodes[6]['info'] = {'dist': (1,2), 'color': 60}
g.add_edges_from([(1,2), (1,3), (2,1), (2, 4), (3, 1), (3, 4), (3, 5), (4, 2), (4, 3), (4, 6), (5, 3), (5, 6), (6, 4), (6, 5)], weight=0)

graph = {1: {'color':40, 'dist':(0,0), 2:{'weight': None}, 3:{'weight': None}},
     2: {'color':30, 'dist':(1,0), 1:{'weight': None}, 4:{'weight': None}},
     3: {'color':1, 'dist':(0,1),  1:{'weight': None}, 4:{'weight': None}, 5:{'weight': None}},
     4: {'color':2, 'dist':(1,1),  2:{'weight': None}, 3:{'weight': None}, 6:{'weight': None}},
     5: {'color':50, 'dist':(0,2),  3:{'weight': None}, 6:{'weight': None}},
     6: {'color':60, 'dist':(1,2),  4:{'weight': None}, 5:{'weight': None}}
     }
def test1(label=0):
    #Calcula os superpixels da imagem
    image_ = np.copy(image)
    superpixels = cv2.ximgproc.createSuperpixelLSC(image_, 40)
    superpixels.iterate(20)
    s_number = superpixels.getNumberOfSuperpixels()
    print("O número de superpixels encontrados: {}.".format(s_number))
    masks = superpixels.getLabelContourMask()
    labels = superpixels.getLabels()

    #Obtem as coordenadas e as adjacências de cada superpixels.
    coords, adjcs = get_coordinates(labeled_image=labels,masks=masks, length=s_number)

    zeros = np.zeros(image_.shape, dtype=np.uint8)
    zeros[masks==255] = 255
    for c in coords:
        mean_r = int(np.mean(coords[c][0]))
        mean_c = int(np.mean(coords[c][1]))
        zeros[(mean_r, mean_c)] = 255

    zeros[coords[label]] = 255
    cv2.imwrite("./outputs/test.png", zeros)

def test2():
    superpixels, adjacency = return_superpixels(image, info=True)
    image_ = np.zeros((512, 512), dtype=np.uint8)
    for s in superpixels:
        image_[superpixels[s]['coordinates']] = superpixels[s]['color']
    print("Superpixel: Adjacentes")
    
    adjacencies = dict()
    for key in adjacency:
        adjacencies[key] = dict()
        if adjacency[key]:
            for a in adjacency[key]:
                color1 = superpixels[key]['color']
                color2 = superpixels[a]['color']
                mean = (color1 + color2)/2
                soma = ((color1 - mean)**2) + ((color2 - mean)**2)
                weight = math.exp(-(abs(color1 - color2) * abs(color1 - color2))/((math.sqrt(soma/2))+1e-5))
                adjacencies[key][a] = {'weight': weight}
        print("{0}: {1}".format(key, adjacencies[key]))

    cv2.imwrite("./outputs/test.png", image_)
def test3(graph):
    number = len(graph.keys())
    d = list()
    r = 3
    #Determinando matriz W
    w = np.zeros((number, number))
    for i in graph:
        color1 = graph[i]['color']
        soma_ = 0
        for j in graph[i]:
            if str(j).isdigit() and (j in graph.keys()):
                color2 = graph[j]['color']
                mean = (color1 + color2)/2
                soma = ((color1 - mean)**2) + ((color2 - mean)**2)
                p1 = (color1 - color2) ** 2
                p2 = p1 / ((math.sqrt(soma/2)**2) + 1e-5)
                d1 = graph[i]['dist']
                d2 = graph[j]['dist']
                p3 = math.sqrt((((d1[0]-d2[0])**2)+((d1[1]-d2[1])**2)))
                if p3 < r:
                    graph[i][j]['weight'] =  math.exp(-(p2)) * math.exp(-p3) #math.exp(-(abs(color1 - color2) * abs(color1 - color2))/(2*((math.sqrt(soma/2))**2)))
                else:
                    graph[i][j]['weight'] = 0
                soma_ += graph[i][j]['weight']
                w[i-1][j-1] = graph[i][j]['weight']
        d.append(soma_)
        print("{0} : {1}".format(i, graph[i]))
    #Determinando matriz D
    d = np.diag(d)
    result = np.linalg.eigh(d-w)
    print("The second smallest eignvalue: {0}\nEignvector: {1}".format(result[0][1], result[1][:, 1]))
               
if __name__ == "__main__":
    #test3(graph)
    sub1 = g.subgraph([1,2,3,4])
    sub2 = g.subgraph([5,6])
    print("First graph")
    print(g.nodes[1]['info']['color'])
    print(g.edges.data())