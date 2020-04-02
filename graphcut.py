import numpy as np 
import networkx as nx
import math

def cut(graph):
    d = list()
    for i in graph.nodes:
        soma = 0
        for j in graph[i]:
            soma += graph[i][j]['weight']
        d.append(soma)
    #Determinando matriz D
    d = np.diag(d)
    w = nx.to_numpy_array(graph)
    result = np.linalg.eigh(d-w)
    eignvector = result[1][:,1]
    #print("The second smallest eignvalue: {0}\nEignvector: {1}".format(result[0][1], eignvector))
    if ((eignvector < 0).all() or (eignvector >= 0).all()) or (len(graph.nodes) == 2):
        return list(graph.nodes)
    else:
        sub1 = graph.copy()
        sub2 = graph.copy()
        sub1.remove_nodes_from(np.array(list(graph.nodes))[(eignvector < 0)])
        sub2.remove_nodes_from(np.array(list(graph.nodes))[(eignvector >= 0)])
        return [cut(sub1), cut(sub2)]
    
    

if __name__ == "__main__":
    g = nx.Graph()
    g.add_nodes_from([1,2,3,4,5,6])
    g.nodes[1]['info'] = {'dist': (0,0), 'color': 40}
    g.nodes[2]['info'] = {'dist': (1,0), 'color': 30}
    g.nodes[3]['info'] = {'dist': (0,1), 'color': 1}
    g.nodes[4]['info'] = {'dist': (1,1), 'color': 2}
    g.nodes[5]['info'] = {'dist': (0,2), 'color': 50}
    g.nodes[6]['info'] = {'dist': (1,2), 'color': 60}
    g.add_edges_from([(1,2), (1,3), (2,1), (2, 4), (3, 1), (3, 4), (3, 5), (4, 2), (4, 3), (4, 6), (5, 3), (5, 6), (6, 4), (6, 5)], weight=0)
    number = len(g.nodes)
    d = list()
    r = 3
    for i in g.nodes:
        color1 = g.nodes[i]['info']['color']
        soma_ = 0
        for j in g[i]:
            color2 = g.nodes[j]['info']['color']
            mean = (color1 + color2)/2
            soma = ((color1 - mean)**2) + ((color2 - mean)**2)
            p1 = (color1 - color2) ** 2
            p2 = p1 / ((math.sqrt(soma/2)**2) + 1e-5)
            d1 = g.nodes[i]['info']['dist']
            d2 = g.nodes[j]['info']['dist']
            p3 = math.sqrt((((d1[0]-d2[0])**2)+((d1[1]-d2[1])**2)))
            if p3 < r:
                g[i][j]['weight'] =  math.exp(-(p2)) * math.exp(-p3) #math.exp(-(abs(color1 - color2) * abs(color1 - color2))/(2*((math.sqrt(soma/2))**2)))
            else:
                g[i][j]['weight'] = 0
    print(cut(g))