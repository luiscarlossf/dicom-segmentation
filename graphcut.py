import numpy as np 
import networkx as nx
import math
import matplotlib.pyplot as plt
def distance(x1, x2):
    if type(x1) == type((0,0)):
        return math.sqrt(((x1[0] - x2[0])**2) + ((x1[1]-x2[1])**2))
    return math.sqrt((x1 - x2)**2)

def similarity_distance(i1, i2, d1, d2, di=51, dd=102.4, r=2):
    """
    Calcula a similaridade entre dois pontos, levando em conta a 
    intensidade e a distância.

    @param i1: float - nível de intensidade do primeiro ponto.
    @param i2: float - nível de intensidade do segundo ponto.
    @param d1: float - localização do primeiro ponto.
    @param d2: float - localização do segundo ponto.
    @param di: float - desvio padrão das distâncias, 10% do valor máximo do conjunto.
    @param dd: float - desvio padrão das distâncias, 10% do valor máximo do conjunto.
    @param r: int - limite de proximidade

    @return similarity:
    """
    intensity = distance(i1, i2)
    distances = distance(d1, d2)
    if distances < r:
        return math.exp(-(intensity**2/di)) * math.exp(-(distances**2/dd))
    else:
        return 0

def cut(graph):
    if (len(graph.nodes) < 2):
        return list(graph.nodes)
    laplacian = nx.normalized_laplacian_matrix(graph).toarray()
    eignvector = np.linalg.eigh(laplacian)[1][:,1]
    eignvalues = nx.normalized_laplacian_spectrum(graph)
    #print("The second smallest eignvalue: {0}\nEignvector: {1}".format(result[0][1], eignvector))
    print("Eignvalues {0}".format(eignvalues))
    print("Eignvector {0}".format(eignvector))
    histogram = np.histogram(eignvalues, bins=np.size(eignvalues), density=True)
    if len(eignvalues) > 2:
        ratio = eignvalues[2] - eignvalues[1]
    else:
        ratio = eignvalues[1] - eignvalues[1]
    print("Ratio {0:.2f}".format(ratio))
    threshold = 0 #np.mean(eignvector)
    res = input("Divide?")
    if res == 'n': #(ratio < 0.06) :#((eignvector < threshold).all() or (eignvector >= threshold).all()) or (len(graph.nodes) <= 2): #len(graph.nodes) <= 2
        return list(graph.nodes)
    else:
        sub1 = graph.copy()
        sub2 = graph.copy()
        sub1.remove_nodes_from(np.array(list(graph.nodes))[(eignvector < threshold)])
        sub2.remove_nodes_from(np.array(list(graph.nodes))[(eignvector >= threshold)])
        return [cut(sub1), cut(sub2)]
    
    

if __name__ == "__main__":
    g = nx.Graph()
    g.add_nodes_from([1,2,3,4,5,6])
    g.nodes[1]['info'] = {'dist': (0,0), 'color': 1}
    g.nodes[2]['info'] = {'dist': (1,0), 'color': 2}
    g.nodes[3]['info'] = {'dist': (0,1), 'color': 20}
    g.nodes[4]['info'] = {'dist': (1,1), 'color': 20}
    g.nodes[5]['info'] = {'dist': (0,2), 'color': 50}
    g.nodes[6]['info'] = {'dist': (1,2), 'color': 60}
    g.add_edges_from([(1,2), (1,3), (2, 4), (3, 4), (3, 5), (4, 6), (5, 6)], weight=0)
    r = 2
    
    for (i, j) in g.edges:
        color1 = g.nodes[i]['info']['color']
        color2 = g.nodes[j]['info']['color']
        d1 = g.nodes[i]['info']['dist']
        d2 = g.nodes[j]['info']['dist']
        g[i][j]['weight'] =  similarity_distance(color1, color2, d1, d2, 25.5, 102.4, 2)
    laplacian = nx.laplacian_matrix(g).toarray()
    eign = np.linalg.eigh(laplacian)
    print(eign)
    print(cut(g))
    #nx.draw(g, with_labels=True, font_weight='bold')
    #plt.show()