import numpy as np 
import networkx as nx
import math
import matplotlib.pyplot as plt

def cut(graph):
    laplacian = nx.laplacian_matrix(graph).toarray()
    #print(np.linalg.eigh(laplacian))
    eignvector = np.array(nx.fiedler_vector(graph))
    #print("The second smallest eignvalue: {0}\nEignvector: {1}".format(result[0][1], eignvector))
    print("Eignvector {0}".format(eignvector))
    histogram = np.histogram(eignvector, bins=np.size(eignvector), density=True)
    ratio =  (np.max(histogram[0])/np.min(histogram[0]) - 1) * 100
    print("Histogram {0}\nRatio {1}".format(histogram, ratio))
    threshold = np.mean(eignvector)
    if ((eignvector < threshold).all() or (eignvector >= threshold).all()) or (len(graph.nodes) <= 2):
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
    g.nodes[1]['info'] = {'dist': (0,0), 'color': 40}
    g.nodes[2]['info'] = {'dist': (1,0), 'color': 30}
    g.nodes[3]['info'] = {'dist': (0,1), 'color': 1}
    g.nodes[4]['info'] = {'dist': (1,1), 'color': 2}
    g.nodes[5]['info'] = {'dist': (0,2), 'color': 50}
    g.nodes[6]['info'] = {'dist': (1,2), 'color': 60}
    g.add_edges_from([(1,2), (1,3), (2,1), (2, 4), (3, 1), (3, 4), (3, 5), (4, 2), (4, 3), (4, 6), (5, 3), (5, 6), (6, 4), (6, 5)], weight=0)
    r = 2
    colors = list()
    distances = list()

    for i in g.nodes:
        colors.append(g.nodes[i]['info']['color'])
        for j in g[i]:
            d1 = g.nodes[i]['info']['dist']
            d2 = g.nodes[j]['info']['dist']
            distances.append(math.sqrt((((d1[0]-d2[0])**2)+((d1[1]-d2[1])**2))))
    aux = [((color - np.mean(colors))**2) for color in colors]
    deviation_colors = math.sqrt(sum(aux)/len(aux))
    aux = [((dist - np.mean(distances))**2) for dist in distances]
    deviation_distances = math.sqrt(sum(aux)/len(aux))
    for i in g.nodes:
        color1 = g.nodes[i]['info']['color']
        soma_ = 0
        for j in g[i]:
            color2 = g.nodes[j]['info']['color']
            mean = (color1 + color2)/2
            soma = ((color1 - mean)**2) + ((color2 - mean)**2)
            p1 = math.sqrt((color1 - color2) ** 2)
            p2 = p1 / ((deviation_colors**2))
            d1 = g.nodes[i]['info']['dist']
            d2 = g.nodes[j]['info']['dist']
            p3 = (math.sqrt((((d1[0]-d2[0])**2)+((d1[1]-d2[1])**2)))) #/ ((deviation_distances ** 2) + 1e-11)
            if p3 < r:
                g[i][j]['weight'] =  math.exp(-(p2)) #* math.exp(-p3) #math.exp(-(abs(color1 - color2) * abs(color1 - color2))/(2*((math.sqrt(soma/2))**2)))
            else:
                g[i][j]['weight'] = 0
    print(cut(g))
    #nx.draw(g, with_labels=True, font_weight='bold')
    #plt.show()