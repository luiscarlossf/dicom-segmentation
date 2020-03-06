from igraph import *
#from superpixel import return_superpixels
#import cv2

g = Graph()

#image = cv2.imread("./outputs/lung.png")
#spixels, adjac = return_superpixels(image)

g.add_vertices(8)
g.vs["color"] = [100, 20,2,30,4,0.1,40,50]
g.vs["centroid"] = [(-1,-1), (0,0), (0,2), (2, 0), (2,2), (1,1), (0,3), (2,3)]
g.vs["flag"] = 8 * [False]
g.add_edge(1,2)
g.add_edge(1,3)
g.add_edge(1,5)
g.add_edge(2,4)
g.add_edge(2,5)
g.add_edge(2,6)
g.add_edge(3,4)
g.add_edge(3,5)
g.add_edge(4,5)
g.add_edge(4,6)
g.add_edge(4,7)
g.add_edge(6,7)
edge = None
for idx, e in enumerate(g.es) :
    edge = e.tuple
    color1 = g.vs[edge[0]]['color']
    color2 = g.vs[edge[1]]['color']
    mean = (color1 + color2)/2
    soma = ((color1 - mean)**2) + ((color2 - mean)**2)
    weight = math.exp(-(abs(color1 - color2) * abs(color1 - color2))/(math.sqrt(soma/2)))
    g.es[idx]["capacity"] = weight
    if weight > 0.0000001:
        print("Aresta", str(edge), "tem o peso ", str(weight))

adjlist = g.get_adjlist()

seeds = {5}
for seed in seeds:
    g.vs[seed]['flag'] = True







