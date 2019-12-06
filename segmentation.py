from superpixel import return_superpixels
import cv2
import networkx as nx
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    g = nx.Graph()
    image = cv2.imread("./outputs/lung.png", 0)
    sp, adjacency = return_superpixels(image)
    g.add_nodes_from(range(0, len(sp)))
    g.add_edges_from(list(adjacency))
    plt.subplot(121)
    nx.draw(g, with_labels=True)
    plt.show()
