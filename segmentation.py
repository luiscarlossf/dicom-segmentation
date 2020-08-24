from superpixel import return_superpixels, pre_process, get_image_lut, show, get_datamark, load_datasets, get_superpixels, get_mark
from graphcut import cut, similarity_distance
from kmeans import graphcut_kmeans
import cv2
import networkx as nx
import matplotlib.pyplot as plt 
import math
from skimage.morphology import opening
import pydicom
import numpy as np
import time
from measures import dice

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
    start = time.time()
    datasets = load_datasets('C:/Users/Luis Carlos/Documents/LCTSC/LCTSC-Test-S1-101')
    end = time.time() - start
    print("{0} segundos para carregar as imagens".format(end))
    start = time.time()
    datamarked = get_datamark(datasets, 'LUNG')
    end = time.time() - start
    print("{0} segundos para carregar as imagens marcadas.".format(end))
    #dataset = pydicom.dcmread('./outputs/000075.dcm')
    end_pre = 0
    end_lut = 0
    end_super = 0
    end_graph = 0
    end_remove = 0
    end_components = 0
    end_lungs = 0
    dices = 0
    for dataset in datamarked:
        start = time.time()
        pixel_array = get_image_lut(dataset)
        end_lut += time.time() - start
        start = time.time()
        image = pre_process(pixel_array)
        end_pre += time.time() - start
        start = time.time()
        sp, adjacency = return_superpixels(image, info=False)
        end_super += time.time() - start
        #g.add_nodes_from(list(sp.keys()))
        
        start = time.time()
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
        end_graph += time.time() - start
        start = time.time()
        remove_whites = [ i for i in g.nodes if g.nodes[i]['info']['color'] >= 150]
        sub1 = g.copy()
        sub1.remove_nodes_from(remove_whites)
        g = sub1
        end_remove += time.time() - start
        start = time.time()
        clusters = list()
        cont =0
        for i in nx.connected_components(g):
            cont += 1
            if len(i) < 350:
                clusters.append(i)
        end_components += time.time() - start
        
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
        start = time.time()
        lungs = clusters
        #print("Lungs: ", lungs)
        image = cv2.imread("./outputs/saida-superpixels.png", 0)
        image1 = np.zeros((512, 512))
        image2 = np.zeros((512,512))
        #show(image)
        #show(pixel_array)
        for i in lungs:
            for j in i:
                pixel_array[tuple(g.nodes[j]['info']['coordinates'])] = 255
                image1[tuple(g.nodes[j]['info']['coordinates'])] = 255
        #Obtendo coordenadas da marcação do especialista.
        coordinates = get_mark(datasets[0], position=dataset.ImagePositionPatient,spacing=dataset.PixelSpacing, roi="Lung")
        for c in coordinates:
            image2 = cv2.fillPoly(image2, np.int32([np.array(c)]), 255)
            #image2[c] = 255
        filename_image = str(round(dataset.ImagePositionPatient[2], 2))
        #show(image1)
        #show(image2)
        d = dice(image1, image2)
        print("DICE Coefficient em {0}: {1}".format(filename_image, d))
        dices += d

        filename = './outputs/lungs/'+ filename_image + '0.png'
        
        cv2.imwrite(filename, pixel_array)
        end_lungs += time.time() - start
        print("Saved: ", filename)
    amount_data = len(datamarked)
    print("{0} segundos para conseguir a imagem em LUT.".format(end_lut/amount_data))
    print("{0} segundos para pré processar a imagem.".format(end_pre/amount_data))
    print("{0} segundos para produzir e definir os superpixels da imagem.".format(end_super/amount_data))
    print("{0} segundos para gerar o grafo da imagem.".format(end_graph/amount_data))
    print("{0} segundos para aplicar o threshold nos nós do grafo.".format(end_remove/amount_data))
    print("{0} segundos para carregar e selecionar os componentes do grafo.".format(end_components/amount_data))
    print("{0} segundos para segmentar os pulmões e salvar a imagem.".format(end_lungs /amount_data))
    print("Total DICE Coefficient: {0}".format(dices/amount_data))
        #show(pixel_array)
        
    #print(graphcut_kmeans(graph=g, m=len(g.nodes), k=4, info=True))
    #nx.draw(g, with_labels=True, font_weight='bold')
    #plt.show()
