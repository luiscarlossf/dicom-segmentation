import os
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

path = 'C:/Users/Luis Carlos/Documents/LCTSC'

def concat_dir(entrada):
    return path + '/' + entrada

def test_in(entrada):
    return 'Test' in entrada

def train_in(entrada):
    return 'Train' in entrada

if __name__ == "__main__":
    dir_list = os.listdir(path)
    lista = filter(train_in, dir_list)
    arquivo = open('./outputs/tests/dice-training','w')
    cont_dice = 0
    
    for set_ in lista:
        datasets = load_datasets(concat_dir(set_))
        datamarked = get_datamark(datasets, 'LUNG')
        flag_greater = 0
        flag_less = 0
        dices = 0
        g = nx.Graph()
        nome_conjunto = set_.replace('-', '_')
        for dataset in datamarked:
            pixel_array = get_image_lut(dataset)
            image = pre_process(pixel_array)
            sp, adjacency = return_superpixels(image, info=False)

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
            
            lungs = clusters
            image = cv2.imread("./outputs/saida-superpixels.png", 0)
            image1 = np.zeros((512, 512))
            image2 = np.zeros((512,512))
            for i in lungs:
                for j in i:
                    pixel_array[tuple(g.nodes[j]['info']['coordinates'])] = 255
                    image1[tuple(g.nodes[j]['info']['coordinates'])] = 255
            #Obtendo coordenadas da marcação do especialista.
            coordinates = get_mark(datasets[0], position=dataset.ImagePositionPatient,spacing=dataset.PixelSpacing, roi="Lung")
            for c in coordinates:
                image2 = cv2.fillPoly(image2, np.int32([np.array(c)]), 255)

            filename_image = str(round(dataset.ImagePositionPatient[2], 2))
            d = dice(image1, image2)
            dices += d

            if d < 0.85 and (flag_less < 2):
                filename = './outputs/tests/less/'+ set_ + filename_image + '0_'+ str(d) +'.png'
                cv2.imwrite(filename, pixel_array)
                filename = './outputs/tests/less/'+ set_ + filename_image + '0_ground_truth.png'
                cv2.imwrite(filename, image2)
                flag_less += 1
            if d > 0.97 and (flag_greater < 2):
                filename = './outputs/tests/greater/'+ set_+filename_image + '0_'+str(d)+'.png'
                cv2.imwrite(filename, pixel_array)
                filename = './outputs/tests/greater/'+ set_ + filename_image + '0_ground_truth.png'
                cv2.imwrite(filename, image2)
                flag_greater += 1
        amount_data = len(datamarked)
        dice_total = dices/amount_data
        cont_dice += dice_total
        arquivo.write(nome_conjunto+': '+str(dice_total)+'\n')
        print("Total DICE Coefficient: {0}".format(dices/amount_data))
    media_geral = cont_dice / len(list(lista))
    arquivo.write('MÉDIA GERAL DE DICE: '+str(media_geral))
    arquivo.close()
