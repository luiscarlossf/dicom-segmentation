import cv2
import numpy as np
import math
import networkx as nx
from superpixel import get_coordinates
import pydicom
from graphcut import cut

def show(pixel_array):
    """
    Exibe a imagem a partir de um array de pixels.

    :param pixel_array: numpy array com os pixels da imagem.
    :return:
    """
    print("Show was called.")
    cv2.imshow('image', pixel_array)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()


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

def test4(center=None, window=None):
    """
    Windoe Center Level for Lung = -500
    Window Width for Lung = 1500
    """
    dataset = pydicom.dcmread('./outputs/000075.dcm')
    pixel_array = np.copy(dataset.pixel_array)
    if dataset.RescaleType == 'HU': #O que fazer quando não tem Rescale
        c = center if center else dataset.WindowCenter #center level
        w = window if window else dataset.WindowWidth #window width
        pixel_array = int(dataset.RescaleSlope) * pixel_array + int(dataset.RescaleIntercept)
        condition1 = pixel_array <= (c- 0.5 - (w - 1)/ 2)
        condition2 = pixel_array > (c- 0.5 + (w - 1)/2)
        pixel_array = np.piecewise(pixel_array, [condition1, condition2], [0,255, lambda pixel_array: ((pixel_array - (c - 0.5))/(w-1)+0.5) * (255 - 0)]).astype(np.uint8)
    
    #spixel_array = cv2.GaussianBlur(pixel_array, (5,5), 0.4)
    show(pixel_array)
    pixel_array[pixel_array > 180]= 255
    show(pixel_array)
    #retval = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    #pixel_array = cv2.morphologyEx(pixel_array, cv2.MORPH_CLOSE,retval)
    #p0 = np.array([[int(np.binary_repr(pixel_array[i,j], 8)[0]) * 255 for j in range(0, pixel_array.shape[1])] for i in range(0, pixel_array.shape[0])])
    p1 = np.array([[int(np.binary_repr(pixel_array[i,j], 8)[1]) * 255 for j in range(0, pixel_array.shape[1])] for i in range(0, pixel_array.shape[0])])
    p2 = np.array([[int(np.binary_repr(pixel_array[i,j], 8)[2]) * 255 for j in range(0, pixel_array.shape[1])] for i in range(0, pixel_array.shape[0])])
    p3 = np.array([[int(np.binary_repr(pixel_array[i,j], 8)[3]) * 255 for j in range(0, pixel_array.shape[1])] for i in range(0, pixel_array.shape[0])])
    p4 = np.array([[int(np.binary_repr(pixel_array[i,j], 8)[4]) * 255 for j in range(0, pixel_array.shape[1])] for i in range(0, pixel_array.shape[0])])
    p5 = np.array([[int(np.binary_repr(pixel_array[i,j], 8)[5]) * 255 for j in range(0, pixel_array.shape[1])] for i in range(0, pixel_array.shape[0])])
    p6 = np.array([[int(np.binary_repr(pixel_array[i,j], 8)[6]) * 255 for j in range(0, pixel_array.shape[1])] for i in range(0, pixel_array.shape[0])])
    p7 = np.array([[int(np.binary_repr(pixel_array[i,j], 8)[7]) * 255 for j in range(0, pixel_array.shape[1])] for i in range(0, pixel_array.shape[0])])

    pixel_array = np.copy( p1 * p2 * p3 * p4 * p5 * p6 * p7).astype(np.uint8)
    show(pixel_array)

    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(pixel_array, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 1000

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255
    pixel_array = img2.astype(np.uint8)
    

    retval = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    pixel_array = cv2.morphologyEx(pixel_array, cv2.MORPH_CLOSE, retval)
    show(pixel_array)

    

    
    '''Mais apropriado para imagens binárias'''
    #superpixels = cv2.ximgproc.createSuperpixelLSC(pixel_array, region_size=40)
    '''Mais apropriado para imagens na janela do pulmão'''
    superpixels = cv2.ximgproc.createSuperpixelSEEDS(pixel_array.shape[0], pixel_array.shape[1], image_channels=1, num_superpixels=350, num_levels=20)
    superpixels.iterate(pixel_array, 15)
    masks = superpixels.getLabelContourMask()
    pixel_array[masks == 255] = 200
    labels = superpixels.getLabels()
    number_spixels = superpixels.getNumberOfSuperpixels()
    print("Número de superpixels criados: {}".format(number_spixels))
    #show(pixel_array)
    coordinates, adjacency = get_coordinates(labeled_image=labels, masks=masks, length=number_spixels)
    spixels = dict()
    for key in coordinates:
        mean_r = int(np.mean(coordinates[key][0]))
        mean_c = int(np.mean(coordinates[key][1]))
        centroid = (mean_r, mean_c)
        color_mean = np.mean(pixel_array[tuple(coordinates[key])])
        spixels[key] = {"label": key, "centroid": centroid, "color": color_mean, "coordinates":coordinates[key]}
        cv2.putText(pixel_array,"{0}".format(key), (centroid[1], centroid[0]),  cv2.FONT_HERSHEY_SIMPLEX,0.3,123)
    show(pixel_array)
    """
    g = nx.Graph()
    for key in spixels.keys():
        g.add_node(key, info=spixels[key], color='red')

    colors = list()
    distances = list()
    for i in g.nodes:
        colors.append(g.nodes[i]['info']['color'])
        for j in g[i]:
            d1 = g.nodes[i]['info']['centroid']
            d2 = g.nodes[j]['info']['centroid']
            distances.append(math.sqrt((((d1[0]-d2[0])**2)+((d1[1]-d2[1])**2))))
    aux = [((color - np.mean(colors))**2) for color in colors]
    deviation_colors = math.sqrt(sum(aux)/len(aux)) if sum(aux) != 0 else 0.01
    print(deviation_colors)
    aux = [((dist - np.mean(distances))**2) for dist in distances]
    deviation_distances = math.sqrt(sum(aux)/len(aux)) if sum(aux) != 0 else 0.01
    print(deviation_distances)
    for i in adjacency:
        for j in adjacency[i]:
            g.add_edge(i, j)
            color1 = g.nodes[i]['info']['color']
            color2 = g.nodes[j]['info']['color']
            mean = (color1 + color2)/2
            soma = ((color1 - mean)**2) + ((color2 - mean)**2)
            p1 = math.sqrt((color1 - color2) ** 2)
            p2 = p1 / (deviation_colors**2)
            d1 = g.nodes[i]['info']['centroid']
            d2 = g.nodes[j]['info']['centroid']
            p3 = (math.sqrt((((d1[0]-d2[0])**2)+((d1[1]-d2[1])**2)))) 
            g[i][j]['weight'] =  math.exp(-(p2)) * math.exp(-p3 / (deviation_distances ** 2)) #math.exp(-(abs(color1 - color2) * abs(color1 - color2))/(2*((math.sqrt(soma/2))**2)))

    print(cut(g))
    nx.draw(g, with_labels=True, font_weight='bold')
    plt.show()
    """

def test5():
    laplacian_matrix = np.array([[5,-4,0,0,1], [-4,6,-1,-1,0], [0,-1,2,-1,0], [0,-1,-1,6,-4], [-1, 0, 0, -4, 5]])
    eign = np.linalg.eigh(laplacian_matrix)
    norm = np.linalg.norm(eign[0])
    print(eign[1][:,1])

if __name__ == "__main__":
    test4(-500, 1500)
    