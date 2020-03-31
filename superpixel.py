import cv2
import os
from dicompylercore import dicomparser
import pydicom
import numpy as np
from random import choices
from skimage.measure import regionprops
import time

path = "C:/Users/luisc/Documents/dicom-database/LCTSC/LCTSC-Test-S1-101"

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

def load_datasets(path):
    """
    Carrega todos os arquivos DICOM de um diretório.

    :param path: str indicando o diretório de origem dos
                 dos arquivos que se deseja carregar
    :return volume: lista com os arquivos dicom
    """
    volume = list()
    slice_locations = dict()
    rtstruct = None
    for root, dirs, files in os.walk(path):
        for file in files:
            ds = pydicom.dcmread(root + '/'+ file)
            if ds.Modality == 'CT':
                slice_locations[ds.SliceLocation] = ds
            elif ds.Modality == 'RTSTRUCT':
                rtstruct = ds
    volume = [slice_locations[key] for key in sorted(slice_locations.keys())]
    volume.insert(0, rtstruct)
    return volume

def get_image(slice):
    """
    Retorna um array de pixels com valores LUT

    :param slice: dataset dicom
    :return pixels_slice:  array numpy com os pixels transformados
    """
    ds = dicomparser.DicomParser(slice)
    intercept, slope = ds.GetRescaleInterceptSlope()
    rescaled_image = slice.pixel_array * slope + intercept
    window, level = ds.GetDefaultImageWindowLevel()
    pixels_slice = ds.GetLUTValue(rescaled_image, window, level)

    return pixels_slice

def get_datamark(datasets, roi):
    """
    Retorna apenas os datasets que contenham marcações
    para região de interesse (ROI)

    :param datasets: list()
    :param roi: str()
    :return datasets: list()
    """
    marking = dicomparser.DicomParser(datasets[0])
    structures = marking.GetStructures()
    roi_number = None
    for i in structures:
        if roi in structures[i]['name']:
            roi_number = structures[i]['id']
    if roi_number == None:
        raise NameError(roi + " não está entre as estruturas marcadas")
    marked_slices = marking.GetStructureCoordinates(roi_number)

    return [ i for i in datasets[1:] if str(round(i.ImagePositionPatient[2], 2)) + '0' in marked_slices ]

def get_mark(dataset, position, spacing, roi):
    """
    Retorna a marcação de uma região de interesse (ROI) dado um dataset DICOM
    na modalidade RTSTRUCT

    :param dataset: Dataset da modalidade RTSTRUCT 
    :param position: tuple() com coordenadas x,y,z do canto superior esquerdo
                     da imagem
    :param spacing: tuple() com a distância física no paciente entre o centro de cada pixel, 
                    especificado por um par numérico - espaçamento de linhas adjacentes 
                    (delimitador) espaçamento de colunas adjacentes em mm.
    :param roi: str() representando o nome da região de interesse que se deseja obter a marcação

    :return coordinates: list() com indices das marcação na imagem.
    """
    marking = dicomparser.DicomParser(dataset)
    structures = marking.GetStructures()
    roi_number = None
    for i in structures:
        if roi in structures[i]['name']:
            roi_number = structures[i]['id']
    if roi_number == None:
        raise NameError(roi + " não está entre as estruturas marcadas")
    
    coordinates = list()
    for mark in marking.GetStructureCoordinates(roi_number)[str(round(position[2], 2)) + '0']:
        contour = np.array(mark['data'])
        rows = ((contour[:, 1] - position[1])/spacing[1]).astype(int)
        columns = ((contour[:, 0] - position[0])/spacing[0]).astype(int)
        coordinates.append([rows, columns])
    
    return coordinates

def get_coordinates(labeled_image, masks, length):
    """
    Retorna as coordenadas de cada rótulo na imagem
    De modo que rótulo 1:
    {1: lits() com as coordenadas no eixo x e uma outra lista do eixo y}
    :return coordinates: dict() com coordenadas de cada superpixel
    """
    coordinates = dict()
    adjacency = set()
    for i in np.arange(labeled_image.shape[0]):
        for j in np.arange(labeled_image.shape[1]):
            if (masks[i, j] == 255) and (i > 0) and (i < (labeled_image.shape[0] - 1)) and (j > 0) and (j < (labeled_image.shape[1] - 1)):
                if labeled_image[i+1, j] != labeled_image[i-1, j]:
                    adjacency.add((labeled_image[i+1, j], labeled_image[i-1, j]))
                #if labeled_image[i-1, j] !=  labeled_image[i+1, j]:
                #    adjacency.add((labeled_image[i-1, j], labeled_image[i+1, j]))
                if labeled_image[i, j+1] != labeled_image[i, j-1]:
                    adjacency.add((labeled_image[i, j+1], labeled_image[i, j-1]))
                #if labeled_image[i, j-1] !=  labeled_image[i, j+1]:
                #    adjacency.add((labeled_image[i, j-1], labeled_image[i, j+1]))
                if labeled_image[i+1, j-1] != labeled_image[i-1, j+1]:
                    adjacency.add((labeled_image[i+1, j-1],labeled_image[i-1, j+1]))
                #if labeled_image[i-1, j+1]!= labeled_image[i+1, j-1]:
                #    adjacency.add((labeled_image[i-1, j+1], labeled_image[i+1, j-1]))
                if labeled_image[i+1, j+1] != labeled_image[ i-1, j-1]:
                    adjacency.add((labeled_image[i+1, j+1], labeled_image[ i-1, j-1]))
                #if labeled_image[ i-1, j-1] != labeled_image[i+1, j+1]:
                #    adjacency.add((labeled_image[ i-1, j-1], labeled_image[i+1, j+1]))
                
    
            try:
                coordinates[labeled_image[i, j]][0].append(i)
                coordinates[labeled_image[i, j]][1].append(j)
            except KeyError:
                coordinates[labeled_image[i, j]] = [list(), list()]
                coordinates[labeled_image[i, j]][0].append(i)
                coordinates[labeled_image[i, j]][1].append(j)
    return coordinates , list(adjacency)

class Group:
    def __init__(self, center):
        self.x = center[0]
        self.y = center[1]
        self.samplesx = list()
        self.samplesy = list()
    
    def recalcula(self):
        anterior = (self.x, self.y)
        self.x = sum(self.samplesx)/len(self.samplesx)
        self.y = sum(self.samplesy)/len(self.samplesy)
        self.samplesx = list()
        self.samplesy = list()
        return anterior != (self.x, self.y)
    
    def set_sample(self, sample):
        self.samplesx.append(sample['x'])
        self.samplesx.append(sample['y'])
    
    def __repr__(self):
        return "Group {0}\n center: {1}   samples: {2}\n".format(self, (self.x, self.y), [self.samplesx, self.samplesy])
    
def get_superpixel(image, region_size, smooth, num_iteration=10, compactness=0.075):
    start = time.time()
    image_ = np.copy(image)
    s = cv2.ximgproc.createSuperpixelLSC(image, region_size, compactness)
    print("Foram gerados {0} superpixels.".format(s.getNumberOfSuperpixels))
    s.iterate()
    s.enforceLabelConnectivity(10)
    seconds = time.time() - start 
    print("Levou {0} segundos para gerar superpixels".format(seconds))

    labels = s.getLabels()
    start = time.time()
    coordinates = dict()
    for i in np.arange(512):
        for j in np.arange(512):
            try:
                coordinates[labels[i, j]][0].append(i)
                coordinates[labels[i, j]][1].append(j)
            except KeyError:
                coordinates[labels[i, j]] = [list(), list()]
    seconds = time.time() - start        
    print("Levou {0} segundos para retornar os labels".format(seconds))
    print("Média da cor do superpixel é {0} ".format(np.mean(image_[coordinates[i]])))
    
    """
    props = regionprops(labels)
    for i in props[1100]['coords']:
        image_[i[1], i[0]]=255
    print("{0} labels retornados.".format(len(props)))
    masks = s.getLabelContourMask()
    image_[masks == 255] = 255
    show(image_)

    image_ = np.copy(image)
    s_slic = cv2.ximgproc.createSuperpixelSLIC(image, cv2.ximgproc.SLIC, region_size, smooth)
    #s_slic.enforceLabelConnectivity()
    s_slic.iterate(num_iteration)
    masks = s_slic.getLabelContourMask()
    image_[masks == 255] = 255
    cv2.imwrite("./outputs/slic-{0}-{1}-{2}.png".format(region_size, smooth, num_iteration), image_)

    image_ = np.copy(image)
    s_slic = cv2.ximgproc.createSuperpixelSLIC(image, cv2.ximgproc.SLICO, region_size, smooth)
    s_slic.iterate(num_iteration)
    masks = s_slic.getLabelContourMask()
    image_[masks == 255] = 255
    cv2.imwrite("./outputs/slico-{0}-{1}-{2}.png".format(region_size, smooth, num_iteration), image_)

    image_ = np.copy(image)
    s_slic = cv2.ximgproc.createSuperpixelSLIC(image, cv2.ximgproc.MSLIC, region_size, smooth)
    s_slic.iterate(num_iteration)
    masks = s_slic.getLabelContourMask()
    image_[masks == 255] = 255
    cv2.imwrite("./outputs/mslic-{0}-{1}-{2}.png".format(region_size, smooth, num_iteration), image_)

    image_ = np.copy(image)
    s_slic = cv2.ximgproc.createSuperpixelLSC(image, region_size)
    s_slic.iterate()
    masks = s_slic.getLabelContourMask()
    image_[masks == 255] = 255
    cv2.imwrite("./outputs/lsc-{0}.png".format(region_size), image_)

    image_ = np.copy(image)
    s_slic = cv2.ximgproc.createSuperpixelSEEDS(image.shape[0], image.shape[1], 1, 10000, 2)
    s_slic.iterate(image, 20)
    masks = s_slic.getLabelContourMask()
    image_[masks == 255] = 255
    cv2.imwrite("./outputs/seeds-{0}-{1}-{2}-{3}-{4}.png".format(image.shape[0], image.shape[1], 1, 10000, 2), image_)
    """

def dice():
    k = 1
    seg = np.zeros((100,100), dtype='int')
    seg[30:70, 30:70] = k

    gt = np.zeros((100,100), dtype='int')
    gt[30:70, 40:80] = k

    dice = np.sum(seg[gt==k]) * 2.0 / (np.sum(seg) + np.sum(gt))

    print('Dice similarity score is {}'.format(dice))
 
def kmeans(samples, k):
    groups = [ Group(choices(samples, k=k)) for i in range(k)]
    recalcula = True
    while recalcula:
        recalcula = True
        for sample in samples:
            values = [(((sample['x'] - group.x) + (sample['y'] - group.y))) ** 2 for group in groups]
            minimo = min(values)
            groups[values.index(minimo)].set_sample(sample)
        for group in groups:
            recalcula = recalcula and group.recalcula()
    print(groups)

def return_superpixels(image, info=False):
    """
    Retorna uma tupla com os superpixels da imagem e suas adjacências.
    """
    #p = np.array([[int(np.binary_repr(image[i,j], 8)[7]) * 255 for j in range(0, image.shape[1])] for i in range(0, image.shape[0])])
    #image_ = np.copy(p)
    image_ = cv2.imread("./outputs/saida1.png", 0)
    s_slic = cv2.ximgproc.createSuperpixelLSC(image_, 40)
    s_slic.iterate(20)
    masks = s_slic.getLabelContourMask()
    image_[masks == 255] = 255
    labels = s_slic.getLabels()
    coordinates, adjacency = get_coordinates(labeled_image=labels, masks=masks, length=s_slic.getNumberOfSuperpixels())
    if info:
        arquivo = open("./outputs/superpixels-info.txt","w")
    pixels = dict()
    for key in coordinates:
        rows = np.array(coordinates[key][0])
        columns = np.array(coordinates[key][1])
        max_r = np.max(rows)
        min_r = np.min(rows)
        max_c = np.max(columns)
        min_c = np.min(columns)
        centroid = ((((max_r - min_r)//2) + min_r), (((max_c - min_c)//2) + min_c))
        color_mean = np.mean(image_[coordinates[key]])
        cv2.putText(image_,"{0}".format(key), (centroid[1], centroid[0]),  cv2.FONT_HERSHEY_SIMPLEX,0.4,255)
        #if color_mean < 10:
        #    image_[coordinates[key]] = 255
        #if i in [66, 70, 73, 74, 80, 84,90, 95, 100, 105, 106]:
        #    image_[coordinates[key]] = 255
        if info:
            arquivo.write("Superpixel {0}\n\tCentroid: {1}\n\tColor mean: {2}\n".format(key,centroid, color_mean))
        pixels[key] = {"label": key, "centroid": centroid, "color": color_mean, "coordinates":coordinates[key]}
    if info:
        arquivo.close()
    #cv2.imwrite("./outputs/saida-seg.png", image_)
    return pixels, adjacency

    







#############################################
"""
if __name__ == "__main__":
    image = cv2.imread("./outputs/lung.png", 0)
    p = np.array([[int(np.binary_repr(image[i,j], 8)[7]) * 255 for j in range(0, image.shape[1])] for i in range(0, image.shape[0])])

    image_ = np.copy(p)
    s_slic = cv2.ximgproc.createSuperpixelLSC(image, 40)
    s_slic.iterate(20)
    masks = s_slic.getLabelContourMask()
    image_[masks == 255] = 255
    labels = s_slic.getLabels()
    coordinates, adjacency = get_coordinates(labeled_image=labels, masks=masks, length=s_slic.getNumberOfSuperpixels())
    print(adjacency)
    print(((0,16)  in adjacency) and ((0,1)  in adjacency) and ((1, 19)  in adjacency))
    arquivo = open("./outputs/superpixels-info.txt","w")
    pixels = list()
    for key in coordinates:
        rows = np.array(coordinates[key][0])
        columns = np.array(coordinates[key][1])
        max_r = np.max(rows)
        min_r = np.min(rows)
        max_c = np.max(columns)
        min_c = np.min(columns)
        centroid = ((((max_r - min_r)//2) + min_r), (((max_c - min_c)//2) + min_c))
        color_mean = np.mean(image_[coordinates[key]])
        cv2.putText(image_,"{0}".format(key), (centroid[1], centroid[0]),  cv2.FONT_HERSHEY_SIMPLEX,0.4,255)
        #if i in [66, 70, 73, 74, 80, 84,90, 95, 100, 105, 106]:
        #    image_[coordinates[key]] = 255
        #pixels.append({"label": key, "centroid": centroid, "color": color_mean, "coordinates":coordinates[key]})
        arquivo.write("Superpixel {0}\n\tCentroid: {1}\n\tColor mean: {2}\n".format(key,centroid, color_mean))

    cv2.imwrite("./outputs/saida-lsc.png", image_)
    cv2.imwrite("./outputs/saida.png", p)
##########
    dataset = load_datasets(path)
    roi = "Lung"
    print("{0} aquisições carregadas no dataset!".format(len(dataset)))
    mark_data = get_datamark(datasets=dataset, roi=roi)
    media = 0
    cv2.imwrite("./outputs/lung.png", get_image(mark_data[len(mark_data)//2]))
    for i in mark_data:
        coordinates = get_mark(dataset[0], position=i.ImagePositionPatient,spacing=i.PixelSpacing, roi=roi)
        image = get_image(i)
        mean_image = 0
        for c in coordinates:
            mean_image += np.mean(image[c])
        media += (mean_image / len(coordinates))
    print("A média de intensidade de pixel do pulmão é {0}".format(media/len(mark_data)))
    #get_superpixel(get_image(dataset[50]),region_size=10, smooth=10., num_iteration=10, compactness=0.075)
    """