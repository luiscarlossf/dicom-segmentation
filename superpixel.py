import cv2
import os
from dicompylercore import dicomparser
import pydicom
import numpy as np
from random import choices
from skimage.measure import regionprops
from skimage.morphology import opening
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
    coordinates = {1: list() com as coordenadas no eixo x e uma outra list() do eixo y}
    adjacency = {1: set() com os rótulos dos superpixels adjacentes a 1}

    @param labeled_image: numpy.array() - imagem rotulada  por superpixels.
    @param masks: numpy.array() - mascará das bordas dos superpixels. 
    @param length: int() - quantidade de superpixels na imagem rotulada
    @return (coordinates, adjacency): dict(), dict() - dicionários com coordenadas 
        adjacências de cada superpixel, respectivamente.
    """
    coordinates = dict()
    adjacency = dict()
    for i in range(length):
        adjacency[i] = set()
    for i in np.arange(labeled_image.shape[0]):
        for j in np.arange(labeled_image.shape[1]):
            if (masks[i, j] == 255) and (i > 0) and (i < (labeled_image.shape[0] - 1)) and (j > 0) and (j < (labeled_image.shape[1] - 1)):
                if labeled_image[i+1, j] != labeled_image[i-1, j]:
                    adjacency[labeled_image[i+1, j]].add(labeled_image[i-1, j])
                    adjacency[labeled_image[i-1, j]].add(labeled_image[i+1, j])

                if labeled_image[i, j+1] != labeled_image[i, j-1]:
                    adjacency[labeled_image[i, j+1]].add(labeled_image[i, j-1])
                    adjacency[labeled_image[i, j-1]].add(labeled_image[i, j+1])

                if labeled_image[i+1, j-1] != labeled_image[i-1, j+1]:
                    adjacency[labeled_image[i+1, j-1]].add(labeled_image[i-1, j+1])
                    adjacency[labeled_image[i-1, j+1]].add(labeled_image[i+1, j-1])

                if labeled_image[i+1, j+1] != labeled_image[ i-1, j-1]:
                    adjacency[labeled_image[i+1, j+1]].add(labeled_image[ i-1, j-1])
                    adjacency[labeled_image[i-1, j-1]].add(labeled_image[ i+1, j+1])
    
            try:
                coordinates[labeled_image[i, j]][0].append(i)
                coordinates[labeled_image[i, j]][1].append(j)
            except KeyError:
                coordinates[labeled_image[i, j]] = [list(), list()]
                coordinates[labeled_image[i, j]][0].append(i)
                coordinates[labeled_image[i, j]][1].append(j)
                
    return coordinates , adjacency

def dice():
    k = 1
    seg = np.zeros((100,100), dtype='int')
    seg[30:70, 30:70] = k

    gt = np.zeros((100,100), dtype='int')
    gt[30:70, 40:80] = k

    dice = np.sum(seg[gt==k]) * 2.0 / (np.sum(seg) + np.sum(gt))

    print('Dice similarity score is {}'.format(dice))

def get_image_lut(dataset, center=-500, window=1500):
    """
    Retorna a image com os valores LUT.

    @param dataset: FileDataset com as metadados da TC.
    @param window: float da Window Center Level, por padrão a janela para o pulmão.
    @param level: float da Window Width, por padrão a largura da janela para o pulmão.

    @return new_image: numpy array com os valores LUT dos pixels da imagem de entrada.
    """
    new_image = np.copy(dataset.pixel_array)
    if dataset.RescaleType == 'HU': #O que fazer quando não tem Rescale
        c = center if center else dataset.WindowCenter #center level
        w = window if window else dataset.WindowWidth #window width
        new_image = int(dataset.RescaleSlope) * new_image + int(dataset.RescaleIntercept)
        condition1 = new_image <= (c- 0.5 - (w - 1)/ 2)
        condition2 = new_image > (c- 0.5 + (w - 1)/2)
        new_image = np.piecewise(new_image, [condition1, condition2], [0,255, lambda new_image: ((new_image - (c - 0.5))/(w-1)+0.5) * (255 - 0)]).astype(np.uint8)
    else:
        #PRECISA SER IMPLEMENTADO A PARTE DE QUANDO NÃO TEM RESCALE.
        pass
    return new_image

def pre_process(image):
    """
    Aplica filtros e transformações na imagem de entrada para 
    ajudar na segementação.

    @param image: numpy array com os pixels da imagem a ser pré-processada.

    @return new_image: numpy array da image de entrada depois de aplicada as transformações.
    """
    
    new_image = np.copy(image)
    #Threshold
    new_image[new_image > 180]= 255
    
    #Fatiamento de pixels
    p1 = np.array([[int(np.binary_repr(new_image[i,j], 8)[1]) * 255 for j in range(0, new_image.shape[1])] for i in range(0, new_image.shape[0])])
    p2 = np.array([[int(np.binary_repr(new_image[i,j], 8)[2]) * 255 for j in range(0, new_image.shape[1])] for i in range(0, new_image.shape[0])])
    p3 = np.array([[int(np.binary_repr(new_image[i,j], 8)[3]) * 255 for j in range(0, new_image.shape[1])] for i in range(0, new_image.shape[0])])
    p4 = np.array([[int(np.binary_repr(new_image[i,j], 8)[4]) * 255 for j in range(0, new_image.shape[1])] for i in range(0, new_image.shape[0])])
    p5 = np.array([[int(np.binary_repr(new_image[i,j], 8)[5]) * 255 for j in range(0, new_image.shape[1])] for i in range(0, new_image.shape[0])])
    p6 = np.array([[int(np.binary_repr(new_image[i,j], 8)[6]) * 255 for j in range(0, new_image.shape[1])] for i in range(0, new_image.shape[0])])
    p7 = np.array([[int(np.binary_repr(new_image[i,j], 8)[7]) * 255 for j in range(0, new_image.shape[1])] for i in range(0, new_image.shape[0])])

    new_image = np.copy( p1 * p2 * p3 * p4 * p5 * p6 * p7).astype(np.uint8)
    

    #Encontra elementos conectados na imagem.
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(new_image, connectivity=8)
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
    new_image = img2.astype(np.uint8)
    
    #Fechamento
    retval = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    new_image = cv2.morphologyEx(new_image, cv2.MORPH_CLOSE, retval)

    return new_image


def return_superpixels(image, info=False):
    """
    Retorna uma tupla com os superpixels da imagem e suas adjacências.

    @param image: numpy.array() - imagem a ser segmentada.
    @param info: boolean() - indica se um arquivo e uma imagem com informações 
        sobre os superpixels serão gerados.
    @return pixel: dict() - dicionário com os superpixels gerados, a quantidade de 
        chaves do dicionário é equivalente ao número de superpixels gerados.
    @return adjacency: dict() - dicionário de adjacência dos superpixels.
    """
    image_ = np.copy(image)
    #superpixels = cv2.ximgproc.createSuperpixelLSC(image_, 40)
    #superpixels.iterate(20)
    #masks = superpixels.getLabelContourMask()
    #image_[masks == 255] = 123
    #labels = superpixels.getLabels()
    #number_spixels = superpixels.getNumberOfSuperpixels()
    '''Mais apropriado para imagens na janela do pulmão'''
    superpixels = cv2.ximgproc.createSuperpixelSEEDS(image_.shape[0], image_.shape[1], image_channels=1, num_superpixels=200, num_levels=5)
    superpixels.iterate(image_, 30)
    masks = superpixels.getLabelContourMask()
    image_[masks == 255] = 200
    labels = superpixels.getLabels()
    number_spixels = superpixels.getNumberOfSuperpixels()
    coordinates, adjacency = get_coordinates(labeled_image=labels, masks=masks, length=number_spixels)
    if info:
        arquivo = open("./outputs/superpixels-info.txt","w")
    spixels = dict()
    for key in coordinates:
        mean_r = int(np.mean(coordinates[key][0]))
        mean_c = int(np.mean(coordinates[key][1]))
        centroid = (mean_r, mean_c)
        color_mean = round(np.mean(image_[tuple(coordinates[key])]), 3)
        if info:
            cv2.putText(image_,"{0}".format(key), (centroid[1], centroid[0]),  cv2.FONT_HERSHEY_SIMPLEX,0.3,200)
            cv2.imwrite("./outputs/saida-superpixels.png", image_)
            arquivo.write("Superpixel {0}\n\tCentroid: {1}\n\tColor mean: {2}\n".format(key,centroid, color_mean))
        spixels[key] = {"label": key, "centroid": centroid, "color": color_mean, "coordinates":coordinates[key]}
    if info:
        arquivo.close()
    #cv2.imwrite("./outputs/saida-seg.png", image_)
    return spixels, adjacency

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