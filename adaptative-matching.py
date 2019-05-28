import cv2
import numpy as np
import os
import pydicom
from random import choice
from dicompylercore import dicomparser
"""
ADAPTIVE TEMPLATE MATCHING
"""


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
    :return volume: numpy array com os arquivos dicom
    """
    volume = list()

    for root, dirs, files in os.walk(path):
        for file in files:
            ds = pydicom.dcmread(root + '/'+ file)
            if ds.Modality == 'CT':
                volume.append(ds)
    return volume


def load_volumes(path):
    """
    Gera uma lista com os paths dos volumes no path

    :param path: path onde os volumes estão armazenados
    :return volumes: paths dos volumes encontrados
    """
    volumes = None

    for root, dirs, files in os.walk(path):
        for file in files:
            volumes.append(pydicom.dcmread(root + '/' + file))
        break

    return volumes


def choice_volume(path):
    """
    Seleciona aleatoriamente um volume do path
    :param path: path onde os volumes estão armazenado
    :return volume: lista com rtstruct dataset e lista de CT datasets
    """
    path_volume = None
    ds = list()
    marks = None

    for root, dirs, files in os.walk(path):
        path_volume = root+choice(dirs)
        print(path_volume)
        break
    for root, dirs, files in os.walk(path_volume):
        for file in files:
            dataset = pydicom.dcmread(root + '/' + file)
            if dataset.Modality == 'CT':
                ds.append(dataset)
            else:
                print(dataset.Modality)
                marks = dataset

    volume = [marks, ds]

    return volume


def getImage(slice):
    ds = dicomparser.DicomParser(slice)
    intercept, slope = ds.GetRescaleInterceptSlope()
    rescaled_image = slice.pixel_array * slope + intercept
    window, level = ds.GetDefaultImageWindowLevel()
    pixels_slice = ds.GetLUTValue(rescaled_image, window, level)

    return pixels_slice

def getTemplate():
    pass

standard_path = "C:/Users/luisc/Documents/dicom-database/LCTSC/Train/"
###########Suponha que o volume tenha sido selecionado aleatoriamente
#Gerando o template padrão
#- Seleciona aleatoriamente um volume dentre todos os outros do banco de dados.

volume = choice_volume(standard_path)
slice = choice(volume[1])
#- Encontrar a marcação do especialista, encontrar o centro da massa dessa marcação
marking = dicomparser.DicomParser(volume[0])
structures = marking.GetStructures()
number_roi = None
for i in structures:
    if structures[i]['name'] == 'SpinalCord':
        number_roi = structures[i]['id']


while True:
    try:
        contour = np.array(marking.GetStructureCoordinates(number_roi)['{0:.2f}'.format(slice.ImagePositionPatient[2])][0]['data'])
        break
    except KeyError:
        slice = choice(volume[1])

rows = ((contour[:, 1] - slice.ImagePositionPatient[1])/slice.PixelSpacing[1]).astype(int)
columns = ((contour[:, 0] - slice.ImagePositionPatient[0])/slice.PixelSpacing[0]).astype(int)
#Conseguindo LUT Values

pixels_slice = getImage(slice)

diameter_x = rows.max() - rows.min()
diameter_y = columns.max() - columns.min()
center_x = int(diameter_x//2 + rows.min())
center_y = int(diameter_y//2 + columns.min())
show(pixels_slice)


#- Recortar duas vezes o tamanho da região correspondente de todos os lados.
print("[{0}:{1}, {2}:{3}]".format(rows.min() - (2*diameter_x), rows.max() + (2*diameter_x), columns.min() - (2*diameter_y),columns.max() + (2*diameter_y)))
standard_template = pixels_slice[rows.min() - (2*diameter_x): rows.max() + (2*diameter_x), columns.min() - (2*diameter_y):columns.max() + (2*diameter_y)]

show(standard_template)
#Gerando o template inicial
#- O algoritmo Template Matching é executado em cada slice do volume, onde o template é o
#template padrão definido anteriormente;

path_volumes = load_volumes(standard_path)
results = list()
for path_volume in path_volumes:
    datasets = load_datasets(path_volume)
    maxs = [cv2.minMaxLoc(cv2.matchTemplate(getImage(ds), standard_template, cv2.TM_CCORR_NORMED)) for ds in datasets]
    maximo = max(maxs[:][1])
    indice = maxs[:][1].index(maximo)



#- Calcular a similaridade em cada slice;

#- O template inicial selecionado é aquele que tem maior similaridade com o template padrão,
#o número do slice do melhor template inicial é salvo.

#Adaptando o template para cada slice
#- Executa o Template Matching no primeiro slice onde a combinação ocorre, como resultado temos:
#o Novo Template e uma imagem.

#- O Novo Template é executado no próximo slice e, novamente, o resultado será um Novo Template e uma image.

#- No final, teremos um conjunto de imagens onde á apenas a região da medula segmentada.
