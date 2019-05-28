import cv2
import numpy as np
import os
import pydicom
from random import choice
from skimage.feature import match_template
from skimage.exposure import rescale_intensity
from dicompylercore import dicomparser
#ADAPTIVE TEMPLATE MATCHING

def show(pixel_array):
    """
    Exibe a imagem a partir de um array de pixels.

    :param pixel_array: numpy array com os pixels da imagem.
    :return:
    """
    print("Show was called.")
    cv2.imshow('image',pixel_array)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

def load_files(path):
    """
    Carrega todos os arquivos DICOM de um diretório.

    :param path: str indicando o diretório de origem dos
                 dos arquivos que se deseja carregar
    :return volume: numpy array com os arquivos dicom
    """
    volume = None

    for root, dirs, files in os.walk(path):
        volume = [pydicom.dcmread(root + '/'+ file) for file in files]

    return volume

def load_volumes(path):
    """
    Gera uma lista com os paths dos volumes no path

    :param path: path onde os volumes estão armazenados
    :return volumes: paths dos volumes encontrados
    """
    volumes = None

    for root, dirs, files in os.walk(path):
        volumes = [root+'/'+dir for dir in dirs]
        break

    return volumes

def choice_volume(path):
    """
    Seleciona aleatoriamente um volume do path
    :param path: path onde os volumes estão armazenado
    :return volume: volume seleciona aleatoriamente
    """
    path_volume = None
    volume = None

    for root, dirs, files in os.walk(path):
        path_volume = root+'/'+choice(dirs)
        break
    for root, dirs, files in os.walk(path_volume):
        volume = [pydicom.dcmread(root + '/' + file) for file in files]
        break

    return volume

###########Suponha que o volume tenha sido selecionado aleatoriamente
#Gerando o template padrão
#- Seleciona aleatoriamente um volume dentre todos os outros do banco de dados.
volume = choice_volume("C:/Users/luisc/Documents/dicom-database/LCTSC/Train")
slice = choice(volume)

#- Encontrar a marcação do especialista, encontrar o centro da massa dessa marcação
marking = dicomparser.DicomParser("C:/Users/luisc/Documents/dicom-database/LCTSC/Train/LCTSC-Train-S1-001/11-16-2003-RTRCCTTHORAX8FLow Adult-39664/1-.simplified-62948/000000.dcm")
contour = np.array(marking.GetStructureCoordinates(1)[str(slice.ImagePositionPatient[2]) + ".00"][0]['data'])

rows = ((contour[:, 1] - slice.ImagePositionPatient[1])/slice.PixelSpacing[1]).astype(int)
columns = ((contour[:, 0] - slice.ImagePositionPatient[0])/slice.PixelSpacing[0]).astype(int)
pixels = np.copy(slice.pixel_array)
diameter_x = rows.max() - rows.min()
diameter_y = columns.max() - columns.min()
center_x = int(diameter_x//2 + rows.min())
center_y = int(diameter_y//2 + columns.min())
#pixels[center_x, center_y] = 65535



#- Recortar duas vezes o tamanho da região correspondente de todos os lados.
standard_template = pixels[rows.min() - (2*diameter_y): rows.max() + (2*diameter_y), columns.min() - (2*diameter_x):columns.max() + (2*diameter_x)]

#Gerando o template inicial
#- O algoritmo Template Matching é executado em cada slice do volume, onde o template é o
#template padrão definido anteriormente;

results = [cv2.minMaxLoc(cv2.matchTemplate(ds.pixel_array, standard_template, cv2.TM_CCORR_NORMED)) for ds in volume if ds.Modality == 'CT']

#- Calcular a similaridade em cada slice;

#- O template inicial selecionado é aquele que tem maior similaridade com o template padrão,
#o número do slice do melhor template inicial é salvo.

#Adaptando o template para cada slice
#- Executa o Template Matching no primeiro slice onde a combinação ocorre, como resultado temos:
#o Novo Template e uma imagem.

#- O Novo Template é executado no próximo slice e, novamente, o resultado será um Novo Template e uma image.

#- No final, teremos um conjunto de imagens onde á apenas a região da medula segmentada.
