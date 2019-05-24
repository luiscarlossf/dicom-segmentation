import cv2
import numpy as np
import os
import pydicom
from random import choice
from skimage.exposure import rescale_intensity
from dicompylercore import dicomparser
#ADAPTIVE TEMPLATE MATCHING

def show(pixel_array):
    """
    Exibe a imagem a partir de um array de pixels.

    :param pixel_array: numpy array com os pixels da imagem.
    :return:
    """
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
    volumes = list()
    for root, dirs, files in os.walk(path):
        for file in files:
            volumes.append(pydicom.dcmread(root + '/'+ file))
    return volumes
###########Suponha que o volume tenha sido selecionado aleatoriamente
volume = load_files("C:/Users/luisc/Documents/dicom-database/LCTSC/LCTSC-Train-S1-001")

marking = dicomparser.DicomParser("C:/Users/luisc/Documents/dicom-database/LCTSC/LCTSC-Train-S1-001/11-16-2003-RTRCCTTHORAX8FLow Adult-39664/1-.simplified-62948/000000.dcm")
index_slice = 75
markings = pydicom.dcmread("C:/Users/luisc/Documents/dicom-database/LCTSC/LCTSC-Train-S1-001/11-16-2003-RTRCCTTHORAX8FLow Adult-39664/1-.simplified-62948/000000.dcm")
contoursequence = markings.ROIContourSequence[0].ContourSequence
print(markings.ROIContourSequence[0].ReferencedROINumber)
print(contoursequence[index_slice].ContourImageSequence)
contour = contoursequence[index_slice].ContourData
contour = np.array(contour).reshape(contoursequence[index_slice].NumberOfContourPoints, 3)
#Gerando o template padrão
#- Seleciona aleatoriamente um volume dentre todos os outros do banco de dados.
slice = None

for index, i in enumerate(volume):
    if i.SOPInstanceUID == contoursequence[index_slice].ContourImageSequence[0].ReferencedSOPInstanceUID:
        print("Index: ", index)
        slice = i
print(slice.SOPInstanceUID)
print(slice.ImagePositionPatient[0])
rows = ((contour[:, 1] - slice.ImagePositionPatient[1])/slice.PixelSpacing[1]).astype('int')
columns = ((contour[:, 0] - slice.ImagePositionPatient[0])/slice.PixelSpacing[0]).astype('int')
pixels = np.copy(slice.pixel_array)
pixels[rows, columns] = 65535

show(pixels)

#- Encontrar a marcação do especialista, encontrar o centro da massa dessa marcação

#- Recortar duas vezes o tamanho da região correspondente de todos os lados.

#Gerando o template inicial
#- O algoritmo Template Matching é executado em cada slice do volume, onde o template é o
#template padrão definido anteriormente;

#- Calcular a similaridade em cada slice;

#- O template inicial selecionado é aquele que tem maior similaridade com o template padrão,
#o número do slice do melhor template inicial é salvo.

#Adaptando o template para cada slice
#- Executa o Template Matching no primeiro slice onde a combinação ocorre, como resultado temos:
#o Novo Template e uma imagem.

#- O Novo Template é executado no próximo slice e, novamente, o resultado será um Novo Template e uma image.

#- No final, teremos um conjunto de imagens onde á apenas a região da medula segmentada.
