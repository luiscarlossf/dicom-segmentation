import cv2
import os
from dicompylercore import dicomparser
import pydicom
import numpy as np
from skimage.measure import regionprops

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

def get_superpixel(image, region_size, smooth, num_iteration=10, compactness=0.075):
    image_ = np.copy(image)
    s = cv2.ximgproc.createSuperpixelLSC(image, region_size, compactness)
    print("Foram gerados {0} superpixels.".format(s.getNumberOfSuperpixels))
    s.iterate()
    s.enforceLabelConnectivity(10)
    labels = s.getLabels()
    props = regionprops(labels)
    for i in props[1100]['coords']:
        image_[i[1], i[0]]=255
    print("{0} labels retornados.".format(len(props)))
    masks = s.getLabelContourMask()
    image_[masks == 255] = 255
    show(image_)

    """
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

if __name__ == "__main__":
    dataset = load_datasets(path)
    print("{0} aquisições carregadas no dataset!".format(len(dataset)-1))
    get_superpixel(get_image(dataset[50]),region_size=10, smooth=10., num_iteration=20, compactness=0.075)