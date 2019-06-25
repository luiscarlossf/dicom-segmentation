import cv2
import numpy as np
import os
import pydicom
from random import choice
from dicompylercore import dicomparser
from skimage.measure import regionprops
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
    :return volume: lista com os arquivos dicom
    """
    volume = list()
    slice_locations = dict()

    for root, dirs, files in os.walk(path):
        for file in files:
            ds = pydicom.dcmread(root + '/'+ file)
            if ds.Modality == 'CT':
                slice_locations[ds.SliceLocation] = ds
    volume = [slice_locations[key] for key in sorted(slice_locations.keys())]
    return volume


def loadpath_volumes(path):
    """
    Gera uma lista com os paths dos volumes no path

    :param path: path onde os volumes estão armazenados
    :return volumes: paths dos volumes encontrados
    """
    volumes = list()

    for root, dirs, files in os.walk(path):
        for dir in dirs:
            volumes.append(root + dir)
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
    slice_locations = dict()
    for root, dirs, files in os.walk(path):
        path_volume = root+choice(dirs)
        break
    for root, dirs, files in os.walk(path_volume):
        for file in files:
            dataset = pydicom.dcmread(root + '/' + file)
            if dataset.Modality == 'CT':
                slice_locations[ds.SliceLocation] = dataset
            else:
                marks = dataset
    ds = [slice_locations[key] for key in sorted(slice_locations.keys())]
    volume = [marks, ds]

    return volume


def getImage(slice):
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

def getImages(datasets):
    """
    Retorna um array de pixels com valores LUT para cada dataset

    :param datasets: lista de datasets dicom
    :return pixels_slice:  lista de arrays numpy com os pixels transformados para
                           cada dataset
    """
    pixels_slice = list()
    for slice in datasets:
        ds = dicomparser.DicomParser(slice)
        intercept, slope = ds.GetRescaleInterceptSlope()
        rescaled_image = slice.pixel_array * slope + intercept
        window, level = ds.GetDefaultImageWindowLevel()
        pixels_slice.append(ds.GetLUTValue(rescaled_image, window, level))

    return pixels_slice

def getStandardTemplate(standard_path):
    ###########Suponha que o volume tenha sido selecionado aleatoriamente
    #Gerando o template padrão
    #- Seleciona aleatoriamente um volume dentre todos os outros do banco de dados.

    volume = choice_volume(standard_path)
    quant_slices = len(volume)
    slice = choice(volume[1])
    #- Encontrar a marcação do especialista, encontrar o centro da massa dessa marcação
    marking = dicomparser.DicomParser(volume[0])
    structures = marking.GetStructures()
    number_roi = None
    for i in structures:
        if structures[i]['name'] == 'SpinalCord':
            number_roi = structures[i]['id']


    while True:
        #Sorteia um novo slice caso o slice sorteado não contenha a medula
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


    #- Recortar duas vezes o tamanho da região correspondente de todos os lados.
    #print("[{0}:{1}, {2}:{3}]".format(rows.min() - (2*diameter_y), rows.max() + (2*diameter_y), columns.min() - (2*diameter_x), columns.max() + (2*diameter_x)))
    standard_template = pixels_slice[rows.min() - (2*diameter_y): rows.max() + (2*diameter_y), columns.min() - (2*diameter_x):columns.max() + (2*diameter_x)]
    cv2.imwrite("standard_template.png", standard_template)
    return standard_template


def getInitialTemplates(standard_template, volume_paths):
    results = list()
    cont = 0
    for path_volume in volume_paths:
        datasets = load_datasets(path_volume)
        mins = list()
        locations = list()
        # - Calcular a similaridade em cada slice;
        for i, ds in enumerate(datasets):
            image = getImage(ds)[200:400, 150:400]
            blur = cv2.blur(image, (5, 5))
            minmaxloc = cv2.minMaxLoc(cv2.matchTemplate(blur, standard_template, cv2.TM_SQDIFF))
            mins.append(minmaxloc[0])
            locations.append(minmaxloc[2])
        minimo = min(mins)
        indice = mins.index(minimo)

        image = getImage(datasets[indice])[200:400, 150:400]

        w, h = standard_template.shape
        top_left = [locations[indice][0], locations[indice][1]]

        if top_left[0] + w >= image.shape[1]:
            w = image.shape[1] - top_left[0] - 1
        if top_left[1] + h >= image.shape[0]:
            h = image.shape[0] - top_left[1] - 1

        # image[top_left[1], top_left[0]:top_left[0] + w] = 255
        # image[top_left[1] + h, top_left[0]:top_left[0] + w] = 255
        # image[top_left[1]:top_left[1]+ h, top_left[0]] = 255
        # image[top_left[1]:top_left[1] + h, top_left[0] + w] = 255
        # show(image)
        # - O template inicial selecionado é aquele que tem maior similaridade com o template padrão,
        # o número do slice do melhor template inicial é salvo.
        cv2.imwrite("./outputs/initial_templates/{0}-{1}.png".format(cont, indice),
                    image[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w])
        cont += 1
        results.append(
            {'initial_template': image[top_left[1]:top_left[1] + h, top_left[0]:top_left[0] + w], 'indice': indice,
             'path_volume': path_volume})
    return results

def getAdaptativeTemplates(results):
    for initial_info in results: # Informações do template inicial
        # - Executa o Template Matching no primeiro slice onde a combinação ocorre, como resultado temos:
        # o Novo Template e uma imagem.
        volume = load_datasets(initial_info['path_volume'])
        datasets = getImages(volume)
        tam = len(datasets)
        # - O Novo Template é executado no próximo slice e, novamente, o resultado será um Novo Template e uma image.
        # - No final, teremos um conjunto de imagens onde á apenas a região da medula segmentada.
        results2 = list()
        adaptative_template = initial_info['initial_template'][::]
        print(initial_info['path_volume'])
        dirname = "./outputs/adaptative_templates/"+os.path.basename(initial_info['path_volume'])

        try:
            os.mkdir(dirname)
        except FileExistsError as e:
            pass

        for i in reversed(range(0, initial_info['indice'])):
            blur_image = cv2.blur(datasets[i][180:400, 150:400], (5, 5))
            adaptative_template = cv2.blur(adaptative_template, (5, 5))
            min, max, minloc, maxloc = cv2.minMaxLoc(cv2.matchTemplate(blur_image, adaptative_template, cv2.TM_SQDIFF))
            w, h = adaptative_template.shape
            if minloc[0] + w >= blur_image.shape[0]:
                w = blur_image.shape[0] - minloc[0] - 1
            if minloc[1] + h >= blur_image.shape[1]:
                h = blur_image.shape[1] - minloc[1] - 1
            adaptative_template = datasets[i][180:400, 150:400][minloc[1]:minloc[1] + h, minloc[0]:minloc[0] + w]
            cv2.imwrite(dirname + "/{0}.png".format(i), adaptative_template)
            results2.append(adaptative_template)

        results2 = results2[::-1]
        adaptative_template = initial_info['initial_template'][::]
        for i in range(initial_info['indice'], tam):
            blur_image = cv2.blur(datasets[i][180:400, 150:400], (5, 5))
            adaptative_template = cv2.blur(adaptative_template, (5, 5))
            min, max, minloc, maxloc = cv2.minMaxLoc(cv2.matchTemplate(blur_image, adaptative_template, cv2.TM_SQDIFF))
            w, h = adaptative_template.shape
            if minloc[0] + w >= blur_image.shape[0]:
                w = blur_image.shape[0] - minloc[0] - 1
            if minloc[1] + h >= blur_image.shape[1]:
                h = blur_image.shape[1] - minloc[1] - 1
            adaptative_template = datasets[i][180:400, 150:400][minloc[1]:minloc[1] + h, minloc[0]:minloc[0] + w]
            cv2.imwrite(dirname + "/{0}.png".format(i), adaptative_template)
            results2.append(adaptative_template)
    return results2

def getCandidatesSegmentation(path):
    for root, dirs, files in os.walk(path):
        dir_name = os.path.basename(root)
        if 'fail' in dir_name:
            continue
        try:
            os.mkdir('./outputs/candidates_segmentation/' + dir_name)
        except FileExistsError as e:
            pass

        for file in files:
            root = root.replace('\\', '/')
            filename = root + '/' + file
            seg_filename = './outputs/candidates_segmentation/' + dir_name + '/' + file
            print(filename)
            im = cv2.imread(filename, 0)
            print(im.shape)
            blur = cv2.GaussianBlur(im, (3, 3), 0)
            retval = cv2.ximgproc.createSuperpixelSLIC(blur, cv2.ximgproc.MSLIC, region_size=12, ruler=110.)
            #retval.enforceLabelConnectivity(12)
            retval.iterate()
            a = retval.getLabelContourMask()
            im[a == 255] = 255
            cv2.imwrite(seg_filename, im)


def get_bounding_boxes(image, width, height):
    bboxes = list()
    top = int(image.shape[0] * 0.5)
    bottom = top
    left = int(image.shape[1] * 0.5)
    right = left
    blur = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    retval = cv2.ximgproc.createSuperpixelSLIC(blur, cv2.ximgproc.MSLIC, region_size=12, ruler=110.)
    retval.iterate()
    a = retval.getLabelContourMask()
    labels = retval.getLabels()
    for prop in regionprops(labels):
        x = int(prop.centroid[0]) + int(blur.shape[0] * 0.5)
        y = int(prop.centroid[1]) + int(blur.shape[1] * 0.5)
        top = x + height//2
        bottom = x - height//2
        right = y + width//2
        left = y - width//2
        bboxes.append(image[bottom:top, left:right])
    return bboxes


"""
standard_path = "C:/Users/luisc/Documents/dicom-database/LCTSC/Train/"
standard_template = cv2.imread("./outputs/standard_template1.png", 0) #getStandardTemplate(standard_path)
#############################GERANDO TEMPLATE INITIAL ############################
#- O algoritmo Template Matching é executado em cada slice do volume, onde o template é o
#template padrão definido anteriormente;
volume_paths = loadpath_volumes(standard_path)
results = list() #Armazena ps templates iniciais e indices dos respectivos slices no seu volume
standard_template = cv2.blur(standard_template,(5,5))
results = getInitialTemplates(standard_template, volume_paths)


#Adaptando o template para cada slice

results2 = getAdaptativeTemplates(results)

getCandidatesSegmentation("./outputs/adaptative_templates")
"""
imag = cv2.imread("./outputs/adaptative_templates/100.png")
print(imag.shape)
get_bounding_boxes(imag, 30, 30)
