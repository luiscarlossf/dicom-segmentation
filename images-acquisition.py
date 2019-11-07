import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

stantard_path = "./outputs/adaptative_templates/100.png"

def show(pixel_array):
    """
    Exibe a imagem a partir de um array de pixels.

    :param pixel_arrahttps://ih1.redbubble.net/image.425603049.2805/mp,840x830,matte,f8f8f8,t-pad,1000x1000,f8f8f8.jpg: numpy array com os pixels da imagem.
    :return:
    """
    cv2.imshow('image', pixel_array)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

im = cv2.imread(stantard_path, 0)
blur = cv2.GaussianBlur(im,(3,3),0)
retval = cv2.ximgproc.createSuperpixelSLIC(blur, cv2.ximgproc.MSLIC, region_size=12, ruler=100.)
retval.iterate()
segments = retval.getLabels()
a = retval.getLabelContourMask()
im[a == 255] = 255
print("Region_size {0} - Number of Superpixels {1}".format(12, retval.getNumberOfSuperpixels()))
show(im)
show(cv2.copyMakeBorder(im, int(im.shape[0] * 0.5),int(im.shape[0] * 0.5), int(im.shape[1] * 0.5), int(im.shape[1] * 0.5), cv2.BORDER_CONSTANT))
#Bounding Box was 30 x 30