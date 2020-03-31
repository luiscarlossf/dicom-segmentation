import cv2
import numpy as np
from superpixel import get_coordinates

image = cv2.imread("./outputs/saida1.png", 0)

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

zeros[coords[100]] = 255
cv2.imwrite("./outputs/test.png", zeros)
