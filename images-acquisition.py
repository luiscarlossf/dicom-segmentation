import pydicom
import os
import numpy as np

example_path = "C:/Users/luisc/Documents/dicom-database/JOELHO2/image-000016.dcm"
stantard_path = "C:/Users/luisc/Documents/dicom-database/JOELHO"

dicom_images = None
lista = list()

#AQUISIÇÃO DE IMAGEM
for root, dirs, files in os.walk(stantard_path):
    for file in files:
        lista.append(pydicom.dcmread(root + '/'+ file))

dicom_images = np.array(lista)

print(len(dicom_images))