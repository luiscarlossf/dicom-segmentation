import numpy as np
def local_refinement_error(s1, s2, p):
    """
    Mede o grau que duas segmentações s1 e s2 concordam no pixel p.

    @param s1 [set]: conjunto de pixels da segmentação 1
    @param s2 [set]: conjunto de pixels da segmentação 2
    @param p : pixel no qual os dois conjuntos possuem ou não.

    @return e [float]: error entre as duas segmentação.
    """
    s_1 = set(s1)
    s_2 = set(s2)
    e = len(s_1.difference(s_2))/len(s_1)

    return e

def global_consistency_error(s1, s2, n):
    """
    Permite o refinamento em diferentes direções em diferentes 
    partes da imagem.

    @param s1 [set]: conjunto de pixels da segmentação 1
    @param s2 [set]: conjunto de pixels da segmentação 2

    @return lce [float]: resultado a medida.
    """
    gce = (1/n) * min([local_refinement_error(s1, s2), local_refinement_error(s2,s1)])
    return gce

def dice(image1, image2):
    return (2 * np.sum(image1*image2)) / (np.sum(image1**2) + np.sum(image2**2))