import numpy as np
import math

"""
Single Pass Seed selection algorithm (SPSS)

Apesar do k-means++ ser O(log k), ele produz diferentes clusters em diferentes execuções
devido os passos 1 e 2 no algoritmo. O SPSS produz uma solução única nos passos 1 e 3.

Passo 1: Inicializa o primeiro centróide com um ponto que é próximo a mais números 
    de outros ponto no conjunto de dados.
Passo 3: Assume que m (número de pontos) pontos são distribuidos uniformimente para
    os k (número de clusteres) clusteres então cada cluster é esperado conter m/k pontos. 
    Computa a soma das distâncias do ponto selecionado para os primeiros m/k pontos mais
    próximos e assume isso como y.
"""

def distance(x1, x2):
    """
    Calcula a distância euclidiana entre os pontos x1 e x2.

    @param x1: numpy array representando o primeiro ponto.
    @param x2: numpy array representando o segundo ponto.

    @return dist: float igual a distância entre x1 e x2
    """
    soma = 0
    if len(x1) != len(x2):
        raise ValueError("Os argumentos tem formas diferentes. x1.shape != x2.shape | "+ str(x1.shape)+ " != " + str(x2.shape))

    for idx in range(len(x1)):
        soma += ((x1[idx] - x2[idx])**2)

    dist =  math.sqrt(soma)

    return dist


# 1 Calculate distance matrix Distmxm in which dist(Xi, Xj) represents distance from Xi to Xj
def calculate_dist(points, m):
    """
    Retorna a matrix de distâncias dos pontos.

    @param points: numpy array representando os pontos

    @return matrix: numpy array com todas as distâncias entre os pontos.
    """
    matrix = np.zeros((m, m))
    for idx in range(m):
        for jdx in range(m):
            matrix[idx, jdx] = distance(points[idx], points[jdx])
    
    return matrix

# 2. Find Sumv in which Sumv(i) is the sum of the distances from Xith point to all other points.
def sumv(idx, matrix):
    """
    Retorna a soma das distâncias do ponto idx para todos os outros pontos.

    @param idx: int do indíce de um ponto
    @param matrix: numpy array com todas as distâncias entre os pontos.

    @return soma: float - soma das distância de idx para todos os outros.
    """
    if matrix.shape[0] < idx:
        raise ValueError("Os argumentos tem tamanhos conflitantes. matrix.shape[0] < idx")

    soma = 0

    for d in matrix[idx]:
        soma += d
    
    return soma
    
# 3. Find the index,h of minimum value of Sumv and find highest density point Xh .
def find_minimum(matrix, m):
    """
    Encontra índice do ponto com maior densidade, do ponto que tem o valor mínimo de Sumv.

    @param matrix: numpy array com todoas as distâncias entre os pontos.
    @param m: int representando a quantidade de pontos

    @return idx: int com o indíce do elemento com maior densidade.
    """
    if matrix.shape[0] < m:
        raise ValueError("Os argumentos tem tamanhos conflitantes. matrix.shape[0] < m | "+ str(matrix.shape[0]) +" < "+str(m))
    
    idx = 0
    minimum = sumv(idx, matrix)

    for i in range(m):
        aux = sumv(i, matrix)
        if aux <= minimum:
            idx = i
            minimum = aux
    
    return idx

# 4. Add Xh to C as the first centroid. 
def add_point(idx, points, centers):
    """
    Adiciona o ponto com índice idx no conjunto de centróides

    @param idx: int do indice  do elemento 
    @param points: numpy array com os pontos 
    @param centers: list dos centróides.

    @return centers: list com o ponto idx adicionado.
    """

    centers.append(points[idx])

    return centers

# 5. For each point Xi, set d (Xi) to be the distance between Xi and the nearest point in C.
def set_dist(points, centers):
    """
    Retorna as distâncias entre cada ponto e o centróide mais próximo em centers.

    @param points: numpy array com os pontos 
    @param centers: list dos centróides.

    @return distances: numpy array com as menores distâncias para cada ponto.
    """
    distances = list()
    for point in points:
        distances.append(min([distance(point,center) for center in centers]))
    
    return distances


# 6. Find y as the sum of distances of first m/k nearest points from the Xh.
def find_y(xh, points, m, k):

    """
    Calcula a soma das distâncias dos primeiros m/k pontos mais próximos de xh.

    @param xh: float sendo o ponto de maior densidade (calculado no passo 3)
    @param points: numpy array com os pontos 
    @param m: int - número de pontos
    @param k: int - número de clusters

    @return y: int - soma das distâncias dos primeiros m/k pontos mais próximos de xh
    """
    distances = sorted([ distance(point, xh) for point in points ])
    y = sum(distances[:(m//k)])
    return y

# 7. Find the unique integr i so that
def find_unique(distances, y):
    """
    Retorna um inteiro único tal que:
        d(X1)2+d(X2)2+...+d(Xi)2> = y>d(X1)2+d(X2)2+...+d(X(i-1))2 
    
    @param distances: list das distâncias de cada ponto para o ponto mais próximo em C
    @param y: int - soma das distâncias dos primeiros m/k pontos mais próximos de Xh.

    @return i: int - inteiro único para indexar o ponto a ser adicionado no conjunto de centróids.
    """
    soma = distances[0] ** 2
    i = 0
    flag = True
    for index, dist in enumerate(distances[1:]):
        if ((soma + (dist ** 2)) >= y) and (y > soma):
            i = index
            break
        soma += (dist ** 2)
    return i

# 8. d(X1)2+d(X2)2+...+d(Xi)2> = y>d(X1)2+d(X2)2+...+d(X(i-1))2 
# 9. Add Xi to C
###add_point() 
# 10. Repeat steps 5-8 until k centroids are found

def spss(points, m, k):
    """
    Calcula os centróides inicias para o k-means usando o algoritmo
    Single Pass Seed selection.

    @param points: numpy.array com os pontos que serão clusterizados pelo kmeans.
    @param m: int - número de pontos.
    @param k: int - número de clusters.

    @return centers: numpy array com os centróides iniciais.
    """
    centers = list()
    #1
    matrix = calculate_dist(points, m)
    #2
    #3
    idx = find_minimum(matrix, m)
    xh = points[idx]
    #4
    centers = add_point(idx, points, centers)
    for index in range(k-1):
        #5
        distances = set_dist(points, centers)
        #6
        y = find_y(xh, points, m, k)
        #7 e 8
        i = find_unique(distances, y)
        #9
        centers = add_point(i, points, centers)
       
        #10
    return np.array(centers)

