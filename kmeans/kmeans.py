import os
import sys

import numpy as np


current_root = os.path.dirname(__file__)
test_set = "/testSet.txt"

inf = 100000

def loadSet(fileName):
    dataMat = []
    file = open(fileName, 'r')
    lines = file.readlines()
    for line in lines:
        currentline = line.strip().split('\t')
        float_line = list(map(float, currentline))
        dataMat.append(float_line)
    return dataMat

def distEclud(VecA,VecB):
    return np.sqrt(sum(np.square(np.power(VecA-VecB, 2))))

def randCent(dataSet, k):
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangJ * np.random.rand(k,1)
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))
    centroids = randCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = np.mat(np.zeros((1,2)))
            for j in range(k):
                distJI = distMeas(centroids[j,:], dataSet[i,:])
                if distJI.all < minDist.all:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != j:
                clusterChanged = True
            clusterAssment[i,:] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A == cent)[0]]
            centroids[cent,:] = np.mean(ptsInClust, axis= 0)
    return centroids, clusterAssment






if __name__ == '__main__':
    dataMat = np.mat(loadSet(current_root+test_set))
    randCent(dataMat, 3)

    dist = distEclud(dataMat[0],dataMat[1])
    print(dist)
    kMeans(dataMat, 4)
