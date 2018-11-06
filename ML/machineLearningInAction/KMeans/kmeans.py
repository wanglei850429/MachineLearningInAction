#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataList = []
    with open(fileName) as fr:
        for line in fr:
            currentLine = line.strip().split('\t')
            floatLine = [float(column) for column in currentLine]
            dataList.append(floatLine)
    return dataList

def distance(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB,2)))

def randCent(dataSet, k):
    n = dataSet.shape[1]
    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = minJ + rangeJ*np.random.rand(k,1)
    return centroids

def kMeans(dataSet,k):
    m,n = np.mat(dataSet).shape
    clusterAssement = np.zeros((m,2))
    centroids = randCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf;clusterIndex = -1
            for j in range(k):
                distIJ = distance(dataSet[i,:],centroids[j,:])
                if distIJ < minDist:
                    minDist = distIJ
                    clusterIndex = j
            if clusterAssement[i,0] != clusterIndex:clusterChanged=True
            clusterAssement[i,:] = clusterIndex,minDist**2
        print(centroids)
        for cent in range(k):
            eachCent = dataSet[np.nonzero(clusterAssement[:,0] == cent)[0]]
            centroids[cent,:] = np.mean(eachCent,axis=0)
    return centroids,clusterAssement

def biKMeans(dataSet,k):
    m,n = np.mat(dataSet).shape
    clusterAssement = np.zeros((m,2))
    centroid0 = np.mean(dataSet,0).tolist()[0]
    centList = [centroid0]
    for i in range(m):
        clusterAssement[i,1] = distance(np.mat(centroid0), dataSet[i,:])**2
    while len(centList) < k:
        lowestSSE = np.inf
        for j in range(len(centList)):
            cluster = dataSet[np.nonzero(clusterAssement[:,0].A == i)[0],:]
            splitCentroids,splitClusterAssement = kMeans(cluster,2)
            sseSplit = np.sum(splitClusterAssement[:,1])
            sseNotSplit = np.sum(clusterAssement[np.nonzero(clusterAssement[:,0].A != i)[0],1])
            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCents = splitCentroids
                bestClusterAss = splitClusterAssement.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClusterAss[np.nonzero(bestClusterAss[:,0].A == 1)[0],0] = len(centList)
        bestClusterAss[np.nonzero(bestClusterAss[:,0].A == 0)[0],0] = bestCentToSplit
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssement[np.nonzero(clusterAssement[:,0].A == bestCentToSplit)[0],:] = bestClusterAss
    return np.mat(centList).clusterAssement
    
# dataMat = np.mat(loadDataSet("testSet.txt"))
dataMat = np.mat(loadDataSet("testSet2.txt"))
# myCent,clusterAssement = kMeans(dataMat,4)
myCent,clusterAssement = kMeans(dataMat,3)
clusterShape = ["o","s","D","^"]
for i in range(clusterAssement.shape[0]):
    plt.scatter(dataMat[i,0],dataMat[i,1],c="b",marker=clusterShape[int(clusterAssement[i,0])])
plt.scatter(myCent[:,0].A.flatten(),myCent[:,1].A.flatten(),c="r",marker="x")
plt.show()

                
    