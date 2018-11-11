#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName,delimt="\t"):
    dataSet = []
    with open(fileName) as fr:
        for line in fr:
            lineList = line.strip().split(delimt)
            floatLine = [float(column) for column in lineList]
            dataSet.append(floatLine)
    return np.mat(dataSet)

def pca(dataMat,topNFeat):
    meanValues = np.mean(dataMat,0)
    centralize = dataMat - meanValues
#     stdValues = np.std(centralize)
#     standard = dataMat/stdValues
    covMat = np.cov(centralize,rowvar=0)
    eigValues,eigVectors = np.linalg.eig(np.mat(covMat))
    eigValueIndex = np.argsort(eigValues)
    eigValueIndex = eigValueIndex[:-(topNFeat+1):-1]
    regEigVectors = eigVectors[:,eigValueIndex]
    lowDDataMat = centralize * regEigVectors
    return lowDDataMat,lowDDataMat*regEigVectors.T+meanValues

# dataMat = loadDataSet("testSet.txt")
# lowDMat,chgMat = pca(dataMat,1)
#  
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker="^",s=90)
# ax.scatter(chgMat[:,0].flatten().A[0],chgMat[:,1].flatten().A[0],marker="o",s=50,c="red")
# plt.show()

def replaceNanWithMean():
    dataMat = loadDataSet("secom.data", " ")
    numFeat = dataMat.shape[1]
    for i in range(numFeat):
        meanValue = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:,i].A))[0],i])
        dataMat[np.nonzero(np.isnan(dataMat[:,i].A))[0],i] = meanValue
    return dataMat
 
dataMat = replaceNanWithMean()
meanValues = np.mean(dataMat,0)
centralize = dataMat - meanValues
cov = np.cov(centralize, rowvar=0)
eigValues,eigVector = np.linalg.eig(np.mat(cov))
print(sorted(eigValues,reverse=True))





    
            