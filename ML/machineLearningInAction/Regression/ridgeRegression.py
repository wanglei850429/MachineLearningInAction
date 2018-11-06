import numpy as np
import matplotlib.pyplot as plt 
from regression import standReg

def ridgeRegre(xMat, yMat, lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + np.eye(xMat.shape[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("denom is singular")
        return
    w = denom.I*xMat.T*yMat
    return w

def ridgeTest(xArray, yArray):
    xMat, yMat = regularize(xArray,yArray)
    numTestPts = 30
    wMat = np.zeros((numTestPts,xMat.shape[1]))
    for i in range(numTestPts):
        w = ridgeRegre(xMat,yMat,np.exp(i - 10))
        wMat[i,:] = w.T
        print("lambda=%f" % np.exp(i - 10))
        print(w.T)
    return wMat

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    fr = open(fileName)
    dataList = [];labelList = []
    for line in fr.readlines():
        currentLine = line.strip().split('\t')
        lineArray = [float(column) for column in currentLine[:numFeat]]
        dataList.append(lineArray)
        labelList.append(float(currentLine[-1]))
    return dataList,labelList

def rssError(yArray, yHatArray):
    return ((yArray - yHatArray)**2).sum()

def stageWise(xArray,yArray,eps=0.001,numIt=5000):
    xMat, yMat = regularize(xArray,yArray)
    m,n = xMat.shape
    retMat = np.zeros((numIt,n))
    w = np.zeros((n,1));wTest = w.copy();wMax = w.copy()
    for i in range(numIt):
        print(w.T)
        lowestError = np.inf
        for j in range(n):
           for sign in [-1,1]: 
               wTest = w.copy()
               wTest[j] +=  sign*eps
               yTest = xMat*wTest
               rssE = rssError(yMat.A,yTest.A)
               if rssE < lowestError:
                   lowestError = rssE
                   wMax = wTest
        w = wMax.copy()
        retMat[i,:] =  w.T
        print(w.T)
    return retMat
               
def regularize(xArray,yArray):      
    xMat = np.mat(xArray)
    yMat = np.mat(yArray).T    
    yMean = np.mean(yMat,0)
    yMat = yMat - yMean
    xMean = np.mean(xMat,0)
    xVar = np.var(xMat,0)
    xMat = (xMat - xMean)/xVar
    return xMat,yMat
        
    

abX,abY = loadDataSet("abalone.txt")
xMat,yMat = regularize(abX,abY)
print(standReg(xMat,yMat).T)
# wMAT = ridgeTest(abX,abY)
# stageWise(abX,abY)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(wMAT)
# plt.show()

    