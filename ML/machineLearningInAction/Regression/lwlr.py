import numpy as np
import matplotlib.pyplot as plt


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

def lwlr(testPoint, xList, yList,k=1.0):
    xMat = np.mat(xList);yMat = np.mat(yList).T
    m = xMat.shape[0]
    weights = np.eye(m)
    for i in range(m):
        diffMat = testPoint - xMat[i,:]
        weights[i,i] = np.exp(diffMat*diffMat.T/(-2*k**2))
    xTx = xMat.T*weights*xMat
    if np.linalg.det(xTx) == 0.0:
        print("singular matrix")
        return
    w = xTx.I*xMat.T*weights*yMat
    return testPoint*w

def lwlrTest(testArrays,xList, yList,k=1.0):
    m = len(testArrays)
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArrays[i],xList,yList,k)
    return yHat

def rssError(yArray, yHatArray):
    return ((yArray - yHatArray)**2).sum()

# dataList,labelList = loadDataSet("ex0.txt")
# yHat = lwlrTest(dataList,dataList,labelList,k=0.01)
# xMat = np.mat(dataList);yMat = np.mat(labelList).T
# sortIndex = np.argsort(xMat[:,1],0)
# xSort = xMat[sortIndex][:,0,:]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:,1].flatten().A[0],yMat[:,0].flatten().A[0],s=2,c='red')
# ax.plot(xSort[:,1],yHat[sortIndex],c='black')
# plt.show()
dataList,labelList = loadDataSet("abalone.txt")
yHat01 = lwlrTest(dataList[0:99],dataList[0:99],labelList[0:99],0.1)
yHat1 = lwlrTest(dataList[0:99],dataList[0:99],labelList[0:99],1)
yHat10 = lwlrTest(dataList[0:99],dataList[0:99],labelList[0:99],10)
print("yhat0.1",rssError(labelList[0:99],yHat01.T))
print("yhat1",rssError(labelList[0:99],yHat1.T))
print("yhat10",rssError(labelList[0:99],yHat10.T))


    
