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

def standReg(dataList,labelList):
    if isinstance(dataList,list):
        xMat = np.mat(dataList); yMat = np.mat(labelList).T
    else :
        xMat = dataList;yMat = labelList
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) == 0.0:
        print("need to use gradient descent")
        return
    w = xTx.I *xMat.T*yMat
    return w

def plotPredict(dataList,labelList,w):
    xMat = np.mat(dataList)
    yMat = np.mat(labelList).T
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:,1].flatten().A[0],yMat[:,0].flatten().A[0],s=2,c='red')
    xCopy = xMat.copy()
    xCopy = sorted(xCopy.A,key = lambda item:item[1])
    # xCopy = xCopy.sort(0)
    yPredict = np.mat(xCopy) * w
    ax.plot(np.mat(xCopy)[:,1],yPredict,c='black')
    plt.show()


dataList,labelList = loadDataSet("ex0.txt")
w = standReg(dataList,labelList)
plotPredict(dataList,labelList,w)

        
        