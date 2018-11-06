import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    fr = open(fileName)
    dataSet=[]
    labels=[]
    for line in fr.readlines():
        column = line.strip().split()
        dataSet.append([1,float(column[0]),float(column[1])])
        labels.append(float(column[2]))
    return dataSet,labels

def sigmoid(wx):
    return 1.0/(1.0+np.exp(-wx))

def costFunction(w,dataSet,labels):
    dataMat = np.mat(dataSet)
    labelsMat = np.mat(labels).T
    ones = np.ones((len(labels),1))
    cost = labelsMat.T*np.log(sigmoid(dataMat*w)) + (ones-labelsMat).T*(np.log(ones-sigmoid(dataMat*w)))
    return cost

def batchGradAscentTrain(dataSet, labels):
    dataMat = np.mat(dataSet)
    label = np.mat(labels).T
    m,n = dataMat.shape
    w = np.ones((n,1))
    alpha = 0.001
    costValue = 0.0
    iterCostValue = costFunction(w,dataSet,labels)
    cnt = 0
    while abs(iterCostValue - costValue) > 0.001 :
        costValue = iterCostValue
        error = label - sigmoid(dataMat*w)
        w = w + alpha*dataMat.T*error
        iterCostValue = costFunction(w,dataSet,labels)
        cnt += 1
    print(cnt)
    return w

def randomGradAscentTrain0(dataSet, labels):
    w = np.ones(len(dataSet[0]))
    alpha = 0.05
    costValue = 0.0
    iterCostValue = costFunction(np.mat(w).reshape(3,1),dataSet,labels)
    j = 0
    while True :
        for i in range(len(dataSet)):
            data = np.array(dataSet[i])
            error = float(labels[i]) - sigmoid(np.dot(data,w))
            w = w + alpha*data*error
            costValue = iterCostValue
            iterCostValue = costFunction(np.mat(w).reshape(3,1),dataSet,labels)
            j += 1
            if abs(iterCostValue - costValue) < 0.00001 :
                print(j)
                return w

def randomGradAscentTrain1(dataSet, labels):
    w = np.ones(len(dataSet[0]))
    costValue = 0.0
    iterCostValue = costFunction(np.mat(w).reshape(3,1),dataSet,labels)
    j = 0
    k = 0
    while True :
        dataIndex = list(range(len(dataSet)))
        for i in range(len(dataSet)):
            alpha = 4/(1.0+i+j)+0.01
            index = int(np.random.uniform(0,len(dataSet)-1))
            data = np.array(dataSet[index])
            del(dataIndex[index])
            error = float(labels[i]) - sigmoid(np.dot(data,w))
            w = w + alpha*data*error
            costValue = iterCostValue
            iterCostValue = costFunction(np.mat(w).reshape(3,1),dataSet,labels)
            k += 1
            if abs(iterCostValue - costValue) < 0.0000001 :
                print(k)
                return w
        j += 1

def denominator(hypo,row):
    temp=np.zeros((row,1))
    temp.fill(1)
    temp=temp-hypo
    temp=np.dot(hypo.T,temp)
    return temp
 
def numerator(hypo,y,x):
    temp = y-hypo
    temp = np.dot(temp.T,x)
    return temp.T


def newtonTrain(dataSet, labels):
    dataMat = np.mat(dataSet)
    m,n = dataMat.shape
    label = np.mat(labels).T
    w = np.zeros((n,1))
    tempxxt = np.dot(dataMat.T,dataMat).I
    for i in range(0,2000):
        temphypo=sigmoid(dataMat*w)
        tempdenominator=denominator(temphypo,m)
        tempnumerator=numerator(temphypo,label,dataMat)
        w = w-np.dot(tempxxt,tempnumerator)/tempdenominator
    return w

def plotBestFit(w,dataSet,labels):
    classAX= [];classAY=[]
    classBX=[];classBY=[]
    dataArray = np.array(dataSet)
    m = len(dataSet)
    for i in range(m):
        if labels[i] == 1:
            classAX.append(dataArray[i,1]);classAY.append(dataArray[i,2])
        else :
            classBX.append(dataArray[i,1]);classBY.append(dataArray[i,2])
    flg = plt.figure()
    ax = flg.add_subplot(111)
    ax.scatter(classAX,classAY,s=30, c='red',marker = 'o')
    ax.scatter(classBX,classBY,s=30, c='blue',marker = 'o')
    x = np.arange(-3.0,3.0,0.1)
    y = (-w[0]-w[1]*x)/w[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()
    

dataSet, labels = loadDataSet("testSet.txt")
# w = randomGradAscentTrain0(dataSet,labels)
w = newtonTrain(dataSet,labels)
# plotBestFit(w,dataSet,labels)
plotBestFit(w.getA(),dataSet,labels)
        