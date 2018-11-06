import numpy as np

def sigmoid(wx):
    return 1.0/(1.0+np.exp(-wx))

def costFunction(w,dataSet,labels):
    dataMat = np.mat(dataSet)
    labelsMat = np.mat(labels).T
    ones = np.ones((len(labels),1),dtype='float')
    cost = labelsMat.T*np.log(sigmoid(dataMat*w)) + (ones-labelsMat).T*(np.log(ones-sigmoid(dataMat*w)))
    return cost

def randomGradAscentTrain0(dataSet, labels):
    w = np.ones(len(dataSet[0]),dtype='float')
    alpha = 0.05
    costValuePrePre = 0.0
    costValuePre = 0.0
    costValueNow = costFunction(np.mat(w).reshape(len(dataSet[0]),1),dataSet,labels)
    j = 0
    while True :
        dataIndex = list(range(len(dataSet)))
        for i in range(len(dataSet)):
            index = int(np.random.uniform(0,len(dataIndex)-1))
            data = np.array(dataSet[index])
            error = float(labels[index]) - sigmoid(np.dot(data,w))
            w = w + alpha*data*error
            del(dataIndex[index])
            costValuePrePre = costValuePre
            costValuePre = costValueNow
            costValueNow = costFunction(np.mat(w).reshape(len(dataSet[0]),1),dataSet,labels)
            j += 1
#             print(abs(costValueNow - costValuePre))
            if abs(costValueNow - costValuePre) + abs(costValueNow - costValuePrePre) < 0.00001 :
#                 print(j)
                return w

def norm(dataSet):
    dataMat = np.mat(dataSet).T
    n,m = dataMat.shape
    for i in range(n):
        minValues = dataMat[i].min()
        maxValues = dataMat[i].max()
        ranges = maxValues - minValues
        dataMat[i] = dataMat[i] - np.tile(minValues, (1,m))
        dataMat[i] = dataMat[i] / np.tile(ranges, (1,m))
    return dataMat.T
       
def loadHorseColic(fileName):
    fr = open(fileName)
    trainSet = [];labelSet=[]
    featureNum = 0
    for line in fr.readlines():
        currentLine = line.strip().split('\t')
        if featureNum == 0:
            featureNum = len(currentLine) - 1
        lineArray = [float(currentLine[i]) for i in range(featureNum)]
        trainSet.append(lineArray)
        labelSet.append(float(currentLine[featureNum]))
    return trainSet,labelSet

def predictClassify(inx,w):
    if sigmoid(sum(inx*w)) > 0.5:
        return 1.0
    else:
        return 0.0
    
def colicTest(dataSet, labels,w):
    predictResult = [predictClassify(data,w) for data in dataSet]
    errorArray = np.array(predictResult) - np.array(labels)
    errorArray = np.array([abs(error) for error in errorArray])
    errorRate = float(errorArray.sum())/len(dataSet)
    print(errorRate)
    return errorRate

def multiTest():
    dataSet, labels = loadHorseColic("horseColicTraining.txt")
    testSet, testLabels = loadHorseColic("horseColicTest.txt")
    numTests =10 ;errorSum = 0.0
    for k in range(numTests):  
        w = randomGradAscentTrain0(norm(dataSet).getA(),labels)
        errorSum += colicTest(norm(testSet).getA(),testLabels,w)
    print("average error rate of 10 times is", errorSum/(float(numTests)))
    

# dataSet, labels = loadHorseColic("horseColicTraining.txt")
# w = randomGradAscentTrain0(norm(dataSet).getA(),labels)
# testSet, testLabels = loadHorseColic("horseColicTest.txt")
# colicTest(norm(testSet).getA(),testLabels,w)
multiTest()