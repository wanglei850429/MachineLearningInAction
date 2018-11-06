import numpy as np
import matplotlib.pyplot as plt

def loadSimpleData():
    dataMat = np.mat([[1.0,2.1],
                      [2.0,1.1],
                      [1.3,1.0],
                      [1.0,1.0],
                      [2.0,1.0]])
    classLabel = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabel

def stumpClassify(dataMat,dimen,threshVal,threshInequ):
    m = dataMat.shape[0]
    ret = np.ones((m,1))
    if threshInequ == "lt":
        ret[dataMat[:,dimen] <= threshVal] = -1.0
    else :
        ret[dataMat[:,dimen] > threshVal] = -1.0
    return ret

def buildStump(dataArray, classLabel,D):
    dataMat = np.mat(dataArray);labelMat=np.mat(classLabel).T
    m,n = dataMat.shape
    numSteps = 10.0;bestStump={};bestClassEst = np.mat(np.zeros((m,1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMat[:,i].min();rangeMax = dataMat[:,i].max() 
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1, int(numSteps)+1):
            for inequal in ["lt","gt"]:
                threshVal = rangeMin + float(j)*stepSize
                predictValue = stumpClassify(dataMat,i,threshVal,inequal)
                errArray = np.mat(np.ones((m,1)))
                errArray[predictValue == labelMat] = 0
                weightedError = D.T*errArray
#                 print("split: dim %d, thresh %.2f,thresh inequal: %s, weighted error is %.3f" \
#                       % (i,threshVal,inequal,weightedError))
                if weightedError < minError :
                    minError = weightedError
                    bestClassEst = predictValue.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['inequal'] = inequal
    return bestStump,minError,bestClassEst

def adaBoostTrainDS(dataMat,classLabel,numIter=40):
    weakClassArray = []
    m = dataMat.shape[0]
    D = np.mat(np.ones((m,1))/m)
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(numIter):
        bestStump,minError,bestClassEst = buildStump(dataMat,classLabel,D)
        print("D:",D.T)
        alpha = float(0.5*np.log((1-minError)/max(minError,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArray.append(bestStump)
        print("classEst:",bestClassEst.T)
        expon = np.multiply(-1*alpha*np.mat(classLabel).T,bestClassEst)
        D = np.multiply(D,np.exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*bestClassEst
        print("aggClassEst:",aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabel).T,np.ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate,"\n")
        if errorRate == 0.0: break
    return weakClassArray,aggClassEst

def adaClassify(dataIn, weakClassArray):
    dataMat = np.mat(dataIn)
    m = dataMat.shape[0]
    aggClassEst = np.zeros((m,1))
    for weakClass in weakClassArray:
        classEst = stumpClassify(dataMat,weakClass["dim"],weakClass["thresh"],weakClass["inequal"])
        aggClassEst += weakClass["alpha"]*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

def loadDataSet(fileName):
    column = 0;
    dataList = [];labelList=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArray = []
        if column == 0:column = len(line.split('\t'))
        currentLine = line.strip().split('\t')
        for i in range(column):
            if i != column -1:
                lineArray.append(float(currentLine[i]))
            elif i == column -1:
                labelList.append(float(currentLine[i]))
        dataList.append(lineArray)
    return dataList,labelList
            
def plotROC(predStrength, classLabels):
    cur = (0.0,0.0)
    ySum = 0.0
    numPosClass = float(sum(np.array(classLabels) == 1.0))
    numNegClass = float(len(classLabels) - numPosClass)
    TP = 0.0;FP=0.0
    TPR = 0.0;FPR=0.0
    TPROld = 0.0;FPROld = 0.0
#     yStep = 1.0 / float(numPosClass)
#     xStep = 1.0 / float(len(classLabels) - numPosClass)
    sortedIndex = np.argsort(-predStrength)
    flg = plt.figure()
    flg.clf()
    ax = plt.subplot(111)
    for index in sortedIndex.tolist()[0]:
        if classLabels[index] == 1.0:
#             delX = 0;delY = yStep
            TP += 1
            TPROld = TPR
            TPR = TP/(numPosClass)
        else:
#             delX = xStep;delY = 0
            ySum += TPR
            FP += 1
            FPROld = FPR
            FPR = FP /(numNegClass)
            
#         ax.plot([cur[0], cur[0] - delX],[cur[1], cur[1] - delY],c='b')
        ax.plot([FPROld, FPR],[TPROld, TPR],c='b')
#         cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel("FP Rate");plt.ylabel("TP Rate")
    plt.title("ROC")
    ax.axis([0,1,0,1])
    print("AUC=",ySum*(1/numNegClass))
    plt.show()
    
        
   
        
dataList,labelList = loadDataSet("horseColicTraining2.txt")
weakClassArray,aggClassEst = adaBoostTrainDS(np.mat(dataList),labelList,10)
testDataList,testlabelList = loadDataSet("horseColicTest2.txt")
predict = adaClassify(testDataList,weakClassArray)
m = len(testlabelList)
errMat= np.mat(np.zeros((m,1)))
errMat[predict!=np.mat(testlabelList).T] = 1
print("errRate is %.2f" % (errMat.sum()/m))

plotROC(aggClassEst.T,labelList)

# dataMat,classLabel = loadSimpleData()

# bestStump,minError,bestClassEst = buildStump(dataMat,classLabel,np.mat(np.ones((5,1))/5))
# print(bestStump)
# print(bestClassEst)       
# classArray = adaBoostTrainDS(dataMat,classLabel,9)
# adaClassify([5,5],classArray)


                
            
        
        
        