from math import log

def createDataSet(fileName):
    fr = open(fileName)
    dataSet = []
    labels = ['age','prescript','astigmatic','tearRate']
    for line in fr.readlines():
        line = line.strip()
        data = line.split('\t')
        dataSet.append(data)
    return dataSet,labels

def calcShanoonEnt(dataSet):
    m = len(dataSet)
    labels = [example[-1] for example in dataSet]
    labelCntDict = {}
    shanoonEnt = 0.0
    for label in labels:
        if label not in labelCntDict.keys():
            labelCntDict[label] = 0
        labelCntDict[label] += 1
    for label in labelCntDict.keys() :
        prob = float(labelCntDict[label])/m
        shanoonEnt += -prob*log(prob,2)
    return shanoonEnt

def splitDataSet(dataSet, axis, value):
    newDataSet = []
    for record in dataSet:
        if record[axis] == value :
            newData = record[:axis]
            newData.extend(record[axis+1:])
            newDataSet.append(newData)
    return newDataSet

def chooseBestFeature(dataSet):
    featureCnt =  len(dataSet[0]) -1
    HD = calcShanoonEnt(dataSet)
    bestInfoRate = 0.0;bestInfoFeature = -1
    for i in range(featureCnt):
        featureIValues = [example[i] for example in dataSet]
        uniqueValues = set(featureIValues)
        featureEnt = 0.0
        HAD = 0.0
        for uniqueValue in uniqueValues:
            subDataSet = splitDataSet(dataSet,i,uniqueValue)
            prob = len(subDataSet)/float(len(dataSet))
            featureEnt += prob*calcShanoonEnt(subDataSet)
            HAD += (prob)*log(prob,2)
        infoGainRate = (HD - featureEnt)/-HAD
        if infoGainRate > bestInfoRate:
            bestInfoRate = infoGainRate
            bestInfoFeature = i
    return bestInfoFeature

def majorCnt(classList):
    classCnt = {}
    for vote in classList:
        if vote not in classCnt.keys():
            classCnt[vote] = 0
        classCnt[vote] += 1
    sortedClassCnt = sorted(classCnt.items(), key=lambda item:item[1], reverse=True)
    return sortedClassCnt[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorCnt
    bestFeature = chooseBestFeature(dataSet)
    bestFeatureLabel =  labels[bestFeature] 
    treeResultMap = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    bestFeatureValue = [example[bestFeature] for example in dataSet]
    uniqueValue = set(bestFeatureValue)
    for value in uniqueValue:
        subLables = labels[:]
        treeResultMap[bestFeatureLabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),subLables)
    return treeResultMap
    
def classify(inputTree, featLabels, testVec):  
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict,featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,fileName):
    import pickle
    fw = open(fileName,'w')
    pickle.dump(inputTree,fw)
    fw.close
    
def loadTree(fileName):
    import pickle
    fr = open(fileName)
    return pickle.load(fr)

dataSet,labels = createDataSet("lenses.txt")
print(classify(createTree(dataSet,labels),['age','prescript','astigmatic','tearRate'],['pre','myope','no','reduced']))