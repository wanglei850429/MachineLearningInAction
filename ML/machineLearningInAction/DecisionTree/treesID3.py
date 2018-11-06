from math import log

def calcShannonEnt(dataSet):
    m = len(dataSet)
    labelDict = {}
    for data in dataSet:
        label = data[-1]
        if label not in labelDict.keys():
            labelDict[label] = 0
        labelDict[label] += 1
    shannonEnt = 0.0
    for label in labelDict.keys():
        prob = float(labelDict[label])/m
        shannonEnt += -prob*log(prob,2)
    return shannonEnt     

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for feature in dataSet:
        if feature[axis] == value :
            reducedFeatureVec = feature[:axis]
            reducedFeatureVec.extend(feature[axis+1:])
            retDataSet.append(reducedFeatureVec)
    return retDataSet

def chooseBestSplitFeature(dataSet):
    numberFeatures = len(dataSet[0]) - 1
    baseEnt = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numberFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEnt = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i , value)
            prob = len (subDataSet)/float(len(dataSet))
            newEnt+= prob*calcShannonEnt(subDataSet)
        infoGain = baseEnt- newEnt
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
        
def majorCnt(classList):
    labelCnt = {}
    for vote in classList:
        if vote not in labelCnt.keys():
            labelCnt[vote] = 0
        labelCnt[vote] += 1
    sortedLabelCnt = sorted(labelCnt.items(), key=lambda item:item[1], reverse=True)
    return sortedLabelCnt[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList) :
        return classList[0]
    if len(dataSet[0]) == 1 :
        return majorCnt(classList)
    bestFeature = chooseBestSplitFeature(dataSet)
    bestFeatureLabels = labels[bestFeature]
    myTree = {bestFeatureLabels:{}}
    del(labels[bestFeature])
    featureValues = [example[bestFeature] for example in dataSet]
    uniqueValues = set(featureValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatureLabels][value] = createTree(splitDataSet\
            (dataSet, bestFeature, value), subLabels)
    return myTree

dataSet, labels = createDataSet()
trees = createTree(dataSet, labels)    
print(trees)

# print(chooseBestSplitFeature(createDataSet()))
