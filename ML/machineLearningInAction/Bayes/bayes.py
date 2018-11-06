import numpy as np

def loadDataSet():
    message = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
               ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
               ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
               ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
               ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
               ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    label = [0,1,0,1,0,1]
    return message,label

def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
       vocabSet = vocabSet | set(doc)
    return sorted(list(vocabSet),key=lambda x:x[0])

def setWord2Vec(vocabList, inputSet):
    returnVec = np.zeros(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("word is not in vocabList")
    return returnVec

def bagWord2Vec(vocabList, inputSet):
    returnVec = np.zeros(len(vocabList))
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("word is not in vocabList")
    return returnVec

def trainNavieBayes(dataSet, labels):
    mDocs = len(dataSet)
    pc = sum(labels)/float(mDocs)
    p0Num =np.ones(len(dataSet[0]));p1Num =np.ones(len(dataSet[0]));
    p0All=2.0;p1All=2.0
    i = 0
    for doc in dataSet:
        if labels[i] == 1:
            p1Num += doc
            p1All += sum(doc)
        else:
            p0Num += doc
            p0All += sum(doc)
        i += 1
    p1Vec = np.log(p1Num/p1All)
    p0Vec = np.log(p0Num/p0All)
    return p0Vec,p1Vec,pc

def classifyNavieBayes(inputVec,p0Vec,p1Vec,pc):
    p1 = sum(inputVec*p1Vec) + np.log(pc)
    p0 = sum(inputVec*p0Vec) + np.log(1.0-pc)
    if p1 > p0:
        return 1
    else:
        return 0
    

message, label = loadDataSet()
trainDocs = []
vocabList = createVocabList(message)
for doc in message:
    trainDocs.append(setWord2Vec(vocabList,doc))
p0Vec,p1Vec,pc = trainNavieBayes(trainDocs,label)
print(classifyNavieBayes(setWord2Vec(vocabList,['my', 'dog', 'has', 'flea', 'problems', 'help', 'please']),p0Vec,p1Vec,pc))
print(classifyNavieBayes(setWord2Vec(vocabList,['stupid','dog']),p0Vec,p1Vec,pc))
