import numpy as np
import re

def createVocabList(dataSet):
    vocabSet = set([])
    for doc in dataSet:
       vocabSet = vocabSet | set(doc)
    return sorted(list(vocabSet),key=lambda x:x[0])

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

def textParse(string):
    listOfTokens = re.split(r'\W*', string)
    return [token.lower() for token in listOfTokens if len(token) > 2]

def spamTest():
    docList = []; classList = [];
    for i in range(1,26):
        wordList = textParse(open("email/spam/%d.txt" % i).read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open("email/ham/%d.txt" % i).read())
        docList.append(wordList)
        classList.append(0)
        print(i)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50));testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []; trainClass = []
    for index in trainingSet:
        trainMat.append(bagWord2Vec(vocabList,docList[index]))
        trainClass.append(classList[index])
    p0Vec,p1Vec,pc = trainNavieBayes(trainMat,trainClass)
    errorCount = 0.0
    for index in testSet:
        words = bagWord2Vec(vocabList,docList[index])
        if classifyNavieBayes(words,p0Vec,p1Vec,pc) != classList[index]:
            errorCount +=1
    print("error rate is: ",float(errorCount)/len(testSet))
    
spamTest()
    