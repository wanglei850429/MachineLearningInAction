import numpy as np
import operator as op
import matplotlib.pyplot as plt

# def createDataSet():
#     group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
#     label = ["A","A","B","B"]
#     return group, label
# 
# group, label = createDataSet();

def classify(inX, group, label, k):
    dataSetSize = group.shape[0]
#   inX= [0.8,0.9]    diffMat=[[-0.2 -0.2][-0.2 -0.1][ 0.8  0.9][ 0.8  0.8]]
    diffMat = np.tile(inX, (dataSetSize, 1)) - group
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance ** 0.5
    sortedDistnace = distance.argsort()
    classCount = {}
    for i in range(k):
        votelabel = label[sortedDistnace[i]]
        classCount[votelabel] = classCount.get(votelabel,0) + 1
#     sortedClassCount = sorted(classCount.items(), key=op.itemgetter(1),reverse=True)
    sortedClassCount = sorted(classCount.items(), key=lambda item:item[1],reverse=True)
    return sortedClassCount[0][0]

def file2Matrix(fileName):
    file = open(fileName)
    lines = file.readlines()
    featuresMat = np.zeros((len(lines),3))
    labels = []
    index = 0
    for line in lines:
        line = line.strip()
        listFromLine = line.split('\t')
        featuresMat[index,:] = listFromLine[0:3]
        labels.append(listFromLine[-1])
        index += 1
    return featuresMat,labels
 
def norm(dataSet):
    minValues = dataSet.min(0)
    maxValues = dataSet.max(0)
    ranges = maxValues - minValues
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minValues, (m,1))
    normDataSet = normDataSet / np.tile(ranges, (m,1))
    return normDataSet, minValues, ranges
     
def datingClassTest():
    rate = 0.1
    datingFeatures, datingLabels = file2Matrix("datingTestSet2.txt")
    datingFeatures,minValues, ranges = norm(datingFeatures)
    m = datingFeatures.shape[0]
    numTest = int(m * rate)
    errCount = 0.0
    for i in range(numTest):
        result = classify(datingFeatures[i,:],datingFeatures[numTest:m,:],\
        datingLabels[numTest:m],3)
        print("the predict result is %s and the real result is %s" %(result, datingLabels[i]) )
        if result != datingLabels[i] :
            errCount += 1.0
    print("The total error rate is: %f" %(errCount/numTest))
    
def predictPerson():
    results = ["not at all","in small doses","in large doses"]
    flight = float(input("flier miles per year"))
    game = float(input("percentage of time spent playing games"))
    iceCream = float(input("iceCream liter"))
    datingFeatures, datingLabels = file2Matrix("datingTestSet2.txt")
    datingFeatures,minValues, ranges= norm(datingFeatures)
    inputData = np.array([flight,game,iceCream])
    normInputData = (inputData - minValues)/ranges
    result = classify(normInputData,datingFeatures,datingLabels,3)
    print(results[int(result) - 1])
    
    
def img2Vector(fileName):
    file = open(fileName)
    lines = file.readlines()
    numberArray = []
    for line in lines:
        line.strip()
        lineArray = []
        for char in line[0:-1]:
            lineArray.append(char)
        numberArray.append(lineArray)
    mat = np.array(numberArray)
    mat = mat.reshape([1,1024])
    
img2Vector("F:/machineLearninginAction/trainingDigits/0_14.txt")
# predictPerson()
# datingClassTest()
# datingFeatures, datingLabels = file2Matrix("datingTestSet2.txt")
# datingFeatures = norm(datingFeatures)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingFeatures[:,0],datingFeatures[:,1],c=datingLabels,marker='o',cmap='autumn')
# plt.show()


# print(classify([0,0],group,label,3))

    
    
    