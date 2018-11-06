#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataList = []
    fr = open(fileName)
    for line in fr.readlines():
        currentLine = line.strip().split('\t')
        floatLine = [float(column) for column in currentLine]
        dataList.append(floatLine)
    return dataList

def splitDataSet(dataSet,feature,value):
#     mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
#     mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:]
    lData = [];rData = []
    for data in dataSet:
        if data[feature] < value:
           lData.append(data) 
        else:
            rData.append(data)
    return lData,rData


def regLeaf(dataSet):
    dataArray = np.array(dataSet)
    return np.mean(dataArray[:,-1])

def regErr(dataSet):
    dataArray = np.array(dataSet)
    return np.var(dataArray[:,-1]) * dataArray.shape[0]

def chooseBestSplit(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    tolS=ops[0];tolN=ops[1]
    dataSet = np.array(dataSet)
    if len(set(dataSet[:,-1])) == 1:
        return None,leafType(dataSet)
    m,n = dataSet.shape
    s = errType(dataSet)
    bestS = np.inf;bestfeat = 0;bestValue = 0
    for featIndex in range(n-1):
        for splitValue in set(dataSet[:,featIndex]):
            list0,list1 = splitDataSet(dataSet.tolist(),featIndex,splitValue)
            if len(list0) < tolN or len(list1) < tolN:
                continue
            newS = errType(list0) + errType(list1)
            if newS < bestS:
                 bestfeat=featIndex
                 bestValue = splitValue
                 bestS = newS
    if abs(s - bestS) < tolS:
        return None,leafType(dataSet)
    list0,list1 = splitDataSet(dataSet,bestfeat,bestValue)
    if (len(list0) < tolN or len(list1) < tolN):
        return None,leafType(dataSet)
    return bestfeat,bestValue
 
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,value = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None: return value
    retTree = {}
    retTree['spFeat'] = feat
    retTree['spValue'] = value
    lSet,rSet = splitDataSet(dataSet,feat,value)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree
    
def isTree(obj):
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    if not isTree(tree):
        return tree
    ltree, rtree = tree['left'], tree['right']
    return (getMean(ltree) + getMean(rtree))/2

def prune(tree, testData):
    ''' 根据测试数据对树结构进行后剪枝
    '''
    if not isTree(tree):
        return tree

    # 若没有测试数据则直接返回树平均值
    if not testData:
        return getMean(tree)

    ltree, rtree = tree['left'], tree['right']

    if not isTree(ltree) and not isTree(rtree):
        # 分割数据用于测试
        ldata, rdata = splitDataSet(testData, tree['spFeat'], tree['spValue'])
        # 分别计算合并前和合并后的测试数据误差
        err_no_merge = (np.sum((np.array(ldata) - ltree)**2) +
                        np.sum((np.array(rdata) - rtree)**2))
        err_merge = np.sum((np.array(testData) - (ltree + rtree)/2)**2)

        if err_merge < err_no_merge:
            print('merged')
            return (ltree + rtree)/2
        else:
            return tree

    tree['left'] = prune(tree['left'], testData)
    tree['right'] = prune(tree['right'], testData)

    return tree
    
def treePredict(data,tree):
    if type(tree) is not dict:
        return tree
    feature = tree['spFeat'];value = tree['spValue']
    if data[feature] < value:
        subTree = tree["left"]
    else:
        subTree = tree["right"]
    return treePredict(data,subTree)
        
    
    
    

# mat0,mat1 = splitDataSet(np.mat(np.eye(4)),1,0.5)
# print(mat0)
# print(mat1)

myData = loadDataSet("ex00.txt")
tree = createTree(myData,ops=(0,1))
print(tree)
testData = loadDataSet("ex2test.txt")
treePred = prune(tree,testData)
print(treePred)
dataSet = np.array(myData)
plt.scatter(dataSet[:,0],dataSet[:,1])
x = np.linspace(0, 1,50)
y = [treePredict([i],tree) for i in x]
plt.plot(x,y,c='r')
plt.show()

