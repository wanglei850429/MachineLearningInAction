#!/usr/bin/env python
# -*- coding: utf-8 -*-
class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        #节点名称
        self.name = nameValue
        #节点出现次数
        self.count = numOccur
        #同样节点链表指针指向名字相同的treeNode
        self.nodeLink = None
        #父节点
        self.parent = parentNode
        #子节点
        self.children = {}
    
    def add(self,numOccur):
        self.count += numOccur
        
    def display(self,ind=1):
        #“ ”*2=“  ”
        print(" "*ind,self.name," ",self.count)
        for child in self.children.values():
            child.display(ind+1)
            
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
#     simpDat = [['A', 'B','C', 'E','F','O'],
#                ['A', 'C', 'G'],
#                ['E', 'I'],
#                ['A', 'C', 'D', 'E', 'G'],
#                ['A', 'C', 'E', 'G', 'L'],
#                ['E', 'J'],
#                ['A', 'B','C', 'E','F', 'P'],
#                ['A', 'C','D'],
#                ['A', 'C','E', 'G','M'],
#                ['A', 'C','E', 'G','N']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for data in dataSet:
        if frozenset(data) not in retDict.keys():
            retDict[frozenset(data)] = 1
        else:retDict[frozenset(data)] += 1
    return retDict

def updateHeader(nodeToTest,targetNode):
    #利用循环找到链表顶端指针为None的节点
    while(nodeToTest.nodeLink!=None):
        nodeToTest = nodeToTest.nodeLink
    #赋值给链表顶端指针为None的节点
    nodeToTest.nodeLink = targetNode

def updateTree(items,inTree,headerTable,count):
    #items中的第一个节点是否已经是根节点的子节点，如果是增加该子节点的
    if items[0] in inTree.children:
        inTree.children[items[0]].add(count)
    else:
        #如果不是追加父节点的子节点
        inTree.children[items[0]] = treeNode(items[0],count,inTree)
        #如果项头表该项目的链表指向没有设定，则指向该节点
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            #如果有设定更新链表
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
    #递归调用updateTree直至叶子节点
    if len(items) > 1:
        updateTree(items[1:],inTree.children[items[0]],headerTable,count)
            
def createTree(dataSet,minSup=1):
    headerTable = {}
    #第一遍扫描数据，生成项头表，剔除非频繁项集
    for data in dataSet:
        for item in data:
            headerTable[item] = headerTable.get(item, 0) + dataSet[data]
    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del[headerTable[k]]
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0: return None,None
    #修改headerTable的形式，使得作为value可以同时存储出现次数和同节点指针
    for k in headerTable:
        headerTable[k] = [headerTable[k],None]
    retTree = treeNode('Null Set',1,None)
    #第二次扫描数据
    for tranSet, count in dataSet.items():
        tempRoute = {}
        for item in tranSet:
            #剔除非频繁项集
            if item in freqItemSet:
                tempRoute[item] = headerTable[item][0]
        if len(tempRoute) > 0:
            #频繁项集按从大到小排序
            sortedItems = [v[0] for v in sorted(tempRoute.items(),key=lambda x:x[1], reverse=True)]
            #填充FPTree
            updateTree(sortedItems,retTree,headerTable,count)
    return retTree,headerTable
            
def ascendTree(leafNode,prefixPath):     
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent,prefixPath)

     
def findPrefixPath(basePat,treeNode):
    condPats={}
    while treeNode != None:
        prefixPath = []
        #找到一条路径条件模式基
        ascendTree(treeNode,prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        #切换为另一个相同元素
        treeNode = treeNode.nodeLink
    return condPats

#挖掘频繁项集
def mineTree(inTree, headerTable, minSup,prefix,freqItemList):
    #从树的底部开始挖掘，支持度由小到大排序
    bigL = [v[0] for v in sorted(headerTable.items(),key=lambda x:x[1][0])]
    for basePat in bigL:
        newFreqSet = prefix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        #各元素的所有路径（条件模式基）并将该元素的支持度赋给所有parent节点
        condPattBases = findPrefixPath(basePat,headerTable[basePat][1])
        #构建FPTree剔除条件模式基中小于支持度的元素
#         myCondTree,myHead = createTree(condPattBases,minSup)
#         if myHead != None:
#             #递归调用如[A:2->C:2->D:2]->[A,D][C,D][A,C,D]
#             mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)
            
dataSet = loadSimpDat()
initSet = createInitSet(dataSet)
myTree, myHeaderTable = createTree(initSet,3)
# myTree.display()
freqItems = []
mineTree(myTree,myHeaderTable,3,set([]),freqItems)
print(freqItems)
# dataList = []
# with open("kosarak.dat") as f:
#     for line in f:
#         dataList.append(line.split())
# initSet = createInitSet(dataList)
# myTree, myHeaderTable = createTree(initSet,100000)
# freqItems = []
# mineTree(myTree,myHeaderTable,100000,set([]),freqItems)
# print(freqItems)
        
