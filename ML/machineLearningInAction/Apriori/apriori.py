#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    C1=[]
    for trans in dataSet:
        for item in trans:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return [frozenset(i) for i in C1]

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for data in D:
        for item in Ck:
            if item.issubset(data):
                if not item in ssCnt.keys():
                    ssCnt[item]=1
                else:
                    ssCnt[item]+=1
    numData = float(len(D))
    retList = [];supportData={}
    for key in ssCnt:
        support = ssCnt[key]/numData
        if support >= minSupport:
            retList.append(key)
        supportData[key] = support
    return retList,supportData

def createCN(itemList,k):
    item = itemList[k-2]
    retList = []
    lenItemList = len(item)
    for i in range(lenItemList):
        for j in range(i+1,lenItemList):
            itemTemp = item[i]|item[j]
            if len(itemTemp)>k or itemTemp in retList:
                continue
            retList.append(item[i]|item[j])
    return retList

def apriori(dataSet,minSupport=0.5):
    C1 = createC1(dataSet)
    D=[set(i) for i in dataSet]
    L1,supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while(True):
        Ck = createCN(L,k)
        Lk, supK = scanD(D,Ck,minSupport)
        if Lk == []:
            k -= 1
            break
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData,k

def calcConf(freqSet,H1,supportData,bigRuleList,minConf):
    prunedH = []
    for conseq in H1:
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf>= minConf:
            print(freqSet-conseq,"-->",conseq,"conf:",conf)
            bigRuleList.append((freqSet-conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf):
    m = len(H1[0])
    if (len(freqSet) > (m+1)):
        hmp1 = createCN(H1,m+1)
        hmp1 = calcConf(freqSet,hmp1,supportData,bigRuleList,minConf)
        if (len(hmp1) > 1):
            rulesFromConseq(freqSet,hmp1,supportData,bigRuleList,minConf)

def generateRules(L,supportData,minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
            else:
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

dataSet = loadDataSet()
# C1= createC1(dataSet)
# print(C1)
# D=[set(i) for i in dataSet]
# print(D)
# L1,supportData0 = scanD(D,C1,0.5)
# print(L1)
mushroom = []
with open("mushroom.dat") as fr:
    for line in fr.readlines():
        currentLine = line.strip().split()
        mushroom.append(currentLine)
L,supportData,k = apriori(mushroom,minSupport=0.3)
rules = generateRules(L,supportData)
        

# L,supportData,k = apriori(dataSet)
# rules = generateRules(L,supportData,minConf=0.5)
# print(rules)

