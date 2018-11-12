#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

def loadExData():
    return [[1,1,1,0,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [1,1,0,2,2],
            [0,0,0,3,3],
            [0,0,0,1,1]]

def loadRecData():
    return [[4,4,0,2,2],
            [4,0,0,3,3],
            [4,0,0,1,1],
            [1,1,1,2,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0]]

#欧式距离计算相似度    
def euclidSim(inA,inB):
    return 1.0/(1.0+np.linalg.norm(inA-inB))

#相关系数计算相似度
def pearsSim(inA,inB):
    if len(inA) < 3: return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]

#余弦相似度 cos(theta) = AB/||A|| ||B||
def cosSim(inA,inB):
    num = float(inA.T*inB)
    return 0.5+0.5*(num/(np.linalg.norm(inA)*np.linalg.norm(inB)))
    
    
#     
#     
#     
# data = loadExData()
# # u,sigma,vt = np.linalg.svd(data)
# dataArray = np.array(data)
# # print(euclidSim(dataArray[:,0],dataArray[:,4]))
# # print(cosSim(dataArray[:,0],dataArray[:,4]))
# print(pearsSim(dataArray[:,0],dataArray[:,4]))

def standEst(dataMat,user,simMeas,item):
    n = dataMat.shape[1]
    simTotal = 0.0; ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        #遍历所有菜寻找与未评价菜的相似度，跳过未评价的菜
        if userRating == 0:continue
        #挑出其他用户对未评价菜做过评价的同时，还做过对其他菜（J）的评价，计算彼此的相似度
        overlap = np.nonzero(np.logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0]
        if len(overlap) == 0: similarity = 0
        else: similarity = simMeas(dataMat[overlap,item],dataMat[overlap,j])
        simTotal += similarity
        ratSimTotal += similarity*userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
    
def svdEst(dataMat,user,simMeas,item):
    n = dataMat.shape[1]
    simTotal = 0.0; ratSimTotal = 0.0
    u,sigma,vt = np.linalg.svd(dataMat)
    #sigmaX平方和大于总平方和的90%，X=4
    sigma4 = np.mat(np.eye(4)*sigma[:4])
    xformedItems = dataMat.T*u[:,:4]*sigma4.I
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j == item:continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        simTotal += similarity
        ratSimTotal += similarity*userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal
    
def recommend(dataMat,user,N=3,simMeas=cosSim,estMethod=svdEst):
    #找出未评分的菜
    unratedItems = np.nonzero(dataMat[user,:].A==0)[1]
    if len(unratedItems) == 0: return "you rated everything"
    itemScores = []
    for item in unratedItems:
        estScore = estMethod(dataMat,user,simMeas,item)
        itemScores.append((item,estScore))
    return sorted(itemScores,key=lambda x:x[1],reverse=True)[:N]

# print(recommend(np.mat(loadRecData()),2))

def printMat(inMat,thresh=0.8):
    for i in range(32):
        for j in range(32):
            if float(inMat[i,j]) > thresh:
                print(1)
            else:
                print(0)
        print(" ")
        
def imgCompress(numSV=3,thresh=0.8):
    myl = []
    for line in open("0_5.txt").readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print("orignal matrix")
    printMat(myMat,thresh)
    print(myMat.shape)
    u,sigma,vt = np.linalg.svd(myMat)
    sigRecon = np.mat(np.zeros((numSV,numSV)))
    for k in range(numSV):
        sigRecon[k,k] = sigma[k]
    reconMat = u[:,:numSV]*sigRecon*vt[:numSV,:]
    print("svd matrix")
    printMat(reconMat,thresh)
    print(reconMat.shape)
    
imgCompress(2)
    

            



