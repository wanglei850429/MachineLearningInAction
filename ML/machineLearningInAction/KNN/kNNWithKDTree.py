import numpy as np
'''
Created on 2018年9月14日

@author: DELL
'''

class KD_node:
    def __init__(self,elt=None,split=None,LL=None,RR=None):
        self.elt = elt
        self.split = split
        self.LL = LL
        self.RR = RR
        
def createKDTree(dataList):
    if len(dataList) == 0:
        return
    dimension = len(dataList[0])
    split = 0.0
    max_var = 0.0
    for i in range(dimension):
        ll = [example[i] for example in dataList]
        var = computeVariance(ll)
        if var >= max_var :
            max_var = var
            split = i
    sortedDataList = sorted(dataList,key= lambda x : x[split],reverse=False)
    elt = sortedDataList[int(len(sortedDataList)/2)]
    root = KD_node(elt,split)
    root.LL = createKDTree(sortedDataList[0:int(len(dataList)/2)])
    root.RR = createKDTree(sortedDataList[int(len(dataList)/2+1):])
    return root
    
def computeVariance(list):
    list = [float(example) for example in list]
    array = np.array(list).reshape(-1,1)
    sum = array.sum()
    mean = sum/float(len(list))
    varVec = (array - np.tile([mean],(len(list),1)))**2/float(len(list))
    return varVec.sum()

def findNN(root, query,k=3):
    nnList = []
    NN = root.elt
    min_dist = computeDist(query, NN)  
    nnList.append([NN,min_dist])
    nodeList = []  
    temp_root = root  
    while temp_root:  
        nodeList.append(temp_root)  
        dd = computeDist(query, temp_root.elt)  
        if min_dist > dd:  
            NN = temp_root.elt
            min_dist = dd
            nnList.append([NN,min_dist])
        ss = temp_root.split  
        if query[ss] <= temp_root.elt[ss]:  
            temp_root = temp_root.LL  
        else:  
            temp_root = temp_root.RR  
    while nodeList:  
        back_elt = nodeList.pop()  
        ss = back_elt.split  
        print ("back.elt = ", back_elt.elt)
        if abs(query[ss] - back_elt.elt[ss]) < min_dist:  
            if query[ss] <= back_elt.elt[ss]:  
                temp_root = back_elt.RR  
            else:  
                temp_root = back_elt.LL  
  
            if temp_root:  
                nodeList.append(temp_root)  
                curDist = computeDist(query, temp_root.elt)  
                if min_dist > curDist:  
                    min_dist = curDist  
                    NN = temp_root.elt
                    nnList.append([NN,min_dist])
    return sorted(nnList,key=lambda item:item[1]) 
  
  
def computeDist(pt1, pt2):  
    pt1Array = np.array(pt1);pt2Array = np.array(pt2)
    return (((pt1Array-pt2Array)**2).sum())**(1/2) 

    

    
dataList = [[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]]
print(findNN(createKDTree(dataList=dataList),[3,4.5]))

        