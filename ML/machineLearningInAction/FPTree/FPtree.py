
class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
    
    def add(self,numOccur):
        self.count += numOccur
        
    def display(self,ind=1):
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
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for data in dataSet:
        retDict[frozenset(data)] = 1
    return retDict
            
def createTree(dataSet,minSup=1):
    headerTable = {}
    for data in dataSet:
        for item in data:
            headerTable[item] = headerTable.get(item, 0) + dataSet[data]
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            del[headerTable[k]]
            
            
dataSet = loadSimpDat()
initSet = createInitSet(dataSet)
createTree(initSet)