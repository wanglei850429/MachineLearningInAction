import matplotlib.pyplot as plt
import treesID4dot5
from matplotlib.pyplot import yticks

decisionNode = dict(boxstyle="sawtooth",fc="0.8")
leafNode = dict(boxstyle="round4",fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,\
        xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',\
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
    
# def createPlot():
#     fig = plt.figure(1,facecolor='white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111,frameon=False)
#     plotNode('decisionNode',(0.5,0.1),(0.1,0.5),decisionNode)
#     plotNode('leafNode',(0.8,0.1),(0.3,0.8),leafNode)
#     plt.show()
    
def getLeafNums(tree):
    leafNum = 0
    keys = list(tree.keys())
    secondDict = tree[keys[0]]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
             leafNum += getLeafNums(secondDict[key])
        else:
            leafNum += 1
    return leafNum

def getTreeDepth(tree):
    maxDepth = 0
    keys = list(tree.keys())
    secondDict = tree[keys[0]]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth : maxDepth = thisDepth
    return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0] - cntrPt[0]) /2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) /2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid,txtString)
    
def plotTree(myTree, parentPt, nodeTxt):
    leafNum = getLeafNums(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(leafNum))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr,cntrPt,parentPt,decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt,str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    flg = plt.figure(1, facecolor='white')
    flg.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False, **axprops)
    plotTree.totalW = float(getLeafNums(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
    plotTree(inTree,(0.5,1.0),'')
    plt.show()
    
    

dataSet, labels = treesID4dot5.createDataSet("lenses.txt") 
tree = treesID4dot5.createTree(dataSet, labels)
createPlot(tree)
