import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    fr = open(fileName)
    dataMat = []; labelMat = []
    for line in fr.readlines():
        lineArray = line.strip().split('\t')
        dataMat.append([float(lineArray[0]),float(lineArray[1])])
        labelMat.append(float(lineArray[2]))
    return dataMat,labelMat

def selectJRand(i,m):
    j =i
    while j==i:
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    elif aj < L:
        aj = L
    return aj

def smoSimple(dataSet,labelSet,C,toler,maxIter):
    dataMat = np.mat(dataSet);labelMat = np.mat(labelSet).T
    b = 0;m,n = dataMat.shape
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while iter < maxIter:
        alphaPairChanged = 0
        for i in range(m):
            fxi = float(np.multiply(alphas,labelMat).T*(dataMat*dataMat[i,:].T)) + b
            Ei = fxi - float(labelMat[i])
            if((labelMat[i]*Ei < -toler) and (alphas[i]<C) or\
               ((labelMat[i]*Ei > toler) and (alphas[i]>0))):
                j = selectJRand(i,m)
                fxj = float(np.multiply(alphas,labelMat).T*(dataMat*dataMat[j,:].T)) + b
                Ej = fxj - float(labelMat[j])
                alphaIOld = alphas[i].copy()
                alphaJOld = alphas[j].copy()
                if labelMat[i] != labelMat[j]:
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[j]+alphas[i]-C)
                    H = min(C, alphas[j]+alphas[i])
                if L==H:print("L=H");continue
                eta = -2.0*dataMat[i,:]*dataMat[j,:].T + dataMat[i,:]*dataMat[i,:].T +\
                    dataMat[j,:]*dataMat[j,:].T
                if eta <= 0: print("eta<=0");continue
                alphas[j] += labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if abs(alphas[j] - alphaJOld) < 0.00001:print("j not moving enough");continue
                alphas[i]+=labelMat[j]*labelMat[i]*(alphaJOld-alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIOld)*dataMat[i,:]*dataMat[i,:].T-\
                    labelMat[j]*(alphas[j]-alphaJOld)*dataMat[i,:]*dataMat[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIOld)*dataMat[i,:]*dataMat[j,:].T-\
                    labelMat[j]*(alphas[j]-alphaJOld)*dataMat[j,:]*dataMat[j,:].T
                if 0<alphas[i] and alphas[i]<C : b=b1
                elif 0<alphas[j] and alphas[j]<C : b=b2
                else: b=(b1+b2)/2.0
                alphaPairChanged += 1
        if alphaPairChanged == 0: iter+=1
        else: iter=0
        print("iteration number: %d" %iter)
    return b,alphas

def plotBestFit(alphas,b,dataSet,labels):
    dataArray = np.array(dataSet)
    labelArray = np.array(labels)
    dataMat = np.mat(dataSet)
    labelMat = np.mat(labels).T
    m = len(dataSet)
    flg = plt.figure()
    ax = flg.add_subplot(111)
    ax.scatter(dataArray[:,0],dataArray[:,1],c=labelArray[:],s=30, marker = 'o')
    x = np.arange(-2.0,12.0,0.1)
    w = np.multiply(alphas,labelMat).T*dataMat
    y = (-b.getA()[0]-w.getA()[0][0]*x)/w.getA()[0][1]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()
    
class optStruct:
    def __init__(self, dataMatIn, classLabels,C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.toler = toler
        self.m = dataMatIn.shape[0]
        self.alphas = np.mat(np.zeros((self.m,1)))
        self.b = 0
        self.eCache = np.mat(np.zeros((self.m,2)))
        self.K = np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X,self.X[i,:],kTup)
            
            
def calcEk(o,k):
#     fxk = float(np.multiply(o.alphas,o.labelMat).T*(o.X*o.X[k,:].T)) + o.b
    fxk = float(np.multiply(o.alphas,o.labelMat).T*(o.K[:,k])) + o.b
    Ek = fxk - float(o.labelMat[k])
    return Ek

def selectJ(i,o,Ei):
    maxK = -1;maxDelta = 0;Ej = 0
    o.eCache[i] = [1,Ei]
    validECacheList = np.nonzero(o.eCache[:,0].A)[0]
    if len(validECacheList) > 1:
        for k in validECacheList:
            if k == i: continue
            Ek = calcEk(o,k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDelta:
                maxK = k;maxDelta=deltaE;Ej=Ek
        return maxK, Ej
    else:
        j = selectJRand(i,o.m)
        Ej = calcEk(o,j)
        return j, Ej
    
def updateEk(o,k):
    Ek = calcEk(o,k)
    o.eCache[k] = [1,Ek]
    
def innerL(i,o):
    Ei = calcEk(o,i)
    if((o.labelMat[i]*Ei < -o.toler) and (o.alphas[i]<o.C) or\
       ((o.labelMat[i]*Ei > o.toler) and (o.alphas[i]>0))):
        j,Ej = selectJ(i,o,Ei)
        alphaIOld = o.alphas[i].copy()
        alphaJOld = o.alphas[j].copy()
        if o.labelMat[i] != o.labelMat[j]:
            L = max(0,o.alphas[j]-o.alphas[i])
            H = min(o.C, o.C+o.alphas[j]-o.alphas[i])
        else:
            L = max(0,o.alphas[j]+o.alphas[i]-o.C)
            H = min(o.C, o.alphas[j]+o.alphas[i])
        if L==H:print("L=H");return 0
        eta = -2.0*o.X[i,:]*o.X[j,:].T + o.X[i,:]*o.X[i,:].T +\
            o.X[j,:]*o.X[j,:].T
        if eta <= 0: print("eta<=0");return 0
        o.alphas[j] += o.labelMat[j]*(Ei-Ej)/eta
        o.alphas[j] = clipAlpha(o.alphas[j],H,L)
        if abs(o.alphas[j] - alphaJOld) < 0.00001:print("j not moving enough");return 0
        o.alphas[i]+=o.labelMat[j]*o.labelMat[i]*(alphaJOld-o.alphas[j])
        b1 = o.b - Ei - o.labelMat[i]*(o.alphas[i]-alphaIOld)*o.X[i,:]*o.X[i,:].T-\
            o.labelMat[j]*(o.alphas[j]-alphaJOld)*o.X[i,:]*o.X[j,:].T
        b2 = o.b - Ej - o.labelMat[i]*(o.alphas[i]-alphaIOld)*o.X[i,:]*o.X[j,:].T-\
            o.labelMat[j]*(o.alphas[j]-alphaJOld)*o.X[j,:]*o.X[j,:].T
        if 0<o.alphas[i] and o.alphas[i]<o.C : o.b=b1
        elif 0<o.alphas[j] and o.alphas[j]<o.C : o.b=b2
        else: o.b=(b1+b2)/2.0
        return 1
    else:
        return 0
    
def innerLKernel(i,o):
    Ei = calcEk(o,i)
    if((o.labelMat[i]*Ei < -o.toler) and (o.alphas[i]<o.C) or\
       ((o.labelMat[i]*Ei > o.toler) and (o.alphas[i]>0))):
        j,Ej = selectJ(i,o,Ei)
        alphaIOld = o.alphas[i].copy()
        alphaJOld = o.alphas[j].copy()
        if o.labelMat[i] != o.labelMat[j]:
            L = max(0,o.alphas[j]-o.alphas[i])
            H = min(o.C, o.C+o.alphas[j]-o.alphas[i])
        else:
            L = max(0,o.alphas[j]+o.alphas[i]-o.C)
            H = min(o.C, o.alphas[j]+o.alphas[i])
        if L==H:print("L=H");return 0
        eta = -2.0*o.K[i,j] + o.K[i,i] +o.K[j,j]
        if eta <= 0: print("eta<=0");return 0
        o.alphas[j] += o.labelMat[j]*(Ei-Ej)/eta
        o.alphas[j] = clipAlpha(o.alphas[j],H,L)
        if abs(o.alphas[j] - alphaJOld) < 0.00001:print("j not moving enough");return 0
        o.alphas[i]+=o.labelMat[j]*o.labelMat[i]*(alphaJOld-o.alphas[j])
        b1 = o.b - Ei - o.labelMat[i]*(o.alphas[i]-alphaIOld)*o.K[i,i]-\
            o.labelMat[j]*(o.alphas[j]-alphaJOld)*o.K[i,j]
        b2 = o.b - Ej - o.labelMat[i]*(o.alphas[i]-alphaIOld)*o.K[i,j]-\
            o.labelMat[j]*(o.alphas[j]-alphaJOld)*o.K[j,j]
        if 0<o.alphas[i] and o.alphas[i]<o.C : o.b=b1
        elif 0<o.alphas[j] and o.alphas[j]<o.C : o.b=b2
        else: o.b=(b1+b2)/2.0
        return 1
    else:
        return 0

def smoP(dataSet,labelSet,C,toler,maxIter,kTup):
    o = optStruct(np.mat(dataSet), np.mat(labelSet).T,C,toler,kTup)
    iter = 0
    entireSet = True; alphaPairChanged = 0
    while(iter < maxIter and ((alphaPairChanged > 0 ) or entireSet)):
        alphaPairChanged = 0
        if entireSet:
            for i in range(o.m):
#                 alphaPairChanged += innerL(i,o)
                alphaPairChanged += innerLKernel(i,o)
            iter += 1
        else:
            nonBoundIs = np.nonzero((o.alphas.A >0) * (o.alphas.A <C))[0]
            for i in nonBoundIs:
                alphaPairChanged += innerL(i,o)
            iter += 1
        if entireSet : entireSet = False
        elif (alphaPairChanged == 0) : entireSet=True
    return o.b,o.alphas
        
def kernelTrans(X,A,kTup):
    m,n = X.shape
    K =np.mat(np.zeros((m,1)))
    if kTup[0] == 'linear' : K = X*A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow * deltaRow.T
        K = np.exp(-K/(kTup[1]**2))
    return K

def testRbf(k1=1.7):
    trainDataArray, trainLabelArray = loadDataSet("testSetRBF.txt")
    b, alphas = smoP(trainDataArray,trainLabelArray,200,0.0001,10000,('rbf',k1))
    dataMat = np.mat(trainDataArray); labelMat = np.mat(trainLabelArray).T
    svIndex = np.nonzero(alphas.A > 0)[0]
    sv = dataMat[svIndex]
    lablSV = labelMat[svIndex]
    print("There are %d support vectors " % sv.shape[0])
    m,n = dataMat.shape
    errCount=0
    for i in range(m):
        kernelEval = kernelTrans(sv,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(lablSV,alphas[svIndex]) + b
        if np.sign(predict) != np.sign(trainLabelArray[i]): errCount+=1
    print("The training error rate is %f" % float(errCount/m))
    dataArray, labelArray = loadDataSet("testSetRBF2.txt")
    dataMat = np.mat(dataArray); labelMat = np.mat(labelArray).T
    m,n = dataMat.shape
    errCount=0
    for i in range(m):
        kernelEval = kernelTrans(sv,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * np.multiply(lablSV,alphas[svIndex]) + b
        if np.sign(predict) != np.sign(labelArray[i]): errCount+=1
    print("The test error rate is %f" % float(errCount/m))
    xInArray = np.array(trainDataArray);yArray = np.array(trainLabelArray)
    plt.scatter(xInArray[:,0],xInArray[:,1],c=yArray,s=50,cmap='autumn')
    plt.gca().scatter(sv.A[:,0],sv.A[:,1],s=300,linewidth=1,edgecolors="blue",facecolors='none')
    plt.show()
    

# dataMat, labelMat = loadDataSet("testSet.txt")
# # b, alphas = smoSimple(dataMat,labelMat,0.6,0.001,40)
# b, alphas = smoP(dataMat,labelMat,0.6,0.001,40)
# print(b)
# print(alphas[alphas>0])
# plotBestFit(alphas,b,dataMat,labelMat)
testRbf()
        