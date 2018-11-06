import tkinter
import numpy as np
import model_tree
import regression_tree
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def reDraw(tolS,tolN):
    reDraw.f = Figure(figsize=(5,4),dpi=100)
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
    reDraw.canvas.draw()
    reDraw.canvas.get_tk_widget().grid(row=0,columnspan=3)
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    if chkBtnVar.get():
        if tolN <2:tolN=2
        myTree = regression_tree.create_tree(reDraw.rawData,model_tree.fleaf,model_tree.ferr,{"err_tolerance":tolS,"n_tolerance":tolN})
        yHat = np.zeros(reDraw.testData.shape[0])
        for i in range(reDraw.testData.shape[0]):
            yHat[i] = model_tree.tree_predict([1.0,reDraw.testData[i]],myTree)
    else:
        myTree = regression_tree.create_tree(reDraw.rawData,regression_tree.fleaf,regression_tree.ferr,{"err_tolerance":tolS,"n_tolerance":tolN})
        yHat = np.zeros(reDraw.testData.shape[0])
        for i in range(reDraw.testData.shape[0]):
            yHat[i] = regression_tree.tree_predict([1.0,reDraw.testData[i]],myTree)
    reDraw.a.scatter(reDraw.rawData[:,1],reDraw.rawData[:,2],s=5)
    reDraw.a.plot(reDraw.testData,yHat,c="r")
    reDraw.canvas.draw()

def getInputs():
    try:tolN = int(tolNEntry.get())
    except:
        tolN = 10
    try:tolS = int(tolSEntry.get())
    except:
        tolS = 1.0
    return tolN,tolS

def drawNewTree():
    tolN,tolS = getInputs()
    reDraw(tolS,tolN)

root = tkinter.Tk()

tkinter.Label(root,text="Plot Place Holder").grid(row=0,columnspan=3)
tkinter.Label(root,text="tolN").grid(row=1,column=0)
tolNEntry = tkinter.Entry(root)
tolNEntry.grid(row=1,column=1)
tolNEntry.insert(0,'10')

tkinter.Label(root,text="tolS").grid(row=2,column=0)
tolSEntry = tkinter.Entry(root)
tolSEntry.grid(row=2,column=1)
tolSEntry.insert(0,'1.0')
tkinter.Button(root,text="redraw",command=drawNewTree).grid(row=1,column=2,rowspan=3)

chkBtnVar = tkinter.IntVar()
chkBtn = tkinter.Checkbutton(root, text="Model Tree",variable = chkBtnVar)
chkBtn.grid(row=3,column=0,columnspan=2)

data = regression_tree.load_data("sine.txt")
data = np.mat(data)
m,n = data.shape
reDraw.rawData = np.ones((m,n+1))
reDraw.rawData[:,1:] = data
reDraw.testData = np.arange(min(reDraw.rawData[:,1]),max(reDraw.rawData[:,1]),0.01)
reDraw(1.0,10)


root.mainloop()