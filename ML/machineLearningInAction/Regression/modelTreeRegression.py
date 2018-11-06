#!/usr/bin/env python
# -*- coding: utf-8 -*-

from treeRegression import *

def linear_regression(dataset):
    ''' 获取标准线性回归系数
    '''
    dataset = np.matrix(dataset)
    # 分割数据并添加常数列
    X_ori, y = dataset[:, :-1], dataset[:, -1]
    X_ori, y = np.matrix(X_ori), np.matrix(y)
    m, n = X_ori.shape
    ones = np.ones(m)
#     X = np.matrix(np.ones((m, n+1)))
#     X[:, 1:] = X_ori
    X = np.insert(X_ori,0,ones,axis=1)

    # 回归系数
    w = (X.T*X).I*X.T*y
    return w, X, y

def fleaf(dataset):
    ''' 计算给定数据集的线性回归系数
    '''
    w, X, y = linear_regression(dataset)
    return w

def ferr(dataset):
    ''' 对给定数据集进行回归并计算误差
    '''
    w, X, y = linear_regression(dataset)
    y_prime = X*w
    return np.sum(np.power(y_prime-y,2))

# def get_nodes_edges(tree, root_node=None):
#     ''' 返回树中所有节点和边
#     '''
#     Node = namedtuple('Node', ['id', 'label'])
#     Edge = namedtuple('Edge', ['start', 'end'])
# 
#     nodes, edges = [], []
# 
#     if type(tree) is not dict:
#         return nodes, edges
# 
#     if root_node is None:
#         label = '{}: {}'.format(tree['feat_idx'], tree['feat_val'])
#         root_node = Node._make([uuid.uuid4(), label])
#         nodes.append(root_node)
# 
#     for sub_tree in (tree['left'], tree['right']):
#         if type(sub_tree) is dict:
#             node_label = '{}: {}'.format(sub_tree['feat_idx'], sub_tree['feat_val'])
#         else:
#             node_label = '{}'.format(np.array(sub_tree.T).tolist()[0])
#         sub_node = Node._make([uuid.uuid4(), node_label])
#         nodes.append(sub_node)
# 
#         edge = Edge._make([root_node, sub_node])
#         edges.append(edge)
# 
#         sub_nodes, sub_edges = get_nodes_edges(sub_tree, root_node=sub_node)
#         nodes.extend(sub_nodes)
#         edges.extend(sub_edges)
# 
#     return nodes, edges
# 
# def dotify(tree):
#     ''' 获取树的Graphviz Dot文件的内容
#     '''
#     content = 'digraph decision_tree {\n'
#     nodes, edges = get_nodes_edges(tree)
# 
#     for node in nodes:
#         content += '    "{}" [label="{}"];\n'.format(node.id, node.label)
# 
#     for edge in edges:
#         start, end = edge.start, edge.end
#         content += '    "{}" -> "{}";\n'.format(start.id, end.id)
#     content += '}'
# 
#     return content

def tree_predict(data, tree):
    if type(tree) is not dict:
        w = tree
        y = np.matrix(data)*w
        return y[0, 0]

    feat_idx, feat_val = tree['spFeat'], tree['spValue']
    if data[feat_idx+1] < feat_val:
        return tree_predict(data, tree['left'])
    else:
        return tree_predict(data, tree['right'])

# dataset = loadDataSet('exp2.txt')
# tree = createTree(dataset, fleaf, ferr, ops=(0.1,4))
# print(tree)

#     # 生成模型树dot文件
#     with open('exp2.dot', 'w') as f:
#         f.write(dotify(tree))

# dataset = np.array(dataset)
# 绘制散点图
# plt.scatter(dataset[:, 0], dataset[:, 1])

# 绘制回归曲线
# x = np.sort(dataset[:, 0])
# y = [tree_predict([1.0,i], tree) for i in x]
# plt.plot(x, y, c='r')
# plt.show()

def regTreeEval(model,inData):
    return float(model)

def modeTreeEval(model,inData):
    m = inData.shape[0]
    ones = np.ones(m)
    X = np.insert(inData,0,ones,axis=1)
    return  float(X*model)

def treeForeCast(tree, inData,modeEval=regTreeEval):
    if not isTree(tree) : return modeEval(tree,inData)
    if inData[tree['spFeat']]  > tree['spValue']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'],inData,modeEval)
        else:
            return modeEval(tree['left'],inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'],inData,modeEval)
        else:
            return modeEval(tree['right'],inData)
        
def creatForeCast(tree, testData,modeEval=regTreeEval):
    m = len(testData)
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = treeForeCast(tree, testData[i],modeEval)
    return yHat
   
trainData = loadDataSet('bikeSpeedVsIq_train.txt')
testData = loadDataSet('bikeSpeedVsIq_test.txt')
tree = createTree(trainData,ops=(1,20))
yhat = creatForeCast(tree,np.array(testData)[:,0])
print(np.corrcoef(yhat, np.array(testData)[:,1], rowvar=0)[0,1])
