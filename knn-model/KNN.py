from numpy import *
import operator

def knn_classify(inX,dataSet,lables,k):
    dataSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSet.shape[0],1))-dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis = 1)
    distances = sqDistance**0.5
    sortDistance = distances.argsort()
    classcount = {}
    for i in range(k):
        voteIlabel = labels[sortDistance[i]]
        classcount[voteIlabel] = classcount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classcount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount,sortedClassCount[0][0] 

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

    