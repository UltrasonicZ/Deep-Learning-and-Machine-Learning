import operator
import numpy as np

def classify(inX, dataSet, labels, k):
    distances = np.sum((inX - dataSet) ** 2, axis=1) ** 0.5
    sortedDisIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]