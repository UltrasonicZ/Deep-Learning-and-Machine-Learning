from classify_KNN import classify
from autoNorm_matrix import autoNorm
# from file_to_matrix import file2matrix
from file_to_matrix import file2matrix
import numpy as np


def flowerClassTest(filename):
    hoRatio = 0.1
    flowerDataMat, flowerLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(flowerDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    normMat_test_1 = np.append(normMat[0:10], normMat[50:60], axis=0)
    normMat_test = np.append(normMat_test_1, normMat[100:110], axis=0)

    normMat_sample_1 = np.append(normMat[10:50], normMat[60:100], axis=0)
    normMat_sample = np.append(normMat_sample_1, normMat[110:150], axis=0)

    flowerLabels_test_1 = np.append(flowerLabels[0:10], flowerLabels[50:60], axis=0)
    flowerLabels_test = np.append(flowerLabels_test_1, flowerLabels[100:110], axis=0)

    flowerLabels_sample_1 = np.append(flowerLabels[10:50], flowerLabels[60:100], axis=0)
    flowerLabels_sample = np.append(flowerLabels_sample_1, flowerLabels[110:150], axis=0)

    for i in range(30):
        classifierResult = \
            classify(normMat_test[i, :],
                      normMat_sample,
                      flowerLabels_sample, 10)
        print("The classifier came back with: %d, "
              "the real answer is:%d" % (classifierResult, flowerLabels_test[i]))
        if classifierResult != flowerLabels_test[i]:
            errorCount += 1.0
    print("The total error rate is: %f" % (errorCount / float(numTestVecs)))


if __name__ == '__main__':
    filename = 'iris.csv'
    flowerClassTest(filename)