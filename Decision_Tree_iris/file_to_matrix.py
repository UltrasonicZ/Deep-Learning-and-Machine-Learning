import numpy as np
import csv


def file2matrix(filename):
    returnMat = np.zeros((150, 4))
    classLabelVector = []
    index = 0
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            data_row = np.array(row)
            returnMat[index, ] = data_row[0:4]
            classLabelVector.append(int(data_row[-1]))
            index += 1
    return returnMat.tolist(), classLabelVector