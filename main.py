# ********************************************
# Author: Nora Coler
# Date: April 10, 2015
#
#
# ********************************************
import numpy
import argparse
import os
from sklearn import metrics
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import timeit
from math import sqrt

size = 444075

def writeFileArray(dictionary, fileName):
    # output feature importance for graphs
    outfile = file(fileName, "w")
    keys = []
    header = ""
    for key in dictionary[0].keys():
        keys.append(key)
        header += '"'+str(key) + '",'
    outfile.write(header + '\n');

    for i in range(len(dictionary)):
        outLine = ""
        for key in keys:
            outLine += str(dictionary[i][key]) + ","
        outLine += "\n"
        outfile.write(outLine)

    outfile.close();



# read a dat file back into python. 
def read_dat_file(myfile, size):
    array = np.fromfile(myfile, dtype=np.float64, count=-1, sep="")
    array = np.reshape(array,(size,-1))
    return array

def readInputFile(inFileName):
    row = []
    column = []
    data = []
    inFile = open(inFileName)
    lines = inFile.readlines()
    for i in range(len(lines)):
        numbers = lines[i].split()
        row.append(int(numbers[0]))
        column.append(int(numbers[1]))
        data.append(int(numbers[2]))

    return [row, column, data]

def createMatrix(row, column, data):
    matrix = sparse.coo_matrix((data, (row, column)), shape=(size, size), dtype=numpy.float64)
    return matrix

def createSymetricMatrix(row, column, data):
    dataDict = {}
    for i in range(len(row)):
        key = "%s-%s"%(row[i], column[i])
        dataDict[key] = data[i]

    newData = []
    newRow = []
    newCol = []
    alreadySeen = {}
    for i in range(len(row)):
        key = "%s-%s"%(row[i], column[i])
        if(key in alreadySeen):
            continue
        total = data[i]
        keySym = "%s-%s"%(column[i], row[i])
        if(keySym in dataDict):
            total += dataDict[keySym]
        newRow.append(row[i])
        newCol.append(column[i])
        newData.append(total)

        newRow.append(column[i])
        newCol.append(row[i])
        newData.append(total)

        alreadySeen[key] = 1
        alreadySeen[keySym] = 1

    return createMatrix(newRow, newCol, newData)

def predict(clf, X, row, column):
    start = timeit.default_timer()
    clf.fit(X)
    print "   fitting done.";
    stop = timeit.default_timer()
    print "   runtime: " + str(stop - start)

    if args.SVD:
        matrixY = clf.components_ 
        probsY = []
        for i in range(len(row)):
            probsY.append(matrixY[row[i]][column[i]])
        return probsY

    if args.Factorization:
        return clf.coef_
    if args.Mixature:
        return clf.estimator_.coef_[0]


if __name__ == '__main__':
    # argument parsing.
    parser = argparse.ArgumentParser(description='Predict Bitcoin.')
    parser.add_argument("-S", "--SVD", action="store_true", help="run SVD")
    parser.add_argument("-F", "--Factorization", action="store_true", help="run non-negative factorization model")
    parser.add_argument("-M", "--Mixature", action="store_true", help="run mixature model")

    # OTHER INPUT VARIABLES
    outname = "" # assigned later
    inX = "txTripletsCounts.txt"
    inY = "testTriplets.txt"

    args = parser.parse_args()
    print args;

    [row, column, data] = readInputFile(inX)
    [rowY, columnY, dataY] = readInputFile(inY)
    print "row max: %i" % max(row)
    print "col max: %i" % max(column)

    X = createMatrix(row, column, data) # matrix of the data
    symX = createSymetricMatrix(row, column, data)

    print "X    shape: %s    nonZero entries: %i" % (str(X.shape), X.nnz)
    print "Xsym shape: %s    nonZero entries: %i" % (str(symX.shape), symX.nnz)
    print "\n"

    # CLASSIFY!
    if args.SVD:
        print "SVD"
        outname = "SVD"
        clf = TruncatedSVD()

    if args.Factorization:
        print "Factorization"
        outname = "Factorization"
        clf = linear_model.Lasso(alpha=alphaIn)

    if args.Mixature:
        print "Mixature"
        outname = "Mixature"
        clf = linear_model.RANSACRegressor(linear_model.LinearRegression())


    if args.SVD or args.Factorization or args.Mixature:
        probsY = predict(clf, X, rowY, columnY)

        probsDict = []
        for i in range(len(probsY)):
            probsDict.append({"TestValue":dataY[i], "Probability":probsY})

        fpr, tpr, thresholds = metrics.roc_curve(probsY, dataY, pos_label=1)
        rocDict = []
        for i in range(len(fpr)):
            rocDict.append({"fpr":fpr[i], "tpr":tpr[i]})

        writeFileArray(rocDict, "%s_roc.csv" % outname)
        writeFileArray(probsDict, "%s_probs.csv" % outname)




