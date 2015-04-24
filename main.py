#!/usr/bin/env python
# ********************************************
# Author: Nora Coler, Nicholas Turner
# Date: April 2015
#
#
# ********************************************
import numpy as np
import argparse
import os
from sklearn import metrics
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import scale
from sklearn.mixture import DPGMM
from scipy import sparse
import timeit
from math import sqrt

# ********************************************
#Constants 
inX = "txTripletsCounts.txt"
inY = "testTriplets.txt"

size = 444075
# ********************************************


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
    matrix = sparse.csc_matrix((data, (row, column)), shape=(size, size), dtype=np.float64)
    return matrix

def createSymmetricMatrix(row, column, data):
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

def import_file(filename, symmetric = True, normalize = True):
    '''Reads in an input file, returns the sparse matrix'''

    # print "Importing..." #Debug output
    [r, c, d] = readInputFile(filename)

    # print "Creating Matrix..."
    if symmetric:
        res = createSymmetricMatrix(r, c, d)
    else:
        res = createMatrix(r, c, d)

    # print "Normalizing..."
    if normalize:
        res = res.tocsr() # apparently this makes things more efficient
        res = scale(res, with_mean=False, copy=False)

    return res

def predictSVD(clf, X, row, column):
    start = timeit.default_timer()
    v = clf.fit_transform(X)
    print "   fitting done.";
    stop = timeit.default_timer()
    print "   runtime: " + str(stop - start)
    u = clf.components_ 
    d = clf.explained_variance_

    matrixY = clf.components_ 
    probsY = []
    for i in range(len(row)):
        prob = np.sum(np.dot(u[:,column[i]], v[row[i],:]) * d)
        if(prob < 0): prob = 0
        if(prob > 1): prob = 1
        probsY.append(prob)
    return probsY

def testSVD(clf, X, row, column, outname):
    start = timeit.default_timer()
    clf.fit(X)
    print "   fitting done.";
    stop = timeit.default_timer()
    print "   runtime: " + str(stop - start)

    ratio = clf.explained_variance_ratio_
    ratioDict = []
    for i in range(len(ratio)):
        ratioDict.append({"Value":i, "Ratio":ratio[i]})
    writeFileArray(ratioDict, "%s_ratio.csv" % outname)

def predict_DPGMM(max_n_comp=100, max_n_iter=500):
    '''Imports Data, Trains a DPGMM, Generates predictions'''

    print "Importing Data..."
    X = import_file(inX)
    Y = import_file(inY)
    [r, c, d] = sparse.find(Y)

    print "Performing Dimension Reduction to 15 components"
    rX = TruncatedSVD(15).fit_transform(X)

    print "Training Model..."
    gmm = DPGMM(max_n_comp, n_iter=max_n_iter)

    start = timeit.default_timer()
    gmm.fit(rX)
    end = timeit.default_timer()

    print "Converged = "
    print gmm.converged_
    print "Ran for %f seconds" % (end-start)

    print "Generating Mixture Probabilities..."
    probs = gmm.predict_proba(X)

    preds = []

    for i in range(len(r)):

        sender = r[i]
        receiver = c[i]

        all_comp_probs = np.multiply(
            probs[sender, :],
            probs[receiver, :]
            )

        final_prob = np.max(all_comp_probs)
        preds.append({
            "ids":(sender, receiver),
            "probability": final_prob})
    writeFileArray(preds, "GPMM_preds.csv")

if __name__ == '__main__':
    # argument parsing.
    parser = argparse.ArgumentParser(description='Predict Bitcoin.')
    parser.add_argument("-S", "--SVD", action="store_true", help="run SVD")
    parser.add_argument("-F", "--Factorization", action="store_true", help="run non-negative factorization model")
    parser.add_argument("-M", "--Mixature", action="store_true", help="run mixature model")

    # OTHER INPUT VARIABLES
    outname = "" # assigned later
    probsY = None

    args = parser.parse_args()
    print args;

    # [row, column, data] = readInputFile(inX)
    [rowY, columnY, dataY] = readInputFile(inY)
    # print "row max: %i" % max(row)
    # print "col max: %i" % max(column)

    X = import_file(inX, symmetric=False) #createMatrix(row, column, data) # matrix of the data
    symX = import_file(inX) #createSymmetricMatrix(row, column, data)

    print "X    shape: %s    nonZero entries: %i" % (str(X.shape), X.nnz)
    print "Xsym shape: %s    nonZero entries: %i" % (str(symX.shape), symX.nnz)
    print "\n"

    # CLASSIFY!
    if args.SVD:
        print "SVD"
        outname = "SVD"
        clf = TruncatedSVD(n_components=90)
        #testSVD(clf, X, rowY, columnY, outname)
        probsY = predictSVD(clf, X, rowY, columnY)


    if args.Factorization:
        print "Factorization"
        outname = "Factorization"
        clf = linear_model.Lasso(alpha=alphaIn)

    if args.Mixature:
        print "Mixature"
        outname = "Mixature"
        clf = linear_model.RANSACRegressor(linear_model.LinearRegression())


    if ((args.SVD or args.Factorization or args.Mixature) and probsY != None):
        probsDict = []
        for i in range(len(probsY)):
            probsDict.append({"TestValue":dataY[i], "Probability":probsY[i]})

        fpr, tpr, thresholds = metrics.roc_curve(probsY, dataY, pos_label=1)
        rocDict = []
        for i in range(len(fpr)):
            rocDict.append({"fpr":fpr[i], "tpr":tpr[i]})

        writeFileArray(rocDict, "%s_roc.csv" % outname)
        writeFileArray(probsDict, "%s_probs.csv" % outname)




