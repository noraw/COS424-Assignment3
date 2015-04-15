# ********************************************
# Author: Nora Coler
# Date: April 10, 2015
#
#
# ********************************************
import numpy as np
import argparse
import os
from sklearn import metrics
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn.pipeline import Pipeline
from scipy import sparse
import timeit
from math import sqrt

# read a dat file back into python. 
def read_dat_file(myfile, size):
    array = np.fromfile(myfile, dtype=np.float64, count=-1, sep="")
    array = np.reshape(array,(size,-1))
    return array

def readInputFile(inFileName):
    row = []
    column = []
    data = []
    inFile = open(inFileNmae)
    lines = inFile.readlines()
    for i in range(len(lines)):
        numbers = lines[i].split()
        row.append(numbers[0])
        column.append(numbers[1])
        data.append(numbers[2])
    return [row, column, data]

def createMatrix([row, column, data]):
    matrix = sparse.coo_matrix(data, (row, column), dtype=numpy.float64)
    return matrix

def createSymetricMatrix(matrix, row, column):
    symMatrix = sparse.coo_matrix(matrix.shape, dtype=numpy.float64)
    for i in range(len(row)):
        total = matrix[row[i], column[i]] + matrix[column[i], row[i]]
        symMatrix[row[i], column[i]] = total
        symMatrix[column[i], row[i]] = total
    return symMatrix

def predict(clf, X, y, X_test, y_test):
    start = timeit.default_timer()
    clf.fit(X, y)
    print "   predictions done.";
    predicted = clf.predict(X_test)
    #probs_train = clf.predict_proba(X)    
    #probs_test = clf.predict_proba(X_test0)
    print "   R2:      " + str(metrics.r2_score(y_test, predicted)) # classifier accuracy
    print "   RSME:    " + str(sqrt(metrics.mean_squared_error(y_test, predicted))) # classifier accuracy
    #print(metrics.classification_report(y_test, predicted))
    stop = timeit.default_timer()
    print "   runtime: " + str(stop - start)

    if args.LinearRegression:
        return clf.coef_[0]
    if args.Lasso:
        return clf.coef_
    if args.RANSAC:
        return clf.estimator_.coef_[0]





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
X = createMatrix([row, column, data]) # matrix of the frequency data
symX = createSymetricMatrix(X, row, column)

print "X shape" + str(X0.shape)
print "X test shape " + str(X_test0.shape)
print "Y shape" + str(y.shape)
print "Y test shape " + str(y_test.shape)


# CLASSIFY!
if args.LinearRegression:
    print "Linear Regression"
    outname = "linearRegression"
    clf = linear_model.LinearRegression()
    #LinearRegression(copy_X=True, fit_intercept=True, normalize=False)

if args.Lasso:
    alphaIn=5.0
    print "Lasso: " + str(alphaIn)
    outname = "lasso"+str(alphaIn)
    clf = linear_model.Lasso(alpha=alphaIn)

if args.RANSAC:
    print "RANSAC"
    outname = "ransac"
    clf = linear_model.RANSACRegressor(linear_model.LinearRegression())


if args.LinearRegression or args.Lasso or args.RANSAC:
    print "Nan = 0"
    feature_importance0 = predict(clf, X0, y, X_test0, y_test)

    print "Nan = Average"
    feature_importanceA = predict(clf, XA, y, X_testA, y_test)

    print "Nan = Random"
    feature_importanceR = predict(clf, XR, y, X_testR, y_test)

    # output feature importance for graphs
    outfile = file("feature_importance_" + outname + ".csv", "w")
    outfile.write('"Feature ID","0 Nans","Average Nans","Random Nans"\n');
    print len(feature_importance0)
    for i in range (len(feature_importance0)):
        outLine = str(i) + ","
        outLine += str(feature_importance0[i]) + ","
        outLine += str(feature_importanceA[i]) + ","
        outLine += str(feature_importanceR[i])
        outLine += "\n"
        outfile.write(outLine)
    outfile.close();





