#!/usr/bin/env python
# ********************************************
# Author: Nora Coler, Nicholas Turner
# Date: April 2015
#
#
# ********************************************
import numpy as np
import argparse, os, random
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.mixture import GMM, DPGMM
from sklearn.preprocessing import scale
from sklearn.manifold import Isomap
from scipy import sparse
from scipy.sparse.linalg import svds
import timeit
from math import sqrt
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib

# ********************************************
#Constants 
inX = "txTripletsCounts.txt"
inY = "testTriplets.txt"

size = 444075
# ********************************************
# ********************************************
# Utility Functions

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

def element_inverse(D):
    '''Take the elementwise inverse of a sparse matrix'''

    [r, c, d] = sparse.find(D)
    dinv = 1 / d
    return sparse.csr_matrix((dinv, (r, c)), shape=D.shape)

def make_graph_laplacian(W):
    '''Creates the (normalized) graph laplacian of the weight matrix W'''

    D = sparse.diags(W.sum(0),[0], shape=W.shape)
    Dinv = element_inverse(D) # this works since it's diagonal

    I = sparse.eye(D.shape[0],D.shape[1])

    L = I - Dinv * W

    return L

def data_log_likelihood(model, data):
    '''Determines the log likelihood over the passed dataset'''
    return sum(model.score(data))

def isomap_data(symmetric=True, num_data_points=-1):
    '''Manifold learning function for visualization, currently doesn't work
    since it tries to form the dense matrix (which all manifold learning strats
    seem to do)'''

    print "Importing Data..."
    if symmetric:
        X = import_file(inX)
    else:
        X = import_file(inX, symmetric=False)

    print "Performing initial SVD to 15 dimensions..."
    rX = TruncatedSVD(15).fit_transform(X)

    #Limiting projection to input number of data points
    if num_data_points > 0:
        rX = rX[:num_data_points,:]

    print "Initial data dimensions"
    print rX.shape

    rrX = Isomap().fit_transform(rX)

    return rrX

def plot_projected_data(d):
    '''plots a 2D projected version of the data'''
    plt.scatter(d[:,0],d[:,1])
    plt.show()

def plot_roc_curve(probs):

    [r, c, d] = readInputFile(inY)
    d = np.array(d)

    fpr, tpr, _ = roc_curve(d, probs)

    print "Area Under Curve: "
    print auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.show()

def augment_with_negative_data(positive_senders, positive_receivers, values, num_ids):

    positive_coords = zip(positive_senders, positive_receivers)

    num_positive_coords = len(positive_coords)

    negative_coords = []
    while len(negative_coords) < num_positive_coords:
        sender = random.randint(0,num_ids-1)
        receiver = random.randint(0,num_ids-1)

        if (sender, receiver) not in positive_coords:
            negative_coords.append((sender, receiver))
            print len(negative_coords)

    negative_senders, negative_receivers = zip(*negative_coords)

    print negative_senders
    all_senders = np.hstack((positive_senders,negative_senders))
    all_receivers = np.hstack((positive_receivers,negative_receivers))
    all_values = np.hstack((values,[0 for elem in negative_senders]))

    return all_senders, all_receivers, all_values

def number_of_common_neighbors(Xsym, sender, receiver):
    '''Takes the symmetric version of the matrix, and
    returns the number of common neighbors for a given
    sender/receiver pair'''

    sender_neighbors = Xsym[sender,:].toarray()
    receiver_neighbors = Xsym[receiver,:].toarray()

    common_neighbors = np.multiply(
        sender_neighbors,
        receiver_neighbors)

    #binarizing
    common_neighbors[common_neighbors != 0] = 1

    return np.sum(common_neighbors)

# ********************************************
# ********************************************
# Training and Prediction Functions

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

def train_vanilla_GMM(num_components=2, num_trials=1, num_data_points=-1):

    print 'Importing Data...'
    X = import_file(inX, symmetric=False, normalize=False)

    print 'Performing svd to %d components...' % (num_components)
    u, s, vt = svds(X,num_components)
    rX = vt.transpose()

    if num_data_points > 0:
        rX = rX[:num_data_points,:]

    gmm = GMM(num_components, covariance_type='spherical',
        n_iter=1000, n_init=num_trials)

    print "Training GMM (%d initializations)..." % (num_trials)
    start = timeit.default_timer()
    gmm.fit(rX)
    end = timeit.default_timer()

    print "Training completed in %f seconds" % (end-start)
    print "Converged = "
    print gmm.converged_

    print "Forming predictions..."
    start = timeit.default_timer()
    probs = gmm.predict_proba(rX)
    end = timeit.default_timer()
    print "Prediction completed in %f seconds" % (end-start)

    return gmm, probs

def train_spectral_GMM(num_components=2, num_trials=1, num_data_points=-1):

    print "Importing Data..."
    X = import_file(inX, symmetric=False, normalize=False)

    print "Forming Graph Laplacian..."
    L = make_graph_laplacian(X)

    print "Performing svd to %d components..." % (num_components)
    u, s, vt = svds(L, num_components)
    components = u

    if num_data_points > 0:
        components = components[:num_data_points,:]

    gmm = GMM(num_components, covariance_type='spherical', 
        n_iter=1000, n_init=num_trials)

    print "Training GMM (%d initializations)..." % num_trials
    start = timeit.default_timer()
    gmm.fit(components)
    end = timeit.default_timer()

    print "Fitting completed in %f seconds" % (end-start)
    print "Converged = "
    print gmm.converged_

    print "Forming predictions..."
    start = timeit.default_timer()
    probs = gmm.predict_proba(components)
    end = timeit.default_timer()
    print "Prediction completed in %f seconds" % (end - start)

    return gmm, probs, components

def train_DPGMM(d, max_n_comp=100, max_n_iter=500):
    '''Imports Data, Trains a DPGMM, Generates predictions testing'''

    print "Training Model..."
    gmm = DPGMM(max_n_comp, n_iter=max_n_iter)

    start = timeit.default_timer()
    gmm.fit(d)
    end = timeit.default_timer()

    print "Training completed in %f seconds" % (end-start)

    print
    print "Converged: "
    print gmm.converged_
    print

    return gmm

def GMM_prediction_probs_max(probs, save=True):

    print "Importing Data..."
    [r, c, d] = readInputFile(inY)

    preds = []

    print "Forming predictions..."
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
        
    if save:
        print "Saving predictions..."
        writeFileArray(preds, outname)

    return np.array([elem['probability'] for elem in preds])

def GMM_prediction_probs_dot(probs, save=True):

    print "Importing Data..."
    [r, c, d] = readInputFile(inY)

    preds = []

    print "Forming predictions..."
    for i in range(len(r)):

        sender = r[i]
        receiver = c[i]

        all_comp_probs = np.multiply(
            probs[sender, :],
            probs[receiver, :]
            )

        final_prob = np.sum(all_comp_probs)
        preds.append({
            "ids": (sender, receiver),
            "probability": final_prob
            })

    if save:
        print "Saving predictions..."
        writeFileArray(preds, outname)

    return np.array([elem['probability'] for elem in preds])

def train_degree_logistic_regression():

    X = import_file(inX, symmetric=False, normalize=False)
    [r, c, d] = readInputFile

    sender_degree = X.sum(1)
    receiver_degree = X.sum(0)


# ********************************************
# ********************************************
# Command-Line Functionality

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


