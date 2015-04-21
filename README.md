# COS424-Assignment3

With and without directionality(Nora)

1. SVD with numbers of interactions(Nora)
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html

try with n_components = ~40,000
save out explained_variance_ratio_ : array, [n_components] and plot it to see best ones
from those best one use that number as n_components and run again
read out components small matrix: transpose * components = big matrix
for each data point do row* column to get value
truncate between 0 and 1

2. Poisson-gamma matrix factorization model(non-negative factorization model)(Nora)
    http://scikit-learn.org/stable/auto_examples/applications/topics_extraction_with_nmf.html
3. Mixture model (Nick)
    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.mixture


Outputs:
roc file(csv): fpr. tpr
probs(csv): test value, probability

Create pretty graphs(Nick)

Results:
violin plots
ROC curves

Nick's Current Agenda Items:
-Perform 2D plot of data clusters
-Derive # of cluster centers from plot
-Run GMM
-Derive probabilities
-Plot ROC curves, etc.
