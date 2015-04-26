library(ggplot2)
library(Matrix)

setwd('~/GitHub//COS424-Assignment3')

train <- read.table('txTripletsCounts.txt',nrows=3348026)
colnames(train) <- c('giver','receiver','N')


##############################
# Plotting degree distribution
# (along with log deg, and log log deg)
unique_addr <- sort(unique(c(train$giver,train$receiver)))
num_addr <- length(unique_addr)

d <- spMatrix(num_addr, num_addr,
              i=train$giver + 1,
              j=train$receiver + 1,
              x=train$N)

degree <- colSums(d)
ldegree <- log10(degree)
lldegree <- log10(ldegree)

qplot(degree, geom='histogram')
qplot(ldegree, geom='histogram')
qplot(lldegree, geom='histogram')

##############################
# Plotting ROC curves

#Copied from http://www.rrandomness.com/page/2/
rocdata <- function(grp, pred){
  # Produces x and y co-ordinates for ROC curve plot
  # Arguments: grp - labels classifying subject status
  #            pred - values of each observation
  # Output: List with 2 components:
  #         roc = data.frame with x and y co-ordinates of plot
  #         stats = data.frame containing: area under ROC curve, p value, upper and lower 95% confidence interval
  
  grp <- as.factor(grp)
  if (length(pred) != length(grp)) {
    stop("The number of classifiers must match the number of data points")
  } 
  
  if (length(levels(grp)) != 2) {
    stop("There must only be 2 values for the classifier")
  }
  
  cut <- unique(pred)
  tp <- sapply(cut, function(x) length(which(pred > x & grp == levels(grp)[2])))
  fn <- sapply(cut, function(x) length(which(pred < x & grp == levels(grp)[2])))
  fp <- sapply(cut, function(x) length(which(pred > x & grp == levels(grp)[1])))
  tn <- sapply(cut, function(x) length(which(pred < x & grp == levels(grp)[1])))
  tpr <- tp / (tp + fn)
  fpr <- fp / (fp + tn)
  roc = data.frame(x = fpr, y = tpr)
  roc <- roc[order(roc$x, roc$y),]
  
  i <- 2:nrow(roc)
  auc <- (roc$x[i] - roc$x[i - 1]) %*% (roc$y[i] + roc$y[i - 1])/2
  
  pos <- pred[grp == levels(grp)[2]]
  neg <- pred[grp == levels(grp)[1]]
  q1 <- auc/(2-auc)
  q2 <- (2*auc^2)/(1+auc)
  se.auc <- sqrt(((auc * (1 - auc)) + ((length(pos) -1)*(q1 - auc^2)) + ((length(neg) -1)*(q2 - auc^2)))/(length(pos)*length(neg)))
  ci.upper <- auc + (se.auc * 0.96)
  ci.lower <- auc - (se.auc * 0.96)
  
  se.auc.null <- sqrt((1 + length(pos) + length(neg))/(12*length(pos)*length(neg)))
  z <- (auc - 0.5)/se.auc.null
  p <- 2*pnorm(-abs(z))
  
  stats <- data.frame (auc = auc,
                       p.value = p,
                       ci.upper = ci.upper,
                       ci.lower = ci.lower
  )
  
  return (list(roc = roc, stats = stats))
}

#SVD
svd_probs <- read.csv('SVD_probs.csv')
svd_roc <- rocdata(svd_probs$TestValue, svd_probs$Probability)
ggplot(svd_roc$roc, aes(x=x,y=y)) +
  geom_line(colour='blue') +
  geom_abline(intercept=0, slope=1) +
  theme_bw()
