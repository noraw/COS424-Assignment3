library(ggplot2)
library(Matrix)

#setwd('~/GitHub//COS424-Assignment3')
setwd('~/Development_Workspaces/COS424/COS424-Assignment3/')

train <- read.table('txTripletsCounts.txt',nrows=3348026)
colnames(train) <- c('giver','receiver','N')
train$indicator <- 1

##############################
# Plotting transaction number distribution

train$lN <- log10(train$N)
train$llN <- log10(train$lN)
ggplot(train, aes(x=llN)) + 
  geom_histogram() +
  labs(x='log_10',
       title='Testing')

##############################
# Plotting degree distribution
# (along with log deg, and log log deg)
unique_addr <- sort(unique(c(train$giver,train$receiver)))
num_addr <- length(unique_addr)

d <- spMatrix(num_addr, num_addr,
              i=train$giver + 1,
              j=train$receiver + 1,
              x=train$N)

binary_d <- spMatrix(num_addr, num_addr,
              i=train$giver + 1,
              j=train$receiver + 1,
              x=train$indicator)

#weighted in-degree
in_degree <- colSums(d)
lin_degree <- log10(in_degree)
llin_degree <- log10(lin_degree)

qplot(in_degree, geom='histogram')
qplot(lin_degree, geom='histogram')
qplot(llin_degree, geom='histogram')

#weighted out-degree
out_degree <- rowSums(d)
lout_degree <- log10(out_degree)

qplot(lout_degree, geom='histogram')

#binary in-degree
bin_degree <- colSums(binary_d)
lbin_degree <- log10(bin_degree)

qplot(lbin_degree, geom='histogram')

#binary out-degree
bout_degree <- rowSums(binary_d)
lbout_degree <- log10(bout_degree)

qplot(lbout_degree, geom='histogram')
##############################
# Plotting ROC curves

#Function for assembling together tpr/fpr data
# (see datafile_mat structure below)
assemble_roc_data <- function(file_mat){
  
  tprfprs <- data.frame(
    'fpr'=c(0,1),
    'tpr'=c(0,1),
    'NAME'=c('Chance','Chance')
  )
  
  for( i in 1:dim(file_mat)[1]){
    
    new_data <- read.csv(file_mat[i,1], header=FALSE)
    new_data$NAME <- file_mat[i,2]
    colnames(new_data) <- c('fpr','tpr','NAME')
    
    tprfprs <- rbind(tprfprs, new_data)
  }
  tprfprs
}

#Constructing datafile_mat
SVD_file = c('SVD_tprfpr.csv','SVD')
GMM_file = c('head_against_wall.csv','GMM')

datafile_mat <- rbind(SVD_file, GMM_file)

#Applying function
roc_data <- assemble_roc_data(datafile_mat)

#Plotting
ggplot(roc_data, aes(x=fpr, y=tpr, colour=NAME)) +
  geom_line()
