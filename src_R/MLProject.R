#rm(list=ls())

#Import Data
setwd("C:/Users/anand/Documents/MLProject")
raw<-na.omit(read.csv("train.csv",sep = ",",header = T,stringsAsFactors = F))

#Load the package
#install.packages("caret")
#install.packages("lattice")
#install.packages("ggplot2")
#install.packages("'nnet'")
library(ggplot2)
library(lattice)
library(caret)
library(e1071)
library(nnet)

#Preprocess Data
raw_def = na.omit(raw[,10:42])
raw_off = na.omit(raw[,c(9,11:42)])
raw_def<-raw_def[raw_def$defensive_work_rate %in% c("high","low","medium"),]
raw_off<-raw_off[raw_off$attacking_work_rate %in% c("high","low","medium"),]

#Creating Training Data
train_off = raw_off[1:120000,]
train_def = raw_def[1:120000,]

#Creating Test Data
test_off = raw_off[120001:162209,]
test_def = raw_def[120001:162209,]


#Creating Training Set and Labels for offensive data
off_train_X = train_off[,2:33]
off_labels = train_off[,1]

#Creating Training Set and Labels for defensive data
def_train_X = train_def[,2:33]
def_labels = train_def[,1]

#Creating Test Sets
off_test = test_off[,2:33]
def_test = test_def[,2:33]

###################################################################################################

#Training the model on Naive Bayes Classifier on Offensive Data
off_model<-train(off_train_X, off_labels, method="nb",verbose=FALSE,trControl=trainControl(method = "cv",number=3))

off_pred_table<-predict(off_model,off_test)

acc_mat<-data.frame(test_off$attacking_work_rate,off_pred_table)
names(acc_mat)<-c("actual","obs")
str(acc_mat)
off_nb_accuracy <- (nrow(acc_mat[acc_mat$actual == acc_mat$obs,]))/ nrow(acc_mat)


###################################################################################################

#Training the model on Naive Bayes Classifier on Defensive Data
def_model<-train(def_train_X, def_labels, method="nb",verbose=FALSE,trControl=trainControl(method = "cv",number=3))

def_pred_table<-predict(def_model,def_test)

acc_mat<-data.frame(test_def$defensive_work_rate,def_pred_table)
names(acc_mat)<-c("actual","obs")
str(acc_mat)
def_nb_accuracy <- (nrow(acc_mat[acc_mat$actual == acc_mat$obs,]))/ nrow(acc_mat)

###################################################################################################

#Training the model on Neural Network on Defensive Data
numFolds <- trainControl(method = 'cv', number = 3, classProbs = TRUE, verboseIter = TRUE,  preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))

def_nn_model <- train(def_train_X, def_labels, method = 'nnet', preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=expand.grid(size=c(20), decay=c(0.1)))


def_pred_table<-predict(def_nn_model,def_test)

acc_mat<-data.frame(test_def$defensive_work_rate,def_pred_table)
names(acc_mat)<-c("actual","obs")
str(acc_mat)
def_nn_accuracy <- (nrow(acc_mat[acc_mat$actual == acc_mat$obs,]))/ nrow(acc_mat)

###################################################################################################

#Training the model on Neural Network on Offensive Data
numFolds <- trainControl(method = 'cv', number = 3, classProbs = TRUE, verboseIter = TRUE,  preProcOptions = list(thresh = 0.75, ICAcomp = 3, k = 5))

off_nn_model <- train(off_train_X, off_labels, method = 'nnet', preProcess = c('center', 'scale'), trControl = numFolds, tuneGrid=expand.grid(size=c(20), decay=c(0.1)))

off_pred_table<-predict(off_nn_model,off_test)

acc_mat<-data.frame(test_off$attacking_work_rate,off_pred_table)
names(acc_mat)<-c("actual","obs")
str(acc_mat)
off_nn_accuracy <- (nrow(acc_mat[acc_mat$actual == acc_mat$obs,]))/ nrow(acc_mat)

##################################################################################################

#Training the model on Random Forest Classifier on Offensive Data
off_rf_model<-train(off_train_X, off_labels, method="rf",verbose=FALSE,trControl=trainControl(method = "cv",number=3))

off_pred_table<-predict(off_model,off_test)

acc_mat<-data.frame(test_off$attacking_work_rate,off_pred_table)
names(acc_mat)<-c("actual","obs")
str(acc_mat)
off_rf_accuracy <- (nrow(acc_mat[acc_mat$actual == acc_mat$obs,]))/ nrow(acc_mat)


###################################################################################################

#Training the model on Random Forest Classifier on Defensive Data
def_rf_model<-train(def_train_X, def_labels, method="rf",verbose=FALSE,trControl=trainControl(method = "cv",number=3))

def_pred_table<-predict(def_model,def_test)

acc_mat<-data.frame(test_def$defensive_work_rate,def_pred_table)
names(acc_mat)<-c("actual","obs")
str(acc_mat)
def_rf_accuracy <- (nrow(acc_mat[acc_mat$actual == acc_mat$obs,]))/ nrow(acc_mat)