#classification for credit data
### Install required packages.
library(e1071)
library(ROCR)
library(SDMTools)
library(rpart)
library(randomForest)

#processing data
data<-read.csv("creditdata.csv") 

#see missing data
#require(Amelia)
#missmap(data, main="Consumer Credit Data - Missings Map", col=c("yellow", "black"), legend=FALSE)

#processing purpose data
data$Purpose<-as.integer(data$Purpose)
for (purp in data$Purpose){
	data$Purpose[which(data$Purpose[purp]==NA)]<-10
}
data<-as.data.frame(data)

# Split data into training and test set.
nsplit<-nrow(data)/2
train<-data[1:nsplit,]
test<-data[(nsplit+1):nrow(data),]

#LOGISTIC REGRESSION#

#learning from training
logitmodel<-glm(Class~.,data=train,family=binomial("logit"))

#predicting the test data
logitmodel.probs<-predict(logitmodel, test, type = "response")
logitmodel.class<-predict(logitmodel, test)
logitmodel.labels<-test$Class

#analyzing result
logitmodel.confusion<-confusion.matrix(logitmodel.labels,logitmodel.class)
logitmodel.accuracy<-prop.correct(logitmodel.confusion)

#roc analysis for test data
logitmodel.prediction<-prediction(logitmodel.probs,logitmodel.labels)
logitmodel.performance<-performance(logitmodel.prediction,"tpr","fpr")
logitmodel.auc<-performance(logitmodel.prediction,"auc")@y.values[[1]]


#SUPPORT VECTOR MACHINE#

#learning from training
svmmodel<-svm(as.factor(Class)~., data=train, method="C-classification", kernel="radial",cross=5, probability=TRUE)

#predicting the test data
svmmodel.predict<-predict(svmmodel,subset(test,select=-Class),decision.values=TRUE)
svmmodel.probs<-attr(svmmodel.predict,"decision.values")
svmmodel.class<-svmmodel.predict[1:nsplit]
svmmodel.labels<-test$Class

#analyzing result
svmmodel.confusion<-confusion.matrix(svmmodel.labels,svmmodel.class)
svmmodel.accuracy<-prop.correct(svmmodel.confusion)

#roc analysis for test data
svmmodel.prediction<-prediction(svmmodel.probs,svmmodel.labels)
svmmodel.performance<-performance(svmmodel.prediction,"tpr","fpr")
svmmodel.auc<-performance(svmmodel.prediction,"auc")@y.values[[1]]


#DECISION-TREE LEARNING#
			
#learning from training
treemodel<-rpart(Class~.,data = train,method="class")

#predicting the test
treemodel.class<-predict(treemodel,test,type="class")
treemodel.probs<-predict(treemodel,test,type="prob")
treemodel.labels<-test$Class

#analyzing result
treemodel.confusion<-confusion.matrix(treemodel.labels,treemodel.class)
treemodel.accuracy<-prop.correct(treemodel.confusion)

#roc analysis for test data
treemodel.prediction<-prediction(treemodel.probs[,2],treemodel.labels)
treemodel.performance<-performance(treemodel.prediction,"tpr","fpr")
treemodel.auc<-performance(treemodel.prediction,"auc")@y.values[[1]]


#RANDOM FOREST#

#learning from training
rfmodel<-randomForest(Class~.,data = train,importance=TRUE)

#predicting the test
rfmodel.class<-predict(rfmodel,test,type="response")
rfmodel.probs<-predict(rfmodel,test)
rfmodel.labels<-test$Class

#analyzing result
rfmodel.confusion<-confusion.matrix(rfmodel.labels,rfmodel.class)
rfmodel.accuracy<-prop.correct(rfmodel.confusion)

#roc analysis for test data
rfmodel.prediction<-prediction(rfmodel.probs,rfmodel.labels)
rfmodel.performance<-performance(rfmodel.prediction,"tpr","fpr")
rfmodel.auc<-performance(rfmodel.prediction,"auc")@y.values[[1]]

#COMPARING ROC PLOT of 4 Model#
windows()
plot(logitmodel.performance,col="red",lwd=2)
plot(svmmodel.performance,add=TRUE,col="green",lwd=2)
plot(treemodel.performance,add=TRUE,col="blue",lwd=2)
plot(rfmodel.performance,add=TRUE,col="black",lwd=2)
title(main="ROC Curve of 4 models", font.main=4)
plot_range<-range(0,0.5,0.5,0.5,0.5)
legend(0.5, plot_range[2], c("logistic regression","svm","decision tree","random forest"), cex=0.8, 
   col=c("red","green","blue","black"), pch=21:22, lty=1:2)