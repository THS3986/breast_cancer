#=========================Data loading and cleaning=========================================
library(readr)
library(dplyr)
library(ggplot2)
library(corrplot)
library(gridExtra)
library(pROC)
library(MASS)
library(caTools)
library(caret)
library(caretEnsemble)


data <- read.csv("C:/Users/HP/Desktop/BAMS3216 Project/data.csv")

str(data)

data$diagnosis <- as.factor(data$diagnosis)
data[,33] <- NULL

summary(data)

prop.table(table(data$diagnosis))

corr_mat <- cor(data[,3:ncol(data)])
corrplot(corr_mat, order = "hclust", tl.cex = 1, addrect = 8)


#==============================Modelling==========================================
# separate data to training data(0.7) and testing data(0.3)
set.seed(7)
data_index <- createDataPartition(data$diagnosis, p=0.7, list = FALSE)
train <- data[data_index, -33]
test <- data[-data_index, -33]

# write the .csv file
write.csv(train, 'train.csv')
write.csv(test, 'test.csv')

# delete the id column
train_data <- train[,-1]
test_data <- test[,-1]

# view the percentage for the both dataset
cbind(freq=table(train_data$diagnosis), percentage=prop.table(table(train_data$diagnosis)))
cbind(freq=table(test_data$diagnosis), percentage=prop.table(table(test_data$diagnosis)))


#======================Applying machine learning models==========================================
fitControl <- trainControl(method="cv",number = 10)
metric <- "Accuracy"


#------Random Forest------
set.seed(7)
model_rf <- train(diagnosis~.,
                  train_data,
                  method="ranger",
                  metric=metric,
                  preProcess = c('center', 'scale'),
                  trControl=fitControl)

model_rf$resample
pred_rf <- predict(model_rf, test_data)
cm_rf <- confusionMatrix(pred_rf, test_data$diagnosis)
cm_rf


#---------------GLMNET----------------------------
set.seed(7)
model_glmnet <- train(diagnosis~.,
                  train_data,
                  method="glmnet",
                  metric=metric,
                  preProcess = c('center', 'scale'),
                  trControl=fitControl,
                  na.action=na.omit)

model_glmnet$resample
pred_glmnet <- predict(model_glmnet, test_data)
cm_glmnet <- confusionMatrix(pred_glmnet, test_data$diagnosis)
cm_glmnet


#------------GLM----------------
set.seed(7)
model_glm <- train(diagnosis~.,
                      train_data,
                      method="glm",
                      metric=metric,
                      preProcess = c('center', 'scale'),
                      trControl=fitControl,
                      na.action=na.omit)

model_glm$resample
pred_glm <- predict(model_glm, test_data)
cm_glm <- confusionMatrix(pred_glm, test_data$diagnosis)
cm_glm


#------------DTREE----------------
set.seed(7)
model_dtree <- train(diagnosis~.,
                   train_data,
                   method="ctree",
                   metric=metric,
                   preProcess = c('center', 'scale'),
                   trControl=fitControl)

model_dtree$resample
pred_dtree <- predict(model_dtree, test_data)
cm_dtree <- confusionMatrix(pred_dtree, test_data$diagnosis)
cm_dtree


#------KNN------
set.seed(7)
model_knn <- train(diagnosis~.,
                   train_data,
                   method="knn",
                   metric=metric,
                   preProcess = c('center', 'scale'),
                   trControl=fitControl)

model_knn$resample
pred_knn <- predict(model_knn, test_data)
cm_knn <- confusionMatrix(pred_knn, test_data$diagnosis)
cm_knn
model_knn

#------Neural Networks (NNET)------
set.seed(7)
model_nnet <- train(diagnosis~.,
                    train_data,
                    method="nnet",
                    metric=metric,
                    preProcess=c('center', 'scale'),
                    trace=FALSE,
                    trControl=fitControl)

model_nnet$resample
pred_nnet <- predict(model_nnet, test_data)
cm_nnet <- confusionMatrix(pred_nnet, test_data$diagnosis)
cm_nnet


#------SVM with radial kernel------
set.seed(7)
model_svm <- train(diagnosis~.,
                   train_data,
                   method="svmRadial",
                   metric=metric,
                   preProcess=c('center', 'scale'),
                   trace=FALSE,
                   trControl=fitControl)

model_svm$resample
pred_svm <- predict(model_svm, test_data)
cm_svm <- confusionMatrix(pred_svm, test_data$diagnosis)
cm_svm


#------Naive Bayes------
set.seed(7)
model_nb <- train(diagnosis~.,
                  train_data,
                  method="nb",
                  metric=metric,
                  preProcess=c('center', 'scale'),
                  trace=FALSE,
                  trControl=fitControl)

model_nb$resample
pred_nb <- predict(model_nb, test_data)
cm_nb <- confusionMatrix(pred_nb, test_data$diagnosis)
cm_nb


#===============================Model result comparasion==============================================
set.seed(7)
model_list <- list(RF=model_rf, GLMNET=model_glmnet, GLM=model_glm,
                   DTREE=model_dtree, KNN = model_knn, 
                   NNET=model_nnet, SVM=model_svm, NB=model_nb)
# collect resamples
resamples <- resamples(model_list)

# summarize the distributions
summary(resamples)
dotplot(resamples)


#===============================Tuning model==============================================
fitControl <- trainControl(method="cv",number = 10)
metric <- "Accuracy"
set.seed(7)
grid <- expand.grid(alpha = 0:1, lambda = seq(0.0001, 1, length = 20))
model_glmnet <- train(diagnosis~.,
                      train_data,
                      method="glmnet",
                      metric=metric,
                      tuneGrid=grid,
                      trControl=fitControl,
                      na.action=na.omit)
print(model_glmnet)
plot(model_glmnet)


#==================================Correlation between models=========================================
model_cor <- modelCor(resamples)


#===========================================Plot==================================================
corrplot(model_cor)


#==========================================Data==================================================
model_cor


#=========================================Comparasion===========================================
bwplot(resamples, metric=metric)

cm_list <- list(RF=cm_rf, GLMNET=cm_glmnet, GLM=cm_glm,
                   DTREE=cm_dtree, KNN = cm_knn, 
                   NNET=cm_nnet, SVM=cm_svm, NB=cm_nb)

cm_list_results <- sapply(cm_list, function(x) x$byClass)
cm_list_results

set.seed(7)
cm_results_max <- apply(cm_list_results, 1, which.max)

output_report <- data.frame(metric=names(cm_results_max), 
                            best_model=colnames(cm_list_results)[cm_results_max],
                            value=mapply(function(x,y) {cm_list_results[x,y]}, 
                                         names(cm_results_max), 
                                         cm_results_max))
rownames(output_report) <- NULL
output_report


# compare original data and predicted data
result_nnet<- data.frame('Id' = test$id,
                         'ORIGINAL_data' = test$diagnosis,
                         'PREDICT_data' = pred_nnet)
write.csv(result_nnet, 'resultNNET.csv')









