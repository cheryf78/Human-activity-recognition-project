---
title: 'Final project practical machine learning : Human activity recognition'
author: "Cheryf LALEYE"
date: "16 mai 2017"
output:
  
  html_document: 
    toc: yes
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)

```


###Goal of the project

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did.

###Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

###Data sources

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


###Loading packages 

```{r}

library(caret)
library(rattle)
library(corrplot)

```

###Loading and reading data

```{r}

raw.training <- read.table("./pml-training.csv",na.strings=c("NA",""), sep=",", header=T)
raw.testing <- read.table("./pml-testing.csv",na.strings=c("NA",""), sep=",", header=T)

dim(raw.training)
dim(raw.testing)

```

Let's tcheck the structure of our training data : 

```{r}
str(raw.training)

```



The training data set contains 159 variables/predictors and 19622 obsertvations. The test data contains 159 variables/predictors and 20 obsertvations. The outcome to predict here is the variable **classe**.

###Cleaning data 

Firstly, we will clean our data set by removing some meaningless variables. We know that the data which are useful are the data from accelerometers on the belt, forearm, arm and dumbell. So we'll keep only these measurements for our model prediction.



```{r}
  
  VarTokeep <- grep("belt|arm|dumbbell|classe",colnames(raw.training))
  cleaned.training.1 <- raw.training[,VarTokeep]
  
  #We'll do the same processing to the testing raw data.
  cleaned.testing.1 <- raw.testing[,VarTokeep]
  
```

Secondly, we'll remove variables/predictors for which we have more than 15% NAs of total observations.

```{r}
  
  ColToRemove <- which(colMeans(is.na(cleaned.training.1)) > 0.15)
  cleaned.training.2 <- cleaned.training.1[,-ColToRemove]
  
  #We'll do the same processing to the testing raw data.
  ColToRemove <- which(colMeans(is.na(cleaned.testing.1)) > 0.15)
  cleaned.testing.2 <- cleaned.testing.1[,-ColToRemove]
  
```

```{r}

dim(cleaned.training.2)
dim(cleaned.testing.2)

rm(cleaned.training.1)
rm(cleaned.testing.1)

```


```{r}
str(cleaned.training.2)

```

Set the testing data after cleaning it: 

```{r}
testing <- cleaned.testing.2
rm(cleaned.testing.2)

```

```{r}
str(testing)
```

```{r}
testing <- testing[,-53]

```

There's now 52 variables/predictors for both training and test data set.

###Data partition

In the aim to avoid overfitting and a biased out-of-sample error, we'll create a validation set from the cleaned training  data on which, we'll evaluate the model.

```{r}

set.seed(12345)
index <- createDataPartition(y=cleaned.training.2$classe, p=.7, list = F)
training <- cleaned.training.2[index,]
validation <- cleaned.training.2[-index,] 

```

```{r}
dim(training)
dim(validation)

```

```{r}
outcomeIndex <- which(colnames(training)=="classe")

```


###Data analysis 

* distribution of the outcome

```{r}
qplot(x=classe, data= training, fill=classe)

```

* Plotting the features distributions

```{r}
featurePlot(training[,-outcomeIndex],training$classe,plot="strip")

```

Since there's no real difference between the distributions of the features, we don't need to perform features scaling (normalization).

* Plotting the correlation between features

```{r}
corrplot(cor(training[,-outcomeIndex]),order="hclust", method="number",addCoef.col="grey")

```

Even if there's no high correlation in average, We see that there are some features which have a strong correlations between each other.

```{r}
StrongCorrelated <- findCorrelation(cor(training[,-outcomeIndex]), cutoff =.7)

```

```{r}
data.frame(names=colnames(training[,StrongCorrelated]),column.position=StrongCorrelated)

```

So maybe, we'll need to perform PCA processing in order to :

* reduce noise
* reduce number of predictors
* get high uncorrelated (orthogonal) predictors

###Preprocessing

```{r}

#perform PCA on the training and validation data set 
pcaObj.1 <- preProcess(training[,-outcomeIndex], method = "pca", thresh = .95, pcaComp = 27)
training.pca <- predict(pcaObj.1, newdata=training[,-outcomeIndex])
training.pca$classe <- training$classe 

pcaObj.2 <- preProcess(validation[,-outcomeIndex], method = "pca", thresh = .95, pcaComp = 27)
validation.pca <- predict(pcaObj.2, newdata=validation[,-outcomeIndex])
validation.pca$classe <- validation$classe

rm(pcaObj.1)
rm(pcaObj.2)

```

```{r}
dim(training.pca)
dim(validation.pca)

```


```{r}
str(training.pca)

```


###Training data and model selection

Now we'll perform machine learning algorithms on 3 models :

** The model with PCA processing, the model without **

* Random forest without the PCA processing performed on dataset

```{r}
#Fitting with random forest
ptm <- proc.time()
model.rf.1 <- train(classe~., data=training, method="rf", trControl = trainControl(method = "oob"), ntree = 250)
proc.time() - ptm

```

* Random forest with the PCA processing performed on dataset

```{r}
#Fitting with random forest
ptm <- proc.time()
model.rf.2 <- train(classe~., data=training.pca, method="rf", trControl = trainControl(method = "oob"), ntree = 250)
proc.time() - ptm

```


```{r}
model.rf.1$finalModel

```

```{r}
model.rf.2$finalModel

```

###Evaluation on validation set and out-of-sample error

```{r}
pred.1 <- predict(model.rf.1, newdata = validation)
confusionMatrix(pred.1, validation$classe)

```


```{r}
pred.2 <- predict(model.rf.2, newdata = validation.pca)
confusionMatrix(pred.2, validation.pca$classe)

```

We can see that the best accurracy is the one from the first model without PCA processing. So we will keep this model to make predictions on our test set. 

Let's make in-depth analysis of the choosen model:

```{r}
par(mfrow=c(1,2))
plot(model.rf.1$finalModel,cex=0.7, pch=16, main="Error Vs No. of trees")
varImpPlot(model.rf.1$finalModel, cex=0.7, pch=16, main="Var importance - r.forest")

```

##Prediction on the testing data 

```{r}
predictions <- predict(model.rf.1, newdata=testing)
predictions

  
```





