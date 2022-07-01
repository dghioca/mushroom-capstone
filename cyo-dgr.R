#Choose-Your-Own Capstone Project - DGR, June 2022



##########################################################
# Download Secondary Mushroom Dataset - 
# https://archive.ics.uci.edu/ml/datasets/Secondary+Mushroom+Dataset
# Create edx set, validation set (final hold-out test set)
##########################################################


### Data download

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(mlbench)) install.packages("mlbench", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(data.table)
library(readr)
library(rpart.plot)
library(ggplot2)
library(mlbench)
library(Rborist)

temp <- tempfile()
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00615/MushroomDataset.zip"
download.file(url, temp)
mush <- read.table(unz(temp, "MushroomDataset/secondary_data.csv"), sep = ";", 
                   header = TRUE, stringsAsFactors = TRUE)

#I downloaded the categorical data as factor instead of character variables
#to help during modeling.

##########################################################

###	Data exploration and preparation

str(mush)
head(mush)
summary(mush, maxsum =20)
dim(mush)


# Based on the summary, there are 21 variables (columns), and 61,069 observation (rows).
# Also, it is apparent that this dataset has some variables with
# many NAs, so the first step was to remove 9 of the variables, each with 2,471 or more missing values.

mush <- mush %>%
  select(class, cap.diameter, cap.shape, cap.color, does.bruise.or.bleed,
                        gill.color, stem.height, stem.width, stem.color, has.ring, habitat, season)
str(mush)
summary(mush, maxsum =20)
dim(mush)

# The new dataset has now 12 variables, one of each (class) is the variable that we want to model.
# The other 11 variables are potential predictors.
# Next, I renamed the factor levels for the 9 categorical variables to make more informative graphs.
# Data set description: https://archive.ics.uci.edu/ml/datasets/Secondary+Mushroom+Dataset#

levels(mush$class) <- list("edible" = "e",  "poisonous" = "p")
levels(mush$cap.shape) <- list("bell"="b", "conical"="c", "convex"="x", "flat"="f",
                               "sunken"="s", "spherical"="p", "others"="o")
levels(mush$cap.color) <- list("brown"="n", "buff"="b", "gray"="g", "green"="r", 
                               "pink"="p", "purple"="u", "red"="e", "white"="w", 
                               "yellow"="y", "blue"="l", "orange"="o", "black"="k")
levels(mush$does.bruise.or.bleed) <- list("bruises-or-bleeding"="t", "no"="f")

levels(mush$gill.color) <- list("brown"="n", "buff"="b", "gray"="g", "green"="r", "pink"="p",
                                "purple"="u", "red"="e", "white"="w", "yellow"="y", "blue"="l",
                                "orange"="o", "black"="k", "none"="f")
levels(mush$stem.color) <- list("brown"="n", "buff"="b", "gray"="g", "green"="r", "pink"="p",
                                "purple"="u", "red"="e", "white"="w", "yellow"="y", "blue"="l",
                                "orange"="o", "black"="k", "none"="f")
levels(mush$has.ring) <- list("ring"="t", "none"="f")
levels(mush$habitat) <- list("grasses"="g", "leaves"="l", "meadows"="m", "paths"="p", "heaths"="h",
                             "urban"="u", "waste"="w", "woods"="d")
levels(mush$season) <- list("spring"="s", "summer"="u", "autumn"="a", "winter"="w")

str(mush)



### Data visualization
# Before splitting the dataset into training and testing sets,
# I performed some visualizations to get a better idea of the structure of the datasets.

# all graphs are bar graphs for each variable distribution.


par(mfrow=c(2,2)) # Set up a 2 x 2 plotting space
distributions <-function(index) 
{plot(mush[,index], main=names(mush[index]),pch=24, las=2, 
      xlab="",ylab="")}
lapply(1:12,FUN=distributions)




### Testing Models

###########
#In an initial attempt to build the model with 11 predictors, I observed very long computational times.
# Thus, I looked at methods of trimming down the number of predictors (i.e., feature selection).
# I found that the Recursive Feature Elimination (RFE) method, which uses a simple backwards selection algorithm,
# may work well for this type of dataset. However, it did not work well for my computer due to
# computational times (>40 min). As an alternative, I used variable importance
# after selecting the best model.

#control <- rfeControl(functions=rfFuncs, method="cv", number=10) # this is the RFE code I used;takes >40 min
#results <- rfe(mush[,2:12], mush[,1], sizes=c(5:7), rfeControl=control)
#print(results)  
#predictors(results) 
#plot(results, type=c("g", "o"))
##############


# Next, I created the training and test sets. I initially chose an 80 - 20 partition between
#training and test sets, but, again, computational times for each model I tested were long,
# so I decided to use a 50-50 partition and this worked better, without reducing much of the 
# accuracy of the models. I checked the results of the partition using str() and dim().

set.seed(1, sample.kind="Rounding")
index <- createDataPartition(y = mush$class, times = 1, p = 0.5, list = FALSE)
mush_train <- mush[-index,]
mush_test <- mush[index,]

str(mush_train)
dim(mush_train) 

str(mush_test)
dim(mush_test) 

# In the next step, I tested several models, including logistic, k nearest neighbor,
# classification tree, random forest, and ensemble.
# Methods such as LDA, QDA, and Loess of which we learned in the course, were not
# appropriate because the mushroom dataset includes non-numeric predictors.



##############
# Method 1  - Logistic regression 

train_glm <- train(mush_train[,2:12], mush_train[,1], method = "glm")
glm_preds <- predict(train_glm, mush_test[,2:12])
mean(glm_preds == mush_test[,1])  #accuracy

# I started by testing the simplest model, logistic regression with glm function in caret::train
# The accuracy was 0.731685. This was not great, but it is a starting point.


##############
# Method 2  - knn - slower

#method = "cv" refers to cross-validation, and number = xx tells the algorithm 
#to break the set into xx-folds for cross-validation; p is the partition proportion of the set;
#k is the number of neighbors (this parameter can be tuned)
#tuneGrid parameter tells which values the main parameter will take

control <- trainControl(method = "cv", number = 10, p =.9)
train_knn <- train(class ~ ., method = "knn", data = mush_train,
                   tuneGrid = data.frame(k = seq(1, 13, by = 2)),
                   trControl = control)
plot(train_knn)
train_knn$bestTune
knn_preds <- predict(train_knn, mush_test)
confusionMatrix(knn_preds, mush_test$class)$overall["Accuracy"]
# knn accuracy was 0.9901097.


##############
# Method 3 - Classification/decision tree - rpart
train_rpart <- train(class ~ ., 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0.0, 0.06, len = 25)),
                     data = mush_train)
train_rpart$bestTune
plot(train_rpart)
rpart_preds <- predict(train_rpart, mush_test)
confusionMatrix(rpart_preds, mush_test$class)$overall["Accuracy"]
# rpart accuracy was 0.9697396.


##############
# Method 4 - Random Forest - using the newer "Rborist" package (shorter computational time)

#method = "cv" refers to cross-validation, and number = k tells the algorithm 
#to break the set into k-folds for cross-validation
#nTree - the number of trees to grow
#nSamp - number of rows to sample, per tree
#minNode - parameter that can be tuned and it represents the minimum number of 
#distinct row references to split a node.
# predFixed is also a tuning parameter and is the number of trial predictors 
#for a split (same as mtry in the next model).


control <-trainControl(method = "cv", number = 10, p =0.8)
grid <- expand.grid(minNode = c(1,2,3,4,5), predFixed = c(1,3,5,7,9,15,25))
train_rf_1 <- train(class ~ ., 
                 method = "Rborist", nTree = 100, trControl = control,
                 tuneGrid = grid,
                 data = mush_train,
                 nSamp = 2000)
train_rf_1$bestTune

fit_rf_1 <- Rborist(mush_train[,2:12], mush_train[,1], 
                  nTree = 1000, 
                  minNode =train_rf_1$bestTune$minNode, 
                  predFixed = train_rf_1$bestTune$predFixed)
rf_1_preds <- predict(train_rf_1,  mush_test)
confusionMatrix(rf_1_preds, mush_test$class)$overall["Accuracy"]

#Random forest (Rborist) accuracy was 0.9835271

##############
# Method 5 - Random Forest  - using the caret::train package (longer computational time)

#mtry is a tuning parameter (same as predFixed in the previous model
#and represents the number of variables randomly sampled as candidates at each split. 
# It can be tuned to optimize the model.
#ntree is the number of trees to grow in the forest and it cannot be tuned in
#randomForest, it is just set fixed. However, the larger the more accurate the model. 
#It also means the larger ntree, the longer it takes to run the model.

train_rf_2 <- train(class ~ ., 
                    method = "rf", ntree = 100,
                    tuneGrid = data.frame(mtry =  seq(6, 12, 2)),
                    data = mush_train)
plot(train_rf_2)
rf_2_preds <- predict(train_rf_2, mush_test)
confusionMatrix(rf_2_preds, mush_test$class)$overall["Accuracy"]
# Random forest (train -rf) accuracy was 0.993829.



##############
# Method 6  - Ensemble

ensemble <- cbind(glm = glm_preds == "edible", 
                  knn = knn_preds == "edible", 
                  rpart = rpart_preds == "edible",
                  rf_1 = rf_1_preds == "edible", 
                  rf_2 = rf_2_preds == "edible")

ensemble_preds <- ifelse(rowMeans(ensemble) > 0.5, "edible", "poisonous")
mean(ensemble_preds == mush_test[,1]) #aacuracy
#Ensemble accuracy was 0.9912232


##############
#RESULTS

#Table with the model summary
  models <- c("Logistic regression", "K nearest neighbors", "Classification tree",
  "Random forest-Rborist", "Random forest-rf", "Ensemble")
  accuracy <- c(mean(glm_preds == mush_test[,1]),
                mean(knn_preds == mush_test[,1]),
                mean(rpart_preds == mush_test[,1]),
                mean(rf_1_preds == mush_test[,1]),
                mean(rf_2_preds == mush_test[,1]),
                mean(ensemble_preds == mush_test[,1]))
  data.frame(Model = models, Accuracy = accuracy)
  
#Three models have accuracy above 0.99. The best model is the Random Forest-rf method 
#model with an accuracy of 0.9935.
  
  
#######################  
# Exploring the final best model- variable importance and visualization
  
# Based on accuracy (0.9926314), the best performing model was the one based on rf_2.
# I looked at additional details for this model, such as sensitivity and specificity:
  confusionMatrix(rf_2_preds, mush_test$class)

# Sensitivity : 0.9907  ;  Specificity : 0.9942 and these are looking very good!
# I also looked at variable importance

#Next I looked at variable importance.
imp <- varImp(train_rf_2)
print(imp)
ggplot(imp)

#Most imp variables were: stem.width, stem.height,  cap.diameter, (above 66%), 
#and then does.bruise.or.bleed and stem color. Next I rerun the best rf_2 model 
#with these five most important variables and the accuracy was 0.9165548.
#Still fairly, if I was not going to eat the mushrooms :)

  train_rf_2 <- train(class ~ stem.width + stem.height + cap.diameter +  
                      does.bruise.or.bleed + stem.color,
                      method = "rf", ntree = 100,
                      tuneGrid = data.frame(mtry =  seq(6, 12, 2)),
                      data = mush_train)
  plot(train_rf_2)
  rf_2_preds <- predict(train_rf_2, mush_test)
  confusionMatrix(rf_2_preds, mush_test$class)$overall["Accuracy"]
  train_rf_2$finalModel

  
#Lastly, I made a few graphs using these five most important predictors and the 
#  outcome variable "class" and using the entire 'mush' dataset:
  
  mush %>% group_by(stem.color) %>% 
    ggplot(aes(stem.width, stem.height, col = does.bruise.or.bleed )) +
           geom_point() +facet_grid(stem.color ~ does.bruise.or.bleed)
  
#This first set of scatterplots suggests there may be a difference in stem width 
#between mushrooms that bruise or bleed and those that don't, with the latter being 
  #shorter and also not getting as tall with a stem width increase. There is however
  #a lot more variation in the stem height as stem width increases in the mushrooms 
  #that do not bruise or bleed, especially those with brown, white, or yellow stems. 
  #It seems that mushrooms with brown stems can get wide and tall and less likely to bruise.
                  
 
   mush %>% group_by(class) %>% 
    ggplot(aes(stem.height, cap.diameter, col = does.bruise.or.bleed )) +
    geom_point() +facet_grid( does.bruise.or.bleed ~ class)
  
#The second graph, a series of scatterplots, shows an interesting pattern for edible mushrooms that do not 
   #bruise or bleed as they seem to include some of the largest mushrooms in terms 
   #of height but also two separate groups, one of smaller cap diameter 
   #(up to about 30 mm) and one of larger cap sizes (between 40 and 60 mm). 
   #We can also see that mushrooms that do bruise or bleed tend to be smaller 
   #in height and cap diameter, regardless of edibility.
   
 
  mush %>% ggplot(aes(does.bruise.or.bleed, stem.width, fill = class)) + 
    geom_boxplot()
  #The third graph, a boxplot, compares edible and poisonous mushrooms by stem width
  # and whether they bruise/bleed or not. Mushrooms that do not bruise/bleed can have large stems,
  #although the median is actually lower for both edible and especially for poisonous mushrooms
  #compared to those that bruise/bleed.
  
  mush %>% ggplot(aes(stem.color, cap.diameter, fill = class)) + 
    geom_boxplot()
  
  #This last graph, also a boxplot, compares edible and poisonous mushrooms by stem 
  #color and cap diameter. We can see again that brown is a frequent stem color 
  #and that edible mushrooms can have larger cap diameter.Note also that poisonous
  #mushrooms don't have buff stem color, and if you find a mushroom with black stem, 
  #it is likely edible only if it has a large cap.