# PREDICTION WRITEUP - HUMAN ACTIVITY RECOGNITION 
Norberto Ortigoza
-June 27, 2018

## 1. Executive Summary
This report deals with the prediction of how well an actvity is being done based on modelling the data collected from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.Two models were generated and defined the one with better accuracy of 0.9943 (Random Forest), which was tested on a data set with correct results.

## 2. Reference
The data was extracted from the website: [url](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har).
The reference for the report and data collection is: Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. Wearable Computing: Accelerometers’ Data Classification of Body Postures and Movements. Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.

## 3. Data
We will store the data on the “training”, and “testing” data frames. We will then later perform an exploratory analysis in order to verify if the variables from the data can be reduced so the processing can be optimized.
```
setwd("H:/Courses/Data Analysis Specialization/8. Machine Learning/Project")
library(dplyr)
training<-read.csv("./pml-training.csv")
testing<-read.csv("./pml-testing.csv")
```
## 4. Exploratory Analysis
Let’s print the 5 first columns of the training set.(For purposes of the report the output won’t be displayed as it may take long space).
```
head(training)
```
We can see there are different columns with important amount of NA or missing values that we will not considered for the data processing as these will add noise to the analysis. In addition, we found some variables related to time that won’t add value to the analysis as the outcome will depends on sensor measurements and not the time this has been applied. We consider that having user as variable is also not useful as we want to predict “classe” based on the data from different users, independently of who the measurements are coming from. At the end, the data with direct measurement from accelerometers are the one that should be considered for predictions.

## 5. Cleaning Data
In order to eliminate the columns with NA or missing values, we will run a code that finds the columns where more than 5% of the data are either NA or missing, obtaining a vector of column indexes (called var_vector) that will allow to eliminate later these variables from the dataset.
```
x=as.integer()
y=as.integer()
var_vector=c()
y=0
x=0
for (j in 1:ncol(training)){
      for(i in 1:nrow(training)){
            if (is.na(training[i,j])==TRUE||training[i,j]=="") {
              x=x+1
            }
      }  
      if (x/nrow(training)>0.05){
      y=y+1
      var_vector[y]<-j
      }
x=0
}
length(var_vector)

## [1] 100
```
The result show 100 indexes from columns with more than 5% NA or missing values that we will not consider.
We will now obtained a new set of training and validation set without the variables previously identified with high NA or missing values and the variables related to time and users. The dataframe so far will be reduced from 160 to 56 variables.
```
training_red<-training[,-var_vector]
training_red<-select(training_red,-contains("time"))
training_red<-select(training_red,-contains("user"))

testing_red<-testing[,-var_vector]
testing_red<-select(testing_red,-contains("time"))
testing_red<-select(testing_red,-contains("user"))
Let’s list the remaining variables for the dataset
print(colnames(training_red))

##  [1] "X"                    "new_window"           "num_window"          
##  [4] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [7] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
## [10] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [13] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [16] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [19] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [22] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [25] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [28] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [31] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [34] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [37] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [40] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [43] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [46] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [49] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [52] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [55] "magnet_forearm_z"     "classe"
```
We can see some other variables that do not seem to provide valuable information to the model, like: “X”, “new window”, “num window”. We will eliminate these as well, reducing the number of variables even more to a final 53.
training_red<-select(training_red,-new_window,-X,-num_window)
testing_red<-select(testing_red,-new_window,-X,-num_window)

## 6. Creating and testing models
We will use the “CV” method (Cross Validation) with 5 folds in order to perform the analysis to obtain the model to predict the testing data. We will evaluate 2 different models (rf and gbm) to check for better accuracy;we will then obtain the predicted values for the testing data. In order to have a faster processing we will do parallel processing for each model and using the trainControl feature from caret package to automatically evaluate the different folds.

### 6.1 Random Forest (rf)
```
library(parallel)
library(doParallel)
library(caret)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",number = 5,allowParallel = TRUE)

set.seed=1002
modFit_rf<-train(classe~.,method="rf",data=training_red,trControl=fitControl)

stopCluster(cluster)
registerDoSEQ()

modFit_rf
## Random Forest 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15697, 15698, 15696, 15699, 15698 
## Resampling results across tuning parameters:
## 
##  mtry  Accuracy   Kappa    
##   2    0.9944959  0.9930373
##   27    0.9942411  0.9927149
##   52    0.9883294  0.9852354
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.

modFit_rf$resample
    Accuracy     Kappa Resample
## 1 0.9956688 0.9945214    Fold1
## 2 0.9938838 0.9922626    Fold2
## 3 0.9938838 0.9922624    Fold5
## 4 0.9943920 0.9929055    Fold4
## 5 0.9946510 0.9932344    Fold3

confusionMatrix.train(modFit_rf)
## Cross-Validated (5 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.1  0.0  0.0  0.0
##          B  0.0 19.2  0.1  0.0  0.0
##          C  0.0  0.0 17.3  0.2  0.0
##          D  0.0  0.0  0.0 16.1  0.0
##          E  0.0  0.0  0.0  0.0 18.3
##                             
##  Accuracy (average) : 0.9945

pred_rf<-predict(modFit_rf,testing_red)
```
### 6.2 Generalized Boosted Regression Model (gbm)
```
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

fitControl <- trainControl(method = "cv",number = 5,allowParallel = TRUE)

set.seed=1003
modFit_gbm<-train(classe~.,method="gbm",data=training_red,trControl=fitControl,verbose=FALSE)

stopCluster(cluster)
registerDoSEQ()

modFit_gbm
## Stochastic Gradient Boosting 
## 
## 19622 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## Summary of sample sizes: 15698, 15698, 15697, 15698, 15697 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa    
##   1                   50      0.7515552  0.6849103
##   1                  100      0.8228009  0.7756996
##   1                  150      0.8536845  0.8148498
##   2                   50      0.8563349  0.8179733
##   2                  100      0.9079093  0.8834404
##   2                  150      0.9327797  0.9149300
##   3                   50      0.8969526  0.8695452
##   3                  100      0.9432781  0.9282262
##   3                  150      0.9625932  0.9526712
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using the largest value.
## The final values used for the model were n.trees = 150,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
modFit_gbm$resample
##    Accuracy     Kappa Resample
## 1 0.9638124 0.9542198    Fold4
## 2 0.9589809 0.9480958    Fold3
## 3 0.9617737 0.9516408    Fold2
## 4 0.9633121 0.9535795    Fold5
## 5 0.9650866 0.9558202    Fold1
confusionMatrix.train(modFit_gbm)
## Cross-Validated (5 fold) Confusion Matrix 
## 
## (entries are percentual average cell counts across resamples)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.0  0.7  0.0  0.0  0.0
##          B  0.3 18.1  0.6  0.1  0.2
##          C  0.1  0.5 16.6  0.5  0.2
##          D  0.1  0.0  0.2 15.7  0.2
##          E  0.0  0.0  0.0  0.1 17.8
##                             
##  Accuracy (average) : 0.9626
pred_gbm<-predict(modFit_gbm,testing_red)
```
From the obtained models we found that Random Forest analysis provided for best accuracy (0.9945). Below the summary of accuracy and predicted results for each model.We can see despite the difference in accuracy, both models provided with the same predictions for the testing set.
```
print(paste("rf_accuracy=","0.9945"))
## [1] "rf_accuracy= 0.9945"
print(paste("gbm_accuracy=","0.9626"))
## [1] "gbm_accuracy= 0.9626"
print(paste("Predicted Results rf= "))
## [1] "Predicted Results rf= "
pred_rf
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
print(paste("Predicted Results gbm="))
## [1] "Predicted Results gbm="
pred_gbm
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
## 7. Expected Out of Sample Error
Accuracy is the fraction of properly predicted cases thus we will get the sample error as 1- accuracy. For the selected model using random forest the error rate will be then
```
print(paste("Out of Sample Error=",round((1-0.9945)*100,3),"%"))
## [1] "Out of Sample Error= 0.55 %"
```
It is important to mention that technically this error is the In Sample Error, however it’s the error so far expected for data out of the one provided on the training set.

























