setwd("H:/PS/Logistic_Regression")
adultCensus = read.csv("adultcensus.csv")
install.packages("ggcorrplot")
install.packages("stringr")
install.packages("stringi")
install.packages("devtools")
install.packages("InformationValue")
install.packages("e1071")
install.packages("tidyverse")

library(e1071)
library(stringr)
library(ggcorrplot)
library(stringi)
library(devtools)
library(InformationValue)
library(caret)
library(tidyverse)
library(caret)

options(scipen = 999)

#Check for missing data. The dataset has "?" question marks in place of missing values.
#This has been taken care of using the following:
#Getting rid of such records:
sapply(adultCensus, function(x) sum(is.na(x)))
#checking for each column:
#adultCensus = adultCensus[adultCensus$occupation != "?", ]

#Checking through all the columns:
No_ques_marks <- function(x, column){
  for (column in colnames(adultCensus)){
    print("?" %in% x[, column])
  }
}
No_ques_marks(adultCensus, colnames(adultCensus))

#Viewing data type of each column:
sapply(adultCensus, class)

head(adultCensus)
tail(adultCensus)
#Label Encoding for outcome variable:
unique(adultCensus$income_class)
adultCensus$income_class[adultCensus$income_class == "<=50K."] <- "<=50K"
adultCensus$income_class[adultCensus$income_class == ">50K."] <- ">50K"

adultCensus$class <- factor(adultCensus$income_class,
                   levels = c('<=50K','>50K'),
                   labels = c(0,1))
#Verifying that the outcome variable is binary
unique(adultCensus$class)

#Collecting only continuous variables:
acNumerical <- adultCensus[c(1,3,11,12,13)]
acNumerical <- scale(acNumerical)
acNumerical <- data.frame(acNumerical)
#Visualizing correlation amongst continuous variables:
acNum_cor <- cor(acNumerical)
acNum_cor
ggcorrplot(acNum_cor, lab = TRUE)

acNumerical$class <- adultCensus$class
#checking correlations with continuous variables
ConVar_cor <- manova(cbind(age, fnlwgt, capital.gain, capital.loss, hours_per_week) ~ class, data = acNumerical)
summary(ConVar_cor)
summary.aov(ConVar_cor)
#Thus, variable "fnlwgt" fails to have a significant impact on income class. We exclude this variable from our model

#We first run the model with selected continuous and all categorical variables.
#Deleting edu_num as it is redundant, along with income_class and fnlwgt.
head(adultCensus,1)
ConVarScaled <- adultCensus[-c(3,5,15)] 
head(ConVarScaled,1)
#Feature Scaling for continuous Variables:  
ConVarScaled[c(1,9,10,11)] = scale(ConVarScaled[c(1,9,10,11)])

model_All <- glm(formula = class ~ ., data = ConVarScaled, family = binomial ,control = list(maxit = 50))
summary(model_All)
#Checking Multi-collinearity by examining VIF values for Categorical Variables
car::vif(model_All)

#We need to remove the variable with highest GVIF, which is "relationship". 
ConVarScaled <- ConVarScaled[-c(6)]
model_All <- glm(formula = class ~ ., data = ConVarScaled, family = binomial ,control = list(maxit = 50))
summary(model_All)
#Checking Multi-collinearity by examining VIF values for Categorical Variables
car::vif(model_All)

#Since all other variables have VIF < 5, we can proceed.
#checking significance with remaining categorical variables:
Cat_vars <- c ("workclass", "education", "marital_status", "occupation", "race", "sex", "Native")
Cat_infoval <- data.frame(VARS=Cat_vars, IV=numeric(length(Cat_vars)), STRENGTH=character(length(Cat_vars)), stringsAsFactors = F)
for (Cat_var in Cat_vars){
  Cat_infoval[Cat_infoval$VARS == Cat_var, "IV"] <- InformationValue::IV(X=adultCensus[, Cat_var], Y=adultCensus$class)
  Cat_infoval[Cat_infoval$VARS == Cat_var, "STRENGTH"] <- attr(InformationValue::IV(X=adultCensus[, Cat_var], Y=adultCensus$class), "howgood")
}
Cat_infoval <- Cat_infoval[order(-Cat_infoval$IV), ]
Cat_infoval

#We would continue to include the bottom variables as well.
#Select significant features:
head(ConVarScaled,1)
Sig_Data <- ConVarScaled#[-c(6,11)] 

#Check class distribution
#Baseline accuracy:
table(adultCensus$class)
#Baseline accuracy
baseline <- round(28875/nrow(adultCensus),2)
baseline
#Our accuracy should at least be 75%

#SPlit into training and testing:
set.seed(123)
training_samples <- Sig_Data$class %>% 
  createDataPartition(p = 0.70, list = FALSE)
train  <- Sig_Data[training_samples, ]
test <- Sig_Data[-training_samples, ]

head(test,1)
# Feature Scaling
#train[c(1,6,7,8)] = scale(train[c(1,6,7,8)])
#test[c(1,6,7,8)] = scale(test[c(1,6,7,8)])
#head(train,1)

#Applying model with selected features:
mod_sig <- glm(formula = class ~ ., data = train, family = binomial(link = "logit") ,control = list(maxit = 50))
summary(mod_sig)
# Predicting the Test set results
pred_modsig = predict(mod_sig, type = 'response', newdata = test[-12])

#find optimal threshold:
library(InformationValue)
oc <- optimalCutoff(test$class, pred_modsig)[1] 
oc

p_class_modsig = ifelse(pred_modsig > oc, 1, 0)

roc.curve(test$class, p_class_modsig)
#0.761
plotROC(test$class, p_class_modsig)
#0.7401
ypred <- as.data.frame(p_class_modsig)
sapply(ypred, class)

unique(ypred$p_class_modsig)
ypred$p_class_modsig <- factor(ypred$p_class_modsig,
                               levels = c('0','1'),
                               labels = c(0,1))
confusionMatrix(reference = test$class, data = ypred$p_class_modsig, positive = '1')

#I would like to treat class imbalance and examine results for new training and testing set. 
#This file will be updated.