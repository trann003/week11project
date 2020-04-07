#### R Studio API Code ####
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


#### Libraries ####
library(tidyverse)
library(haven)
library(caret)
library(xgboost)
library(LiblineaR)


#### Data Import and Cleaning ####
# import data
week11_tbl_full <- read_sav("../data/GSS2006.sav")
 
# personality data
week11_tbl_per <- week11_tbl_full %>%
  # select personality variables
  select_at(vars(starts_with("BIG5"))) %>%
  zap_labels(.) %>%
  # missing values for Don't know (98), No answer (99), or Not applicable (-1)
  na_if(98 | 99 | -1) %>%
  # create ID 
  rowid_to_column(., "ID")


# health data
week11_tbl_health <- week11_tbl_full %>%
  # select health variables
  select(HEALTH, HEALTH1) %>%
  # missing values for Don't know (8), No answer (9), or Not applicable (0)
  na_if(8 | 9 | 0) %>%
  zap_labels(.) %>%
  # create ID
  rowid_to_column(., "ID")

# merge personality and health data
week11_tbl <- full_join(week11_tbl_health, week11_tbl_per, by = "ID")


#### Analysis ####

# reproducibility
set.seed(111)

# function to get results
get_best_result = function(caret_fit) {
  best = which(rownames(caret_fit$results) == rownames(caret_fit$bestTune))
  best_result = caret_fit$results[best, ]
  rownames(best_result) = NULL
  best_result
}

# create holdout and test samples
test <- week11_tbl %>%
  sample_n(250)
training <- week11_tbl[-as.numeric(rownames(test)),]

# Pre processing data
training_preproc <- preProcess(training[, 2:13], 
                               method = "medianImpute")
training_data <- predict(training_preproc, training)


# OLS
lm_mod <- train(HEALTH1 ~ BIG5A2 + BIG5D2 + BIG5B2 + 
                  BIG5C2 + BIG5D1 + BIG5B1 +
                  BIG5C1 + BIG5E2 + BIG5E1 + 
                  BIG5A1 + 
                  
                  BIG5A2*BIG5D2 + BIG5A2*BIG5B2 + 
                  BIG5A2*BIG5C2 + BIG5A2*BIG5D1 + 
                  BIG5A2*BIG5B1 + BIG5A2*BIG5C1 +
                  BIG5A2*BIG5E2 + BIG5A2*BIG5E1 +
                  BIG5A2*BIG5A1 + 
                  
                  BIG5D2*BIG5B2 + BIG5D2*BIG5A1 +
                  BIG5D2*BIG5C2 + BIG5D2*BIG5D1 + 
                  BIG5D2*BIG5B1 + BIG5D2*BIG5C1 +
                  BIG5D2*BIG5E2 + BIG5D2*BIG5E1 + 
                  
                  BIG5B2*BIG5A1 +
                  BIG5B2*BIG5C2 + BIG5B2*BIG5D1 + 
                  BIG5B2*BIG5B1 + BIG5B2*BIG5C1 +
                  BIG5B2*BIG5E2 + BIG5B2*BIG5E1 +
                  
                  BIG5C2*BIG5A1 + BIG5B2*BIG5D1 + 
                  BIG5B2*BIG5B1 + BIG5B2*BIG5C1 +
                  BIG5B2*BIG5E2 + BIG5B2*BIG5E1 +
                  
                  BIG5D1*BIG5A1 + 
                  BIG5D1*BIG5B1 + BIG5D1*BIG5C1 +
                  BIG5D1*BIG5E2 + BIG5D1*BIG5E1 +
                  
                  BIG5B1*BIG5A1 + BIG5B1*BIG5C1 +
                  BIG5B1*BIG5E2 + BIG5B1*BIG5E1 +
                  
                  BIG5C1*BIG5A1 + 
                  BIG5C1*BIG5E2 + BIG5C1*BIG5E1 +
                  
                  BIG5E2*BIG5A1 + BIG5E2*BIG5E1 + 
                  
                  BIG5A1*BIG5E1,
                data = training_data,
                # treat missing values
                na.action = na.pass,
                #Set cross-validation to be 10 fold)
                trControl = trainControl("cv", number = 10),
                # OLS
                method = "lm",
                verbose = F)  

get_best_result(lm_mod)

# Elastic net 
elas_mod <- train(HEALTH1 ~ BIG5A2 + BIG5D2 + BIG5B2 + 
                    BIG5C2 + BIG5D1 + BIG5B1 +
                    BIG5C1 + BIG5E2 + BIG5E1 + 
                    BIG5A1 + 
                    
                    BIG5A2*BIG5D2 + BIG5A2*BIG5B2 + 
                    BIG5A2*BIG5C2 + BIG5A2*BIG5D1 + 
                    BIG5A2*BIG5B1 + BIG5A2*BIG5C1 +
                    BIG5A2*BIG5E2 + BIG5A2*BIG5E1 +
                    BIG5A2*BIG5A1 + 
                    
                    BIG5D2*BIG5B2 + BIG5D2*BIG5A1 +
                    BIG5D2*BIG5C2 + BIG5D2*BIG5D1 + 
                    BIG5D2*BIG5B1 + BIG5D2*BIG5C1 +
                    BIG5D2*BIG5E2 + BIG5D2*BIG5E1 + 
                    
                    BIG5B2*BIG5A1 +
                    BIG5B2*BIG5C2 + BIG5B2*BIG5D1 + 
                    BIG5B2*BIG5B1 + BIG5B2*BIG5C1 +
                    BIG5B2*BIG5E2 + BIG5B2*BIG5E1 +
                    
                    BIG5C2*BIG5A1 + BIG5B2*BIG5D1 + 
                    BIG5B2*BIG5B1 + BIG5B2*BIG5C1 +
                    BIG5B2*BIG5E2 + BIG5B2*BIG5E1 +
                    
                    BIG5D1*BIG5A1 + 
                    BIG5D1*BIG5B1 + BIG5D1*BIG5C1 +
                    BIG5D1*BIG5E2 + BIG5D1*BIG5E1 +
                    
                    BIG5B1*BIG5A1 + BIG5B1*BIG5C1 +
                    BIG5B1*BIG5E2 + BIG5B1*BIG5E1 +
                    
                    BIG5C1*BIG5A1 + 
                    BIG5C1*BIG5E2 + BIG5C1*BIG5E1 +
                    
                    BIG5E2*BIG5A1 + BIG5E2*BIG5E1 + 
                    
                    BIG5A1*BIG5E1,
                  data = training_data,
                  # Elastic net
                  method = "glmnet", 
                  # missing values
                  na.action = na.pass,
                  # set cross-validation to be 10 fold)
                  trControl = trainControl("cv", number = 10))  

# Elastic net results
get_best_result(elas_mod)

# Xgboost
xgb_mod <- train(HEALTH1 ~ BIG5A2 + BIG5D2 + BIG5B2 + 
                    BIG5C2 + BIG5D1 + BIG5B1 +
                    BIG5C1 + BIG5E2 + BIG5E1 + 
                    BIG5A1 + 
                    
                    BIG5A2*BIG5D2 + BIG5A2*BIG5B2 + 
                    BIG5A2*BIG5C2 + BIG5A2*BIG5D1 + 
                    BIG5A2*BIG5B1 + BIG5A2*BIG5C1 +
                    BIG5A2*BIG5E2 + BIG5A2*BIG5E1 +
                    BIG5A2*BIG5A1 + 
                    
                    BIG5D2*BIG5B2 + BIG5D2*BIG5A1 +
                    BIG5D2*BIG5C2 + BIG5D2*BIG5D1 + 
                    BIG5D2*BIG5B1 + BIG5D2*BIG5C1 +
                    BIG5D2*BIG5E2 + BIG5D2*BIG5E1 + 
                    
                    BIG5B2*BIG5A1 +
                    BIG5B2*BIG5C2 + BIG5B2*BIG5D1 + 
                    BIG5B2*BIG5B1 + BIG5B2*BIG5C1 +
                    BIG5B2*BIG5E2 + BIG5B2*BIG5E1 +
                    
                    BIG5C2*BIG5A1 + BIG5B2*BIG5D1 + 
                    BIG5B2*BIG5B1 + BIG5B2*BIG5C1 +
                    BIG5B2*BIG5E2 + BIG5B2*BIG5E1 +
                    
                    BIG5D1*BIG5A1 + 
                    BIG5D1*BIG5B1 + BIG5D1*BIG5C1 +
                    BIG5D1*BIG5E2 + BIG5D1*BIG5E1 +
                    
                    BIG5B1*BIG5A1 + BIG5B1*BIG5C1 +
                    BIG5B1*BIG5E2 + BIG5B1*BIG5E1 +
                    
                    BIG5C1*BIG5A1 + 
                    BIG5C1*BIG5E2 + BIG5C1*BIG5E1 +
                    
                    BIG5E2*BIG5A1 + BIG5E2*BIG5E1 + 
                    
                    BIG5A1*BIG5E1,
                  data = training_data,
                  # XGBoost
                  method = "xgbLinear", 
                  # treat missing values
                  na.action = na.pass,
                  #Set cross-validation to be 10 fold
                  trControl = trainControl("cv", number = 10)) 

# XGB results
xbg_results <- get_best_result(xgb_mod)


# Support vector regression
svr_mod <- train(HEALTH1 ~ BIG5A2 + BIG5D2 + BIG5B2 + 
                   BIG5C2 + BIG5D1 + BIG5B1 +
                   BIG5C1 + BIG5E2 + BIG5E1 + 
                   BIG5A1 + 
                   
                   BIG5A2*BIG5D2 + BIG5A2*BIG5B2 + 
                   BIG5A2*BIG5C2 + BIG5A2*BIG5D1 + 
                   BIG5A2*BIG5B1 + BIG5A2*BIG5C1 +
                   BIG5A2*BIG5E2 + BIG5A2*BIG5E1 +
                   BIG5A2*BIG5A1 + 
                   
                   BIG5D2*BIG5B2 + BIG5D2*BIG5A1 +
                   BIG5D2*BIG5C2 + BIG5D2*BIG5D1 + 
                   BIG5D2*BIG5B1 + BIG5D2*BIG5C1 +
                   BIG5D2*BIG5E2 + BIG5D2*BIG5E1 + 
                   
                   BIG5B2*BIG5A1 +
                   BIG5B2*BIG5C2 + BIG5B2*BIG5D1 + 
                   BIG5B2*BIG5B1 + BIG5B2*BIG5C1 +
                   BIG5B2*BIG5E2 + BIG5B2*BIG5E1 +
                   
                   BIG5C2*BIG5A1 + BIG5B2*BIG5D1 + 
                   BIG5B2*BIG5B1 + BIG5B2*BIG5C1 +
                   BIG5B2*BIG5E2 + BIG5B2*BIG5E1 +
                   
                   BIG5D1*BIG5A1 + 
                   BIG5D1*BIG5B1 + BIG5D1*BIG5C1 +
                   BIG5D1*BIG5E2 + BIG5D1*BIG5E1 +
                   
                   BIG5B1*BIG5A1 + BIG5B1*BIG5C1 +
                   BIG5B1*BIG5E2 + BIG5B1*BIG5E1 +
                   
                   BIG5C1*BIG5A1 + 
                   BIG5C1*BIG5E2 + BIG5C1*BIG5E1 +
                   
                   BIG5E2*BIG5A1 + BIG5E2*BIG5E1 + 
                   
                   BIG5A1*BIG5E1,
                 data = training_data,
                 # XGBoost
                 method = "svmLinear3", 
                 # missing values
                 na.action = na.pass,
                 # Set cross-validation to be 10 fold
                 trControl = trainControl("cv", number = 10)) 
svr_results <- get_best_result(svr_mod)

# comparing cross-validated
summary(resamples(list(lm_mod, elas_mod, xgb_mod, svr_mod)))
dotplot(resamples(list(lm_mod, elas_mod, xgb_mod, svr_mod)), metric="RMSE")
dotplot(resamples(list(lm_mod, elas_mod, xgb_mod, svr_mod)), metric="Rsquared")

# predict
test_preproc <- preProcess(test[,2:13], method = "medianImpute")
test_data <- predict(test_preproc, test)


lm_predict <- predict(lm_mod, test_data, na.action = na.pass)
elas_predict <- predict(elas_mod, test_data, na.action = na.pass)
xgb_predict <- predict(xgb_mod, test_data, na.action = na.pass)
svr_predict <- predict(svr_mod, test_data, na.action = na.pass)

lm_cor <- cor(lm_predict, test_data$HEALTH1, use = "complete.obs")
elas_cor <- cor(elas_predict, test_data$HEALTH1, use = "complete.obs")
xgb_cor <- cor(xgb_predict, test_data$HEALTH1, use = "complete.obs")
svr_cor <- cor(svr_predict, test_data$HEALTH1, use = "complete.obs")
