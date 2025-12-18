# In the analysis, please treat the median house value "medv" as the response and the other thirteen 
# variables as the covariates. "chas" is a categorical variable which is beyond the scope of the
# course. You can ignore "chas" and consider the other twelve covariates only.

# ==============================================================================
# Project: Boston Housing Price Analysis
# Description: 
#   1. Data Exploration & Descriptive Statistics
#   2. Multiple Linear Regression：Baseline -> Outlier -> Box-Cox -> Stepwise -> WLS -> Ridge.
#   3. Machine Learning Models (Random Forest, XGBoost)
# ==============================================================================

# Load Required Libraries
library(mlbench)   # Boston Housing dataset
library(caret)     # Train/test split
library(corrplot)  # Correlation visualization
library(ggplot2)   # EDA plots
library(MASS)      # Stepwise regression, Box-Cox
library(car)       # VIF, Cook's Distance
library(glmnet)    # Ridge Regression
library(nlme)      # For GLS (WLS implementation)
library(tidyr)     # For data reshaping (EDA)
library(randomForest) 
library(xgboost)
library(dplyr)      # For data cleaning and manipulation
library(e1071)      # For calculating skewness and kurtosis

# Load data
data("BostonHousing")

# Preprocessing: Remove 'chas' variable as requested in the instructions
# Treating 'medv' as response and others as covariates.
df <- subset(BostonHousing, select = -chas)

# Check structure
str(df)

# 1. Exploratory Data Analysis (EDA) & Data Split

# 1.1 Correlation Matrix
# Identify linear relationships between predictors and target (medv), 
# and detect potential multicollinearity (high correlation between predictors).
cor_matrix <- cor(df)
corrplot(cor_matrix, method = "color", tl.col = "black", tl.cex = 0.7,
         addCoef.col = "black", number.cex = 0.6,
         title = "Correlation Matrix of Variables", mar = c(0,0,1,0))

# 1.2 THE BIG DISTRIBUTION PLOT
# Visual check for non-linear trends, heteroscedasticity, and outliers in the raw relationship between each predictor and the target (medv).
df_long <- pivot_longer(df, cols = -medv, names_to = "feature", values_to = "value")

p_big_eda <- ggplot(df_long, aes(x = value, y = medv)) +
  geom_point(alpha = 0.4, color = "darkgreen", size = 0.8) +
  # Add a linear smooth line to quickly see the general trend
  geom_smooth(method = "lm", color = "orange", se = FALSE) +
  facet_wrap(~feature, scales = "free_x") +
  theme_minimal() +
  labs(title = "Feature vs Target (medv) Distribution Matrix",
       x = "Feature Value", y = "Median Home Value")
print(p_big_eda)

# 1.3 Data Split: training (70%) for model fitting and testing (30%) for evaluation.
set.seed(2025) 
trainIndex <- createDataPartition(df$medv, p = 0.7, list = FALSE) 
trainData <- df[trainIndex, ]
testData  <- df[-trainIndex, ]


# ------------------------------------------------------------------------------
## 2. Multiple Linear Regression
# ------------------------------------------------------------------------------

### 2.0 Model 1: Baseline OLS Model
# Goal: Establish the initial model benchmark before any transformations or cleaning.
baseline_ols <- lm(medv ~ ., data = trainData)
summary(baseline_ols)

# Diagnostic: Check for high VIF and plot initial diagnostics
vif_baseline <- vif(baseline_ols)
cat("VIF Check: ", vif_baseline, "\n") 
par(mfrow = c(2, 2))
plot(baseline_ols, main = "Model 1 Diagnostics (Baseline OLS)")
par(mfrow = c(1, 1))
# Observation: Residual plots show non-linearity, high variance, and non-normality.

# --- Box-Cox Helper Functions ---
# Note: These robust functions handle the critical lambda = 0 (log transformation) case and prevent errors during inverse transformation (non-positive base).
apply_boxcox <- function(y, lambda) {
  if (abs(lambda) < 1e-4) return(log(y)) # Lambda close to zero
  return((y^lambda - 1) / lambda)
}
inverse_boxcox <- function(y_bc, lambda) {
  if (abs(lambda) < 1e-4) return(exp(y_bc))
  base <- y_bc * lambda + 1
  # Guard against non-positive base due to numerical instability
  if (any(base <= 0)) warning("Non-positive base fixed in inverse Box-Cox.")
  base[base <= 0] <- 1e-8 
  return(base^(1/lambda))
}

### 2.1 Step 1: Outlier Removal (High-Leverage Points): Improve model robustness by removing high-influence outliers based on Cook's Distance.
cooksD <- cooks.distance(baseline_ols)
# Cutoff: 4 / n (A common heuristic)
trainData_clean <- trainData[cooksD < (4 / nrow(trainData)), ]
cat("Removed", nrow(trainData) - nrow(trainData_clean), "outliers.\n")

### 2.2 Step 2: Box-Cox Transformation: Improve residual normality and linear relationship by transforming the target variable (medv).
# Step: Find optimal lambda on the clean training data.
ols_for_bc <- lm(medv ~ ., data = trainData_clean)
bc <- boxcox(ols_for_bc, lambda = seq(-2, 2, 0.1), plotit = FALSE)
best_lambda <- bc$x[which.max(bc$y)]

# Apply transformation to both train and test target variables.
trainData_clean$medv_bc <- apply_boxcox(trainData_clean$medv, best_lambda)
testData$medv_bc <- apply_boxcox(testData$medv, best_lambda)

cat("Optimal Box-Cox Lambda:", round(best_lambda, 4), "\n")

### 2.3 Model 3: OLS on Box-Cox Transformed Target
# Model 3 serves as the benchmark after addressing non-normality and non-linearity (via Box-Cox).
# CRITICAL FIX: Exclude the original 'medv' column to prevent data leakage.
ols_transformed <- lm(medv_bc ~ . - medv, data = trainData_clean)
summary(ols_transformed)

# Diagnostic: Check residuals after Box-Cox
par(mfrow = c(2, 2))
plot(ols_transformed, main = "Model 3 Diagnostics (Post Box-Cox)")
par(mfrow = c(1, 1))
# Observation: Q-Q plot should be straighter, but Scale-Location plot still shows heteroscedasticity.

### 2.4 Step 4: Stepwise Variable Selection (AIC): Achieve a simpler model by selecting the subset of predictors that minimizes the AIC (Akaike Information Criterion).
step_model_bc <- stepAIC(ols_transformed, direction = "both", trace = FALSE)
formula_step <- formula(step_model_bc)
summary(step_model_bc)
# Note: This model's formula (`formula_step`) is used in subsequent steps.

### 2.5 Step 5: Weighted Least Squares (WLS)
# Goal: Address the remaining heteroscedasticity (non-constant variance, often seen in Model 3 Scale-Location plot) to ensure standard errors and p-values are reliable.
# Action: Use Generalized Least Squares (GLS) in 'nlme' to model variance as proportional to `rad` raised to a power (delta), which is estimated via Maximum Likelihood (MLE).

wls_model <- gls(
  model = formula_step, 
  data = trainData_clean, 
  weights = varPower(form = ~ rad) # Variance is modeled as prop. to rad^(2*delta)
)
summary(wls_model) # Output full summary (Requirement 3)

# Diagnostic: Check WLS residuals
par(mfrow = c(1, 1))
plot(fitted(wls_model), residuals(wls_model, type = "normalized"),
     xlab = "Fitted Values (medv_bc)", ylab = "Normalized Residuals (WLS)",
     main = "Model 5 Diagnostics: Residuals vs Fitted (WLS - Homoscedasticity Check)")
abline(h = 0, col = "red")
# Observation: The band of residuals around zero are more horizontal and uniform than Model 3.

### 2.6 Step 6: Ridge Regression
# Goal: Stabilize coefficient estimates against high multicollinearity (VIF check) by applying L2 regularization.
# Action: Use the formula from Stepwise Selection to fit the Ridge model via cross-validation.
x_train <- model.matrix(formula_step, data = trainData_clean)[, -1] # Remove intercept
y_train <- trainData_clean$medv_bc
x_test  <- model.matrix(formula_step, data = testData)[, -1]

set.seed(2025)
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0, nfolds = 10) # alpha=0 for Ridge
best_lambda_ridge <- cv_ridge$lambda.min
ridge_model <- glmnet(x_train, y_train, alpha = 0, lambda = best_lambda_ridge)

cat("Optimal Lambda:", round(best_lambda_ridge, 4), "\n")
print(coef(ridge_model)) # Output coefficients 

# -------------------------- Final Evaluation (Original Scale) --------------
# Goal: Compare model performance using Test RMSE, converted back to the original price scale.

# Prediction & Back-transformation
preds_ols   <- inverse_boxcox(predict(ols_transformed, testData), best_lambda)
preds_wls   <- inverse_boxcox(predict(wls_model, testData), best_lambda)
# For glmnet, predict requires newx matrix and s (lambda)
preds_ridge <- inverse_boxcox(predict(ridge_model, newx = x_test, s = best_lambda_ridge)[,1], best_lambda)

# RMSE Calculation
calc_rmse <- function(p, a) sqrt(mean((p - a)^2))

rmse_results <- data.frame(
  Model = c("OLS (Box-Cox)", "WLS (rad-weighted)", "Ridge Regression"),
  Test_RMSE_medv = round(c(calc_rmse(preds_ols, testData$medv),
                           calc_rmse(preds_wls, testData$medv),
                           calc_rmse(preds_ridge, testData$medv)), 4)
)

# Final Test RMSE Comparison (Original Price Scale) 
print(rmse_results)

# Final Linear Regression Conclusions:
# The iterative refinement process (Outlier Removal, Box-Cox, WLS) successfully addressed major OLS assumptions (Normality, Linearity, Homoscedasticity), 
# as evidenced by the improved model fit (Adj. R-squared: ~0.84 vs. Baseline ~0.74).
# 
# KEY FINDINGS:
# 1. WLS Model improved the reliability of standard errors by correcting for heteroscedasticity 
#    using `rad` as the variance covariate (Power = 0.285).
# 2. Ridge Regression's slight performance drop (RMSE 5.66) suggests the bias introduced by L2 
#    regularization outweighed the benefit of variance reduction in this specific feature set.
# 3. OVERALL PERFORMANCE: Despite rigorous linear modeling, the final test RMSE (5.47 - 5.66) 
#    remains relatively high for this dataset. This suggests that the relationship between 
#    predictors and target may contain significant non-linear components, limiting the efficacy 
#    of purely linear methods.
# 4. NEXT STEP: Explore non-linear models (Random Forest, XGBoost) to potentially achieve 
#    a lower predictive error.


## New split for Machine Learning: Original Data Split (Training (70%) and Testing (30%) sets) 
set.seed(2025)
trainIndex_original <- createDataPartition(df$medv, p = 0.7, list = FALSE)
trainData_original <- df[trainIndex_original, ]
testData_original  <- df[-trainIndex_original, ]

# ------------------------------------------------------------------------------
## 3. Machine Learning: Random Forest 
# ------------------------------------------------------------------------------
## 3.1 Mtry Tuning based on Original Data
###  Initial Fit - Baseline Model (Default Parameters)
set.seed(2025)
rf_baseline <- randomForest(medv ~ ., data = trainData_original, ntree = 500, importance = TRUE)
print(rf_baseline) # Print OOB Error and RMSE as baseline metrics
# Baseline RF RMSE (OOB): sqrt(10.78044) = 3.2834

###  Determine Optimal ntree (Number of Trees)
# The OOB error curve is used to find where error stabilizes.
plot(rf_baseline, main = "OOB Error vs. Number of Trees")
# Conclusion: Error stabilizes around ntree = 100. We will use N_TREES=200 for stability in tuning.

###  3.1 Fine Tuning - mtry (Number of Variables Sampled)
# Use caret's grid search to tune mtry on the original dataset.
rf_grid <- expand.grid(mtry = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
set.seed(2025)
rf_mtry_tuned <- train(
  medv ~ .,
  data = trainData_original,
  method = "rf",
  trControl = ctrl,
  tuneGrid = rf_grid,
  ntree = 200, # Using 200 trees for mtry tuning
  importance = TRUE
)

# Best mtry results
FIXED_MTRY <- rf_mtry_tuned$bestTune$mtry
cat("Best mtry Tuning found (via 5-fold CV): \n")
print(rf_mtry_tuned$bestTune)
plot(rf_mtry_tuned, main = "Random Forest Tuning (RMSE vs mtry)")
cat(sprintf("Best mtry parameter selected: %d\n", FIXED_MTRY))

## 3.2 Data Cleaning: Remove Censored Data 
# The value '50.0' of medv is widely recognized as a censored/truncated value in this dataset which distorts regression models. 
# Removing these rows mitigates the severe overfitting observed in the original analysis (Training RMSE≈1.3 while Test RMSE≈3.86).
clean_df <- df %>%
  filter(medv < 50) 
cat(sprintf("Original Data Size: %d\n", nrow(df)))
cat(sprintf("Cleaned Data Size: %d (Removed %d rows)\n", nrow(clean_df), nrow(df) - nrow(clean_df)))

## 3.3 Data Splitting (Using Clean Data)
set.seed(2025)
# Re-split data based on the cleaned dataset
trainIndex_clean <- createDataPartition(clean_df$medv, p = 0.7, list = FALSE)
trainData_clean <- clean_df[trainIndex_clean, ]
testData_clean  <- clean_df[-trainIndex_clean, ]

## 3.4 Random Forest Hyperparameter Tuning (nodesize only, mtry fixed)
# Define parameter grid: nodesize (min leaf samples)
nodesize_grid <- expand.grid(
  nodesize = c(5, 10, 15, 20, 25, 30), # Testing larger leaf sizes to control tree complexity
  OOB_RMSE = 0                       
)

# Manual Grid Search using OOB Error
set.seed(2025) 
for(i in 1:nrow(nodesize_grid)) {
  
  # Train the model with specific nodesize
  model <- randomForest(
    formula = medv ~ ., 
    data = trainData_clean,
    ntree = 500,  # Using 500 trees for final stability            
    mtry = FIXED_MTRY,
    nodesize = nodesize_grid$nodesize[i],
    importance = FALSE            
  )
  
  # Record OOB RMSE 
  nodesize_grid$OOB_RMSE[i] <- sqrt(model$mse[N_TREES_FINAL])
}

# Find the best nodesize minimizing OOB RMSE
best_nodesize_params <- nodesize_grid[which.min(nodesize_grid$OOB_RMSE), ]
cat("\nBest nodesize found based on OOB RMSE:\n")
print(best_nodesize_params)

## 3.5 Final Optimized Model Training and Evaluation
# Use the fixed mtry (7) and the best nodesize to train the final model
set.seed(2025)
rf_final_model <- randomForest(
  medv ~ ., 
  data = trainData_clean,
  ntree = N_TREES_FINAL, 
  mtry = FIXED_MTRY,
  nodesize = best_nodesize_params$nodesize, 
  importance = TRUE
)


# --- Performance Evaluation ---
#  1. Training Set Evaluation (Overfitting Diagnosis)
rf_train_pred <- predict(rf_final_model, newdata = trainData_clean)
train_rmse <- RMSE(rf_train_pred, trainData_clean$medv)
train_r2   <- R2(rf_train_pred, trainData_clean$medv)

cat("\n--- Final Random Forest Performance Summary ---\n")
cat(sprintf("Hyperparameters: mtry=%d, nodesize=%d\n", FIXED_MTRY, best_nodesize_params$nodesize))
cat("\n# Training Set Performance (Overfitting Check)\n")
cat(sprintf("RF Training RMSE: %.4f\n", train_rmse)) 
cat(sprintf("RF Training R-squared: %.4f\n", train_r2))

#  2. Test Set Evaluation (Generalization Performance)
rf_test_pred <- predict(rf_final_model, newdata = testData_clean)
test_rmse <- RMSE(rf_test_pred, testData_clean$medv)
test_r2   <- R2(rf_test_pred, testData_clean$medv)

cat("\n# Test Set Performance (Generalization)\n")
cat(sprintf("RF Test RMSE: %.4f\n", test_rmse)) # Key metric for generalization
cat(sprintf("RF Test R-squared: %.4f\n", test_r2)) # Key metric for generalization

# Visualize Variable Importance
varImpPlot(rf_final_model, main = "Variable Importance (Optimized Random Forest)")


# ------------------------------------------------------------------------------
## 4. Machine Learning: XGBoost (Extreme Gradient Boosting)
# ------------------------------------------------------------------------------
# Split data into Training (70%) and Testing (30%) sets with stratification
# Stratify by medv bins to ensure similar distribution in train/test
set.seed(2025)
df$medv_bin <- cut(df$medv, breaks = 5)  # Create bins for balanced split
trainIndex <- createDataPartition(df$medv_bin, p = 0.7, list = FALSE)
trainData <- df[trainIndex, ] %>% select(-medv_bin)  # Remove temporary bin column
testData  <- df[-trainIndex, ] %>% select(-medv_bin)

# Setup Cross-Validation Control for XGBoost
ctrl_xgb <- trainControl(
  method = "cv", 
  number = 5, 
  verboseIter = FALSE, 
  allowParallel = TRUE
)

cat("\n--- 4.1 XGBoost Phase I: Estimate nrounds ---\n")

# Initial parameters with mild regularization to avoid overfitting early
xgb_initial_params <- data.frame(
  nrounds = 100,
  max_depth = 4,  # Slightly shallower to reduce complexity
  eta = 0.1,
  gamma = 0.1,    # Small gamma to penalize trivial splits
  colsample_bytree = 0.9,  # Reduce feature sampling to increase diversity
  min_child_weight = 2,    # Higher to prevent overfitting to small leaves
  subsample = 0.9          # Reduce sample fraction for more randomness
)

set.seed(2025)
xgb_cv_nrounds <- xgb.cv(
  data = as.matrix(trainData %>% select(-medv)), 
  label = trainData$medv,
  params = list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    eta = xgb_initial_params$eta,
    max_depth = xgb_initial_params$max_depth,
    gamma = xgb_initial_params$gamma,
    colsample_bytree = xgb_initial_params$colsample_bytree,
    min_child_weight = xgb_initial_params$min_child_weight,
    subsample = xgb_initial_params$subsample,
    reg_lambda = 1  # Add L2 regularization
  ),
  nrounds = 500,
  nfold = 5,
  early_stopping_rounds = 30,  # Tighter early stopping (30 rounds)
  verbose = FALSE
)

optimal_nrounds <- xgb_cv_nrounds$best_iteration
cat(sprintf("Optimal nrounds determined by early stopping (eta=0.1): %d\n", optimal_nrounds))

cat("\n--- 4.2 XGBoost Phase II: Random Search for Complexity Parameters ---\n")

initial_eta <- 0.1

# Narrowed grid focusing on regularization and controlled depth
xgb_grid_fast <- expand.grid(
  nrounds = optimal_nrounds, 
  eta = initial_eta, 
  max_depth = c(3, 4, 5),  # Shallow to moderate depth only
  min_child_weight = c(2, 3, 4),  # Higher weights to limit small leaves
  gamma = c(0.1, 0.3, 0.5),  # Stronger split penalties
  colsample_bytree = c(0.7, 0.8, 0.9),  # More feature subsampling
  subsample = c(0.7, 0.8, 0.9)          # More sample subsampling
)

set.seed(2025)
xgb_tuned_random <- train(
  medv ~ .,
  data = trainData,
  method = "xgbTree",
  trControl = ctrl_xgb,
  tuneGrid = xgb_grid_fast,
  tuneLength = 20,  # Slightly fewer combinations to focus on robust params
  verbose = FALSE
)

best_complexity_params <- xgb_tuned_random$bestTune
cat("XGBoost - Best Random Search Tuning (Complexity):\n")
print(best_complexity_params)

cat("\n--- 4.3 XGBoost Phase III: Fine-Tuning Learning Rate (eta) ---\n")

best_params_fixed <- best_complexity_params
best_params_fixed$nrounds <- NULL
best_params_fixed$eta <- NULL

# Eta grid with early stopping integration
eta_grid <- expand.grid(
  nrounds = optimal_nrounds * 2,  # Allow more rounds for smaller eta
  eta = c(0.03, 0.05, 0.07),  # Slightly larger than before to balance with reg
  max_depth = best_complexity_params$max_depth,
  gamma = best_complexity_params$gamma,
  colsample_bytree = best_complexity_params$colsample_bytree,
  min_child_weight = best_complexity_params$min_child_weight,
  subsample = best_complexity_params$subsample
)

set.seed(2025)
xgb_final_tuned <- train(
  medv ~ .,
  data = trainData,
  method = "xgbTree",
  trControl = ctrl_xgb,
  tuneGrid = eta_grid,
  verbose = FALSE
)

xgb_final_best_params <- xgb_final_tuned$bestTune
cat("XGBoost - Final Optimized Parameters:\n")
print(xgb_final_best_params)
cat(sprintf("XGBoost Final CV RMSE: %.4f\n", min(xgb_final_tuned$results$RMSE)))

# --- 4.4 Performance Evaluation ---

# --- Training set evaluation (for overfitting diagnosis) ---
xgb_final_model <- xgb_final_tuned$finalModel
dtrain_eval <- xgb.DMatrix(data = as.matrix(trainData %>% select(-medv)))

xgb_train_pred <- predict(xgb_final_model, newdata = dtrain_eval)
xgb_rmse_train <- RMSE(xgb_train_pred, trainData$medv)
xgb_rsq_train  <- R2(xgb_train_pred, trainData$medv)

cat("\n--- XGBoost Training Set Performance ---\n")
cat(sprintf("XGBoost Training RMSE: %.4f\n", xgb_rmse_train))
cat(sprintf("XGBoost Training R-squared: %.4f\n", xgb_rsq_train))

# --- Test set evaluation (generalization for reporting) ---
dtest_eval <- xgb.DMatrix(data = as.matrix(testData %>% select(-medv)))
xgb_test_pred <- predict(xgb_final_model, newdata = dtest_eval)

xgb_rmse_test <- RMSE(xgb_test_pred, testData$medv)
xgb_rsq_test  <- R2(xgb_test_pred, testData$medv)

cat("\n--- XGBoost Test Set Performance ---\n")
cat(sprintf("XGBoost Test RMSE: %.4f\n", xgb_rmse_test))
cat(sprintf("XGBoost Test R-squared: %.4f\n", xgb_rsq_test))

# Plot variable importance
importance_matrix <- xgb.importance(model = xgb_final_model)
xgb.plot.importance(importance_matrix, top_n = 12, main = "Variable Importance (Optimized XGBoost)")