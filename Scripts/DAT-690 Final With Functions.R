#install.packages("RSNNS")
#install.packages("here")

# Packages used in this analysis
library(readr)
library(caret)
library(dplyr)
library(here)


# HELPER FUNCTIONS USED IN THE PREPROCESSING PIPELINE

# Fix revol_util_pct values (replaces "-" with NA)
# Input: dataframe
# Output: dataframe with corrected values and data type
clean_raw_columns <- function(data) {
  if (is.character(data$revol_util_pct)) {
    data$revol_util_pct <- na_if(data$revol_util_pct, "-")
    data$revol_util_pct <- as.numeric(data$revol_util_pct)
  }
  data
}

# Removes the columns related to the loans which are decided after the amount
# Input: dataframe
# Output: dataframe with removed columns  
drop_loan_related_columns <- function(data) {
  subset(data, select = -c(
    ID, installment, int_rate_pct, term_60mon, total_rev_hi_lim,
    chargeoff_within_12_mths
  ))
}

# Calculates and removes columns with collinearity
# Input: dataframe and threshold for collinearity
# Output: dataframe with collinear columns removed
remove_multicollinear_columns <- function(data, cutoff = 0.85) {
  cor_matrix     <- cor(data, use = "complete.obs")
  high_corr_vars <- findCorrelation(cor_matrix, cutoff = cutoff)
  data[, -high_corr_vars]
}

# Removed columns that do not correlate with the target beyond a threshold
# Input: dataframe, target variable, minimum absolute correlation to the target
# Output: dataframe with the non-correlating variables removed
filter_by_target_correlation <- function(data, target = "funded_amnt", min_abs_cor = 0.05) {
  target_cor   <- cor(data, use = "complete.obs")[, target]
  vars_to_keep <- names(abs(target_cor)[abs(target_cor) >= min_abs_cor])
  data[, vars_to_keep]
}

# Imputes missing values of all columns with their respective column median value
# Input: dataframe
# Output: dataframe with no missing values
impute_with_column_medians <- function(data) {
  na_counts      <- sapply(data, function(x) sum(is.na(x)))
  vars_to_impute <- names(na_counts[na_counts > 0])
  for (var in vars_to_impute) {
    data[[var]][is.na(data[[var]])] <- median(data[[var]], na.rm = TRUE)
  }
  data
}



# PREPROCESSING PIPELINES

# Executes all the preprocessing steps consecutively, preparing for modeling
# Input: raw dataframe
# Output: polished dataframe ready for use in modeling
preprocess_training_data <- function(data) {
  data <- clean_raw_columns(data)
  data <- drop_loan_related_columns(data)
  data <- remove_multicollinear_columns(data)
  data <- filter_by_target_correlation(data)
  data <- impute_with_column_medians(data)
  data
}

# Executes all the preprocessing steps consecutively, preparing for modeling
# Input: raw dataframe and the names of the columns in the final training dataframe
# Output: polished dataframe with identical columns to the train set
preprocess_test_data <- function(data, training_columns) {
  data <- clean_raw_columns(data)
  data <- data[, training_columns]       # align to training column set
  data <- impute_with_column_medians(data)
  data
}




# Load the data
CreditAmount_Data <- read_csv(here("data", "CreditAmount_Data.csv"))
CreditAmount_Data_Test <- read_csv("DAT 690 Data Files/CreditAmount_Verify.csv")

# Preprocess the data
CreditAmount_Data      <- preprocess_training_data(CreditAmount_Data)
CreditAmount_Data_Test <- preprocess_test_data(CreditAmount_Data_Test, names(CreditAmount_Data))




# EXPLORATORY DATA ANALYSIS

# Distribution of target variable histogram
hist(
  CreditAmount_Data$funded_amnt,
  breaks = 30,
  main   = "Distribution of Funded Loan Amount",
  xlab   = "Funded Amount",
  ylab   = "Frequency"
)

# Summary statistics of target variable
summary(CreditAmount_Data$funded_amnt)
length(unique(CreditAmount_Data$funded_amnt))

# Calculate the correlation of the variables against target
target_cor <- cor(CreditAmount_Data, use = "complete.obs")[, "funded_amnt"]
target_cor <- sort(target_cor, decreasing = TRUE)
target_cor

top_correlated_vars <- names(target_cor[2:6])

# Plots top correlated variables against target with trend line
par(mfrow = c(2, 3))
for (var in top_correlated_vars) {
  plot(log(CreditAmount_Data[[var]] + 1),
       CreditAmount_Data$funded_amnt,
       main = paste("funded_amnt vs log(", var, "+1)", sep = ""),
       xlab = paste("log(", var, "+1)", sep = ""),
       ylab = "funded_amnt")
  abline(lm(CreditAmount_Data$funded_amnt ~ log(CreditAmount_Data[[var]] + 1)), col = "red")
}
par(mfrow = c(1, 1))

# Summary statistics of top correlated variables and exports table as csv
summary_matrix <- t(sapply(top_correlated_vars, function(var) summary(CreditAmount_Data[[var]])))
summary_matrix
write.csv(summary_matrix, "summary_matrix.csv", row.names = TRUE)



# MODELING

# Setting seed for reproducabilty
set.seed(99)

train_x <- subset(CreditAmount_Data, select = -funded_amnt)
train_y <- CreditAmount_Data$funded_amnt

# Stepwise feature selection algorithm
full_formula <- as.formula(paste("funded_amnt ~", paste(names(train_x), collapse = " + ")))
full_model   <- lm(full_formula, data = CreditAmount_Data)
null_model   <- lm(funded_amnt ~ 1, data = CreditAmount_Data)

# Train algorithm
step_model <- step(null_model,
                   scope     = list(lower = null_model, upper = full_model),
                   direction = "both",
                   trace     = 0)

cat("\nSelected features:\n")
print(names(coef(step_model))[-1])

# Narrow down trainset to only selected features and scale them logarithmically 
selected_features    <- names(coef(step_model))[-1]
train_x_selected     <- train_x[, selected_features]
train_x_selected_log <- as.data.frame(lapply(train_x_selected, function(x) log(x + 1)))
train_y_log          <- log(train_y + 1)

# Train the linear regression model
lm_model <- train(
  x         = train_x_selected_log,
  y         = train_y_log,
  method    = "lm",
  trControl = trainControl(method = "cv", number = 5)
)

summary(lm_model$finalModel)
print(lm_model)




# EVALUATE THE TRAIN PREDICTIONS

# Make predictions
train_preds_log <- predict(lm_model, newdata = train_x_selected_log)
train_preds     <- exp(train_preds_log) - 1
train_residuals <- train_y - train_preds

# Calculate evaluation metrics
train_rmse <- RMSE(train_preds, train_y)
train_mape <- mean(abs(train_residuals / train_y)) * 100
train_r2   <- R2(train_preds, train_y)

cat("RMSE: "); print(train_rmse)
cat("MAPE: "); print(train_mape)
cat("R2:   "); print(train_r2)

log_residuals <- train_y_log - train_preds_log


# Plot residuals
par(mfrow = c(2, 2))

# Plot residuals against fitted
plot(train_preds_log, log_residuals,
     main = "Residuals vs Fitted (log scale)",
     xlab = "Fitted Values (log)", ylab = "Residuals",
     pch = 20, col = "steelblue")
abline(h = 0, col = "red", lty = 2)

# Normal Q-Q plot
qqnorm(log_residuals, main = "Normal Q-Q Plot", pch = 20, col = "steelblue")
qqline(log_residuals, col = "red", lty = 2)

# Scale location plot
plot(train_preds_log, sqrt(abs(log_residuals)),
     main = "Scale-Location (log scale)",
     xlab = "Fitted Values (log)", ylab = expression(sqrt("|Residuals|")),
     pch = 20, col = "steelblue")
abline(h = 0, col = "red", lty = 2)

# Distribution of residuals plot
hist(log_residuals,
     breaks = 30,
     main   = "Residual Distribution (log scale)",
     xlab   = "Residuals",
     col    = "steelblue", border = "white")
abline(v = 0, col = "red", lty = 2)

par(mfrow = c(1, 1))

# Summary of residuals
cat("Residual Summary:")
print(summary(log_residuals))




# EVALUATE THE TEST PREDICTIONS

test_x     <- CreditAmount_Data_Test[, selected_features]
test_y     <- CreditAmount_Data_Test$funded_amnt
test_x_log <- as.data.frame(lapply(test_x, function(x) log(x + 1)))

# Make predictions with the test set
test_preds_log <- predict(lm_model, newdata = test_x_log)
test_preds     <- exp(test_preds_log) - 1
test_residuals <- test_y - test_preds

# Test evaluation metrics
test_rmse <- RMSE(test_preds, test_y)
test_mape <- mean(abs(test_residuals / test_y)) * 100
test_r2   <- R2(test_preds, test_y)

cat("--- Test Set Evaluation ---\n")
cat("RMSE: "); print(test_rmse)
cat("MAPE: "); print(test_mape)
cat("R2:   "); print(test_r2)

cat("\n--- Train vs Test Comparison ---\n")
print(data.frame(
  Metric = c("RMSE", "MAPE", "R2"),
  Train  = c(train_rmse, train_mape, train_r2),
  Test   = c(test_rmse,  test_mape,  test_r2)
))

par(mfrow = c(1, 2))

hist(
  CreditAmount_Data$funded_amnt,
  breaks = 30,
  main   = "",
  xlab   = "Funded Amount",
  ylab   = "Frequency"
)

# Distribution of predictions histogram
hist(
  test_preds,
  breaks = 30,
  main   = "",
  xlab   = "Predicted Funded Amount",
  ylab   = "Frequency"
)

par(mfrow = c(1, 1))