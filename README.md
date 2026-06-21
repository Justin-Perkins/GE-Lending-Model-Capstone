# GE Credit Lending Model — Data Analytics Capstone
 
A linear regression model built to predict customer credit/loan amounts for GE's Credit department, developed as a graduate capstone project (SNHU, DAT-690) following the CRISP-DM framework.
 
**Video walkthrough:** https://youtu.be/54aVg6x0ruA
 
## Problem
 
GE's Credit department relies on human judgment and manual formulas to decide how much credit to extend to customers. This process is prone to error and bias, and creates two business risks:
 
- **Under-allocation** — customers don't get the credit they need and go elsewhere
- **Over-allocation** — customers borrow more than they can repay, increasing default risk

**Goal:** train a linear regression model that predicts an appropriate loan amount from customer credit history, account data, and utilization data, without using any demographic information.
 
## Data
 
- 5,000 training observations, 100 validation observations, 74 numeric features
- Categories: borrower information, credit history, credit utilization, account counts
- No personally identifiable information was used (no age, race, ethnicity, gender, or religion), in line with the Equal Credit Opportunity Act (ECOA)
- Missing values made up less than 2% of the dataset; one data entry error (`"-"` in a numeric column) was corrected
## Approach
 
Following CRISP-DM:
 
1. **Business & Data Understanding** — defined the prediction problem, explored feature distributions and correlations to the target
2. **Data Preparation** — removed loan-derived/leaky features, imputed missing values with column medians, applied log transformations to reduce skew, removed multicollinear features, filtered features by minimum correlation to target
3. **Modeling** — bidirectional stepwise feature selection, linear regression trained with 5-fold cross-validation
4. **Evaluation** — assessed with RMSE, MAPE, and R² on both train and held-out test data
5. **Deployment planning** — designed as a decision-support tool (not full automation); GE Credit employees retain final say on loan amounts
## Results
 
| Metric | Value |
|--------|-------|
| RMSE | $7,663.22 |
| MAPE | 55.73% |
| R² | 0.25 |
 
The model explains only about 25% of the variance in loan amounts. The target variable's non-normal, clustered distribution (likely from human-influenced round-number lending decisions) violates a core linear regression assumption, and log transformation only partially addressed the non-linearity in feature relationships.
 
## Limitations & Recommendations
 
- The target variable's distribution suggests the underlying relationship may be more discrete/non-linear than a single linear model can capture
- Business context not present in the data (e.g., stated purpose of the loan, requested amount) likely explains additional variance
- **Next steps:** incorporate requested loan amount and stated purpose as features; explore ensemble or tree-based models better suited to non-linear, discrete-leaning targets
## Repository Structure
 
```
/report
  Final PowerPoint.pdf                - final presentation
  Compiled Milestones.pdf             - full CRISP-DM writeup (data understanding, prep, modeling, evaluation)
  Production Turnover Report.pdf      - production handoff documentation for IT
/diagrams
  Bussiness Process Flowchart.pdf     - CRISP-DM process flowchart with explanation
/scripts
  Model_pipeline.R                    - full preprocessing, modeling, and evaluation pipeline
/data
  CreditAmount_Data.csv               - simulated training data (no PII)
  CreditAmount_DataDictionary         - Explanation of all fields in the data
  CreditAmount_Verify.csv             - simulated validation data (no PII)
  summary_matrix.csv                  - summary statistics for top correlated features
README.md
```
 
## Running the Model
 
The script uses [`here`](https://here.r-lib.org/) for portable, repo-relative file paths.
 
```r
install.packages(c("readr", "caret", "dplyr", "here"))
```
 
Run `scripts/model_pipeline.R` from anywhere within the cloned repository — `here()` resolves the project root automatically.
 
Note: the data included in this repository is simulated and contains no personally identifiable or real customer information. The script expects training and validation CSVs in the `/data` folder, matching the column structure described in the technical writeup.