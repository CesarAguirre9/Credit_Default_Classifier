# Credit Default Classifier

Predicting the probability that a borrower will experience serious financial distress — defined as 90+ days past due — within the next two years. Built on the [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) dataset (150,000 borrowers, 6.7% default rate).

---

## Problem Context

Consumer lenders price credit by estimating the probability of default (PD) for each applicant. A well-calibrated PD model allows a lender to:

- Approve or deny applications based on expected loss vs. expected revenue
- Set risk-based pricing (higher PD → higher rate or lower credit line)
- Satisfy regulatory adverse action requirements (ECOA, Fair Housing Act)
- Generate CECL reserve estimates for accounting purposes

This project replicates the core modeling step that a credit risk data scientist owns in production.

---

## Dataset

- **Source**: [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
- **Size**: 150,000 borrowers, 11 features
- **Target**: `SeriousDlqin2yrs` — 1 if the borrower went 90+ days past due within 2 years, 0 otherwise
- **Class imbalance**: ~93.3% non-default / ~6.7% default

Key features include revolving credit utilization, age, monthly income, debt ratio, and delinquency history across three severity buckets (30–59, 60–89, 90+ days past due).

---

## Methodology

### 1. Exploratory Data Analysis
- Target distribution (countplot with percentage labels)
- Summary statistics and data type inspection
- Correlation heatmap to identify multicollinearity
- Feature distributions split by default status (density-normalized histplots)

### 2. Data Cleaning
- Removed revolving utilization outliers > 200% (likely data entry errors)
- Dropped 1 row with age = 0 (not a valid borrower)
- Imputed `MonthlyIncome` and `NumberOfDependents` missing values with median
- Retained missingness indicator flags as features — missingness was found to carry predictive signal

### 3. Feature Engineering — Handling Multicollinearity

The three delinquency bucket features (30–59, 60–89, 90+ days past due) are highly correlated, causing a **suppressor effect** in logistic regression: the 60–89 day coefficient flipped negative, which is economically nonsensical.

**Solution**: Created a severity-weighted composite feature for the logistic model:

```
delinquency_severity = (30-59 days × 1) + (60-89 days × 2) + (90+ days × 3)
```

The weights reflect escalating severity — consistent with how FICO's payment history component is constructed. Tree-based models retain the three original columns since they handle multicollinearity natively.

This resulted in **two feature sets**:
- `X_lr` (10 features): composite delinquency score replaces the three originals
- `X_tree` (12 features): three original delinquency columns, no composite

### 4. Class Imbalance Handling
- **Logistic Regression / Random Forest**: `class_weight='balanced'` — internally reweights each class inversely proportional to its frequency
- **XGBoost**: `scale_pos_weight = 14` (ratio of negatives to positives) — equivalent effect for gradient boosting

### 5. Models

| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | `C=0.1`, `class_weight='balanced'`, `StandardScaler` pipeline |
| Random Forest | `n_estimators=200`, `class_weight='balanced'` |
| XGBoost | `n_estimators=200`, `learning_rate=0.10`, `max_depth=10`, `scale_pos_weight=14` |

### 6. Evaluation
- **AUC-ROC**: primary metric for imbalanced classification
- **Precision-Recall curves + Average Precision**: measures performance across all thresholds, with a random-classifier baseline equal to the default rate (6.7%)
- **Confusion matrix**: normalized by actual class to show true positive / false positive rates
- **Feature importance**: LR odds ratios, RF and XGBoost impurity-based importance — compared side-by-side and in a combined table

---

## Results

See the notebook for full metrics. Summary:

- **XGBoost** achieves the highest AUC-ROC and Average Precision, driven by its ability to capture non-linear delinquency patterns
- **Random Forest** performs comparably to XGBoost with slightly lower Average Precision
- **Logistic Regression** trades some predictive performance for full interpretability — coefficient signs are all economically sensible after the multicollinearity fix
- All three models substantially outperform the random classifier baseline

**Top predictors across all models**: Revolving Utilization, Delinquency Severity / 90+ Days Late, Age

---

## Real-World Credit Risk Connection

Revolving utilization is the most predictive feature — consistent with how card issuers use real-time utilization signals in line management and adverse action decisions.

The delinquency severity composite mirrors FICO's payment history construction logic (35% of the FICO score). The threshold at which the model flags a borrower as high-risk is not fixed at 0.5 — in production it would be set by balancing the cost of a false negative (approving a future defaulter, LGD ≈ 40–70%) against the cost of a false positive (denying a creditworthy applicant, foregone NIM + customer lifetime value).

Logistic regression remains the regulatory baseline at most U.S. consumer lenders because its coefficients map directly to adverse action reason codes required by ECOA. XGBoost in production requires SHAP values for the same purpose.

---

## How to Run

**Requirements**: Python 3.9+, Jupyter

```bash
pip install numpy pandas scikit-learn xgboost seaborn matplotlib openpyxl
```

1. Download the dataset from Kaggle: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
2. Place `cs-training.csv` and `Data_Dictionary.xls` in the `Data/` folder
3. Open `credit_default_classifier.ipynb` and run all cells top to bottom

---

## Project Structure

```
Credit_Default_Classifier/
├── Data/
│   ├── cs_training.csv.zip      # Training data (Kaggle)
│   └── Data_Dictionary.xls      # Feature descriptions
├── credit_default_classifier.ipynb
└── README.md
```

---

## Tech Stack

Python · pandas · scikit-learn · XGBoost · seaborn · matplotlib

---

## Author

Cesar Martin — Finance & credit risk background (Citi), transitioning to Data Science.
