# Banking Fraud Detection

A machine learning project to detect fraudulent financial transactions using the PaySim synthetic dataset — a simulation of mobile money transfers based on real transaction patterns from an African mobile money service.

## Problem

Fraud detection is a severe class imbalance problem: in this dataset, only **0.13%** of transactions (8,213 out of 6.36 million) are fraudulent. A naive model could achieve 99.8% accuracy by never flagging fraud at all — so this project focuses on the metrics that actually matter for fraud detection: **Precision, Recall, F1, and ROC-AUC**, not accuracy.

## Dataset

- **Source:** [PaySim - Synthetic Financial Datasets For Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle)
- **Size:** 6,362,620 transactions, 11 original columns
- **Not included in this repository** due to size (~490MB) — see [Setup](#setup) below for how to obtain it

## Project Pipeline

| Stage | Script | Description |
|---|---|---|
| 1. Data Cleaning | `src/clean_data.py` | Drops raw ID columns, engineers balance-error features, encodes transaction type |
| 2. Exploratory Data Analysis | `src/eda.py` | Analyzes fraud patterns by transaction type, amount, and time |
| 3. Train/Test Split | `src/split_data.py` | Stratified 80/20 split, SMOTE oversampling on training data only |
| 4. Model Training | `src/train_model.py` | Trains Logistic Regression (baseline) and Random Forest |
| 5. Feature Importance | `src/feature_importance.py` | Analyzes which features drive fraud predictions |
| 6. Prediction | `src/predict.py` | Demonstrates predictions on new transactions |

## Key Findings

- **Fraud only occurs in `TRANSFER` and `CASH_OUT` transactions** — zero fraud cases were found in `CASH_IN`, `DEBIT`, or `PAYMENT` transactions across all 6.36 million records.
- The engineered feature `errorBalanceOrig` (the discrepancy between expected and actual balance after a transaction) is the single strongest predictor of fraud, accounting for 38% of the Random Forest's decision-making importance — nearly 3x more influential than any other feature.
- Class imbalance is severe (0.13% fraud rate) and was addressed using SMOTE oversampling, applied only to the training set to avoid inflating test results.

## Model Results

| Metric | Logistic Regression (baseline) | Random Forest |
|---|---|---|
| Precision | 0.028 | **0.980** |
| Recall | 0.940 | **0.998** |
| F1 Score | 0.054 | **0.989** |
| ROC-AUC | 0.989 | **0.9996** |

The baseline Logistic Regression model catches most fraud (94% recall) but at the cost of an unusably high false-positive rate (only 2.8% precision — roughly 35 false alarms for every real fraud case caught). Random Forest dramatically improves on this, correctly identifying 1,639 of 1,643 fraud cases in the test set with only 33 false positives out of 1.27 million transactions.

**Note:** These results are strong in part because PaySim is a synthetic dataset with clean, consistent fraud patterns. Real-world fraud detection typically involves noisier data and more subtle fraud behavior, so results here should be understood as a strong proof-of-concept rather than a guarantee of real-world performance.

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/avanthikka-n/banking-fraud-detection.git
cd banking-fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download `paysim.csv` from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it at:
```
data/raw/paysim.csv
```

### 4. Run the pipeline in order
```bash
python src/clean_data.py
python src/eda.py
python src/split_data.py
python src/train_model.py
python src/feature_importance.py
python src/predict.py
```

## Project Structure
```
banking-fraud-detection/
├── data/
│   ├── raw/            # Original dataset (not tracked in git)
│   └── processed/      # Cleaned and split data (not tracked in git)
├── models/              # Trained model files (not tracked in git)
├── outputs/              # Charts and analysis outputs
├── src/                 # All pipeline scripts
├── notebooks/            # Jupyter notebooks (optional exploration)
├── tests/                # Unit tests
└── README.md
```

## Future Improvements
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) on the Random Forest model
- Try gradient boosting models (XGBoost, LightGBM) for comparison
- Deploy as a live API (Flask/FastAPI) for real-time transaction scoring
- Test on real-world, noisier fraud data to validate generalization