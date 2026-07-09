"""
train_model.py
---------------
Trains fraud detection models on the prepared train/test data.

Trains two models for comparison:
1. Logistic Regression (baseline)
2. Random Forest (stronger model)

Evaluates both using Precision, Recall, F1, and ROC-AUC
(NOT accuracy, which is meaningless on this imbalanced dataset).

Usage:
    python src/train_model.py

Input:  data/processed/X_train.csv, y_train.csv, X_test.csv, y_test.csv
Output: models/logistic_regression.pkl, models/random_forest.pkl
"""

import pandas as pd
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

DATA_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading train/test data...")
X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

print(f"Train shape: {X_train.shape}")
print(f"Test shape:  {X_test.shape}")


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n--- {name} Results ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))

    return {'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}


# ---- Model 1: Logistic Regression (baseline) ----
print("\n" + "=" * 50)
print("Training Logistic Regression (baseline)...")
print("=" * 50)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
log_reg_results = evaluate_model("Logistic Regression", log_reg, X_test, y_test)

joblib.dump(log_reg, f"{MODEL_DIR}/logistic_regression.pkl")
print(f"\nSaved model: {MODEL_DIR}/logistic_regression.pkl")

# ---- Model 2: Random Forest (stronger model) ----
print("\n" + "=" * 50)
print("Training Random Forest...")
print("=" * 50)
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_results = evaluate_model("Random Forest", rf, X_test, y_test)

joblib.dump(rf, f"{MODEL_DIR}/random_forest.pkl")
print(f"\nSaved model: {MODEL_DIR}/random_forest.pkl")

# ---- Summary comparison ----
print("\n" + "=" * 50)
print("Summary Comparison")
print("=" * 50)
summary = pd.DataFrame({
    'Logistic Regression': log_reg_results,
    'Random Forest': rf_results
})
print(summary)

print("\nDone! Models saved to the 'models' folder.")