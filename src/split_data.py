"""
split_data.py
--------------
Splits the cleaned dataset into train/test sets (stratified by fraud label)
and applies SMOTE to the training set only, to handle severe class imbalance.

Usage:
    python src/split_data.py

Input:  data/processed/paysim_cleaned.csv
Output: data/processed/X_train.csv, y_train.csv, X_test.csv, y_test.csv
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT_PATH = "data/processed/paysim_cleaned.csv"
OUTPUT_DIR = "data/processed"

print("Loading cleaned data...")
df = pd.read_csv(INPUT_PATH)
print(f"Shape: {df.shape}")

if 'type_label' in df.columns:
    df = df.drop(columns=['type_label'])
if 'amount_log' in df.columns:
    df = df.drop(columns=['amount_log'])

X = df.drop(columns=['isFraud'])
y = df['isFraud']

print("\nSplitting into train/test sets (80/20, stratified by isFraud)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train shape: {X_train.shape}, fraud rate: {y_train.mean()*100:.3f}%")
print(f"Test shape:  {X_test.shape}, fraud rate: {y_test.mean()*100:.3f}%")

print("\nApplying SMOTE to training set only...")
try:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE - Train shape: {X_train_resampled.shape}, "
          f"fraud rate: {y_train_resampled.mean()*100:.3f}%")
except ImportError:
    print("imbalanced-learn not installed. Run: pip install imbalanced-learn")
    print("Skipping SMOTE - saving original (imbalanced) training set instead.")
    X_train_resampled, y_train_resampled = X_train, y_train

os.makedirs(OUTPUT_DIR, exist_ok=True)
X_train_resampled.to_csv(f"{OUTPUT_DIR}/X_train.csv", index=False)
y_train_resampled.to_csv(f"{OUTPUT_DIR}/y_train.csv", index=False)
X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

print(f"\nSaved train/test files to {OUTPUT_DIR}/")
print("Done! Ready for model training.")