"""
feature_importance.py
----------------------
Analyzes which features the Random Forest model relies on most
to detect fraud. Useful for explainability and your project write-up.

Usage:
    python src/feature_importance.py

Input:  models/random_forest.pkl, data/processed/X_train.csv
Output: outputs/feature_importance.png, prints ranked list to terminal
"""

import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

MODEL_PATH = "models/random_forest.pkl"
X_TRAIN_PATH = "data/processed/X_train.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading model and feature names...")
model = joblib.load(MODEL_PATH)

# Only need the column headers, not the full 1GB file
X_train_columns = pd.read_csv(X_TRAIN_PATH, nrows=0).columns.tolist()

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train_columns,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n--- Feature Importance Ranking ---")
print(feature_importance_df.to_string(index=False))

# Plot top 10 features
plt.figure(figsize=(10, 6))
top_features = feature_importance_df.head(10)
plt.barh(top_features['feature'][::-1], top_features['importance'][::-1], color='steelblue')
plt.title('Top 10 Most Important Features (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance.png")
print(f"\nSaved chart: {OUTPUT_DIR}/feature_importance.png")

print("\nDone!")