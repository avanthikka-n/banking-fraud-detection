"""
predict.py
-----------
Loads the trained Random Forest model and makes fraud predictions
on new transaction data.

Usage:
    python src/predict.py

This script demonstrates prediction on a few sample transactions
from the test set. Modify the `sample_transaction` section to test
your own custom transaction values.
"""

import pandas as pd
import joblib

MODEL_PATH = "models/random_forest.pkl"
X_TEST_PATH = "data/processed/X_test.csv"
Y_TEST_PATH = "data/processed/y_test.csv"

print("Loading trained model...")
model = joblib.load(MODEL_PATH)

print("Loading a few sample transactions from the test set...")
X_test = pd.read_csv(X_TEST_PATH)
y_test = pd.read_csv(Y_TEST_PATH).squeeze()

sample = X_test.sample(5, random_state=1)
sample_actual = y_test.loc[sample.index]

predictions = model.predict(sample)
probabilities = model.predict_proba(sample)[:, 1]

print("\n--- Sample Predictions ---")
results = pd.DataFrame({
    'Actual': sample_actual.values,
    'Predicted': predictions,
    'Fraud_Probability': probabilities.round(4)
})
print(results.to_string(index=False))

print("\nActual: 1 = Fraud, 0 = Not Fraud")
print("Predicted: 1 = Model flags as Fraud, 0 = Model says Not Fraud")
print("Fraud_Probability: model's confidence that this transaction is fraud (0 to 1)")

print("\nDone!")