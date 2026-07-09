"""
eda.py
------
Explores the cleaned PaySim dataset to understand fraud patterns.

Usage:
    python src/eda.py

Input: data/processed/paysim_cleaned.csv
Output: charts saved to outputs/ folder
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

INPUT_PATH = "data/processed/paysim_cleaned.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading cleaned data...")
df = pd.read_csv(INPUT_PATH)
print(f"Shape: {df.shape}")

# 1. Overall fraud rate
print("\n--- Fraud Rate ---")
fraud_counts = df['isFraud'].value_counts()
fraud_pct = df['isFraud'].mean() * 100
print(f"Non-fraud: {fraud_counts[0]:,}")
print(f"Fraud:     {fraud_counts[1]:,}")
print(f"Fraud rate: {fraud_pct:.3f}%")

plt.figure()
fraud_counts.plot(kind='bar', color=['steelblue', 'crimson'])
plt.title('Transaction Count: Fraud vs Non-Fraud')
plt.xticks([0, 1], ['Non-Fraud', 'Fraud'], rotation=0)
plt.ylabel('Count')
plt.savefig(f"{OUTPUT_DIR}/fraud_count.png")
print(f"Saved chart: {OUTPUT_DIR}/fraud_count.png")

# 2. Fraud by transaction type
print("\n--- Fraud by Transaction Type ---")
type_cols = [c for c in df.columns if c.startswith('type_')]

def get_type(row):
    for c in type_cols:
        if row[c] == 1:
            return c.replace('type_', '')
    return 'CASH_IN'

df['type_label'] = df.apply(get_type, axis=1)

fraud_by_type = df.groupby('type_label')['isFraud'].agg(['sum', 'count', 'mean'])
fraud_by_type.columns = ['fraud_count', 'total_count', 'fraud_rate']
print(fraud_by_type.sort_values('fraud_count', ascending=False))

plt.figure()
fraud_by_type['fraud_count'].sort_values(ascending=False).plot(kind='bar', color='crimson')
plt.title('Fraud Count by Transaction Type')
plt.ylabel('Number of Fraud Cases')
plt.xlabel('Transaction Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fraud_by_type.png")
print(f"Saved chart: {OUTPUT_DIR}/fraud_by_type.png")

# 3. Amount comparison
print("\n--- Amount: Fraud vs Non-Fraud ---")
print(df.groupby('isFraud')['amount'].describe())

plt.figure()
df.boxplot(column='amount', by='isFraud')
plt.title('Amount Distribution by Fraud Status')
plt.suptitle('')
plt.xlabel('isFraud (0=No, 1=Yes)')
plt.ylabel('Amount')
plt.savefig(f"{OUTPUT_DIR}/amount_by_fraud.png")
print(f"Saved chart: {OUTPUT_DIR}/amount_by_fraud.png")

print("\nDone! Check the 'outputs' folder for saved charts.")