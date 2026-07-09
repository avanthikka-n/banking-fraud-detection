"""
clean_data.py
--------------
Cleans and prepares the PaySim fraud detection dataset for modeling.

Usage:
    python src/clean_data.py

Input : data/raw/paysim.csv
Output: data/processed/paysim_cleaned.csv
"""

import pandas as pd
import os

RAW_PATH = "data/raw/paysim.csv"
PROCESSED_PATH = "data/processed/paysim_cleaned.csv"


def load_data(path):
    print(f"Loading data from {path} ...")
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
    return df


def check_data_quality(df):
    print("\n--- Data Quality Check ---")
    print("Missing values per column:")
    print(df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"\nFraud cases: {df['isFraud'].sum():,} out of {len(df):,} "
          f"({df['isFraud'].mean() * 100:.3f}%)")


def clean_data(df):
    print("\n--- Cleaning Data ---")

    # 1. Drop exact duplicate rows, if any
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} duplicate rows.")

    # 2. Extract merchant/customer flag from ID columns, then drop the IDs
    #    (IDs are unique per row and carry no predictive signal on their own)
    df['origIsMerchant'] = df['nameOrig'].str.startswith('M').astype(int)
    df['destIsMerchant'] = df['nameDest'].str.startswith('M').astype(int)
    df = df.drop(columns=['nameOrig', 'nameDest'])
    print("Extracted merchant flags and dropped raw ID columns.")

    # 3. Engineer balance-error features (known strong fraud signal in PaySim)
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    print("Created errorBalanceOrig and errorBalanceDest features.")

    # 4. One-hot encode the transaction type column
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    print("Encoded transaction 'type' column.")

    # 5. Drop isFlaggedFraud (almost entirely zeros, adds little signal)
    if 'isFlaggedFraud' in df.columns:
        df = df.drop(columns=['isFlaggedFraud'])
        print("Dropped isFlaggedFraud column (rarely non-zero).")

    return df


def save_data(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved cleaned dataset to {path}")
    print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")


def main():
    df = load_data(RAW_PATH)
    check_data_quality(df)
    df_clean = clean_data(df)
    save_data(df_clean, PROCESSED_PATH)


if __name__ == "__main__":
    main()