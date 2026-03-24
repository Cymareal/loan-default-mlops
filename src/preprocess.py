import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

RAW_PATH = "data/raw/Loan_default.csv"
PROCESSED_PATH = "data/processed/features.csv"

def preprocess():
    df = pd.read_csv(RAW_PATH)
    print(f"Raw data shape: {df.shape}")

    df = df.drop(columns=["LoanID"], errors="ignore")

    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    df['LoanToIncome']       = df['LoanAmount'] / (df['Income'] + 1)
    df['CreditLinesPerYear'] = df['NumCreditLines'] / ((df['MonthsEmployed'] / 12) + 1)
    df['RiskInteraction']    = df['InterestRate'] * df['DTIRatio']

    df = df.drop(columns=['NumCreditLines', 'LoanAmount', 'MonthsEmployed'], errors="ignore")


    df = df.dropna(subset=["Default"])

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed shape: {df.shape}")
    print(f"Features: {df.columns.tolist()}")
    print(f"Saved to {PROCESSED_PATH}")

if __name__ == "__main__":
    preprocess()