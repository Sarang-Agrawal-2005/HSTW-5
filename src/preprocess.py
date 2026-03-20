# src/preprocess.py

import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import ensure_dir, get_latest_csv


def main():
    print("🔄 PREPROCESSING STARTED...")

    input_csv = get_latest_csv("data")
    print(f"Detected new CSV file: {input_csv.name}")

    df = pd.read_csv(input_csv)
    df = df.dropna()

    # Identify target column = last column
    target_col = df.columns[-1]

    # Convert target to int if needed
    if df[target_col].dtype == float:
        df[target_col] = df[target_col].astype(int)

    # --- STORE ENCODERS ---
    encoders = {}

    # Encode categorical columns except target
    for col in df.select_dtypes(include=['object', 'string']).columns:
        if col == target_col:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Scale numeric columns except target
    numeric_cols = [
        col for col in df.select_dtypes(include=['float64', 'int64']).columns
        if col != target_col
    ]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save encoders and scaler
    ensure_dir("artifacts")
    joblib.dump(encoders, "artifacts/encoders.pkl")
    joblib.dump(scaler, "artifacts/scaler.pkl")

    # Save processed CSV
    ensure_dir("data/processed")
    output_path = f"data/processed/{input_csv.stem}_preprocessed.csv"
    df.to_csv(output_path, index=False)

    print(f"✔️ Preprocessing complete → {output_path}")
    print("✔️ Saved encoders & scaler → artifacts/")


if __name__ == "__main__":
    main()