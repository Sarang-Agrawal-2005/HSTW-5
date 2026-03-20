# src/train.py

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from utils import ensure_dir


def get_all_processed():
    directory = "data/processed"
    return sorted(
        [f for f in os.listdir(directory) if f.endswith(".csv")],
        key=lambda f: os.path.getmtime(os.path.join(directory, f))
    )


def main():
    print("🤖 TRAINING STARTED...")

    processed_files = get_all_processed()

    if not processed_files:
        raise FileNotFoundError("❌ No processed CSV files found in data/processed/")

    print("Found processed files:")
    for f in processed_files:
        print("   •", f)

    # Load and merge all files
    dfs = []
    for file in processed_files:
        df = pd.read_csv(f"data/processed/{file}")
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    print(f"📊 Combined dataset shape: {full_df.shape}")

    # Split into X and y
    target_col = full_df.columns[-1]
    X = full_df.drop(columns=[target_col])
    y = full_df[target_col]

    # Train model
    # rf = RandomForestClassifier(
    #     n_estimators=300,
    #     random_state=42,
    #     class_weight="balanced"
    # )
    # rf.fit(X, y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    ensure_dir("models")
    version = get_next_version()
    model_path = f"models/model_v{version}.pkl"
    joblib.dump(model, model_path)

    print(f"✔️ Model trained on ALL files and saved → {model_path}")


def get_next_version():
    ensure_dir("models")
    files = [f for f in os.listdir("models") if f.startswith("model_v")]

    if not files:
        return 1

    nums = [int(f.split("v")[1].split(".")[0]) for f in files]
    return max(nums) + 1


if __name__ == "__main__":
    main()