# src/evaluate.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
from utils import load_model, save_json


def main():
    print("📊 EVALUATION STARTED...")

    # Load ALL processed data (same as training)
    processed_files = sorted(Path("data/processed").glob("*.csv"))
    dfs = [pd.read_csv(f) for f in processed_files]
    df = pd.concat(dfs, ignore_index=True)

    print(f"📄 Evaluation dataset shape: {df.shape}")

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Create evaluation split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.99,
        random_state=42,
        stratify=y
    )

    metrics = {}

    # Evaluate ALL model versions
    model_files = sorted(Path("models").glob("model_v*.pkl"))

    print(f"🧪 Found {len(model_files)} model versions to evaluate")

    for model_path in model_files:
        model = load_model(model_path)
        acc = accuracy_score(y_test, model.predict(X_test))
        metrics[model_path.stem] = acc
        print(f"  → {model_path.stem} accuracy: {acc}")

    # Save metrics
    save_json("models/metrics.json", metrics)

    print("✔️ Evaluation complete → metrics.json updated with latest accuracies.")


if __name__ == "__main__":
    main()