# src/save_artifacts.py

from utils import load_json, load_model, save_model
from pathlib import Path


def main():
    print("🏆 SELECTING BEST MODEL AMONG ALL VERSIONS...")

    metrics = load_json("models/metrics.json")

    # remove 'old_best' temporary entry if present
    clean_metrics = {k: v for k, v in metrics.items() if k != "old_best"}

    best_model_name = max(clean_metrics, key=clean_metrics.get)

    print(f"Best model identified = {best_model_name}")

    full_path = Path("models") / f"{best_model_name}.pkl"
    best_model = load_model(full_path)

    save_model(best_model, "models/best_model.pkl")

    print("✔️ best_model.pkl updated.")


if __name__ == "__main__":
    main()