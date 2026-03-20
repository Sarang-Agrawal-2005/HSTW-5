# src/utils.py

import json
import joblib
from pathlib import Path
import re


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def get_latest_csv(folder="data"):
    folder_path = Path(folder)
    csv_files = list(folder_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in data/ folder.")

    # Extract number from filename e.g. "3.csv" → 3
    def extract_number(path):
        match = re.match(r"(\d+)\.csv$", path.name)
        if match:
            return int(match.group(1))
        return -1  # ignore non-numbered CSVs

    # Choose file with largest number
    latest = max(csv_files, key=extract_number)

    if extract_number(latest) == -1:
        raise ValueError("CSV files must be named like '1.csv', '2.csv', ...")

    return latest