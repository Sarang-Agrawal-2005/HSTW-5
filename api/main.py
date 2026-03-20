# src/main.py

from fastapi import FastAPI
import joblib
import pandas as pd

MODEL_PATH = "models/best_model.pkl"
ENCODER_PATH = "artifacts/encoders.pkl"
SCALER_PATH = "artifacts/scaler.pkl"

app = FastAPI()

model = joblib.load(MODEL_PATH)
encoders = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)


@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    # Apply label encoders
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    # Scale numeric
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(prob)
    }