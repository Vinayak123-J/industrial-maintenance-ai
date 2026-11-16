from pathlib import Path
import joblib
import pandas as pd
from .config import MODEL_PATH

_model = None
_metrics = {}

def load_model():
    global _model, _metrics
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)
        _model = artifact["model"]
        _metrics = artifact.get("metrics", {})
    return _model

def get_model():
    return _model or load_model()

def get_metrics():
    return _metrics

def predict(payload: dict):
    model = get_model()
    if model is None:
        raise RuntimeError("Model not found. Run: python scripts/train_model.py")
    row = pd.DataFrame([{
        "Type": payload["type"],
        "Air temperature [K]": payload["air_temperature_k"],
        "Process temperature [K]": payload["process_temperature_k"],
        "Rotational speed [rpm]": payload["rotational_speed_rpm"],
        "Torque [Nm]": payload["torque_nm"],
        "Tool wear [min]": payload["tool_wear_min"],
    }])
    probabilities = model.predict_proba(row)[0]
    probability = float(probabilities[1])
    failed = probability >= 0.5
    risk = "High" if probability >= 0.65 else "Medium" if probability >= 0.30 else "Low"
    if risk == "High":
        recommendation = "Schedule an inspection promptly and review operating conditions before continued operation."
    elif risk == "Medium":
        recommendation = "Increase monitoring frequency and plan preventive inspection."
    else:
        recommendation = "Continue normal monitoring and follow the planned maintenance schedule."
    return probability, failed, risk, recommendation
