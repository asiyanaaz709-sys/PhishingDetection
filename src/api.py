"""FastAPI service for phishing URL prediction and model metrics."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .feature_extraction import FEATURE_NAMES, extract_features, feature_vector
from .train_model import DATA_PATH, load_dataset


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
model = joblib.load(MODEL_DIR / "model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")


class PredictionRequest(BaseModel):
    url: str = Field(min_length=1)
    html: str | None = None


app = FastAPI(title="Phishing Website Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _calculate_metrics() -> dict[str, float | list[list[int]]]:
    features, labels = load_dataset(DATA_PATH)
    _, x_test, _, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    scaled_features = scaler.transform(x_test)
    predictions = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


@app.get("/metrics")
def metrics() -> dict[str, float | list[list[int]]]:
    return _calculate_metrics()


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    try:
        features = extract_features(request.url, request.html)
        values = scaler.transform(pd.DataFrame([feature_vector(request.url, request.html)], columns=FEATURE_NAMES))
        probability = float(model.predict_proba(values)[0, 1])
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "url": request.url,
        "prediction": "phishing" if probability >= 0.5 else "legitimate",
        "phishing_probability": probability,
        "confidence": max(probability, 1 - probability),
        "features": features,
        "feature_names": FEATURE_NAMES,
    }