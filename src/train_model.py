"""Train and persist the phishing URL classifier."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.io import arff
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

try:
    from .feature_extraction import FEATURE_NAMES
except ImportError:
    from feature_extraction import FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "uci-phishing-websites" / "Training Dataset.arff"
MODEL_DIR = ROOT / "models"


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the official UCI ARFF and map its labels to phishing=1."""
    records, _ = arff.loadarff(path)
    frame = pd.DataFrame(records)
    frame.columns = [column.decode() if isinstance(column, bytes) else column for column in frame.columns]
    frame = frame.astype(float)
    features = frame[FEATURE_NAMES]
    labels = (frame["Result"] == -1).astype(int)
    return features, labels


def train_model() -> dict[str, float | list[list[int]]]:
    features, labels = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    search = GridSearchCV(
        estimator,
        {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.9],
            "colsample_bytree": [0.9],
        },
        scoring="f1",
        cv=3,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(x_train_scaled, y_train)

    predictions = search.predict(x_test_scaled)
    probabilities = search.predict_proba(x_test_scaled)[:, 1]
    metrics: dict[str, float | list[list[int]]] = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(search.best_estimator_, MODEL_DIR / "model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    (MODEL_DIR / "feature_names.json").write_text(json.dumps(FEATURE_NAMES, indent=2), encoding="utf-8")

    print(f"Best parameters: {search.best_params_}")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    return metrics


if __name__ == "__main__":
    train_model()