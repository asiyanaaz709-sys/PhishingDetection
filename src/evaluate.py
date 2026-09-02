"""Evaluate the persisted phishing classifier and plot feature importance."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from .feature_extraction import FEATURE_NAMES
    from .train_model import DATA_PATH, MODEL_DIR, load_dataset
except ImportError:
    from feature_extraction import FEATURE_NAMES
    from train_model import DATA_PATH, MODEL_DIR, load_dataset


def evaluate_model() -> dict[str, float | list[list[int]]]:
    features, labels = load_dataset(DATA_PATH)
    _, x_test, _, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    model = joblib.load(MODEL_DIR / "model.pkl")
    predictions = model.predict(scaler.transform(x_test))
    probabilities = model.predict_proba(scaler.transform(x_test))[:, 1]
    metrics: dict[str, float | list[list[int]]] = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }

    importance = model.feature_importances_
    order = importance.argsort()
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh([FEATURE_NAMES[index] for index in order], importance[order], color="#0f766e")
    axis.set_title("XGBoost Feature Importance")
    axis.set_xlabel("Importance")
    figure.tight_layout()
    figure.savefig(MODEL_DIR / "feature_importance.png", dpi=160)
    plt.close(figure)

    for name, value in metrics.items():
        print(f"{name}: {value}")
    print(f"feature_importance_chart: {MODEL_DIR / 'feature_importance.png'}")
    return metrics


if __name__ == "__main__":
    evaluate_model()