"""Train and persist an Iris classifier for the FastAPI serving app."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "iris_classifier.joblib"
MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42


def train() -> dict:
    """Train a RandomForest Iris classifier and save the model bundle."""
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(x_test))

    bundle = {
        "model": pipeline,
        "model_name": "iris-random-forest-classifier",
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "accuracy": float(accuracy),
        "feature_names": list(iris.feature_names),
        "target_names": list(iris.target_names),
        "framework": "scikit-learn",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


if __name__ == "__main__":
    metadata = train()
    print(f"Saved model to {MODEL_PATH}")
    print(f"Version: {metadata['model_version']}")
    print(f"Accuracy: {metadata['accuracy']:.4f}")
