"""FastAPI application for serving a trained Iris classifier."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "iris_classifier.joblib"


class IrisFeatures(BaseModel):
    """Input schema for one Iris flower observation."""

    sepal_length: Annotated[float, Field(gt=0, description="Sepal length in cm")]
    sepal_width: Annotated[float, Field(gt=0, description="Sepal width in cm")]
    petal_length: Annotated[float, Field(gt=0, description="Petal length in cm")]
    petal_width: Annotated[float, Field(gt=0, description="Petal width in cm")]

    def as_vector(self) -> list[float]:
        """Return feature values in the training order."""
        return [self.sepal_length, self.sepal_width, self.petal_length, self.petal_width]


class PredictionResponse(BaseModel):
    prediction: int
    class_name: str
    probabilities: dict[str, float]
    model_version: str


class BatchPredictionRequest(BaseModel):
    items: Annotated[list[IrisFeatures], Field(min_length=1, max_length=100)]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    framework: str
    accuracy: float
    trained_at: str
    feature_names: list[str]
    target_names: list[str]


def load_model_bundle() -> dict:
    """Load the trained model bundle from disk."""
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model artifact not found at {MODEL_PATH}. Run `python scripts/train_model.py`."
        )
    return joblib.load(MODEL_PATH)


model_bundle = load_model_bundle()
app = FastAPI(
    title="Iris ML Model Serving API",
    description="Online inference API for a scikit-learn Iris classifier.",
    version=model_bundle["model_version"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request latency for simple operational monitoring."""
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "method=%s path=%s status_code=%s duration_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


def predict_one(features: IrisFeatures) -> PredictionResponse:
    """Run inference for one validated Iris feature payload."""
    model = model_bundle["model"]
    target_names = model_bundle["target_names"]
    vector = [features.as_vector()]
    prediction = int(model.predict(vector)[0])

    if not hasattr(model, "predict_proba"):
        raise HTTPException(status_code=500, detail="Model does not expose probabilities")

    probability_values = model.predict_proba(vector)[0]
    probabilities = {
        target_names[index]: round(float(value), 6)
        for index, value in enumerate(probability_values)
    }
    return PredictionResponse(
        prediction=prediction,
        class_name=target_names[prediction],
        probabilities=probabilities,
        model_version=model_bundle["model_version"],
    )


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for orchestrators and load balancers."""
    return {"status": "ok", "model_version": model_bundle["model_version"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: IrisFeatures) -> PredictionResponse:
    """Return a realtime prediction for one Iris observation."""
    return predict_one(features)


@app.post("/batch-predict", response_model=BatchPredictionResponse)
def batch_predict(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Return predictions for a bounded batch of Iris observations."""
    predictions = [predict_one(item) for item in request.items]
    return BatchPredictionResponse(
        predictions=predictions,
        count=len(predictions),
        model_version=model_bundle["model_version"],
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    """Return model metadata for governance and debugging."""
    return ModelInfoResponse(
        model_name=model_bundle["model_name"],
        model_version=model_bundle["model_version"],
        framework=model_bundle["framework"],
        accuracy=model_bundle["accuracy"],
        trained_at=model_bundle["trained_at"],
        feature_names=model_bundle["feature_names"],
        target_names=model_bundle["target_names"],
    )
