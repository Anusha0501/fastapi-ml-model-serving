# MLOps Notes: ML Model Serving API

## What is model serving?

Model serving is the practice of making a trained machine learning model available to software systems so they can send input data and receive predictions. A serving system usually includes input validation, feature ordering, model loading, inference, response formatting, logging, monitoring, and version management.

In this project, `scripts/train_model.py` trains a classifier on the Iris dataset and writes a model bundle to `models/iris_classifier.joblib`. The FastAPI app loads that artifact and exposes prediction endpoints.

## Offline vs online inference

| Type | Meaning | Common use cases | Tradeoffs |
| --- | --- | --- | --- |
| Offline inference | Predictions are generated later, usually on a schedule or over stored data. | daily churn scoring, weekly risk reports, backfills | high throughput and cheaper compute, but not instant |
| Online inference | Predictions are generated immediately when a request arrives. | fraud checks, recommendations, customer-facing APIs | low latency and fresh decisions, but requires reliable serving infrastructure |

## Batch vs realtime inference

Batch inference processes many records together. It is ideal when latency is not critical and throughput matters. Realtime inference processes one request, or a very small group of records, immediately. It is ideal when another application needs an answer now.

This API supports both styles:

- `POST /predict` is realtime online inference for one Iris record.
- `POST /batch-predict` is online batch inference for up to 100 Iris records in one request.

## Training an Iris classifier

The training script loads the Iris dataset, splits it into train and test sets, trains a `Pipeline` with `StandardScaler` and `RandomForestClassifier`, evaluates accuracy, and saves a model bundle with metadata.

Run:

```bash
python scripts/train_model.py
```

The saved bundle includes:

- model object
- model name
- semantic version
- training timestamp
- evaluation accuracy
- feature names
- class names
- framework name

## Pickle

Pickle is Python's built-in object serialization format. It converts Python objects into bytes and later restores them. Many ML artifacts can be saved with pickle, but pickle files can execute code during loading. Only load pickle artifacts from trusted sources.

## Joblib

Joblib is a persistence library commonly used in the Python ML ecosystem. It is efficient for objects containing large NumPy arrays and is widely used for scikit-learn models. This project uses `joblib.dump` to save the model bundle and `joblib.load` to load it in the API.

## Model versioning

Model versioning records which model artifact is currently deployed. A practical versioning strategy should track:

- model semantic version, such as `1.0.0`
- training code version or Git commit SHA
- dataset version and feature schema
- metrics from validation and production
- deployment environment and release date

The `/model-info` endpoint exposes the model version and metadata so clients and operators can verify which model is serving predictions.

## Monitoring

Production model monitoring should include application and ML-specific signals:

- latency, throughput, and error rate
- CPU, memory, and container restarts
- input feature distributions
- prediction distributions
- data drift and concept drift
- accuracy or business outcome metrics when labels arrive
- model version and endpoint usage

## Logging

Logs help debug incidents and audit behavior. This API includes middleware that logs HTTP method, path, status code, and request duration. In production, prefer structured JSON logs and avoid storing sensitive raw input values unless there is an approved privacy and retention policy.

## Recommended production improvements

- Add authentication and authorization.
- Store models in an artifact registry.
- Add CI tests for schema validation and endpoint behavior.
- Add Prometheus metrics and distributed tracing.
- Validate feature drift with scheduled reports.
- Use canary or shadow deployments before full rollout.
