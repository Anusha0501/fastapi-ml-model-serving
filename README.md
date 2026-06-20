# FastAPI ML Model Serving

A teaching-focused FastAPI project that trains a scikit-learn Iris classifier and serves it with online prediction endpoints.

## What you will build

- `POST /predict` for one realtime Iris prediction.
- `POST /batch-predict` for bounded batch inference.
- `GET /model-info` for model metadata and version details.
- `GET /health` for service health checks.
- Training script that saves a versioned model bundle with Joblib.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_model.py
uvicorn app.main:app --reload
```

Open the API docs at <http://127.0.0.1:8000/docs>.

## Example requests

### Realtime prediction

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```

### Batch prediction

```bash
curl -X POST http://127.0.0.1:8000/batch-predict \
  -H 'Content-Type: application/json' \
  -d '{"items":[{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2},{"sepal_length":6.2,"sepal_width":3.4,"petal_length":5.4,"petal_width":2.3}]}'
```

### Model information

```bash
curl http://127.0.0.1:8000/model-info
```

## Project structure

```text
app/main.py              FastAPI service and schemas
scripts/train_model.py   Iris training pipeline and model persistence
models/                  Generated model artifacts
MLOPS_NOTES.md           MLOps teaching notes
INTERVIEW_GUIDE.md       Interview questions and answers
```
