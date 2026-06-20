# Interview Guide: FastAPI and MLOps

## FastAPI interview questions

1. What is FastAPI? It is a high-performance Python web framework built on Starlette and Pydantic.
2. Why is FastAPI popular for APIs? It provides type-driven validation, automatic OpenAPI docs, async support, and strong developer ergonomics.
3. What is Pydantic used for in FastAPI? It validates, parses, and documents request and response models.
4. What is an endpoint? It is a route plus HTTP method that handles a client request.
5. What is the difference between `GET` and `POST`? `GET` retrieves data; `POST` submits data to create or process something.
6. How do you define a request body? Create a Pydantic model and use it as a function parameter.
7. How do you define a response schema? Set `response_model` on the route decorator.
8. What is automatic API documentation? FastAPI generates Swagger UI and ReDoc from OpenAPI metadata.
9. Where is Swagger UI available by default? At `/docs`.
10. Where is ReDoc available by default? At `/redoc`.
11. What is ASGI? It is the asynchronous server gateway interface used by modern Python web frameworks.
12. What is Uvicorn? It is an ASGI server commonly used to run FastAPI apps.
13. What is dependency injection in FastAPI? A system for declaring reusable dependencies with `Depends`.
14. How do you validate numeric ranges? Use Pydantic `Field` constraints such as `gt`, `ge`, `lt`, and `le`.
15. What status code means validation failed? FastAPI usually returns `422 Unprocessable Entity`.
16. How do you raise an API error? Raise `HTTPException` with a status code and detail.
17. What is middleware? Code that runs before and after request handling.
18. Why add logging middleware? To capture request paths, status codes, and latency.
19. What is async in FastAPI? It allows non-blocking request handlers with `async def`.
20. Should CPU-heavy ML inference be async? Not necessarily; CPU-heavy work can block and may need worker processes or task queues.
21. What is OpenAPI? A standard machine-readable API specification.
22. How does FastAPI use type hints? It uses them for validation, serialization, and docs.
23. How do you handle query parameters? Add typed function parameters not declared as path or body models.
24. How do you handle path parameters? Add variables in the route path and matching typed function parameters.
25. How do you handle headers? Use FastAPI's `Header` dependency.
26. How do you handle authentication? Use dependencies, OAuth2 helpers, API keys, or external identity providers.
27. What is CORS? Browser security policy controlling cross-origin requests.
28. How do you enable CORS? Add `CORSMiddleware` with allowed origins and methods.
29. What is a health check endpoint? A lightweight endpoint used to verify service availability.
30. What is readiness? Whether the service can safely receive traffic.
31. What is liveness? Whether the process is alive and should not be restarted.
32. How can you version an API? Use URL prefixes, headers, or separate routers.
33. What is an APIRouter? A way to group related routes.
34. How do you test FastAPI? Use `TestClient` or async HTTP clients.
35. What is response serialization? Converting Python objects into JSON-compatible responses.
36. How do you avoid leaking internal fields? Use response models and explicit schemas.
37. What is idempotency? Repeating a request has the same effect as making it once.
38. Are prediction requests idempotent? Usually yes if the model and inputs are unchanged.
39. How do you run FastAPI in production? Use Uvicorn or Gunicorn with Uvicorn workers behind a reverse proxy.
40. What controls concurrency? Event loop, worker count, thread pools, and process count.
41. What is a background task? Work scheduled after returning a response.
42. Should long batch jobs run inside an HTTP request? Usually no; use queues for long-running jobs.
43. How does FastAPI handle JSON? It parses JSON into typed Python objects and serializes responses back to JSON.
44. What is schema evolution? Changing request or response schemas without breaking clients.
45. Why cap batch size? To protect latency, memory, and service availability.
46. How do you document examples? Use Pydantic schema examples or route metadata.
47. What is observability? Logs, metrics, and traces that explain service behavior.
48. What is a 500 error? An unexpected server-side failure.
49. Why separate training code from serving code? Training is offline and serving must be stable, fast, and minimal.
50. What makes a good ML API? Clear schema, low latency, versioned models, monitoring, and predictable errors.

## MLOps interview questions

1. What is MLOps? It is the discipline of building, deploying, monitoring, and governing ML systems reliably.
2. What is model serving? Exposing a trained model so applications can request predictions.
3. What is offline inference? Generating predictions on stored data outside the request path.
4. What is online inference? Generating predictions immediately for live requests.
5. What is batch inference? Processing many records at once.
6. What is realtime inference? Returning predictions with low latency for immediate decisions.
7. What is feature engineering? Transforming raw data into model-ready features.
8. What is a training pipeline? Automated steps for data loading, preprocessing, training, evaluation, and artifact creation.
9. What is an inference pipeline? Runtime steps for validation, preprocessing, prediction, and postprocessing.
10. What is training-serving skew? Differences between training-time and serving-time feature logic.
11. How do you reduce training-serving skew? Reuse transformations, version schemas, and validate production data.
12. What is a model artifact? A saved model file plus metadata needed for deployment.
13. What is an artifact registry? A system for storing and versioning model artifacts.
14. What is model versioning? Tracking deployed model versions and their lineage.
15. What is data versioning? Tracking datasets used to train and evaluate models.
16. What is experiment tracking? Recording parameters, metrics, code versions, and artifacts for training runs.
17. What is model registry? A catalog of model versions and promotion stages.
18. What is CI for ML? Automated checks for code, tests, schemas, and model training workflows.
19. What is CD for ML? Automated deployment of validated models and services.
20. What is model validation? Checking model metrics, robustness, and constraints before release.
21. What is drift? A change in production data or relationships compared with training.
22. What is data drift? Input feature distributions change.
23. What is concept drift? The relationship between inputs and labels changes.
24. What is prediction drift? Output prediction distributions change.
25. How do you monitor models? Track service metrics, data distributions, predictions, and delayed labels.
26. What is latency? Time taken to return a prediction.
27. What is throughput? Number of predictions handled per unit time.
28. What is an SLA? A promised service level such as uptime or latency.
29. What is an SLO? An internal reliability objective.
30. What is a canary deployment? Releasing to a small percentage of traffic first.
31. What is a shadow deployment? Running a new model on copied traffic without affecting users.
32. What is A/B testing? Comparing model variants on live traffic or users.
33. What is rollback? Returning to a previous known-good model or service version.
34. What is reproducibility? Ability to recreate a training result from code, data, environment, and parameters.
35. Why pin dependencies? To reduce environment drift and improve reproducibility.
36. What is Pickle? Python object serialization; unsafe for untrusted files.
37. What is Joblib? A persistence tool often used for scikit-learn models and NumPy-heavy objects.
38. Why avoid loading untrusted model files? Deserialization can execute malicious code.
39. What is model governance? Policies for approval, auditability, security, and compliance.
40. What is explainability? Understanding why a model made a prediction.
41. What is bias monitoring? Measuring whether model behavior is unfair across groups.
42. What is a feature store? A platform for consistent feature computation and retrieval.
43. What is online feature serving? Fetching low-latency features for realtime inference.
44. What is offline feature storage? Historical feature data used for training and batch scoring.
45. What is autoscaling? Adjusting compute capacity based on demand.
46. What is containerization? Packaging code and dependencies into portable runtime images.
47. What belongs in model logs? Request metadata, model version, latency, and safe diagnostic details.
48. What should not be logged? Secrets, credentials, and sensitive personal data unless explicitly approved.
49. How do you handle model failures? Return safe errors, alert operators, and support rollback.
50. What is the goal of MLOps? Reliable, repeatable, observable, and governed ML delivery.
