# Production ML Pipeline with MLflow, DVC, FastAPI, and Automated Retraining

This project is a complete local MLOps pipeline that covers the full lifecycle:

- Data versioning with DVC
- Data processing and validation
- Experiment tracking with MLflow
- Model registry with Production-stage serving
- FastAPI inference API
- Drift detection report generation
- Conditional retraining trigger
- Docker Compose orchestration with service health checks
- Test suite with coverage above 60%

The repository is designed for practical submission and reproducibility. You can clone it, run a few commands, and validate each requirement end to end.

## 1) Architecture Overview

The stack runs three services:

- db: PostgreSQL backend for MLflow metadata
- mlflow-server: MLflow Tracking + Model Registry UI
- api: FastAPI prediction service loading the model from MLflow Production stage

Pipeline flow:

1. Raw data is tracked in DVC.
2. Processing creates feature-enriched training data.
3. Validation writes a JSON result file.
4. Training runs multiple experiments, logs params/metrics/artifacts to MLflow.
5. Best run is registered as production-model and promoted to Production.
6. API serves predictions from the Production model.
7. Monitoring generates a drift HTML report.
8. Retraining is triggered only when drift_detected.flag exists.

## 2) Repository Layout

The structure below focuses on the parts used during evaluation and day-to-day development:

```text
building-mlops-pipeline/
├── data/
│   ├── raw/
│   │   ├── data.csv
│   │   └── data.csv.dvc
│   └── processed/
│       └── processed.csv
├── src/
│   ├── api/
│   │   └── main.py
│   ├── __init__.py
│   ├── config.py
│   ├── data_processing.py
│   ├── monitoring.py
│   ├── retrain.py
│   └── train.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_data_processing.py
│   ├── test_monitoring_retrain.py
│   └── test_train.py
├── reports/
│   ├── data_drift_report.html
│   └── validation_result.json
├── gx/
├── notebooks/
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.mlflow
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── README.md
```

Local runtime folders like `mlruns/`, `postgres_data/`, and caches are environment artifacts and are not part of the core source layout.

## 3) Prerequisites

- Python 3.10+
- Docker Desktop (or Docker Engine + Compose plugin)
- Git
- DVC CLI

Optional but recommended:

- A clean virtual environment for local testing

## 4) Quick Start (Clone to Running System)

### 4.1 Clone and enter project

```bash
git clone <your-repo-url>
cd building-mlops-pipeline
```

### 4.2 Create environment file

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

### 4.3 Start full stack

```bash
docker compose up --build -d
```

### 4.4 Verify services

```bash
docker compose ps
curl http://localhost:5000
curl http://localhost:8000/health
```

Expected:

- MLflow UI is reachable on port 5000.
- API health returns:

```json
{"status":"ok"}
```

## 5) Environment Variables

The .env.example template documents all required values:

- POSTGRES_USER
- POSTGRES_PASSWORD
- POSTGRES_DB
- MLFLOW_TRACKING_URI
- MODEL_NAME
- EXPERIMENT_NAME
- RAW_DATA_PATH
- DATA_PATH
- VALIDATION_PATH
- REPORT_PATH
- DRIFT_FLAG_FILE
- DRIFT_THRESHOLD
- RETRAIN_IN_DOCKER

Default model and experiment names used by this project:

- MODEL_NAME=production-model
- EXPERIMENT_NAME=churn-prediction

## 6) Data Versioning and Pipeline

Raw data is DVC-tracked at data/raw/data.csv.dvc.

Run and validate the pipeline:

```bash
dvc repro
dvc status
```

DVC stages in dvc.yaml:

- process_data: generates data/processed/processed.csv
- validate_data: generates reports/validation_result.json

## 7) Training, MLflow Logging, and Model Registry

Run training from inside the API container so networking to MLflow service is always correct:

```bash
docker compose exec -T api python src/train.py
```

What training does:

- Creates or uses experiment churn-prediction
- Executes at least 3 runs (C in [0.1, 1.0, 10.0])
- Logs parameter C
- Logs metric accuracy
- Logs model artifacts
- Registers best model to production-model
- Transitions the new version to Production

## 8) API Usage

### 8.1 Health endpoint

```bash
curl http://localhost:8000/health
```

Response:

```json
{"status":"ok"}
```

### 8.2 Prediction endpoint

```bash
curl -X POST http://localhost:8000/predict \
	-H "Content-Type: application/json" \
	-d '{"feature1": 0.5, "feature2": 1.2}'
```

Response shape:

```json
{"prediction": 0}
```

Invalid payloads return HTTP 422, for example:

```bash
curl -X POST http://localhost:8000/predict \
	-H "Content-Type: application/json" \
	-d '{"feature1": "not-a-float"}'
```

## 9) Drift Detection and Retraining

Generate drift report:

```bash
python src/monitoring.py
```

Output:

- reports/data_drift_report.html
- drift_detected.flag (only when drift is detected)

Run retraining trigger:

```bash
python src/retrain.py
```

Behavior:

- If drift_detected.flag does not exist: retraining is skipped.
- If drift_detected.flag exists: training executes and the flag is removed.

By default retraining runs inside the API container using:

- docker compose exec -T api python src/train.py

## 10) Testing and Coverage

Run tests:

```bash
pytest --cov=src --cov-report=term-missing
```

Current result in this repository:

- 10 tests passed
- Total coverage: 71%

This satisfies the minimum 60% coverage requirement.

## 11) Requirement-by-Requirement Verification Commands

1. Compose stack and health checks

```bash
docker compose up --build -d
docker compose ps
```

2. Environment template present

```bash
ls .env.example
```

3. DVC data + stage definition

```bash
ls data/raw/data.csv.dvc
cat dvc.yaml
dvc status
```

4. Validation result contains success boolean

```bash
python src/data_processing.py --step validate
python -c "import json; d=json.load(open('reports/validation_result.json')); print(type(d.get('success')).__name__, d.get('success'))"
```

5. MLflow UI reachable

```bash
curl -i http://localhost:5000
```

6. Experiment and runs logged

```bash
docker compose exec -T api python src/train.py
```

7. Model registry Production version

Use MLflow UI or client check after training.

8. API health

```bash
curl -i http://localhost:8000/health
```

9. Valid prediction request

```bash
curl -i -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"feature1":1.0,"feature2":2.0}'
```

10. Invalid prediction request returns 422

```bash
curl -i -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"feature1":"bad"}'
```

11. Drift report generation

```bash
python src/monitoring.py
ls reports/data_drift_report.html
```

12. Retraining creates a new model version

```bash
python src/retrain.py
```

13. Retraining trigger condition via flag file

```bash
rm -f drift_detected.flag
python src/retrain.py
touch drift_detected.flag
python src/retrain.py
```

PowerShell equivalent:

```powershell
Remove-Item drift_detected.flag -ErrorAction SilentlyContinue
python src/retrain.py
New-Item drift_detected.flag -ItemType File
python src/retrain.py
```

14. Coverage requirement

```bash
pytest --cov=src --cov-report=term-missing
```

## 12) Helpful Commands

Start stack:

```bash
docker compose up --build -d
```

Stop stack:

```bash
docker compose down
```

Stop stack and remove volumes:

```bash
docker compose down -v
```

Tail logs:

```bash
docker compose logs -f
```

## 13) Notes for Reviewers

- This project is intentionally local-first and does not require Kubernetes.
- The primary objective is robust lifecycle automation (not SOTA model accuracy).
- The codebase includes unit and integration-style tests for API, processing, monitoring, and retraining logic.


