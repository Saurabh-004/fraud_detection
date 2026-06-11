# 🛡️ Credit Card Fraud Detection

A production-grade machine learning system for real-time credit card fraud detection. Built with a FastAPI REST API, XGBoost/scikit-learn models, MLflow experiment tracking, Prometheus metrics, and Grafana dashboards — all containerized with Docker.

---

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start (Docker — Recommended)](#quick-start-docker--recommended)
- [Local Setup (Without Docker)](#local-setup-without-docker)
- [API Usage](#api-usage)
- [Monitoring](#monitoring)
- [Running Tests](#running-tests)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Features

- Real-time fraud prediction via REST API
- Multiple ML models (XGBoost default, scikit-learn alternatives)
- Model registry with versioning
- MLflow experiment tracking
- Prometheus metrics + Grafana dashboards
- Redis caching
- SHAP-based model explainability
- Full test suite with coverage reporting
- CI/CD via GitHub Actions

---

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| ML | XGBoost, scikit-learn, imbalanced-learn |
| Experiment Tracking | MLflow |
| Monitoring | Prometheus + Grafana |
| Caching | Redis |
| Containerization | Docker + Docker Compose |
| Testing | pytest + pytest-cov |
| Python | 3.11+ |

---

## Project Structure

```
fraud_detection/
├── src/                    # Core source code
│   ├── api.py              # FastAPI app & endpoints
│   ├── mlops.py            # MLOps utilities
│   └── evaluation.py       # Model evaluation logic
├── notebooks/              # EDA & model experimentation
│   └── saved_models/       # Trained model files
├── model_registry/         # Versioned model storage
├── monitoring/             # Prometheus & Grafana config
│   ├── prometheus.yml
│   └── grafana/
├── scripts/                # Utility scripts
├── tests/                  # Test suite
├── outputs/                # Evaluation outputs & plots
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start (Docker — Recommended)

This is the easiest way to run the entire stack on any machine. You only need **Git** and **Docker Desktop** installed.

### Prerequisites

- [Git](https://git-scm.com/downloads)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)

Verify both are installed:
```bash
git --version
docker --version
docker compose version
```

### 1. Clone the repository

```bash
git clone https://github.com/Saurabh-004/fraud_detection.git
cd fraud_detection
```

### 2. Start all services

```bash
docker compose up --build
```

This builds and starts:
- **Fraud Detection API** → http://localhost:8000
- **MLflow UI** → http://localhost:5000
- **Prometheus** → http://localhost:9090
- **Grafana** → http://localhost:3000 (login: `admin` / `fraudadmin`)
- **Redis** → localhost:6379

> First build takes 3–5 minutes to pull images and install dependencies. Subsequent starts are instant.

### 3. Verify everything is running

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy"}
```

### 4. Stop the services

```bash
docker compose down
```

To also remove all stored data (models, MLflow runs, Grafana dashboards):
```bash
docker compose down -v
```

---

## Local Setup (Without Docker)

If you prefer to run without Docker, follow these steps.

### Prerequisites

- Python 3.11 or higher
- `pip` or [`uv`](https://github.com/astral-sh/uv) (the project uses `uv`)

Check your Python version:
```bash
python --version   # must be 3.11+
```

### 1. Clone the repository

```bash
git clone https://github.com/Saurabh-004/fraud_detection.git
cd fraud_detection
```

### 2. Create a virtual environment

**Using `uv` (recommended — faster):**
```bash
pip install uv
uv venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
uv pip install -r requirements.txt
```

**Using standard `pip`:**
```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
# macOS / Linux
export PYTHONPATH=.
export MODEL_REGISTRY_DIR=./model_registry
export SAVED_MODELS_DIR=./notebooks/saved_models
export DEFAULT_MODEL=xgboost
export FRAUD_THRESHOLD=0.5

# Windows (Command Prompt)
set PYTHONPATH=.
set MODEL_REGISTRY_DIR=.\model_registry
set SAVED_MODELS_DIR=.\notebooks\saved_models
set DEFAULT_MODEL=xgboost
set FRAUD_THRESHOLD=0.5
```

### 4. Start the API

```bash
python main.py
```

Or directly with uvicorn:
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at http://localhost:8000.

---

## API Usage

### Interactive Docs

Once running, open the auto-generated API documentation in your browser:

- **Swagger UI** → http://localhost:8000/docs
- **ReDoc** → http://localhost:8000/redoc

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/` | API info |
| `POST` | `/predict` | Predict fraud for a single transaction |
| `POST` | `/predict/batch` | Predict fraud for multiple transactions |
| `GET` | `/models` | List available models |
| `GET` | `/metrics` | Prometheus metrics |

### Example: Single Transaction Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
  }'
```

Expected response:
```json
{
  "prediction": 0,
  "fraud_probability": 0.0023,
  "is_fraud": false,
  "model_used": "xgboost"
}
```

### Example: Python Client

```python
import requests

transaction = {
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
    # ... (V1 through V28 + Amount)
    "Amount": 149.62
}

response = requests.post("http://localhost:8000/predict", json=transaction)
result = response.json()

print(f"Fraud: {result['is_fraud']}")
print(f"Probability: {result['fraud_probability']:.4f}")
```

---

## Monitoring

When running via Docker Compose, the full monitoring stack is available:

### Grafana (http://localhost:3000)
- Login: `admin` / `fraudadmin`
- Pre-built dashboards show prediction counts, fraud rate, latency, and model performance.

### MLflow (http://localhost:5000)
- Browse all training runs, compare model metrics, and view registered model versions.

### Prometheus (http://localhost:9090)
- Raw metrics endpoint. Query prediction latency, request counts, and error rates.

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_api.py -v
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `xgboost` | Model to load at startup |
| `FRAUD_THRESHOLD` | `0.5` | Probability threshold to classify as fraud |
| `MODEL_REGISTRY_DIR` | `./model_registry` | Path to versioned models |
| `SAVED_MODELS_DIR` | `./notebooks/saved_models` | Path to trained model files |
| `MLFLOW_TRACKING_URI` | _(not set)_ | MLflow server URI (set automatically in Docker) |

---

## Troubleshooting

**Port already in use**

If port 8000 is taken, change the mapping in `docker-compose.yml`:
```yaml
ports:
  - "8001:8000"   # access at http://localhost:8001
```

**Docker build fails with memory error**

Increase Docker Desktop's memory limit to at least 4 GB under Settings → Resources.

**`ModuleNotFoundError` in local setup**

Make sure `PYTHONPATH` is set to the project root:
```bash
export PYTHONPATH=.     # macOS/Linux
set PYTHONPATH=.        # Windows
```

**Model not found on startup**

Ensure the `notebooks/saved_models/` directory contains trained model files. If empty, check the notebooks for training instructions or run the training script:
```bash
python scripts/train.py   # if available
```

---

## License

This project is open source. See [LICENSE](LICENSE) for details.