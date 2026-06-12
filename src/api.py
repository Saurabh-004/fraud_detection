"""
Sub-Problem D — Production System: FastAPI REST API
- Input validation (Pydantic)
- Real-time inference (p99 < 100 ms)
- Data drift detection (Evidently / statistical tests)
- Model registry with versioning & rollback (MLflow-backed or file-based fallback)
"""

import os
import time
import json
import logging
import hashlib
import pickle
import gzip
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.mlops import MonitoringDashboard, RetrainTrigger
from src.model_loader import bootstrap_registry

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
)
logger = logging.getLogger("fraud_api")

REQUEST_COUNT = Counter("fraud_api_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram(
    "fraud_api_latency_seconds",
    "Request latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5],
)
DRIFT_ALERTS = Counter("fraud_api_drift_alerts_total", "Drift alert count", ["feature"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_DIR = Path(os.getenv("MODEL_REGISTRY_DIR", PROJECT_ROOT / "model_registry"))
REGISTRY_DIR.mkdir(exist_ok=True)


class ModelRegistry:
    """Versioned model registry with rollback capability."""

    def __init__(self, registry_dir: Path = REGISTRY_DIR):
        self.dir = registry_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {"versions": {}, "current": None, "previous": None}

    def _save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def register(self, model, metadata: dict, version: str = None) -> str:
        version = version or datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
        path = self.dir / f"model_{version}.pkl.gz"
        with gzip.open(path, "wb") as f:
            pickle.dump(model, f)

        sha = hashlib.md5(path.read_bytes()).hexdigest()
        entry = {
            "version": version,
            "path": str(path),
            "sha256": sha,
            "metadata": metadata,
            "registered_at": datetime.utcnow().isoformat(),
        }
        self.manifest["versions"][version] = entry
        self.manifest["previous"] = self.manifest["current"]
        self.manifest["current"] = version
        self._save_manifest()
        logger.info(f"Registered model {version}")
        return version

    def load(self, version: str = None):
        version = version or self.manifest.get("current")
        if not version or version not in self.manifest["versions"]:
            raise ValueError(f"Model version '{version}' not found in registry.")
        path = self.manifest["versions"][version]["path"]
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    def rollback(self) -> str:
        prev = self.manifest.get("previous")
        if not prev:
            raise ValueError("No previous version to roll back to.")
        self.manifest["current"] = prev
        self.manifest["previous"] = None
        self._save_manifest()
        logger.warning(f"Rolled back to model version: {prev}")
        return prev

    def list_versions(self) -> List[dict]:
        return list(self.manifest["versions"].values())


class DriftDetector:
    """Statistical drift detection using PSI and Kolmogorov-Smirnov test."""

    PSI_THRESHOLD = 0.2
    KS_THRESHOLD = 0.1

    def __init__(self, reference_stats: dict = None):
        self.reference_stats = reference_stats or {}
        self.reference_samples: Dict[str, np.ndarray] = {}
        self.alerts: List[dict] = []

    def fit(self, X_ref: np.ndarray, feature_names: List[str]):
        for i, name in enumerate(feature_names):
            col = X_ref[:, i].astype(float)
            self.reference_samples[name] = col.copy()
            self.reference_stats[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col) + 1e-9),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
                "percentiles": {str(p): float(np.percentile(col, p)) for p in [5, 25, 50, 75, 95]},
                "hist": np.histogram(col, bins=10)[0].tolist(),
                "hist_edges": np.histogram(col, bins=10)[1].tolist(),
            }
        logger.info(f"Drift detector fitted on {len(feature_names)} features.")

    def _psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        eps = 1e-6
        combined = np.concatenate([expected, actual])
        edges = np.percentile(combined, np.linspace(0, 100, buckets + 1))
        edges[0] -= 1
        edges[-1] += 1
        exp_pct = np.histogram(expected, edges)[0] / len(expected) + eps
        act_pct = np.histogram(actual, edges)[0] / len(actual) + eps
        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    def check(self, X_new: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        if not self.reference_stats:
            return {"status": "no_reference", "alerts": []}
        from scipy import stats as scipy_stats

        drift_report = {"timestamp": datetime.utcnow().isoformat(), "alerts": [], "features": {}}

        for i, name in enumerate(feature_names):
            if name not in self.reference_stats:
                continue
            ref = self.reference_stats[name]
            col = X_new[:, i].astype(float)
            ref_sample = self.reference_samples.get(name)
            if ref_sample is None or len(ref_sample) == 0:
                ref_hist = np.array(ref["hist"], dtype=float)
                ref_edges = np.array(ref["hist_edges"])
                ref_sample = np.repeat(
                    (ref_edges[:-1] + ref_edges[1:]) / 2,
                    ref_hist.astype(int).clip(0),
                )
            if len(ref_sample) == 0:
                continue

            psi = self._psi(ref_sample, col)
            ks_stat, ks_p = scipy_stats.ks_2samp(ref_sample, col)
            mean_shift = abs(np.mean(col) - ref["mean"]) / (ref["std"] + 1e-9)

            drift_report["features"][name] = {
                "psi": round(psi, 4),
                "ks_stat": round(float(ks_stat), 4),
                "ks_p_value": round(float(ks_p), 4),
                "mean_shift_sigma": round(float(mean_shift), 2),
                "drifted": psi > self.PSI_THRESHOLD or ks_stat > self.KS_THRESHOLD,
            }
            if drift_report["features"][name]["drifted"]:
                alert = {"feature": name, "psi": psi, "ks_stat": ks_stat}
                drift_report["alerts"].append(alert)
                DRIFT_ALERTS.labels(feature=name).inc()
                logger.warning(f"DRIFT ALERT: {name} PSI={psi:.3f} KS={ks_stat:.3f}")

        drift_report["n_drifted"] = len(drift_report["alerts"])
        drift_report["status"] = "drift_detected" if drift_report["alerts"] else "ok"
        return drift_report


class TransactionInput(BaseModel):
    transaction_id: Optional[str] = Field(None, description="Unique transaction ID")
    features: List[float] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Raw (30) or scaled (38) feature vector matching training schema",
    )

    @field_validator("features")
    @classmethod
    def no_nan_inf(cls, v):
        for x in v:
            if not np.isfinite(x):
                raise ValueError("Feature vector must not contain NaN or Inf values.")
        return v


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    fraud_probability: float
    is_fraud: bool
    risk_tier: str
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: Optional[str]
    uptime_seconds: float
    timestamp: str


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud scoring with drift monitoring",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_registry = ModelRegistry()
_model = None
_model_ver = "unloaded"
_drift_det = DriftDetector()
_feature_names: List[str] = []
_threshold = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
_start_time = time.time()
_monitor = MonitoringDashboard()
_retrain_trigger = RetrainTrigger()


@app.on_event("startup")
async def load_model():
    global _model, _model_ver, _feature_names, _threshold
    version = bootstrap_registry(_registry)
    if version:
        logger.info(f"Bootstrapped model registry with {version}")

    try:
        _model = _registry.load()
        _model_ver = _registry.manifest.get("current", "unknown")
        meta = _registry.manifest["versions"][_model_ver].get("metadata", {})
        _feature_names = meta.get("feature_columns", [])
        if "threshold" in meta:
            _threshold = float(meta["threshold"])
        logger.info(f"Model {_model_ver} loaded at startup.")
    except Exception as e:
        logger.warning(f"No model in registry at startup: {e}. POST /model/register first.")


def _risk_tier(prob: float) -> str:
    if prob >= 0.8:
        return "HIGH"
    if prob >= 0.4:
        return "MEDIUM"
    return "LOW"


@app.get("/health", response_model=HealthResponse)
async def health():
    REQUEST_COUNT.labels(endpoint="/health", status="200").inc()
    return HealthResponse(
        status="ok" if _model else "model_not_loaded",
        model_version=_model_ver,
        uptime_seconds=round(time.time() - _start_time, 1),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(req: TransactionInput, background: BackgroundTasks):
    if _model is None:
        raise HTTPException(503, "Model not loaded. Register a model first via POST /model/register.")

    t0 = time.perf_counter()
    try:
        X = np.array(req.features, dtype=np.float32).reshape(1, -1)
        prob = float(_model.predict_proba(X)[0, 1])
        latency_ms = (time.perf_counter() - t0) * 1000
        is_fraud = prob >= _threshold

        REQUEST_LATENCY.observe(latency_ms / 1000)
        REQUEST_COUNT.labels(endpoint="/predict", status="200").inc()
        background.add_task(_log_prediction, req.transaction_id, prob, latency_ms, is_fraud)

        return PredictionResponse(
            transaction_id=req.transaction_id,
            fraud_probability=round(prob, 6),
            is_fraud=is_fraud,
            risk_tier=_risk_tier(prob),
            model_version=_model_ver,
            latency_ms=round(latency_ms, 3),
        )
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status="500").inc()
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, f"Inference error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(req: BatchTransactionInput):
    if _model is None:
        raise HTTPException(503, "Model not loaded.")
    t0 = time.perf_counter()
    try:
        X = np.array([t.features for t in req.transactions], dtype=np.float32)
        probs = _model.predict_proba(X)[:, 1]
        latency_ms = (time.perf_counter() - t0) * 1000
        results = []
        for txn, prob in zip(req.transactions, probs):
            p = float(prob)
            results.append(
                {
                    "transaction_id": txn.transaction_id,
                    "fraud_probability": round(p, 6),
                    "is_fraud": p >= _threshold,
                    "risk_tier": _risk_tier(p),
                }
            )
        return {
            "predictions": results,
            "model_version": _model_ver,
            "total_latency_ms": round(latency_ms, 2),
        }
    except Exception as e:
        raise HTTPException(500, f"Batch inference error: {str(e)}")


@app.post("/drift/check")
async def check_drift(req: BatchTransactionInput):
    X = np.array([t.features for t in req.transactions], dtype=np.float32)
    names = _feature_names or [f"f{i}" for i in range(X.shape[1])]
    report = _drift_det.check(X, names)

    if report.get("n_drifted", 0) > 0:
        for alert in report["alerts"]:
            _monitor.record_drift(alert["feature"], alert["psi"])
        _retrain_trigger.should_retrain(n_drifted_features=report["n_drifted"])

    return report


@app.get("/monitoring/summary")
async def monitoring_summary():
    return _monitor.summary()


@app.post("/model/register")
async def register_model(payload: dict):
    global _model, _model_ver, _feature_names
    version = payload.get("version")
    meta = payload.get("metadata", {})

    if payload.get("bootstrap_from_notebook", False):
        version = bootstrap_registry(_registry) or version
        if version:
            _model = _registry.load(version)
            _model_ver = version
            meta = _registry.manifest["versions"][version].get("metadata", {})
            _feature_names = meta.get("feature_columns", [])
            return {
                "status": "registered",
                "version": version,
                "registry_versions": len(_registry.manifest["versions"]),
                "current": _registry.manifest.get("current"),
            }

    return {
        "message": "Use bootstrap_from_notebook=true or ModelRegistry.register() in training pipeline.",
        "registry_versions": len(_registry.manifest["versions"]),
        "current": _registry.manifest.get("current"),
    }


@app.post("/model/rollback")
async def rollback():
    global _model, _model_ver
    try:
        prev_version = _registry.rollback()
        _model = _registry.load(prev_version)
        _model_ver = prev_version
        logger.warning(f"Rolled back to {prev_version}")
        return {"status": "rolled_back", "version": prev_version}
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/model/versions")
async def list_versions():
    return {"versions": _registry.list_versions(), "current": _registry.manifest.get("current")}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/threshold")
async def get_threshold():
    return {"threshold": _threshold}


@app.put("/threshold")
async def set_threshold(t: float):
    global _threshold
    if not 0.0 < t < 1.0:
        raise HTTPException(400, "Threshold must be in (0, 1)")
    _threshold = t
    logger.info(f"Threshold updated to {t}")
    return {"threshold": _threshold}


def _log_prediction(txn_id, prob, latency_ms, is_fraud):
    _monitor.record(prob, is_fraud, latency_ms)
    logger.info(
        json.dumps(
            {
                "event": "prediction",
                "txn_id": txn_id,
                "prob": round(prob, 6),
                "latency_ms": round(latency_ms, 2),
                "model_version": _model_ver,
            }
        )
    )
