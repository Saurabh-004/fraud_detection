"""Load trained joblib artifacts from the notebook pipeline into the API registry."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib

from src.inference import FraudDetectionPipeline


def _resolve_path(env_var: str, default: Path) -> Path:
    return Path(os.getenv(env_var, str(default)))


def load_training_artifacts(
    models_dir: Path | None = None,
    model_name: str | None = None,
) -> FraudDetectionPipeline:
    root = Path(__file__).resolve().parent.parent
    models_dir = models_dir or _resolve_path(
        "SAVED_MODELS_DIR", root / "notebooks" / "saved_models"
    )
    model_name = model_name or os.getenv("DEFAULT_MODEL", "xgboost")

    model_path = models_dir / f"{model_name}.joblib"
    scaler_path = models_dir / "scaler.joblib"
    metadata_path = models_dir / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    feature_columns = None
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
        feature_columns = meta.get("feature_columns")

    return FraudDetectionPipeline(model, scaler, feature_columns)


def bootstrap_registry(registry, models_dir: Path | None = None) -> str | None:
    """Register the best saved model if the registry has no current version."""
    if registry.manifest.get("current"):
        return registry.manifest["current"]

    try:
        pipeline = load_training_artifacts(models_dir=models_dir)
    except FileNotFoundError:
        return None

    version = os.getenv("MODEL_VERSION", "v_notebook_xgboost")
    metadata = {
        "source": "notebooks/saved_models",
        "model": os.getenv("DEFAULT_MODEL", "xgboost"),
        "feature_columns": pipeline.feature_columns,
        "raw_feature_count": pipeline.raw_feature_count,
    }
    return registry.register(pipeline, metadata, version=version)
