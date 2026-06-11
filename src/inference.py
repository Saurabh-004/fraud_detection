"""Inference pipeline: raw transaction -> feature engineering -> scaling -> model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import RAW_FEATURE_NAMES, default_feature_columns, engineer_features, get_feature_columns


class FraudDetectionPipeline:
    """Wraps scaler + classifier for end-to-end fraud scoring."""

    def __init__(self, model, scaler, feature_columns: list[str] | None = None):
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns or default_feature_columns()
        self.raw_feature_count = len(RAW_FEATURE_NAMES)
        self.scaled_feature_count = len(self.feature_columns)

    def _raw_frame_from_array(self, X: np.ndarray) -> pd.DataFrame:
        if X.shape[1] != self.raw_feature_count:
            raise ValueError(
                f"Expected {self.raw_feature_count} raw features "
                f"({', '.join(RAW_FEATURE_NAMES[:3])}, ...), got {X.shape[1]}"
            )
        return pd.DataFrame(X, columns=RAW_FEATURE_NAMES)

    def _prepare_scaled_matrix(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] == self.scaled_feature_count:
            return X
        if X.shape[1] == self.raw_feature_count:
            df = self._raw_frame_from_array(X)
            df_fe = engineer_features(df)
            ordered = df_fe[self.feature_columns].to_numpy(dtype=np.float32)
            return self.scaler.transform(ordered)
        raise ValueError(
            f"Feature count {X.shape[1]} not recognized. "
            f"Provide {self.raw_feature_count} raw or {self.scaled_feature_count} scaled features."
        )

    def predict_proba(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        scaled = self._prepare_scaled_matrix(arr)
        return self.model.predict_proba(scaled)

    def predict(self, X) -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        scaled = self._prepare_scaled_matrix(arr)
        return self.model.predict(scaled)
