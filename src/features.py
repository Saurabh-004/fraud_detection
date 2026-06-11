"""Feature engineering shared between training notebook and inference API."""

import numpy as np
import pandas as pd

RAW_FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
PCA_TOP = ["V1", "V3", "V4", "V9", "V10"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same 8 engineered features used in preprocessing.ipynb."""
    d = df.copy()

    d["fe_log_amount"] = np.log1p(d["Amount"])

    amt_mean = d["Amount"].mean()
    amt_std = d["Amount"].std() + 1e-9
    d["fe_amount_zscore"] = (d["Amount"] - amt_mean) / amt_std

    seconds_in_day = 86400
    d["fe_hour"] = (d["Time"] % seconds_in_day) / 3600
    d["fe_is_night"] = ((d["fe_hour"] >= 22) | (d["fe_hour"] < 6)).astype(int)

    d["fe_v1_v3_interact"] = d["V1"] * d["V3"]
    d["fe_pca_norm"] = np.sqrt((d[PCA_TOP] ** 2).sum(axis=1))
    d["fe_amount_x_log"] = d["Amount"] * d["fe_log_amount"]
    d["fe_v14_signed_sq"] = d["V14"] * np.abs(d["V14"])

    return d


def get_feature_columns(columns) -> list[str]:
    return [c for c in columns if c not in ["Class"] and not str(c).startswith("Unnamed")]


def default_feature_columns() -> list[str]:
    template = pd.DataFrame({name: [0.0] for name in RAW_FEATURE_NAMES})
    return get_feature_columns(engineer_features(template).columns)
