"""
Build a scikit-learn-ready feature matrix from the combined flood/baseline dataset.

Missing indicator values are imputed with the column median computed on the
training split only (to avoid data leakage). Features are then standardised
with StandardScaler so that coefficients are comparable across indicators.

Exported:
  INDICATOR_COLS  — re-exported from load_dataset for convenience
  build_features(df)  -> X (np.ndarray), y (np.ndarray), feature_names (list[str])
  get_preprocessor()  -> sklearn Pipeline (impute + scale)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Re-export so other modules only need to import from here
from Natural_disruptions.validating_indicators.helper_scripts.load_dataset import (
    INDICATOR_COLS,
    load_combined,
)

log = logging.getLogger(__name__)


def get_preprocessor() -> Pipeline:
    """
    Return a sklearn Pipeline that:
      1. Imputes missing values with the column median
      2. Standardises each feature to zero mean, unit variance

    Fit this on training data only; apply transform to both train and test.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])


def build_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract feature matrix X and label vector y from the combined DataFrame.

    Only rows with at least one non-null indicator are kept. Columns that are
    entirely null (e.g. GloFAS when the cache is empty) are dropped automatically
    and a warning is logged.

    Returns:
      X             — float array (n_samples, n_features)
      y             — int array (n_samples,)  0=baseline, 1=flood
      feature_names — list of indicator column names actually used
    """
    available = [c for c in INDICATOR_COLS if c in df.columns]
    null_cols  = [c for c in available if df[c].isna().all()]
    if null_cols:
        log.warning(f"Dropping indicators with 100% missing values: {null_cols}")
    feature_names = [c for c in available if c not in null_cols]

    if not feature_names:
        raise ValueError(
            "No indicator columns have any data. "
            "Ensure the enrichment pipeline has been run."
        )

    # Drop rows missing ALL features (cannot be imputed meaningfully)
    sub = df[feature_names + ["label"]].copy()
    all_null_mask = sub[feature_names].isna().all(axis=1)
    if all_null_mask.any():
        log.warning(f"Dropping {all_null_mask.sum()} rows with all-null features")
        sub = sub[~all_null_mask]

    X = sub[feature_names].values.astype(float)
    y = sub["label"].values.astype(int)

    n_flood    = (y == 1).sum()
    n_baseline = (y == 0).sum()
    log.info(
        f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features  "
        f"(floods={n_flood}, baseline={n_baseline})"
    )
    log.info(f"Features: {feature_names}")

    return X, y, feature_names


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_combined()
    X, y, names = build_features(df)
    print(f"X shape: {X.shape}  |  y: {np.bincount(y)}")
    print("Features:", names)
