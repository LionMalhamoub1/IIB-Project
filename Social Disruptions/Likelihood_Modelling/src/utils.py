from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def expanding_zscore(series: pd.Series, min_obs: int = 3) -> pd.Series:
    exp = series.expanding(min_periods=min_obs)
    mu  = exp.mean()
    sig = exp.std().replace(0.0, np.nan)
    return (series - mu) / sig


def zscore_by_country(
    df: pd.DataFrame,
    col: str,
    country_col: str = "country_iso3",
    min_obs: int = 3,
) -> pd.Series:
    return df.groupby(country_col, sort=False)[col].transform(
        lambda s: expanding_zscore(s, min_obs=min_obs)
    )


def make_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    def _future_sum(s: pd.Series) -> pd.Series:
        return s.rolling(horizon, min_periods=horizon).sum().shift(-horizon)

    future = df.groupby("country_iso3", sort=False)["acled_events"].transform(
        _future_sum
    )
    return (future >= 1).where(future.notna(), other=np.nan)


def make_target_elevated(
    df: pd.DataFrame,
    horizon: int,
    multiplier: float = 1.5,
    min_events: int = 3,
    baseline_window: int = 90,
) -> pd.Series:
    """
    y = 1 if events in the next `horizon` days exceed `multiplier` times the
    rolling `baseline_window`-day mean AND are at least `min_events`.

    Uses a lagged baseline (shift(1)) to avoid look-ahead within the window.
    """
    def _compute(s: pd.Series) -> pd.Series:
        baseline = s.rolling(baseline_window, min_periods=baseline_window).mean().shift(1)
        future   = s.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
        expected = baseline * horizon
        elevated = (future > multiplier * expected) & (future >= min_events)
        valid    = future.notna() & baseline.notna()
        return elevated.where(valid, other=np.nan)

    return df.groupby("country_iso3", sort=False)["acled_events"].transform(_compute)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    yt   = y_true[mask]
    yp   = y_prob[mask]

    if len(np.unique(yt)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan}

    return {
        "roc_auc": roc_auc_score(yt, yp),
        "pr_auc":  average_precision_score(yt, yp),
        "brier":   brier_score_loss(yt, yp),
    }


def calibration_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
) -> pd.DataFrame:
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(n_bins)]
    cut    = pd.cut(y_prob, bins=bins, labels=labels, include_lowest=True)
    cal    = pd.DataFrame({"y_true": y_true, "y_prob": y_prob, "bin": cut})
    return (
        cal.groupby("bin", observed=True)
        .agg(
            mean_pred   = ("y_prob", "mean"),
            mean_actual = ("y_true", "mean"),
            n           = ("y_true", "count"),
        )
        .reset_index()
    )


def missingness_report(df: pd.DataFrame) -> pd.DataFrame:
    n    = len(df)
    miss = df.isna().sum()
    return (
        pd.DataFrame({
            "n_missing":   miss,
            "pct_missing": (miss / n * 100).round(2),
        })
        .sort_values("pct_missing", ascending=False)
    )
