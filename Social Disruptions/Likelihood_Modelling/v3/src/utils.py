# Shared utilities for the GDELT-label modelling pipeline.

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


# ---------------------------------------------------------------------------
# VIF filtering
# ---------------------------------------------------------------------------

def _vif_all(X: np.ndarray) -> np.ndarray:
    """
    Compute VIF for every column of X using numpy least-squares.
    VIF_i = 1 / (1 - R²_i), where R²_i is from regressing column i on all others.
    """
    n, k = X.shape
    vifs = np.empty(k)
    for i in range(k):
        y   = X[:, i]
        Xo  = np.delete(X, i, axis=1)
        # Add intercept
        Xo  = np.hstack([Xo, np.ones((n, 1))])
        try:
            coef, *_ = np.linalg.lstsq(Xo, y, rcond=None)
            ss_res   = np.sum((y - Xo @ coef) ** 2)
            ss_tot   = np.sum((y - y.mean()) ** 2)
            r2       = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r2       = float(np.clip(r2, 0.0, 1.0 - 1e-10))
            vifs[i]  = 1.0 / (1.0 - r2)
        except Exception:
            vifs[i] = np.nan
    return vifs


def vif_filter(
    X_train: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 10.0,
    max_sample: int = 8000,
) -> tuple[list[str], dict[str, float]]:
    """
    Iteratively drop the feature with the highest VIF until all VIFs < threshold.

    Uses numpy least-squares on median-imputed values. Subsamples to max_sample
    rows for speed — collinearity structure is stable across sample sizes.
    Only pass numeric feature columns — do NOT include country_iso3 or other
    categorical columns.

    Returns
    -------
    filtered_features : list[str]   features surviving the filter
    dropped           : dict        {feature_name: vif_when_dropped}
    """
    from sklearn.impute import SimpleImputer

    feats   = [f for f in feature_cols if f in X_train.columns]
    dropped: dict[str, float] = {}
    imp     = SimpleImputer(strategy="median")

    # Subsample once for speed — collinearity doesn't change with more rows
    X_sample = X_train[feats]
    if len(X_sample) > max_sample:
        X_sample = X_sample.sample(max_sample, random_state=42)

    while len(feats) > 1:
        X_imp = imp.fit_transform(X_sample[feats]).astype(float)
        vifs  = _vif_all(X_imp)

        finite_vifs = vifs[np.isfinite(vifs)]
        if len(finite_vifs) == 0 or finite_vifs.max() < threshold:
            break

        worst_idx  = int(np.nanargmax(vifs))
        worst_feat = feats[worst_idx]
        dropped[worst_feat] = float(vifs[worst_idx])
        feats = [f for f in feats if f != worst_feat]
        X_sample = X_sample[feats]

    if dropped:
        logger.info(
            "  VIF filter: dropped %d/%d features (threshold=%.0f) — %s",
            len(dropped), len(feature_cols), threshold,
            ", ".join(f"{f}({v:.1f})" for f, v in dropped.items()),
        )

    return feats, dropped


# ---------------------------------------------------------------------------
# Z-score helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Target helpers
# ---------------------------------------------------------------------------

def get_target_col(event_type: str, horizon: int) -> str:
    """Return the pre-computed GDELT target column name, e.g. 'protest_7d'."""
    if event_type not in ("protest", "strike"):
        raise ValueError(f"event_type must be 'protest' or 'strike', got {event_type!r}")
    if horizon not in (7, 30):
        raise ValueError(f"horizon must be 7 or 30, got {horizon}")
    return f"{event_type}_{horizon}d"


def estimate_reliable_countries(
    df: pd.DataFrame,
    min_median_articles_per_event: float = 2.0,
) -> set[str]:
    """
    Identify countries where GDELT coverage is deep enough to trust zero
    labels as genuine negatives.

    Reliability is measured by the median number of articles per detected
    event cluster for each country.  A high median means that when something
    happens GDELT covers it with multiple articles — routine activity is being
    captured.  A median of 1 means half of all detected events surface as a
    single article, implying only the most internationally visible events are
    detected and the zero labels are unreliable.

    This avoids the circularity of the previous coverage_flag approach, which
    counted event-specific articles per day (partially measuring event frequency
    rather than independent monitoring intensity).

    """
    needed = {"n_articles", "n_protest_events", "n_strike_events"}
    if not needed.issubset(df.columns):
        return set(df["country_iso3"].unique())

    event_days = df[(df["n_protest_events"] + df["n_strike_events"]) > 0].copy()
    if event_days.empty:
        return set(df["country_iso3"].unique())

    event_days["articles_per_event"] = (
        event_days["n_articles"] /
        (event_days["n_protest_events"] + event_days["n_strike_events"])
    )
    median_depth = event_days.groupby("country_iso3")["articles_per_event"].median()
    reliable = set(median_depth[median_depth >= min_median_articles_per_event].index)
    return reliable


def make_target_pu_country(
    df: pd.DataFrame,
    event_type: str,
    horizon: int,
    reliable_countries: set[str],
) -> pd.Series:
    """
    Country-level Positive-Unlabelled (PU) learning target.

    The per-day coverage flag is event-driven: countries only appear in GDELT
    when something newsworthy happens, so "medium/high coverage today" is not
    a reliable signal that GDELT was systematically monitoring the country.

    Instead, we classify countries by their *average* coverage level.
    Countries with consistently high coverage (e.g. USA, India, South Africa)
    are assumed to be systematically monitored; their zero-event days are
    genuine negatives.  Countries with sparse, event-driven coverage
    (e.g. Bolivia, Peru, Namibia) have unreliable zeros, which are masked.

    Label assignment:
      y = 1   Confirmed positive  -- event detected in any country
      y = 0   Reliable negative   -- reliable country AND no event detected
      y = NaN Unlabelled          -- unreliable country AND no event detected

    """
    col = get_target_col(event_type, horizon)
    if col not in df.columns:
        raise KeyError(f"Target column '{col}' not found. Run build_panel.py first.")

    y = df[col].astype(float).copy()

    # Mask the last `horizon` days — truncated forward window
    if "date" in df.columns:
        max_date = df["date"].max()
        cutoff   = max_date - pd.Timedelta(days=horizon)
        y.loc[df["date"] > cutoff] = np.nan

    # Unreliable country + no event = unlabelled
    unreliable_zeros = (~df["country_iso3"].isin(reliable_countries)) & (y == 0)
    y.loc[unreliable_zeros] = np.nan

    return y


def make_target_pu(
    df: pd.DataFrame,
    event_type: str,
    horizon: int,
) -> pd.Series:
    """
    Positive-Unlabelled (PU) learning target.

    Standard binary classification treats every zero label as a confirmed
    negative. This is wrong when the label source (GDELT) has coverage gaps:
    a zero on a low-coverage day means "GDELT wasn't watching", not "nothing
    happened".

    This function assigns three tiers:
      y = 1   Confirmed positive  -- event detected (any coverage level)
      y = 0   Reliable negative   -- medium/high coverage AND no event detected
                                     (GDELT was watching and saw nothing)
      y = NaN Unlabelled          -- low coverage AND no event detected
                                     (GDELT wasn't watching; we don't know)

    Only rows with y in {0, 1} enter the model's loss function.  Unlabelled
    rows are excluded from training but still receive a predicted probability
    at inference time (corrected via estimate_labelling_probability).
    """
    col = get_target_col(event_type, horizon)
    if col not in df.columns:
        raise KeyError(f"Target column '{col}' not found. Run build_panel.py first.")

    y = df[col].astype(float).copy()

    # Mask the last `horizon` days — forward window is truncated at dataset end
    if "date" in df.columns:
        max_date = df["date"].max()
        cutoff   = max_date - pd.Timedelta(days=horizon)
        y.loc[df["date"] > cutoff] = np.nan

    # Core PU assignment: low coverage + no event = unlabelled
    if "coverage_flag" in df.columns:
        unlabelled = (df["coverage_flag"] == "low") & (y == 0)
        y.loc[unlabelled] = np.nan

    return y


def estimate_labelling_probability(
    df: pd.DataFrame,
    event_type: str,
    horizon: int,
) -> float:
    """
    Estimate c = P(GDELT detects event | event actually happened).

    Uses the Elkan-Noto (2008) intuition: among all positive days in the
    training data, what fraction had medium/high coverage?  This estimates
    how often GDELT would detect a real event when it occurs.

    A value of c=0.6 means GDELT detects 60% of real events on average;
    the remaining 40% are in the unlabelled (low-coverage) pool.

    The predicted probability is then corrected as:
        P_true = clip(P_observed / c, 0, 1)

    """
    col = get_target_col(event_type, horizon)
    if col not in df.columns or "coverage_flag" not in df.columns:
        return 1.0

    positive_mask  = df[col] == 1
    n_pos          = positive_mask.sum()
    if n_pos == 0:
        return 1.0

    # Positives that GDELT detected with at least medium coverage
    n_pos_detected = (positive_mask & df["coverage_flag"].isin(["medium", "high"])).sum()

    # All positives on low-coverage days are "lucky detections" despite low coverage
    # Include them — they were detected regardless of the flag
    # c = fraction of positives that had medium/high coverage (the "reliable" detections)
    c = float(n_pos_detected / n_pos)

    # Floor to avoid extreme corrections.
    # A floor of 0.5 means the correction never more than doubles probabilities,
    # keeping predictions calibrated.  The original floor of 0.05 caused
    # near-universal clipping to 1.0 for sparse-coverage targets.
    c = max(c, 0.50)
    return c


def make_target_gdelt(
    df: pd.DataFrame,
    event_type: str,
    horizon: int,
    exclude_low_coverage: bool = False,
) -> pd.Series:
    """
    Return the GDELT-derived binary target column as a Series aligned with df.

    The target columns are pre-computed in build_labels_modelling.py:
      protest_7d  / protest_30d  -- 1 if any protest in next h days
      strike_7d   / strike_30d   -- 1 if any strike in next h days

    exclude_low_coverage=True masks low-coverage country-days as NaN (unreliable negatives).
    """
    col = get_target_col(event_type, horizon)
    if col not in df.columns:
        raise KeyError(
            f"Target column '{col}' not found in panel. "
            "Run build_panel.py first."
        )

    y = df[col].astype(float).copy()

    # Mask the last `horizon` days per country — forward-looking window is truncated
    # at the end of the dataset, making those labels unreliable (filled with 0).
    if "date" in df.columns and "country_iso3" in df.columns:
        max_date = df["date"].max()
        cutoff   = max_date - pd.Timedelta(days=horizon)
        y.loc[df["date"] > cutoff] = np.nan

    # Optionally mask low-coverage zero labels
    if exclude_low_coverage and "coverage_flag" in df.columns:
        low_cov_zeros = (df["coverage_flag"] == "low") & (y == 0)
        y.loc[low_cov_zeros] = np.nan

    return y


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    mask = ~(np.isnan(y_true) | np.isnan(y_prob))
    yt   = y_true[mask]
    yp   = y_prob[mask]

    if len(np.unique(yt)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan,
                "brier": np.nan, "brier_skill_score": np.nan}

    brier = float(brier_score_loss(yt, yp))
    prev  = float(yt.mean())
    # Brier of the naive model that always predicts the prevalence
    brier_naive = prev * (1.0 - prev)
    bss = 1.0 - brier / brier_naive if brier_naive > 0 else np.nan

    return {
        "roc_auc":           float(roc_auc_score(yt, yp)),
        "pr_auc":            float(average_precision_score(yt, yp)),
        "brier":             brier,
        "brier_skill_score": bss,
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
