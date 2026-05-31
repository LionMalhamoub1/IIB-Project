"""
build_leadlag_network.py
------------------------
Construct a directed lead-lag network between panel indicators and the ACLED
social disruption target.

Method
------
For each predictor X and target Y (acled_events):

    corr(X_{t-k}, Y_t)   for k in 1..MAX_LAG days

A directed edge X -> Y is added when the maximum absolute correlation across
all tested lags meets the threshold.

Edge attributes
    weight   : signed Pearson r at the optimal lag
    lag_days : the lag k (in days) at which the maximum |r| was observed

No data leakage: shifts are computed within each country group so that
observations from different countries are never blended across time.

Outputs
-------
  outputs/adjacency_matrices/leadlag_adj.csv   – edge list with weight & lag
  outputs/figures/leadlag_network.png          – directed network figure

Usage
-----
    python Graph_analysis_social/src/build_leadlag_network.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure sibling utils is importable when running as a script
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (  # noqa: E402
    load_panel_data,
    save_network_plot,
    select_features,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH: Path = (
    _SRC.parent.parent
    / "Likelihood_modelling_social"
    / "data"
    / "interim"
    / "modelling_panel.parquet"
)

_ROOT   = _SRC.parent
OUT_ADJ = _ROOT / "outputs" / "adjacency_matrices" / "leadlag_adj.csv"
OUT_FIG = _ROOT / "outputs" / "figures" / "leadlag_network.png"

START_DATE: str | None = "2017-01-01"
END_DATE:   str | None = None

# Target variable
TARGET: str = "acled_events"

# Predictors to evaluate as potential leading indicators
FEATURE_LIST: list[str] = [
    "fx_pct_30d_z",
    "fx_vol_30d_z",
    "oil_brent_pct_30d_z",
    "inflation_cpi_yoy_z",
    "gdp_growth_z",
    "unemployment_total_z",
    "political_stability_est_z",
    "rule_of_law_est_z",
    "government_effectiveness_est_z",
    "voice_accountability_est_z",
    "economic_stress_index_z",
    "labour_conflict_index_z",
    "protest_mobilisation_index_z",
    "food_cpi_inflation_z",
    "energy_cpi_inflation_z",
]

# Maximum lag to test (days)
MAX_LAG: int = 30

# Minimum |correlation| required to draw a directed edge
THRESHOLD: float = 0.05

# True  → pool all country-rows (faster).
# False → compute per-country correlations and average them.
USE_POOLED_DATA: bool = True

# Minimum joint non-NaN observations required for a valid correlation
MIN_OBS: int = 30


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lead-lag correlation helpers
# ---------------------------------------------------------------------------

def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between two 1-D arrays.  Returns NaN when fewer than MIN_OBS
    finite joint observations are available or when one series is constant."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < MIN_OBS:
        return np.nan
    xm = x[mask] - x[mask].mean()
    ym = y[mask] - y[mask].mean()
    denom = np.sqrt((xm ** 2).sum() * (ym ** 2).sum())
    if denom == 0.0:
        return np.nan
    return float(np.dot(xm, ym) / denom)


def _best_leadlag(corrs: dict[int, float]) -> tuple[float, int]:
    """Return (max_abs_r, best_lag) from a {lag: corr} dict."""
    valid = {k: v for k, v in corrs.items() if not np.isnan(v)}
    if not valid:
        return np.nan, -1
    best_lag = max(valid, key=lambda k: abs(valid[k]))
    return valid[best_lag], best_lag


def compute_leadlag_pooled(
    df: pd.DataFrame,
    predictor: str,
    target: str,
    max_lag: int,
) -> tuple[float, int]:
    """Pool all country rows and compute corr(predictor_{t-k}, target_t) for
    k=1..max_lag.  Within-country shifts prevent cross-country time bleed.

    Returns (signed_r_at_best_lag, best_lag_days).
    """
    y = df[target].values
    corrs: dict[int, float] = {}

    for k in range(1, max_lag + 1):
        x_shifted = df.groupby("country_iso3")[predictor].shift(k).values
        corrs[k] = _pearson_r(x_shifted, y)

    return _best_leadlag(corrs)


def compute_leadlag_country_average(
    df: pd.DataFrame,
    predictor: str,
    target: str,
    max_lag: int,
) -> tuple[float, int]:
    """Compute lead-lag correlations per country then average across countries.

    Returns (avg_signed_r_at_best_lag, best_lag_days) where best_lag is
    chosen on the average series.
    """
    lag_corrs: dict[int, list[float]] = {k: [] for k in range(1, max_lag + 1)}

    for _, grp in df.groupby("country_iso3"):
        if len(grp) < MIN_OBS:
            continue
        grp = grp.sort_values("date")
        for k in range(1, max_lag + 1):
            x = grp[predictor].shift(k).values
            y = grp[target].values
            r = _pearson_r(x, y)
            if not np.isnan(r):
                lag_corrs[k].append(r)

    avg_corrs = {k: float(np.mean(v)) for k, v in lag_corrs.items() if v}
    return _best_leadlag(avg_corrs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()
    logger.info("=" * 60)
    logger.info("Lead-Lag Directed Network Builder")
    logger.info("  DATA_PATH       : %s", DATA_PATH)
    logger.info("  TARGET          : %s", TARGET)
    logger.info("  MAX_LAG         : %d days", MAX_LAG)
    logger.info("  THRESHOLD       : %.3f", THRESHOLD)
    logger.info("  USE_POOLED_DATA : %s", USE_POOLED_DATA)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = load_panel_data(DATA_PATH, START_DATE, END_DATE)
    df = df.sort_values(["country_iso3", "date"]).reset_index(drop=True)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in panel.")

    available_features = [f for f in FEATURE_LIST if f in df.columns]
    missing = [f for f in FEATURE_LIST if f not in df.columns]
    if missing:
        logger.warning("Features not found, skipped: %s", missing)

    logger.info(
        "Evaluating %d predictors vs target '%s'.", len(available_features), TARGET
    )

    # Keep only needed columns to reduce memory during shift operations
    keep_cols = ["country_iso3", "date", TARGET] + available_features
    df = df[keep_cols]

    # ------------------------------------------------------------------
    # 2. Compute lead-lag correlations for each predictor
    # ------------------------------------------------------------------
    records: list[dict] = []

    for feat in available_features:
        if USE_POOLED_DATA:
            signed_r, best_lag = compute_leadlag_pooled(df, feat, TARGET, MAX_LAG)
        else:
            signed_r, best_lag = compute_leadlag_country_average(
                df, feat, TARGET, MAX_LAG
            )

        abs_r = abs(signed_r) if not np.isnan(signed_r) else np.nan
        records.append(
            {
                "predictor": feat,
                "target": TARGET,
                "signed_corr": signed_r,
                "max_abs_corr": abs_r,
                "best_lag_days": best_lag,
            }
        )
        logger.info(
            "  %-38s  |r|=%.3f  lag=%d days",
            feat,
            abs_r if not np.isnan(abs_r) else float("nan"),
            best_lag,
        )

    results_df = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 3. Build directed NetworkX graph
    # ------------------------------------------------------------------
    G = nx.DiGraph()
    G.add_nodes_from([TARGET] + available_features)

    for _, row in results_df.iterrows():
        if np.isnan(row["max_abs_corr"]):
            continue
        if row["max_abs_corr"] >= THRESHOLD:
            G.add_edge(
                row["predictor"],
                row["target"],
                weight=float(row["signed_corr"]),
                lag=int(row["best_lag_days"]),
            )

    logger.info(
        "Directed graph: %d nodes | %d edges  (threshold=%.3f)",
        G.number_of_nodes(), G.number_of_edges(), THRESHOLD,
    )

    # ------------------------------------------------------------------
    # 4. Save adjacency edge list (with lag attribute)
    # ------------------------------------------------------------------
    OUT_ADJ.parent.mkdir(parents=True, exist_ok=True)

    edge_rows = [
        {
            "source":    u,
            "target":    v,
            "weight":    data["weight"],
            "abs_corr":  abs(data["weight"]),
            "lag_days":  data["lag"],
        }
        for u, v, data in G.edges(data=True)
    ]
    edge_df = pd.DataFrame(edge_rows).sort_values("abs_corr", ascending=False)
    edge_df.to_csv(OUT_ADJ, index=False)
    logger.info("Saved edge list -> %s", OUT_ADJ)

    # ------------------------------------------------------------------
    # 5. Save network figure
    # ------------------------------------------------------------------
    save_network_plot(
        G,
        OUT_FIG,
        title=(
            f"Lead-Lag Directed Network  (target: {TARGET}, "
            f"|r| >= {THRESHOLD}, max lag {MAX_LAG} days)"
        ),
        directed=True,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
