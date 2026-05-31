"""
build_correlation_network.py
----------------------------
Construct an undirected Pearson correlation network over a configurable set
of panel indicators.

Network definition
------------------
  Nodes  = selected variables
  Edges  = |Pearson r| >= THRESHOLD, pooled across the full panel
            OR averaged across per-country matrices

Outputs
-------
  outputs/adjacency_matrices/correlation_matrix.csv   – full correlation matrix
  outputs/adjacency_matrices/correlation_adj.csv      – thresholded adjacency
  outputs/figures/correlation_network.png             – network figure

Usage
-----
    python Graph_analysis_social/src/build_correlation_network.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure sibling utils is importable when running as a script
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (  # noqa: E402
    build_network_from_matrix,
    country_average_correlation,
    load_panel_data,
    safe_correlation,
    save_network_plot,
    select_features,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Input: modelling panel (actual path in the repo)
DATA_PATH: Path = (
    _SRC.parent.parent
    / "Likelihood_modelling_social"
    / "data"
    / "interim"
    / "modelling_panel.parquet"
)

# Output paths (relative to module root)
_ROOT    = _SRC.parent
OUT_CORR = _ROOT / "outputs" / "adjacency_matrices" / "correlation_matrix.csv"
OUT_ADJ  = _ROOT / "outputs" / "adjacency_matrices" / "correlation_adj.csv"
OUT_FIG  = _ROOT / "outputs" / "figures" / "correlation_network.png"

# Date filter (set to None to use all available data)
START_DATE: str | None = "2017-01-01"
END_DATE:   str | None = None

# Variables to include in the network.
# Use the z-scored variants for comparability across countries.
FEATURE_LIST: list[str] = [
    "acled_events",
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

# Minimum |Pearson r| to draw an edge
THRESHOLD: float = 0.10

# True  → pool all rows and compute a single Pearson matrix.
# False → compute per-country matrices, then average them.
USE_POOLED_DATA: bool = True

# Minimum joint non-NaN observations for a valid correlation
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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()
    logger.info("=" * 60)
    logger.info("Correlation Network Builder")
    logger.info("  DATA_PATH       : %s", DATA_PATH)
    logger.info("  THRESHOLD       : %.2f", THRESHOLD)
    logger.info("  USE_POOLED_DATA : %s", USE_POOLED_DATA)
    logger.info("  MIN_OBS         : %d", MIN_OBS)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    df = load_panel_data(DATA_PATH, START_DATE, END_DATE)

    # ------------------------------------------------------------------
    # 2. Select features
    # ------------------------------------------------------------------
    feature_df = select_features(df, FEATURE_LIST)
    feature_cols = list(feature_df.columns)
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)

    # ------------------------------------------------------------------
    # 3. Compute correlation matrix
    # ------------------------------------------------------------------
    if USE_POOLED_DATA:
        logger.info("Computing pooled Pearson correlation matrix ...")
        corr_matrix = safe_correlation(feature_df, min_obs=MIN_OBS)
    else:
        logger.info("Computing country-average correlation matrix ...")
        corr_matrix = country_average_correlation(
            df[["country_iso3"] + feature_cols],
            feature_cols,
            min_obs=MIN_OBS,
        )

    logger.info("Correlation matrix shape: %s", corr_matrix.shape)

    # ------------------------------------------------------------------
    # 4. Save full correlation matrix
    # ------------------------------------------------------------------
    OUT_CORR.parent.mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(OUT_CORR)
    logger.info("Saved correlation matrix -> %s", OUT_CORR)

    # ------------------------------------------------------------------
    # 5. Apply threshold and build NetworkX graph
    # ------------------------------------------------------------------
    G = build_network_from_matrix(corr_matrix, threshold=THRESHOLD, directed=False)
    logger.info(
        "Undirected graph: %d nodes | %d edges  (threshold=%.2f)",
        G.number_of_nodes(), G.number_of_edges(), THRESHOLD,
    )

    # ------------------------------------------------------------------
    # 6. Save thresholded adjacency matrix
    # ------------------------------------------------------------------
    adj = pd.DataFrame(
        {
            n: {m: G[n][m]["weight"] if G.has_edge(n, m) else 0.0 for m in G.nodes()}
            for n in G.nodes()
        }
    ).T
    adj.to_csv(OUT_ADJ)
    logger.info("Saved adjacency matrix -> %s", OUT_ADJ)

    # ------------------------------------------------------------------
    # 7. Save network figure
    # ------------------------------------------------------------------
    save_network_plot(
        G,
        OUT_FIG,
        title=f"Correlation Network  (|r| >= {THRESHOLD})",
        directed=False,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
