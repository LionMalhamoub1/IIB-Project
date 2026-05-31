"""
build_graphical_lasso_network.py
---------------------------------
Estimate a sparse conditional dependency network using the Graphical Lasso.

Method
------
1. Select numeric features (configurable) and standardise them.
2. Fit sklearn GraphicalLassoCV with cross-validated sparsity penalty alpha.
3. Extract the precision matrix (inverse covariance).
4. Non-zero off-diagonal entries identify pairs of variables that are
   conditionally dependent given all other variables in the model.

Unlike pairwise correlations, the precision matrix reveals *direct* links:
an entry theta_{ij} != 0 means X_i and X_j are associated even after
controlling for all other variables in FEATURE_LIST.

Outputs
-------
  outputs/adjacency_matrices/glasso_precision_matrix.csv  – full precision matrix
  outputs/adjacency_matrices/glasso_adj.csv               – thresholded adjacency
  outputs/figures/glasso_network.png                      – network figure

Usage
-----
    python Graph_analysis_social/src/build_graphical_lasso_network.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Ensure sibling utils is importable when running as a script
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (  # noqa: E402
    build_network_from_matrix,
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

_ROOT    = _SRC.parent
OUT_PREC = _ROOT / "outputs" / "adjacency_matrices" / "glasso_precision_matrix.csv"
OUT_ADJ  = _ROOT / "outputs" / "adjacency_matrices" / "glasso_adj.csv"
OUT_FIG  = _ROOT / "outputs" / "figures" / "glasso_network.png"

START_DATE: str | None = "2017-01-01"
END_DATE:   str | None = None

# All features to include in the conditional dependency estimation.
# Including acled_events as a node allows the model to identify its direct
# conditional dependencies with each indicator.
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

# Threshold on |precision matrix entry| for edge inclusion.
# GraphicalLassoCV already zeros most entries; this guards against numerical
# noise in near-zero residuals.
THRESHOLD: float = 0.01

# Fraction of rows to subsample before fitting (1.0 = use all).
# Reduce for faster iteration; CV is the most compute-intensive step.
SUBSAMPLE_FRAC: float = 1.0

# Number of cross-validation folds for alpha selection
CV_FOLDS: int = 5

# Maximum EM iterations for the Graphical Lasso solver
MAX_ITER: int = 500

# Reproducible subsampling
RANDOM_STATE: int = 42


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
    logger.info("Graphical Lasso Conditional Dependency Network")
    logger.info("  DATA_PATH       : %s", DATA_PATH)
    logger.info("  THRESHOLD       : %.4f", THRESHOLD)
    logger.info("  SUBSAMPLE_FRAC  : %.2f", SUBSAMPLE_FRAC)
    logger.info("  CV_FOLDS        : %d", CV_FOLDS)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data and select features
    # ------------------------------------------------------------------
    df = load_panel_data(DATA_PATH, START_DATE, END_DATE)
    feature_df = select_features(df, FEATURE_LIST)
    feature_cols = list(feature_df.columns)
    logger.info("Using %d features: %s", len(feature_cols), feature_cols)

    # ------------------------------------------------------------------
    # 2. Drop rows with any NaN in the selected feature set
    # ------------------------------------------------------------------
    clean = feature_df.dropna()
    logger.info(
        "Complete rows (no NaN in any feature): %d  (%.1f%% of total)",
        len(clean),
        100.0 * len(clean) / max(len(feature_df), 1),
    )

    if len(clean) < 100:
        raise ValueError(
            f"Too few complete rows ({len(clean)}) to fit Graphical Lasso reliably. "
            "Consider reducing FEATURE_LIST or relaxing date filters."
        )

    # ------------------------------------------------------------------
    # 3. Optional subsampling (for speed during development)
    # ------------------------------------------------------------------
    if SUBSAMPLE_FRAC < 1.0:
        clean = clean.sample(frac=SUBSAMPLE_FRAC, random_state=RANDOM_STATE)
        logger.info("After subsampling: %d rows.", len(clean))

    # ------------------------------------------------------------------
    # 4. Standardise features (Graphical Lasso assumes standardised data)
    # ------------------------------------------------------------------
    scaler = StandardScaler()
    X = scaler.fit_transform(clean.values)
    logger.info("Data matrix shape for fitting: %s", X.shape)

    # ------------------------------------------------------------------
    # 5. Fit GraphicalLassoCV
    # ------------------------------------------------------------------
    logger.info(
        "Fitting GraphicalLassoCV (cv=%d, max_iter=%d) — this may take a moment ...",
        CV_FOLDS, MAX_ITER,
    )
    model = GraphicalLassoCV(cv=CV_FOLDS, max_iter=MAX_ITER, n_jobs=-1)
    model.fit(X)
    logger.info("Best alpha (sparsity penalty): %.4f", model.alpha_)

    # ------------------------------------------------------------------
    # 6. Extract precision matrix
    # ------------------------------------------------------------------
    precision = pd.DataFrame(
        model.precision_,
        index=feature_cols,
        columns=feature_cols,
    )
    logger.info("Precision matrix (rounded to 3dp):\n%s", precision.round(3).to_string())

    # ------------------------------------------------------------------
    # 7. Save precision matrix
    # ------------------------------------------------------------------
    OUT_PREC.parent.mkdir(parents=True, exist_ok=True)
    precision.to_csv(OUT_PREC)
    logger.info("Saved precision matrix -> %s", OUT_PREC)

    # ------------------------------------------------------------------
    # 8. Build undirected network from precision matrix
    # ------------------------------------------------------------------
    G = build_network_from_matrix(precision, threshold=THRESHOLD, directed=False)
    logger.info(
        "Conditional dependency graph: %d nodes | %d edges  (threshold=%.4f)",
        G.number_of_nodes(), G.number_of_edges(), THRESHOLD,
    )

    # ------------------------------------------------------------------
    # 9. Save thresholded adjacency matrix
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
    # 10. Save network figure
    # ------------------------------------------------------------------
    save_network_plot(
        G,
        OUT_FIG,
        title=(
            f"Graphical Lasso: Conditional Dependency Network"
            f"  (alpha={model.alpha_:.4f}, |precision| >= {THRESHOLD})"
        ),
        directed=False,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
