"""
Extract and visualise feature importances from trained flood classifiers.

For Random Forest: mean decrease in impurity (Gini importance), averaged
across all trees. Provides a stable global measure.

For Logistic Regression: absolute standardised coefficients. Because the
features are standardised before training, the magnitude of each coefficient
directly reflects how strongly the model relies on that indicator.

Both are normalised to sum to 1 for easy comparison.

Saves:
  results/predictive_models/feature_importance_rf.png    — RF importances
  results/predictive_models/feature_importance_lr.png    — LR coefficients
  results/predictive_models/feature_importance.json      — raw values

Usage:
  python -m Natural_disruptions.predictive_models.feature_importance
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .build_features import build_features, get_preprocessor, INDICATOR_COLS
from Natural_disruptions.validating_indicators.helper_scripts.load_dataset import load_combined

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "Natural_disruptions" / "results" / "predictive_models"

RANDOM_STATE = 42

INDICATOR_LABELS = {
    "spi_30d":                "SPI-30",
    "chirps_7d_anom_pct":     "CHIRPS 7-day anom.",
    "era5_soil_moisture_day0":"ERA5 soil moisture",
    "gpm_peak_3h_mm":         "GPM peak 3-hour",
    "jrc_recurrence_pct":     "JRC recurrence",
    "pop_density_km2":        "Pop. density",
}


def _bar_plot(importances: np.ndarray, feature_names: list[str],
              title: str, xlabel: str, out_path: Path,
              color: str = "#4472c4") -> None:
    """Horizontal bar chart, sorted by importance descending."""
    order = np.argsort(importances)
    sorted_imp   = importances[order]
    sorted_names = [INDICATOR_LABELS.get(feature_names[i], feature_names[i]) for i in order]

    fig, ax = plt.subplots(figsize=(6, 0.55 * len(feature_names) + 1.5))
    bars = ax.barh(range(len(sorted_names)), sorted_imp, color=color, alpha=0.85)
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    ax.set_xlim(0, sorted_imp.max() * 1.2)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved plot to {out_path}")


def compute_rf_importance(X: np.ndarray, y: np.ndarray,
                           feature_names: list[str]) -> np.ndarray:
    """
    Train a Random Forest on the full dataset and return normalised
    Gini feature importances (averaged across 5-fold CV for stability).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_imps = []

    for train_idx, _ in skf.split(X, y):
        pipe = Pipeline([
            ("pre", get_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5,
                random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
            )),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        all_imps.append(pipe.named_steps["clf"].feature_importances_)

    importances = np.mean(all_imps, axis=0)
    return importances / importances.sum()  # normalise


def compute_lr_importance(X: np.ndarray, y: np.ndarray,
                           feature_names: list[str]) -> np.ndarray:
    """
    Train a Logistic Regression on the full dataset. Return the absolute
    standardised coefficients, normalised to sum to 1.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_coefs = []

    for train_idx, _ in skf.split(X, y):
        pipe = Pipeline([
            ("pre", get_preprocessor()),
            ("clf", LogisticRegression(
                C=1.0, max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced",
            )),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        all_coefs.append(np.abs(pipe.named_steps["clf"].coef_[0]))

    coefs = np.mean(all_coefs, axis=0)
    return coefs / coefs.sum()


def run_feature_importance(out_dir: Path = RESULTS_DIR) -> dict:
    """
    Compute and save feature importances for both RF and LR.

    Returns a dict with "random_forest" and "logistic_regression" sub-dicts,
    each mapping indicator name to importance score.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined()
    X, y, feature_names = build_features(df)

    # Random Forest
    log.info("Computing Random Forest feature importances (5-fold avg)...")
    rf_imp = compute_rf_importance(X, y, feature_names)
    _bar_plot(
        rf_imp, feature_names,
        title="Random Forest — feature importance (Gini, normalised)",
        xlabel="Normalised importance",
        out_path=out_dir / "feature_importance_rf.png",
        color="#4472c4",
    )

    # Logistic Regression
    log.info("Computing Logistic Regression coefficients (5-fold avg)...")
    lr_imp = compute_lr_importance(X, y, feature_names)
    _bar_plot(
        lr_imp, feature_names,
        title="Logistic Regression — |coefficient| (standardised, normalised)",
        xlabel="Normalised |coefficient|",
        out_path=out_dir / "feature_importance_lr.png",
        color="#e05c2a",
    )

    # Save JSON
    result = {
        "random_forest": {
            name: round(float(imp), 5)
            for name, imp in zip(feature_names, rf_imp)
        },
        "logistic_regression": {
            name: round(float(imp), 5)
            for name, imp in zip(feature_names, lr_imp)
        },
    }
    out_path = out_dir / "feature_importance.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved feature importance to {out_path}")

    # Print table
    print("\n=== Feature importances (normalised) ===")
    print(f"{'Indicator':<28} {'RF':>8} {'LR':>8}")
    print("-" * 46)
    for name in feature_names:
        label = INDICATOR_LABELS.get(name, name)
        print(
            f"{label:<28} "
            f"{result['random_forest'][name]:>8.4f} "
            f"{result['logistic_regression'][name]:>8.4f}"
        )

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_feature_importance()
