"""
Leave-one-indicator-out ablation study.

For each indicator, trains a Random Forest with that indicator removed and
measures the drop in AUC-ROC relative to the full model. A large drop means
the removed indicator was carrying significant discriminative signal; a small
drop means the model can compensate with the remaining features.

Also reports the AUC improvement when each indicator is added to an otherwise
empty model (single-feature baseline), showing each indicator's standalone
value.

Saves:
  results/predictive_models/ablation_metrics.json
  results/predictive_models/ablation_loo.png       — AUC drop (leave-one-out)
  results/predictive_models/ablation_single.png    — AUC (single feature)

Usage:
  python -m Natural_disruptions.predictive_models.ablation
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .build_features import build_features, get_preprocessor
from Natural_disruptions.validating_indicators.helper_scripts.load_dataset import load_combined

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "Natural_disruptions" / "results" / "predictive_models"

N_FOLDS      = 5
RANDOM_STATE = 42

INDICATOR_LABELS = {
    "spi_30d":                "SPI-30",
    "chirps_7d_anom_pct":     "CHIRPS 7-day anom.",
    "era5_soil_moisture_day0":"ERA5 soil moisture",
    "gpm_peak_3h_mm":         "GPM peak 3-hour",
    "jrc_recurrence_pct":     "JRC recurrence",
    "pop_density_km2":        "Pop. density",
}


def _make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, min_samples_leaf=5,
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
    )


def _cv_auc(X: np.ndarray, y: np.ndarray, n_folds: int = N_FOLDS) -> float:
    """Cross-validated mean AUC-ROC for a feature matrix X."""
    if X.shape[1] == 0:
        return 0.5

    skf  = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        pipe = Pipeline([
            ("pre", get_preprocessor()),
            ("clf", _make_rf()),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        y_prob = pipe.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], y_prob))
    return float(np.mean(aucs))


def run_ablation(out_dir: Path = RESULTS_DIR) -> dict:
    """
    Run leave-one-out ablation and single-feature analysis.

    Returns a dict with:
      full_auc               — AUC with all features
      leave_one_out          — {indicator: auc_without, auc_drop}
      single_feature         — {indicator: auc_with_only_this_feature}
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined()
    X, y, feature_names = build_features(df)
    n_features = len(feature_names)

    # --- Full model AUC ---
    log.info(f"Computing full model AUC ({n_features} features)...")
    full_auc = _cv_auc(X, y)
    log.info(f"Full model AUC = {full_auc:.4f}")

    # --- Leave-one-out ---
    loo_results = {}
    for i, col in enumerate(feature_names):
        mask   = [j for j in range(n_features) if j != i]
        X_loo  = X[:, mask]
        auc_wo = _cv_auc(X_loo, y)
        drop   = full_auc - auc_wo
        loo_results[col] = {
            "auc_without": round(auc_wo, 4),
            "auc_drop":    round(drop, 4),
        }
        label = INDICATOR_LABELS.get(col, col)
        log.info(f"  LOO [{label}]: AUC={auc_wo:.4f}  drop={drop:+.4f}")

    # --- Single feature ---
    single_results = {}
    for i, col in enumerate(feature_names):
        X_single     = X[:, [i]]
        auc_single   = _cv_auc(X_single, y)
        single_results[col] = round(auc_single, 4)
        label = INDICATOR_LABELS.get(col, col)
        log.info(f"  Single [{label}]: AUC={auc_single:.4f}")

    result = {
        "full_auc":       round(full_auc, 4),
        "n_features":     n_features,
        "feature_names":  feature_names,
        "leave_one_out":  loo_results,
        "single_feature": single_results,
    }

    # Save JSON
    metrics_path = out_dir / "ablation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"Saved ablation metrics to {metrics_path}")

    # Print table
    print(f"\n=== Ablation study (RF, {N_FOLDS}-fold CV) ===")
    print(f"Full model AUC = {full_auc:.4f}\n")
    print(f"{'Indicator':<28} {'AUC w/o':>8} {'Drop':>8} {'Single':>8}")
    print("-" * 58)
    for col in feature_names:
        label = INDICATOR_LABELS.get(col, col)
        print(
            f"{label:<28} "
            f"{loo_results[col]['auc_without']:>8.4f} "
            f"{loo_results[col]['auc_drop']:>+8.4f} "
            f"{single_results[col]:>8.4f}"
        )

    # --- Plots ---
    _plot_loo(loo_results, feature_names, full_auc, out_dir / "ablation_loo.png")
    _plot_single(single_results, feature_names, full_auc, out_dir / "ablation_single.png")

    return result


def _plot_loo(loo: dict, feature_names: list[str], full_auc: float, out_path: Path) -> None:
    """Horizontal bar chart of AUC drop when each indicator is removed."""
    labels = [INDICATOR_LABELS.get(c, c) for c in feature_names]
    drops  = [loo[c]["auc_drop"] for c in feature_names]

    order   = np.argsort(drops)[::-1]
    s_labels = [labels[i] for i in order]
    s_drops  = [drops[i]  for i in order]

    colors = ["#d94f3d" if d >= 0 else "#5aad69" for d in s_drops]

    fig, ax = plt.subplots(figsize=(6, 0.55 * len(feature_names) + 1.8))
    bars = ax.barh(range(len(s_labels)), s_drops, color=colors, alpha=0.85)
    ax.set_yticks(range(len(s_labels)))
    ax.set_yticklabels(s_labels, fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("AUC drop (positive = removing hurts)")
    ax.set_title(f"Leave-one-out ablation\n(full model AUC = {full_auc:.3f})")
    ax.bar_label(bars, fmt="%+.4f", fontsize=8, padding=3)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved LOO plot to {out_path}")


def _plot_single(single: dict, feature_names: list[str], full_auc: float, out_path: Path) -> None:
    """Bar chart of AUC when each indicator is used alone."""
    labels = [INDICATOR_LABELS.get(c, c) for c in feature_names]
    aucs   = [single[c] for c in feature_names]

    order    = np.argsort(aucs)[::-1]
    s_labels = [labels[i] for i in order]
    s_aucs   = [aucs[i]   for i in order]

    fig, ax = plt.subplots(figsize=(6, 0.55 * len(feature_names) + 1.8))
    bars = ax.barh(range(len(s_labels)), s_aucs, color="#4472c4", alpha=0.85)
    ax.set_yticks(range(len(s_labels)))
    ax.set_yticklabels(s_labels, fontsize=10)
    ax.axvline(full_auc, color="red", linewidth=1.2, linestyle="--",
               label=f"Full model ({full_auc:.3f})")
    ax.axvline(0.5, color="grey", linewidth=0.8, linestyle=":",
               label="Random (0.500)")
    ax.set_xlabel("AUC-ROC (single feature only)")
    ax.set_title("Single-feature AUC — standalone discriminability")
    ax.bar_label(bars, fmt="%.4f", fontsize=8, padding=3)
    ax.set_xlim(0.4, max(full_auc + 0.05, max(s_aucs) + 0.05))
    ax.legend(fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved single-feature plot to {out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_ablation()
