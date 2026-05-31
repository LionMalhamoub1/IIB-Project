"""
Train flood vs. non-flood classifiers using hydro-climate indicators as features.

Two models are trained and evaluated with stratified 5-fold cross-validation:
  - Logistic Regression (interpretable, linear baseline)
  - Random Forest (captures non-linear interactions, more powerful)

Metrics reported per fold and averaged:
  AUC-ROC, Precision, Recall, F1, and confusion matrix

Saves:
  results/predictive_models/training_metrics.json
  results/predictive_models/roc_comparison.png
  results/predictive_models/confusion_matrices.png

Usage:
  python -m Natural_disruptions.predictive_models.train
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .build_features import build_features, get_preprocessor
from Natural_disruptions.validating_indicators.helper_scripts.load_dataset import load_combined

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "Natural_disruptions" / "results" / "predictive_models"

N_FOLDS      = 5
RANDOM_STATE = 42


# ------------------------------------------------------------------
# Model definitions
# ------------------------------------------------------------------

def _make_models(n_features: int) -> dict[str, Pipeline]:
    """
    Return a dict of named sklearn Pipelines (preprocessor + classifier).

    Logistic Regression uses a moderate L2 penalty; C is set relative to
    sample count so it doesn't need tuning as dataset size changes.
    Random Forest uses 300 trees — enough for stable importances.
    """
    return {
        "Logistic Regression": Pipeline([
            ("pre",  get_preprocessor()),
            ("clf",  LogisticRegression(
                C=1.0, max_iter=1000, random_state=RANDOM_STATE,
                class_weight="balanced",
            )),
        ]),
        "Random Forest": Pipeline([
            ("pre",  get_preprocessor()),
            ("clf",  RandomForestClassifier(
                n_estimators=300, max_depth=None,
                min_samples_leaf=5, random_state=RANDOM_STATE,
                class_weight="balanced", n_jobs=-1,
            )),
        ]),
    }


# ------------------------------------------------------------------
# Cross-validated evaluation
# ------------------------------------------------------------------

def evaluate_model(
    model: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
) -> dict:
    """
    Stratified k-fold cross-validation.

    Returns a dict with per-fold and mean/std metrics, plus the aggregated
    ROC curve arrays (fpr_grid, mean_tpr) for plotting.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = []
    all_probs    = []
    all_labels   = []
    tprs         = []
    fpr_grid     = np.linspace(0, 1, 200)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = model.predict(X_te)

        all_probs.extend(y_prob.tolist())
        all_labels.extend(y_te.tolist())

        fold_auc = roc_auc_score(y_te, y_prob)
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        tprs.append(np.interp(fpr_grid, fpr, tpr))

        fold_metrics.append({
            "fold":      fold_idx + 1,
            "auc":       round(fold_auc, 4),
            "precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_te, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_te, y_pred, zero_division=0), 4),
        })

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr  = np.std(tprs,  axis=0)
    mean_auc = auc(fpr_grid, mean_tpr)

    # Overall confusion matrix on all out-of-fold predictions
    cm = confusion_matrix(all_labels, [1 if p >= 0.5 else 0 for p in all_probs])

    def _mean_std(key):
        vals = [f[key] for f in fold_metrics]
        return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

    return {
        "folds":        fold_metrics,
        "mean_auc":     round(mean_auc, 4),
        "mean_precision": _mean_std("precision")[0],
        "std_precision":  _mean_std("precision")[1],
        "mean_recall":    _mean_std("recall")[0],
        "std_recall":     _mean_std("recall")[1],
        "mean_f1":        _mean_std("f1")[0],
        "std_f1":         _mean_std("f1")[1],
        "confusion_matrix": cm.tolist(),
        # For plotting
        "_fpr_grid":  fpr_grid.tolist(),
        "_mean_tpr":  mean_tpr.tolist(),
        "_std_tpr":   std_tpr.tolist(),
    }


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_roc_comparison(results: dict[str, dict], out_path: Path) -> None:
    """Overlay mean ROC curves for each model on a single figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.50)")

    colors = ["#4472c4", "#e05c2a", "#2ca02c", "#9467bd"]
    for (name, res), color in zip(results.items(), colors):
        fpr  = np.array(res["_fpr_grid"])
        tpr  = np.array(res["_mean_tpr"])
        std  = np.array(res["_std_tpr"])
        mean_auc = res["mean_auc"]
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name}  (AUC={mean_auc:.3f})")
        ax.fill_between(fpr, tpr - std, tpr + std, alpha=0.15, color=color)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Mean ROC curves ({N_FOLDS}-fold CV)")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved ROC comparison to {out_path}")


def plot_confusion_matrices(results: dict[str, dict], out_path: Path) -> None:
    """Side-by-side confusion matrices for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = np.array(res["confusion_matrix"])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Baseline", "Flood"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10)

    fig.suptitle("Confusion matrices (aggregated across CV folds)", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved confusion matrices to {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def run_training(out_dir: Path = RESULTS_DIR) -> dict:
    """
    Train all models, evaluate with cross-validation, and save results.

    Returns a dict {model_name: evaluation_results} for downstream use
    (feature importance, ablation).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined()
    X, y, feature_names = build_features(df)

    models = _make_models(X.shape[1])
    all_results = {}

    for name, model in models.items():
        log.info(f"Evaluating: {name}")
        res = evaluate_model(model, X, y)
        res["feature_names"] = feature_names
        all_results[name] = res

        log.info(
            f"  AUC={res['mean_auc']:.3f}  "
            f"P={res['mean_precision']:.3f}  "
            f"R={res['mean_recall']:.3f}  "
            f"F1={res['mean_f1']:.3f}"
        )

    # Save metrics (strip private _fpr/_tpr arrays)
    saveable = {
        name: {k: v for k, v in res.items() if not k.startswith("_")}
        for name, res in all_results.items()
    }
    metrics_path = out_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(saveable, f, indent=2)
    log.info(f"Saved training metrics to {metrics_path}")

    # Print summary table
    print("\n=== Model performance (mean ± std across 5-fold CV) ===")
    print(f"{'Model':<22} {'AUC':>7} {'Precision':>12} {'Recall':>9} {'F1':>9}")
    print("-" * 65)
    for name, res in all_results.items():
        print(
            f"{name:<22} {res['mean_auc']:>7.3f} "
            f"{res['mean_precision']:>7.3f}±{res['std_precision']:<4.3f} "
            f"{res['mean_recall']:>5.3f}±{res['std_recall']:<4.3f} "
            f"{res['mean_f1']:>5.3f}±{res['std_f1']:<4.3f}"
        )

    # Plots
    plot_roc_comparison(all_results,      out_dir / "roc_comparison.png")
    plot_confusion_matrices(all_results,  out_dir / "confusion_matrices.png")

    return all_results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_training()
