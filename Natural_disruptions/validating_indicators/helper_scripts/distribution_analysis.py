"""
Compare hydro-climate indicator distributions between flood events and
non-flood baseline samples.

For each indicator this module computes:
  - Coverage rate (fraction of non-null values) for each class
  - Summary statistics: mean, median, std, 10th/90th percentiles
  - AUC-ROC: how well the indicator alone separates flood from non-flood
  - KS statistic: whether the two distributions are significantly different

Saves:
  results/<timestamp>/indicator_metrics.json   — all numeric metrics
  results/<timestamp>/distributions.png        — violin plots per indicator
  results/<timestamp>/roc_curves.png           — ROC curve per indicator

Usage:
  python -m Natural_disruptions.validating_indicators.distribution_analysis
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score, roc_curve

from .load_dataset import INDICATOR_COLS, load_combined

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "Natural_disruptions" / "results" / "validating_indicators"

# Human-readable labels for plots
INDICATOR_LABELS = {
    "spi_30d":                "SPI-30\n(z-score)",
    "chirps_7d_anom_pct":     "CHIRPS 7-day\nanom. (%)",
    "era5_soil_moisture_day0":"ERA5 soil\nmoisture",
    "gpm_peak_3h_mm":         "GPM peak\n3-hour (mm)",
    "jrc_recurrence_pct":     "JRC recurrence\n(%)",
    "pop_density_km2":        "Pop. density\n(km²)",
}


# ------------------------------------------------------------------
# Metric computation
# ------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute per-indicator discriminative metrics.

    Returns a dict structured as:
      {indicator: {coverage_flood, coverage_baseline, auc, ks_stat, ks_pvalue,
                   mean_flood, median_flood, std_flood,
                   mean_baseline, median_baseline, std_baseline, ...}}
    """
    metrics = {}
    floods   = df[df["label"] == 1]
    baseline = df[df["label"] == 0]

    for col in INDICATOR_COLS:
        if col not in df.columns:
            log.warning(f"Indicator {col} not in dataset — skipping")
            continue

        f_vals = floods[col].dropna()
        b_vals = baseline[col].dropna()

        cov_flood    = len(f_vals) / len(floods)   if len(floods)   > 0 else 0
        cov_baseline = len(b_vals) / len(baseline) if len(baseline) > 0 else 0

        # AUC-ROC: treat indicator as score (higher = more flood-like)
        # Both classes need at least a few points
        auc = None
        if len(f_vals) >= 5 and len(b_vals) >= 5:
            all_vals = np.concatenate([f_vals.values, b_vals.values])
            all_labs = np.concatenate([np.ones(len(f_vals)), np.zeros(len(b_vals))])
            # AUC is symmetric around 0.5; take max(auc, 1-auc) so direction doesn't matter
            raw_auc = roc_auc_score(all_labs, all_vals)
            auc = max(raw_auc, 1 - raw_auc)

        # Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = (None, None)
        if len(f_vals) >= 5 and len(b_vals) >= 5:
            res = scipy_stats.ks_2samp(f_vals.values, b_vals.values)
            ks_stat, ks_pvalue = round(float(res.statistic), 4), round(float(res.pvalue), 6)

        def _sumstats(s: pd.Series) -> dict:
            if len(s) == 0:
                return {k: None for k in ("mean", "median", "std", "p10", "p90")}
            return {
                "mean":   round(float(s.mean()), 4),
                "median": round(float(s.median()), 4),
                "std":    round(float(s.std()), 4),
                "p10":    round(float(s.quantile(0.10)), 4),
                "p90":    round(float(s.quantile(0.90)), 4),
            }

        metrics[col] = {
            "coverage_flood":    round(cov_flood, 4),
            "coverage_baseline": round(cov_baseline, 4),
            "auc":               round(auc, 4) if auc is not None else None,
            "ks_stat":           ks_stat,
            "ks_pvalue":         ks_pvalue,
            "n_flood":           len(f_vals),
            "n_baseline":        len(b_vals),
            "flood":             _sumstats(f_vals),
            "baseline":          _sumstats(b_vals),
        }

    return metrics


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def _clip_for_plot(series: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    """Clip outliers for cleaner violin plots."""
    q_lo = series.quantile(lo)
    q_hi = series.quantile(hi)
    return series.clip(q_lo, q_hi)


def plot_distributions(df: pd.DataFrame, out_path: Path) -> None:
    """
    Violin plots comparing flood vs. baseline distribution for each indicator.
    """
    available = [c for c in INDICATOR_COLS if c in df.columns and df[c].notna().any()]
    n = len(available)
    if n == 0:
        log.warning("No indicators with data to plot")
        return

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5))
    if n == 1:
        axes = [axes]

    floods   = df[df["label"] == 1]
    baseline = df[df["label"] == 0]

    for ax, col in zip(axes, available):
        f_vals = _clip_for_plot(floods[col].dropna())
        b_vals = _clip_for_plot(baseline[col].dropna())

        data_to_plot = [b_vals.values, f_vals.values]
        parts = ax.violinplot(data_to_plot, positions=[0, 1],
                              showmedians=True, showextrema=False)

        # Style: baseline = grey, flood = cornflowerblue
        colors = ["#aaaaaa", "#4472c4"]
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "Flood"], fontsize=9)
        ax.set_title(INDICATOR_LABELS.get(col, col), fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Indicator distributions: flood vs. non-flood baseline", fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved distribution plot to {out_path}")


def plot_roc_curves(df: pd.DataFrame, out_path: Path) -> None:
    """
    Per-indicator ROC curves on a single figure.
    The direction (higher or lower = more flood-like) is chosen automatically.
    """
    available = [c for c in INDICATOR_COLS if c in df.columns and df[c].notna().any()]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (AUC=0.50)")

    colors = plt.cm.tab10.colors
    for i, col in enumerate(available):
        sub = df[[col, "label"]].dropna()
        if len(sub) < 10:
            continue
        y_true  = sub["label"].values
        y_score = sub[col].values
        raw_auc = roc_auc_score(y_true, y_score)
        # Flip score if indicator is negatively oriented
        if raw_auc < 0.5:
            y_score = -y_score
            raw_auc = 1 - raw_auc
        fpr, tpr, _ = roc_curve(y_true, y_score)
        label_name = INDICATOR_LABELS.get(col, col).replace("\n", " ")
        ax.plot(fpr, tpr, color=colors[i % len(colors)],
                linewidth=1.8, label=f"{label_name}  (AUC={raw_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — individual indicator discriminability")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved ROC curve plot to {out_path}")


def plot_coverage(metrics: dict, out_path: Path) -> None:
    """
    Horizontal bar chart showing coverage rate (% non-null) per indicator
    for flood events and baseline separately.
    """
    labels = list(metrics.keys())
    cov_flood = [metrics[c]["coverage_flood"] * 100 for c in labels]
    cov_base  = [metrics[c]["coverage_baseline"] * 100 for c in labels]

    y = np.arange(len(labels))
    height = 0.35
    fig, ax = plt.subplots(figsize=(7, 0.8 * len(labels) + 1.5))
    ax.barh(y + height / 2, cov_flood, height, label="Flood events",  color="#4472c4", alpha=0.85)
    ax.barh(y - height / 2, cov_base,  height, label="Baseline",      color="#aaaaaa", alpha=0.85)

    ax.set_yticks(y)
    readable = [INDICATOR_LABELS.get(c, c).replace("\n", " ") for c in labels]
    ax.set_yticklabels(readable, fontsize=9)
    ax.set_xlabel("Coverage (%)")
    ax.set_title("Indicator coverage — flood vs. baseline")
    ax.set_xlim(0, 105)
    ax.axvline(100, color="grey", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved coverage plot to {out_path}")


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def run_distribution_analysis(out_dir: Path = RESULTS_DIR) -> dict:
    """
    Run the full distribution analysis and save all outputs.

    Returns the metrics dict for downstream use.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined()
    metrics = compute_metrics(df)

    # Save numeric metrics
    metrics_path = out_dir / "indicator_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved metrics to {metrics_path}")

    # Print summary table to console
    print("\n=== Indicator discriminability summary ===")
    header = f"{'Indicator':<28} {'Cov-F':>6} {'Cov-B':>6} {'AUC':>6} {'KS':>6} {'Mean-F':>8} {'Mean-B':>8}"
    print(header)
    print("-" * len(header))
    for col, m in metrics.items():
        print(
            f"{col:<28} "
            f"{m['coverage_flood']*100:>5.1f}% "
            f"{m['coverage_baseline']*100:>5.1f}% "
            f"{(m['auc'] or 0):>6.3f} "
            f"{(m['ks_stat'] or 0):>6.3f} "
            f"{(m['flood']['mean'] or 0):>8.3f} "
            f"{(m['baseline']['mean'] or 0):>8.3f}"
        )

    # Save plots
    plot_distributions(df, out_dir / "distributions.png")
    plot_roc_curves(df,    out_dir / "roc_curves.png")
    plot_coverage(metrics, out_dir / "coverage.png")

    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_distribution_analysis()
