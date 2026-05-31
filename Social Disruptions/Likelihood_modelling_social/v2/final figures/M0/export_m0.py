"""
export_m0.py
Generates all M0 (Persistence) figures and saves them to final figures/M0/.

Outputs:
  m0_static_timelines_protest_7d.png   -- country probability timelines, static backtest
  m0_static_timelines_strike_7d.png
  m0_expanding_timelines_protest_7d.png -- country timelines with retraining boundaries
  m0_expanding_timelines_strike_7d.png
  m0_expanding_performance.png          -- monthly ROC-AUC and BSS, both targets
  m0_metrics.csv                        -- all metrics for M0 (static + expanding)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).resolve().parent
_V2      = _HERE.parent
PROC_DIR = _V2 / "data" / "processed"
EXP_DIR  = PROC_DIR / "expanding_lr"
OUT_DIR  = _V2 / "final figures" / "M0"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["protest_7d", "strike_7d"]
TARGET_LABELS = {"protest_7d": "Protest (7-day)", "strike_7d": "Strike (7-day)"}

ILLUSTRATIVE_COUNTRIES = {"ARG": "Argentina", "CHL": "Chile", "BRA": "Brazil",
                           "TUR": "Turkiye",   "KEN": "Kenya"}

STYLE = {
    "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
}
plt.rcParams.update(STYLE)

MODEL_NAME = "model0_persistence"
COLOR      = "#1f77b4"


# ---------------------------------------------------------------------------
# Static backtest: country timelines
# ---------------------------------------------------------------------------

def fig_m0_static_timelines(target: str) -> None:
    p = PROC_DIR / target / "preds.parquet"
    if not p.exists():
        print(f"No static preds for {target}")
        return

    preds = pd.read_parquet(p)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[preds["model_name"] == MODEL_NAME]
    if preds.empty:
        print(f"No M0 predictions in {target}")
        return

    countries = [c for c in ILLUSTRATIVE_COUNTRIES if c in preds["country_iso3"].values]

    n = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    fig.suptitle(
        f"M0 Persistence — Predicted Probability Timelines\n"
        f"{TARGET_LABELS[target]} | Static backtest (test folds 2020–2021)",
        fontsize=11, fontweight="bold",
    )

    for ax, iso3 in zip(axes[:, 0], countries):
        c = preds[preds["country_iso3"] == iso3].sort_values("date")
        if c.empty:
            continue

        ax.fill_between(c["date"], c["y_pred"], alpha=0.15, color=COLOR)
        ax.plot(c["date"], c["y_pred"], color=COLOR, lw=1.5, label="M0 Persistence")

        events = c[c["y_true"] == 1]
        if not events.empty:
            ax.vlines(events["date"], 0, 0.12, color="grey", lw=0.8,
                      alpha=0.5, label="Event day")

        # Fold boundary
        boundary = pd.Timestamp("2021-01-01")
        d_min, d_max = c["date"].min(), c["date"].max()
        if d_max >= boundary >= d_min:
            ax.axvline(boundary, color="grey", lw=1, ls=":", alpha=0.7)
            ax.text(boundary, 0.95, " 2021", fontsize=7, color="grey",
                    va="top", transform=ax.get_xaxis_transform())
        ax.set_xlim(d_min, d_max)

        name = ILLUSTRATIVE_COUNTRIES.get(iso3, iso3)
        ax.set_title(f"{name} ({iso3})", fontsize=9, loc="left", fontweight="bold")
        ax.set_ylabel("P(event)", fontsize=8)
        ax.set_ylim(0, 1.05)
        if ax is axes[0, 0]:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / f"m0_static_timelines_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Expanding window: country timelines with retraining boundaries
# ---------------------------------------------------------------------------

def fig_m0_expanding_timelines(target: str) -> None:
    p = EXP_DIR / f"preds_{target}.parquet"
    if not p.exists():
        print(f"No expanding preds for {target}")
        return

    preds = pd.read_parquet(p)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[preds["model_name"] == MODEL_NAME]
    if preds.empty:
        print(f"No M0 expanding predictions for {target}")
        return

    countries = [c for c in ILLUSTRATIVE_COUNTRIES if c in preds["country_iso3"].values]
    retrain_months = sorted(pd.to_datetime(preds["retrain_month"].unique()))

    n = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    fig.suptitle(
        f"M0 Persistence — Expanding Window Timelines\n"
        f"{TARGET_LABELS[target]} | Monthly retraining 2020–2021",
        fontsize=11, fontweight="bold",
    )

    for ax, iso3 in zip(axes[:, 0], countries):
        c = preds[preds["country_iso3"] == iso3].sort_values("date")
        if c.empty:
            continue

        ax.fill_between(c["date"], c["y_pred"], alpha=0.15, color=COLOR)
        ax.plot(c["date"], c["y_pred"], color=COLOR, lw=1.5)

        events = c[c["y_true"] == 1]
        if not events.empty:
            ax.vlines(events["date"], 0, 0.12, color="grey", lw=0.8, alpha=0.5)

        d_min, d_max = c["date"].min(), c["date"].max()

        # Red dotted lines at each retraining boundary
        first = True
        for rm in retrain_months:
            if d_min <= rm <= d_max:
                ax.axvline(rm, color="#d62728", lw=0.7, ls=":", alpha=0.6,
                           label="Retrain" if first else None)
                first = False

        ax.set_xlim(d_min, d_max)
        name = ILLUSTRATIVE_COUNTRIES.get(iso3, iso3)
        ax.set_title(f"{name} ({iso3})", fontsize=9, loc="left", fontweight="bold")
        ax.set_ylabel("P(event)", fontsize=8)
        ax.set_ylim(0, 1.05)
        if ax is axes[0, 0]:
            handles = [
                mpatches.Patch(color=COLOR, label="M0 Persistence"),
                plt.Line2D([0], [0], color="#d62728", lw=1, ls=":", label="Retrain boundary"),
                plt.Line2D([0], [0], color="grey", lw=1, alpha=0.5, label="Event day"),
            ]
            ax.legend(handles=handles, fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / f"m0_expanding_timelines_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Expanding window: monthly ROC-AUC and BSS, one figure per target
# ---------------------------------------------------------------------------

def fig_m0_expanding_performance(target: str) -> None:
    p = EXP_DIR / f"metrics_{target}.csv"
    if not p.exists():
        print(f"No expanding metrics for {target}")
        return

    df = pd.read_csv(p)
    df = df[df["model_name"] == MODEL_NAME].copy()
    df["month_dt"] = pd.to_datetime(df["month"])
    df = df.sort_values("month_dt")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"M0 Persistence — Expanding Window Monthly Performance\n"
        f"{TARGET_LABELS[target]}, 2020–2021",
        fontsize=11, fontweight="bold",
    )

    axes[0].plot(df["month_dt"], df["roc_auc"],
                 color=COLOR, lw=1.8, marker="o", ms=4)
    axes[1].plot(df["month_dt"], df["brier_skill_score"],
                 color=COLOR, lw=1.8, marker="o", ms=4)

    boundary = pd.Timestamp("2021-01-01")
    for ax in axes:
        ax.axvline(boundary, color="grey", lw=1.0, ls=":", alpha=0.7)
        ax.text(boundary, ax.get_ylim()[0] + 0.01, " 2021", fontsize=7, color="grey")
        ax.set_xlabel("Month", fontsize=9)
        ax.tick_params(axis="x", rotation=30)

    axes[0].set_title("ROC-AUC", fontsize=10)
    axes[0].set_ylim(0.7, 1.0)
    axes[1].set_title("Brier Skill Score", fontsize=10)
    axes[1].axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    plt.tight_layout()
    out = OUT_DIR / f"m0_expanding_performance_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Coefficient importance
# ---------------------------------------------------------------------------

FEATURE_LABELS = {
    "gdelt_protest_7d_lag":    "Protests last 7d",
    "gdelt_protest_28d_lag":   "Protests last 28d",
    "gdelt_strike_7d_lag":     "Strikes last 7d",
    "gdelt_strike_28d_lag":    "Strikes last 28d",
    "gdelt_protest_region_14d": "Regional protests 14d",
    "gdelt_strike_region_14d":  "Regional strikes 14d",
}


def fig_m0_coefficients(target: str) -> None:
    p = PROC_DIR / target / "coefs_lr.csv"
    if not p.exists():
        print(f"No coefs_lr.csv for {target}")
        return

    coefs = pd.read_csv(p)
    coefs = coefs[coefs["model_name"] == MODEL_NAME].copy()
    coefs = coefs[~coefs["feature"].str.startswith("fe__")]
    if coefs.empty:
        return

    coefs["feat_clean"] = coefs["feature"].str.replace("num__", "").str.replace("remainder__", "")
    coefs["label"] = coefs["feat_clean"].map(FEATURE_LABELS).fillna(
        coefs["feat_clean"].str.replace("_", " ").str.title()
    )

    avg = (coefs.groupby(["feat_clean", "label"])["coefficient"]
           .mean().reset_index())
    avg["abs_coef"] = avg["coefficient"].abs()
    avg = avg.sort_values("abs_coef")

    fig, ax = plt.subplots(figsize=(8, max(4, len(avg) * 0.55)))
    ax.barh(avg["label"], avg["coefficient"], color=COLOR,
            edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Coefficient (standardised features)", fontsize=10)
    ax.set_title(
        f"M0 Persistence — Feature Coefficients\n{TARGET_LABELS[target]} | averaged over folds",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = OUT_DIR / f"m0_coefficients_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Metrics CSV
# ---------------------------------------------------------------------------

def save_m0_metrics() -> None:
    rows = []

    # Static backtest
    for target in TARGETS:
        p = PROC_DIR / target / "metrics.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["model_name"] == MODEL_NAME].copy()
        df["evaluation"] = "static"
        df["target"] = target
        rows.append(df[["target", "evaluation", "fold_id",
                         "roc_auc", "pr_auc", "brier", "brier_skill_score", "pos_rate"]])

    # Expanding window (monthly summary)
    for target in TARGETS:
        p = EXP_DIR / f"metrics_{target}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["model_name"] == MODEL_NAME].copy()
        df["evaluation"] = "expanding"
        df["target"] = target
        df.rename(columns={"month": "fold_id"}, inplace=True)
        rows.append(df[["target", "evaluation", "fold_id",
                         "roc_auc", "pr_auc", "brier", "brier_skill_score"]])

    if rows:
        out = pd.concat(rows, ignore_index=True)
        for c in ["roc_auc", "pr_auc", "brier", "brier_skill_score"]:
            out[c] = out[c].round(4)
        out.to_csv(OUT_DIR / "m0_metrics.csv", index=False)
        print(f"Saved: {OUT_DIR / 'm0_metrics.csv'}  ({len(out)} rows)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    print(f"Saving M0 outputs to {OUT_DIR}\n")
    for target in TARGETS:
        fig_m0_static_timelines(target)
        fig_m0_expanding_timelines(target)
        fig_m0_expanding_performance(target)
        fig_m0_coefficients(target)
    save_m0_metrics()

    # Copy this script into the M0 folder for reproducibility
    shutil.copy2(__file__, OUT_DIR / "export_m0.py")
    print(f"Saved: {OUT_DIR / 'export_m0.py'}")
    print("\nDone.")
