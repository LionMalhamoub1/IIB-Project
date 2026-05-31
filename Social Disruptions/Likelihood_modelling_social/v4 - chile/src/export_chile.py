"""
export_chile.py
===============
Generates all Chile v4 figures and a summary comparison CSV.

Outputs (final figures/Chile/):
  chile_metrics.csv              — all metrics, both folds, both targets
  chile_bss_comparison.png       — BSS bar chart M0-M7, protest vs strike
  chile_roc_comparison.png       — ROC-AUC bar chart
  chile_timelines.png            — predicted probability over time for Chile
  chile_lr_coefficients.png      — LR coefficients for best LR model (M4)
  chile_shap.png                 — SHAP importance for M6 XGBoost Full
  chile_vs_pooled.png            — Chile BSS vs v3 pooled-panel Chile BSS
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_SRC     = Path(__file__).resolve().parent
_V4      = _SRC.parent
_V3      = _V4.parent / "v3"
PROC_DIR = _V4 / "data" / "processed"
V3_PROC  = _V3 / "data" / "processed"
OUT_DIR  = _V4 / "final figures" / "Chile"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["protest_7d", "strike_7d"]
TARGET_LABELS = {"protest_7d": "Protest (7-day)", "strike_7d": "Strike (7-day)"}
YEAR_LABELS   = {1: "2020 (Fold 1)", 2: "2021 (Fold 2)"}

MODEL_ORDER = [
    "model0_persistence", "model1_markets", "model2_full",
    "model3_structural",  "model4_fao",     "model_lr_nolag",
    "model5_xgb",         "model6_xgb_nolag",
]
MODEL_LABELS = {
    "model0_persistence": "M0\nPersistence",
    "model1_markets":     "M1\nMarkets",
    "model2_full":        "M2\nFull Macro",
    "model3_structural":  "M3\nStructural",
    "model4_fao":         "M4\nFAO+GTA",
    "model_lr_nolag":     "M5\nNo Lags",
    "model5_xgb":         "M6\nXGB Full",
    "model6_xgb_nolag":   "M7\nXGB\nNo Lags",
}
MODEL_COLORS = {
    "model0_persistence": "#1f77b4",
    "model1_markets":     "#ff7f0e",
    "model2_full":        "#2ca02c",
    "model3_structural":  "#d62728",
    "model4_fao":         "#9467bd",
    "model_lr_nolag":     "#17becf",
    "model5_xgb":         "#8c564b",
    "model6_xgb_nolag":   "#e377c2",
}

STYLE = {
    "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
}
plt.rcParams.update(STYLE)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_metrics() -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        p = PROC_DIR / target / "metrics.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["target"] = target
        df["year"]   = df["fold_id"].map({1: 2020, 2: 2021})
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_v3_metrics() -> pd.DataFrame:
    """Load v3 pooled-panel metrics (full 39-country model) for comparison."""
    rows = []
    for target in TARGETS:
        p = V3_PROC / target / "metrics.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["target"] = target
        df["year"]   = df["fold_id"].map({1: 2020, 2: 2021})
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_metric_comparison(data: pd.DataFrame, metric: str,
                           ylabel: str, fname: str) -> None:
    """Grouped bar chart: one group per model, 4 bars (2 targets × 2 years)."""
    BAR_STYLES = {
        ("protest_7d", 2020): {"alpha": 0.90, "hatch": ""},
        ("protest_7d", 2021): {"alpha": 0.55, "hatch": "..."},
        ("strike_7d",  2020): {"alpha": 0.90, "hatch": "///"},
        ("strike_7d",  2021): {"alpha": 0.55, "hatch": "xx"},
    }
    BAR_LABELS = {
        ("protest_7d", 2020): "Protest 2020",
        ("protest_7d", 2021): "Protest 2021",
        ("strike_7d",  2020): "Strike 2020",
        ("strike_7d",  2021): "Strike 2021",
    }
    n_models = len(MODEL_ORDER)
    n_bars   = 4
    bar_w    = 0.18
    group_w  = n_bars * bar_w + 0.08
    x_centres = np.arange(n_models) * group_w
    offsets   = np.array([(i - (n_bars - 1) / 2) * bar_w for i in range(n_bars)])

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(f"Chile — {ylabel} by Model\n(Static backtest, 2020–2021)",
                 fontsize=11, fontweight="bold")

    legend_handles = []
    bar_order = [("protest_7d", 2020), ("protest_7d", 2021),
                 ("strike_7d", 2020),  ("strike_7d", 2021)]
    for bar_idx, (target, year) in enumerate(bar_order):
        style  = BAR_STYLES[(target, year)]
        subset = data[(data["target"] == target) & (data["year"] == year)]
        subset = subset.set_index("model_name")
        vals   = [subset.loc[m, metric] if m in subset.index else np.nan
                  for m in MODEL_ORDER]
        colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
        ax.bar(x_centres + offsets[bar_idx], vals, width=bar_w,
               color=colors, linewidth=0.4, **style)
        legend_handles.append(mpatches.Patch(
            facecolor="grey", alpha=style["alpha"], hatch=style["hatch"],
            edgecolor="black", linewidth=0.5, label=BAR_LABELS[(target, year)],
        ))

    if metric == "brier_skill_score":
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    ax.set_xticks(x_centres)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", ncol=2)
    plt.tight_layout()
    out = OUT_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_timelines(target: str) -> None:
    p = PROC_DIR / target / "preds.parquet"
    if not p.exists():
        return
    preds = pd.read_parquet(p)
    preds["date"] = pd.to_datetime(preds["date"])

    models_to_show = ["model0_persistence", "model4_fao",
                      "model5_xgb", "model6_xgb_nolag"]
    colors = {m: MODEL_COLORS[m] for m in models_to_show}

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle(f"Chile — Predicted P(event) over Time\n{TARGET_LABELS[target]}",
                 fontsize=11, fontweight="bold")

    for model_name in models_to_show:
        sub = preds[preds["model_name"] == model_name].sort_values("date")
        if sub.empty:
            continue
        ax.plot(sub["date"], sub["y_pred"],
                color=colors[model_name], lw=1.5, alpha=0.85,
                label=MODEL_LABELS[model_name].replace("\n", " "))

    # Event markers
    events = preds[preds["y_true"] == 1]["date"].drop_duplicates().sort_values()
    for d in events:
        ax.axvline(d, color="grey", lw=0.4, alpha=0.3)

    ax.axvline(pd.Timestamp("2021-01-01"), color="black", lw=1.0, ls=":", alpha=0.6)
    ax.text(pd.Timestamp("2021-01-01"), 0.02, " 2021", fontsize=8, color="black")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("P(event)", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", ncol=2, frameon=False)
    plt.tight_layout()
    out = OUT_DIR / f"chile_timelines_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_lr_coefs(target: str) -> None:
    p = PROC_DIR / target / "coefs_lr.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df["model_name"] == "model4_fao"].copy()
    if df.empty:
        return

    # Average across folds
    df["label"] = (df["feature"]
                   .str.replace("num__", "").str.replace("remainder__", "")
                   .str.replace("_", " ").str.title())
    avg = df.groupby("label")["coefficient"].mean().reset_index()
    avg = avg.reindex(avg["coefficient"].abs().sort_values().index)
    avg = avg.tail(25)

    colors = ["#d62728" if c >= 0 else "#1f77b4" for c in avg["coefficient"]]

    fig, ax = plt.subplots(figsize=(9, max(4, len(avg) * 0.4)))
    ax.barh(avg["label"], avg["coefficient"], color=colors,
            edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("LR Coefficient (M4 FAO+GTA, avg over folds)", fontsize=10)
    ax.set_title(f"Chile — LR Feature Coefficients\n{TARGET_LABELS[target]} | Top 25",
                 fontsize=11, fontweight="bold")
    handles = [mpatches.Patch(color="#d62728", label="Positive (increases P)"),
               mpatches.Patch(color="#1f77b4", label="Negative (decreases P)")]
    ax.legend(handles=handles, fontsize=9, frameon=False)
    plt.tight_layout()
    out = OUT_DIR / f"chile_lr_coefs_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_shap(target: str) -> None:
    p = PROC_DIR / target / "shap_importance.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df["model_name"] == "model5_xgb"].copy()
    if df.empty:
        return
    df = df.sort_values("mean_abs_shap").tail(25)
    df["label"] = (df["feature"]
                   .str.replace("num__", "").str.replace("remainder__", "")
                   .str.replace("_", " ").str.title())

    has_sign = "mean_shap" in df.columns
    colors = (["#d62728" if s >= 0 else "#1f77b4" for s in df["mean_shap"]]
              if has_sign else ["#8c564b"] * len(df))

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.42)))
    ax.barh(df["label"], df["mean_abs_shap"], color=colors,
            edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(f"Chile — M6 XGBoost SHAP Importance\n{TARGET_LABELS[target]} | Top 25",
                 fontsize=11, fontweight="bold")
    if has_sign:
        handles = [mpatches.Patch(color="#d62728", label="Increases P(event)"),
                   mpatches.Patch(color="#1f77b4", label="Decreases P(event)")]
        ax.legend(handles=handles, fontsize=9, loc="lower right", frameon=False)
    plt.tight_layout()
    out = OUT_DIR / f"chile_shap_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_vs_pooled(chile_data: pd.DataFrame, v3_data: pd.DataFrame) -> None:
    """Line chart: Chile-only BSS vs pooled-panel BSS for Chile, across models."""
    if chile_data.empty or v3_data.empty:
        print("Skipping vs-pooled chart: missing data.")
        return

    YEAR_STYLE = {
        2020: {"color": "#2166ac", "ls": "-"},
        2021: {"color": "#d62728", "ls": "--"},
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)
    fig.suptitle(
        "Chile-Only Model vs Pooled-Panel Model (39 countries)\n"
        "Brier Skill Score — Static Backtest",
        fontsize=11, fontweight="bold",
    )
    handles_added = False
    for ax, target in zip(axes, TARGETS):
        c_sub = chile_data[chile_data["target"] == target].set_index(["model_name", "year"])
        v_sub = v3_data[v3_data["target"] == target].set_index(["model_name", "year"])

        for year, style in YEAR_STYLE.items():
            c_vals = [c_sub.loc[(m, year), "brier_skill_score"]
                      if (m, year) in c_sub.index else np.nan for m in MODEL_ORDER]
            v_vals = [v_sub.loc[(m, year), "brier_skill_score"]
                      if (m, year) in v_sub.index else np.nan for m in MODEL_ORDER]

            ax.plot(range(len(MODEL_ORDER)), c_vals,
                    color=style["color"], ls=style["ls"], lw=2.0,
                    marker="o", ms=7,
                    label=f"Chile-only {year}" if not handles_added else "_")
            ax.plot(range(len(MODEL_ORDER)), v_vals,
                    color=style["color"], ls=style["ls"], lw=1.2,
                    marker="s", ms=5, alpha=0.5,
                    label=f"Pooled {year}" if not handles_added else "_")
        handles_added = True

        ax.axhline(0, color="black", lw=0.8, ls=":", alpha=0.5)
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([MODEL_LABELS[m].replace("\n", " ") for m in MODEL_ORDER],
                           rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel("Brier Skill Score", fontsize=10)
        ax.set_title(TARGET_LABELS[target], fontsize=11, fontweight="bold")

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = OUT_DIR / "chile_vs_pooled.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_metrics_csv(data: pd.DataFrame) -> None:
    if data.empty:
        return
    cols = ["target", "model_name", "year", "fold_id",
            "roc_auc", "pr_auc", "brier", "brier_skill_score",
            "n_train", "n_test", "pos_rate"]
    cols = [c for c in cols if c in data.columns]
    out = data[cols].copy()
    for c in ["roc_auc", "pr_auc", "brier", "brier_skill_score"]:
        if c in out.columns:
            out[c] = out[c].round(4)
    out["model_label"] = out["model_name"].map(
        {k: v.replace("\n", " ") for k, v in MODEL_LABELS.items()})
    out.to_csv(OUT_DIR / "chile_metrics.csv", index=False)
    print(f"Saved: {OUT_DIR / 'chile_metrics.csv'}  ({len(out)} rows)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import shutil
    print(f"Saving Chile figures to {OUT_DIR}\n")

    data    = load_metrics()
    v3_data = load_v3_metrics()

    if data.empty:
        print("No Chile metrics found — run train_backtest_chile.py first.")
    else:
        save_metrics_csv(data)
        fig_metric_comparison(data, "brier_skill_score", "Brier Skill Score",
                              "chile_bss_comparison.png")
        fig_metric_comparison(data, "roc_auc", "ROC-AUC",
                              "chile_roc_comparison.png")
        for target in TARGETS:
            fig_timelines(target)
            fig_lr_coefs(target)
            fig_shap(target)
        fig_vs_pooled(data, v3_data)

    shutil.copy2(__file__, OUT_DIR / "export_chile.py")
    print(f"\nDone.")
