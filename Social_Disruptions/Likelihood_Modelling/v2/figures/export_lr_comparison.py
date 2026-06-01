# Generates LR model comparison figures (M0-M5) and saves to final_figures/LR_comparison/.
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

_HERE    = Path(__file__).resolve().parent
_V2      = _HERE.parent
EXP_DIR  = _V2 / "data" / "processed" / "expanding_lr"
OUT_DIR  = _V2 / "final_figures" / "LR_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["protest_7d", "strike_7d"]

MODELS = [
    "model0_persistence",
    "model1_markets",
    "model2_full",
    "model3_structural",
    "model4_fao",
    "model_lr_nolag",
]
MODEL_LABELS = {
    "model0_persistence":  "M0\nPersistence",
    "model1_markets":      "M1\nMarkets",
    "model2_full":         "M2\nFull Macro",
    "model3_structural":   "M3\nStructural",
    "model4_fao":          "M4\nFAO+GTA",
    "model_lr_nolag":      "M5\nNo Lags",
}
MODEL_COLORS = {
    "model0_persistence":  "#1f77b4",
    "model1_markets":      "#ff7f0e",
    "model2_full":         "#2ca02c",
    "model3_structural":   "#d62728",
    "model4_fao":          "#9467bd",
    "model_lr_nolag":      "#17becf",
}

# 4 bar styles: (target, year) -> (alpha, hatch, edge)
BAR_STYLES = {
    ("protest_7d", 2020): {"alpha": 0.90, "hatch": "",    "edgecolor": "white"},
    ("protest_7d", 2021): {"alpha": 0.55, "hatch": "...", "edgecolor": "white"},
    ("strike_7d",  2020): {"alpha": 0.90, "hatch": "///", "edgecolor": "white"},
    ("strike_7d",  2021): {"alpha": 0.55, "hatch": "xx",  "edgecolor": "white"},
}
BAR_ORDER = [
    ("protest_7d", 2020),
    ("protest_7d", 2021),
    ("strike_7d",  2020),
    ("strike_7d",  2021),
]
BAR_LABELS = {
    ("protest_7d", 2020): "Protest 2020",
    ("protest_7d", 2021): "Protest 2021",
    ("strike_7d",  2020): "Strike 2020",
    ("strike_7d",  2021): "Strike 2021",
}

STYLE = {
    "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150,
}
plt.rcParams.update(STYLE)

METRICS = {
    "roc_auc":           ("ROC-AUC",          "ROC-AUC",    (0.7, 1.0)),
    "pr_auc":            ("PR-AUC",            "PR-AUC",     (0.4, 1.0)),
    "brier":             ("Brier Score",       "Brier score", None),
    "brier_skill_score": ("Brier Skill Score", "BSS",         None),
}


def load_yearly_averages() -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        p = EXP_DIR / f"metrics_{target}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["model_name"].isin(MODELS)].copy()
        df["year"] = pd.to_datetime(df["month"]).dt.year
        avg = (df.groupby(["model_name", "year"])
               [["roc_auc", "pr_auc", "brier", "brier_skill_score"]]
               .mean()
               .reset_index())
        avg["target"] = target
        rows.append(avg)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def fig_metric(data: pd.DataFrame, metric_key: str) -> None:
    title, ylabel, ylim = METRICS[metric_key]

    n_models  = len(MODELS)
    n_bars    = len(BAR_ORDER)        # 4 bars per model
    bar_w     = 0.18
    group_gap = 0.08                  # extra space between model groups
    group_w   = n_bars * bar_w + group_gap
    x_centres = np.arange(n_models) * group_w
    offsets   = np.array([(i - (n_bars - 1) / 2) * bar_w for i in range(n_bars)])

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(
        f"LR Model Comparison — {title}\n"
        f"Expanding window averages by year (2020 vs 2021)",
        fontsize=11, fontweight="bold",
    )

    legend_handles = []

    for bar_idx, (target, year) in enumerate(BAR_ORDER):
        style   = BAR_STYLES[(target, year)]
        subset  = data[(data["target"] == target) & (data["year"] == year)]
        subset  = subset.set_index("model_name")

        vals = [
            subset.loc[m, metric_key] if m in subset.index else np.nan
            for m in MODELS
        ]
        colors = [MODEL_COLORS[m] for m in MODELS]

        bars = ax.bar(
            x_centres + offsets[bar_idx],
            vals,
            width=bar_w,
            color=colors,
            linewidth=0.4,
            **style,
        )

        # One legend patch per bar style (use neutral grey to show pattern only)
        legend_handles.append(
            mpatches.Patch(
                facecolor="grey",
                alpha=style["alpha"],
                hatch=style["hatch"],
                edgecolor="black",
                linewidth=0.5,
                label=BAR_LABELS[(target, year)],
            )
        )

    ax.set_xticks(x_centres)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)

    if metric_key == "brier_skill_score":
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    ax.legend(handles=legend_handles, fontsize=8,
              loc="upper right" if metric_key != "brier" else "upper left",
              ncol=2)

    plt.tight_layout()
    out = OUT_DIR / f"lr_comparison_{metric_key}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_comparison_metrics(data: pd.DataFrame) -> None:
    if data.empty:
        return
    out = data[["target", "model_name", "year",
                "roc_auc", "pr_auc", "brier", "brier_skill_score"]].copy()
    out["model_label"] = out["model_name"].map(MODEL_LABELS).str.replace("\n", " ")
    for c in ["roc_auc", "pr_auc", "brier", "brier_skill_score"]:
        out[c] = out[c].round(4)
    order = {m: i for i, m in enumerate(MODELS)}
    out["_order"] = out["model_name"].map(order)
    out = out.sort_values(["target", "year", "_order"]).drop(columns=["_order"])
    out.to_csv(OUT_DIR / "lr_comparison_metrics.csv", index=False)
    print(f"Saved: {OUT_DIR / 'lr_comparison_metrics.csv'}  ({len(out)} rows)")


if __name__ == "__main__":
    import shutil
    print(f"Saving LR comparison outputs to {OUT_DIR}\n")

    data = load_yearly_averages()
    if data.empty:
        print("No expanding window data found — check EXP_DIR.")
    else:
        for metric_key in METRICS:
            fig_metric(data, metric_key)
        save_comparison_metrics(data)

    shutil.copy2(__file__, OUT_DIR / "export_lr_comparison.py")
    print(f"Saved: {OUT_DIR / 'export_lr_comparison.py'}")
    print("\nDone.")
