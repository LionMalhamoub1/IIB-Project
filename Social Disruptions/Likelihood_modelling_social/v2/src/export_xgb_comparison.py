"""
export_xgb_comparison.py
Generates XGB comparison figures saved to final figures/XGB_comparison/.

Outputs:
  xgb_performance.csv
  xgb_bss_comparison.png
  lr_vs_xgb_bss.png
  xgb_calibration.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.calibration import calibration_curve
from matplotlib.lines import Line2D

BASE    = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent.parent / "final figures" / "XGB_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS    = [("protest_7d", "Protest 7d"), ("strike_7d", "Strike 7d")]
XGB_MODELS = {"model5_xgb": "M6 XGBoost Full", "model6_xgb_nolag": "M7 XGBoost No Lags"}
LR_MODELS  = {
    "model0_persistence": "M0 Persistence",
    "model4_fao":         "M4 FAO/GTA",
    "model_lr_nolag":     "M5 No Lags",
}
ALL_COMPARE = {**LR_MODELS, **XGB_MODELS}
METRICS     = ["roc_auc", "pr_auc", "brier", "brier_skill_score"]

COLOR_2020 = "#2166ac"
COLOR_2021 = "#d6604d"

plt.rcParams.update({
    "font.size": 11, "font.family": "sans-serif",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_static(models):
    rows = []
    for target, _ in TARGETS:
        df = pd.read_csv(BASE / target / "metrics.csv")
        df = df[df["model_name"].isin(models)].copy()
        df["year"]   = df["fold_id"].map({1: 2020, 2: 2021})
        df["target"] = target
        rows.append(df[["target", "model_name", "year"] + METRICS])
    return pd.concat(rows, ignore_index=True)


def save_xgb_performance():
    rows = []
    for target, _ in TARGETS:
        # Static
        df = pd.read_csv(BASE / target / "metrics.csv")
        df = df[df["model_name"].isin(XGB_MODELS)].copy()
        df["year"]      = df["fold_id"].map({1: 2020, 2: 2021})
        df["eval_type"] = "static"
        df["period"]    = df["year"].astype(str)
        df["target"]    = target
        rows.append(df[["target", "model_name", "eval_type", "period"] + METRICS])
        # Monthly expanding avg
        exp = pd.read_csv(BASE / "expanding_xgb_monthly" / f"metrics_{target}.csv")
        exp = exp[exp["model_name"].isin(XGB_MODELS)].copy()
        exp["year"]      = exp["month"].str[:4].astype(int)
        exp["eval_type"] = "expanding_monthly_avg"
        exp["period"]    = exp["year"].astype(str)
        exp["target"]    = target
        avg = exp.groupby(["target", "model_name", "eval_type", "period"])[METRICS].mean().reset_index()
        rows.append(avg)

    out = pd.concat(rows, ignore_index=True)
    out["model_label"] = out["model_name"].map(XGB_MODELS)
    type_ord  = {"static": 0, "expanding_monthly_avg": 1}
    model_ord = {"model5_xgb": 0, "model6_xgb_nolag": 1}
    out["_t"] = out["eval_type"].map(type_ord)
    out["_m"] = out["model_name"].map(model_ord)
    out = out.sort_values(["target", "_m", "_t", "period"]).drop(columns=["_t", "_m"])
    for c in METRICS:
        out[c] = out[c].round(4)
    out = out[["target", "model_label", "model_name", "eval_type", "period"] + METRICS]
    out.to_csv(OUT_DIR / "xgb_performance.csv", index=False)
    print(f"Saved: {OUT_DIR / 'xgb_performance.csv'}  ({len(out)} rows)")


def fig_xgb_bss_comparison():
    data   = load_static(XGB_MODELS)
    mkeys  = list(XGB_MODELS.keys())
    xlbls  = list(XGB_MODELS.values())
    xpos   = list(range(len(mkeys)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    fig.suptitle("XGBoost Model Comparison — BSS (Static Backtest)",
                 fontsize=12, fontweight="bold")

    for ax, (target, title) in zip(axes, TARGETS):
        sub = data[data["target"] == target].set_index(["model_name", "year"])
        for year, color, ls, lbl in [
            (2020, COLOR_2020, "-",  "2020"),
            (2021, COLOR_2021, "--", "2021"),
        ]:
            vals = [sub.loc[(m, year), "brier_skill_score"]
                    if (m, year) in sub.index else np.nan for m in mkeys]
            ax.plot(xpos, vals, color=color, ls=ls, lw=2.0,
                    marker="o", ms=7, label=lbl, zorder=3)

        ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.6, zorder=2)
        ax.text(len(xlbls) - 1, 0.006, "naive baseline",
                ha="right", va="bottom", fontsize=9, color="black", alpha=0.7)
        ax.yaxis.grid(True, color="#cccccc", lw=0.6, alpha=0.8, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlbls, rotation=45, ha="right", fontsize=11)
        ax.set_ylim(0, 0.6)
        ax.set_ylabel("Brier Skill Score" if ax is axes[0] else "", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "xgb_bss_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def load_expanding_avg(models, lr=True):
    rows = []
    for target, _ in TARGETS:
        if lr:
            df = pd.read_csv(BASE / "expanding_lr" / f"metrics_{target}.csv")
            df["year"] = pd.to_datetime(df["month"]).dt.year
        else:
            df = pd.read_csv(BASE / "expanding_xgb_monthly" / f"metrics_{target}.csv")
            df["year"] = df["month"].str[:4].astype(int)
        df = df[df["model_name"].isin(models)].copy()
        avg = df.groupby(["model_name", "year"])["brier_skill_score"].mean().reset_index()
        avg["target"] = target
        rows.append(avg)
    return pd.concat(rows, ignore_index=True)


def fig_lr_vs_xgb_bss():
    data_lr  = load_expanding_avg(LR_MODELS,  lr=True)
    data_xgb = load_expanding_avg(XGB_MODELS, lr=False)
    data     = pd.concat([data_lr, data_xgb], ignore_index=True)
    mkeys  = list(ALL_COMPARE.keys())
    xlbls  = list(ALL_COMPARE.values())
    n_lr   = len(LR_MODELS)
    n_xgb  = len(XGB_MODELS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle("LR vs XGBoost — BSS (Expanding Window, yearly avg)",
                 fontsize=12, fontweight="bold")

    for ax, (target, title) in zip(axes, TARGETS):
        sub = data[data["target"] == target].set_index(["model_name", "year"])
        for year, color, ls, lbl in [
            (2020, COLOR_2020, "-",  "2020"),
            (2021, COLOR_2021, "--", "2021"),
        ]:
            vals_lr  = [sub.loc[(m, year), "brier_skill_score"]
                        if (m, year) in sub.index else np.nan
                        for m in list(LR_MODELS.keys())]
            vals_xgb = [sub.loc[(m, year), "brier_skill_score"]
                        if (m, year) in sub.index else np.nan
                        for m in list(XGB_MODELS.keys())]
            ax.plot(list(range(n_lr)), vals_lr,
                    color=color, ls=ls, lw=2.0, marker="o", ms=7, label=lbl, zorder=3)
            ax.plot(list(range(n_lr, n_lr + n_xgb)), vals_xgb,
                    color=color, ls=ls, lw=2.0, marker="s", ms=7, zorder=3)

        ax.axvspan(-0.5,        n_lr - 0.5,         alpha=0.04, color="#2166ac", zorder=0)
        ax.axvspan(n_lr - 0.5,  n_lr + n_xgb - 0.5, alpha=0.04, color="#d62728", zorder=0)
        ax.axvline(n_lr - 0.5, color="grey", lw=1.0, ls=":", alpha=0.5)
        ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.6
        ax.text(n_lr / 2 - 0.5,           0.52, "LR",  ha="center", fontsize=10,
                color="#2166ac", alpha=0.8, fontweight="bold")
        ax.text(n_lr + n_xgb / 2 - 0.5,   0.52, "XGB", ha="center", fontsize=10,
                color="#d62728", alpha=0.8, fontweight="bold")
        ax.axhline(0, color="black", lw=1.2, ls="--", alpha=0.6, zorder=2)
        ax.yaxis.grid(True, color="#cccccc", lw=0.6, alpha=0.8, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_xticks(list(range(len(mkeys))))
        ax.set_xticklabels(xlbls, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(0, 0.55)
        ax.set_ylabel("Brier Skill Score" if ax is axes[0] else "", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    legend_handles = [
        Line2D([0], [0], color=COLOR_2020, lw=2, marker="o", ms=7, label="2020"),
        Line2D([0], [0], color=COLOR_2021, lw=2, marker="o", ms=7, label="2021"),
        Line2D([0], [0], color="grey",     lw=2, marker="o", ms=7, label="LR (circle)"),
        Line2D([0], [0], color="grey",     lw=2, marker="s", ms=7, label="XGB (square)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "lr_vs_xgb_bss.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_xgb_calibration():
    fold_colors = {1: COLOR_2020, 2: COLOR_2021}
    fold_labels = {1: "2020 (fold 1)", 2: "2021 (fold 2)"}

    fig = plt.figure(figsize=(13, 8))
    fig.suptitle("M6 XGBoost Full — Calibration (Reliability Diagrams)",
                 fontsize=12, fontweight="bold")

    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.28)

    for col, (target, title) in enumerate(TARGETS):
        preds = pd.read_parquet(BASE / target / "preds.parquet")
        preds = preds[preds["model_name"] == "model5_xgb"].copy()

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col], height_ratios=[3, 1], hspace=0.08)
        ax_rel = fig.add_subplot(inner[0])
        ax_his = fig.add_subplot(inner[1], sharex=ax_rel)

        ax_rel.plot([0, 1], [0, 1], color="grey", ls="--", lw=1.2,
                    alpha=0.7, label="Perfect calibration", zorder=2)

        bins  = np.linspace(0, 1, 11)
        bw    = (bins[1] - bins[0]) * 0.44

        for fold_id in [1, 2]:
            sub    = preds[preds["fold_id"] == fold_id]
            y_true = sub["y_true"].dropna().values.astype(int)
            y_pred = sub.loc[sub["y_true"].notna(), "y_pred"].values

            frac_pos, mean_pred = calibration_curve(
                y_true, y_pred, n_bins=10, strategy="uniform")
            color = fold_colors[fold_id]
            ax_rel.plot(mean_pred, frac_pos, color=color, lw=2.0,
                        marker="o", ms=6, label=fold_labels[fold_id], zorder=3)

            hist, _ = np.histogram(y_pred, bins=bins)
            hist_n  = hist / hist.sum()
            offset  = (fold_id - 1.5) * bw
            ax_his.bar(bins[:-1] + (bins[1] - bins[0]) / 2 + offset,
                       hist_n, width=bw, color=color, alpha=0.75,
                       edgecolor="white", lw=0.3)

        ax_rel.set_xlim(0, 1)
        ax_rel.set_ylim(0, 1)
        ax_rel.set_ylabel("Fraction of positives", fontsize=11)
        ax_rel.set_title(title, fontsize=11, fontweight="bold")
        ax_rel.yaxis.grid(True, color="#cccccc", lw=0.5, alpha=0.7, zorder=0)
        ax_rel.xaxis.grid(False)
        ax_rel.set_axisbelow(True)
        ax_rel.spines["left"].set_linewidth(0.8)
        ax_rel.spines["bottom"].set_linewidth(0.8)
        ax_rel.legend(fontsize=9, loc="upper left", frameon=False)
        plt.setp(ax_rel.get_xticklabels(), visible=False)

        ax_his.set_xlabel("Mean predicted probability", fontsize=11)
        ax_his.set_ylabel("Fraction\nof samples", fontsize=9)
        ax_his.yaxis.grid(True, color="#cccccc", lw=0.5, alpha=0.7, zorder=0)
        ax_his.xaxis.grid(False)
        ax_his.set_axisbelow(True)
        ax_his.spines["left"].set_linewidth(0.8)
        ax_his.spines["bottom"].set_linewidth(0.8)
        ax_his.spines["top"].set_visible(False)
        ax_his.spines["right"].set_visible(False)

    out = OUT_DIR / "xgb_calibration.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    import shutil
    print(f"Saving XGB comparison outputs to {OUT_DIR}\n")
    save_xgb_performance()
    fig_xgb_bss_comparison()
    fig_lr_vs_xgb_bss()
    fig_xgb_calibration()
    shutil.copy2(__file__, OUT_DIR / "export_xgb_comparison.py")
    print(f"\nDone.")
