# Generates LR vs XGBoost BSS comparison figures.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent.parent / "final_figures" / "XGB_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [("protest_7d", "Protest 7d"), ("strike_7d", "Strike 7d")]

LR_MODELS = {
    "model0_persistence": "M0\nPersistence",
    "model3_structural":  "M3\nStructural",
    "model_lr_nolag":     "M5\nNo Lags",
}
XGB_MODELS = {
    "model5_xgb":        "M6\nXGB Full",
    "model6_xgb_nolag":  "M7\nXGB No Lags",
}
ALL_MODELS = {**LR_MODELS, **XGB_MODELS}

COLOR_2020 = "#2166ac"
COLOR_2021 = "#d6604d"

plt.rcParams.update({
    "font.size": 11, "font.family": "sans-serif",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})


# ── helpers ──────────────────────────────────────────────────────────────────

def load_expanding_bss():
    rows = []
    for target, _ in TARGETS:
        # LR
        df = pd.read_csv(BASE / "expanding_lr" / f"metrics_{target}.csv")
        df["year"] = pd.to_datetime(df["month"]).dt.year
        df = df[df["model_name"].isin(LR_MODELS)]
        avg = df.groupby(["model_name", "year"])["brier_skill_score"].mean().reset_index()
        avg["target"] = target
        rows.append(avg)
        # XGB monthly
        df = pd.read_csv(BASE / "expanding_xgb_monthly" / f"metrics_{target}.csv")
        df["year"] = df["month"].str[:4].astype(int)
        df = df[df["model_name"].isin(XGB_MODELS)]
        avg = df.groupby(["model_name", "year"])["brier_skill_score"].mean().reset_index()
        avg["target"] = target
        rows.append(avg)
    return pd.concat(rows, ignore_index=True)


def wilson_ci(n_pos, n_total, z=1.96):
    if n_total == 0:
        return np.nan, np.nan
    p = n_pos / n_total
    denom = 1 + z**2 / n_total
    centre = (p + z**2 / (2 * n_total)) / denom
    half   = z * np.sqrt(p * (1 - p) / n_total + z**2 / (4 * n_total**2)) / denom
    return float(np.clip(centre - half, 0, 1)), float(np.clip(centre + half, 0, 1))


# ── Figure 1: lr_vs_xgb_bss.png ─────────────────────────────────────────────

def fig_lr_vs_xgb_bss():
    data = load_expanding_bss()

    mkeys  = list(ALL_MODELS.keys())
    xlbls  = list(ALL_MODELS.values())
    n_lr   = len(LR_MODELS)
    n_xgb  = len(XGB_MODELS)
    n_tot  = len(mkeys)

    bar_w  = 0.35
    x      = np.arange(n_tot)
    offsets = {2020: -bar_w / 2, 2021: bar_w / 2}
    colors  = {2020: COLOR_2020, 2021: COLOR_2021}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        "LR vs XGBoost — BSS (Expanding Window, yearly avg)",
        fontsize=12, fontweight="bold",
    )

    for ax, (target, title) in zip(axes, TARGETS):
        sub = data[data["target"] == target].set_index(["model_name", "year"])

        for year in [2020, 2021]:
            vals = [
                sub.loc[(m, year), "brier_skill_score"]
                if (m, year) in sub.index else np.nan
                for m in mkeys
            ]
            ax.bar(x + offsets[year], vals, bar_w,
                   color=colors[year], alpha=0.85, label=str(year), zorder=3)

        # Light separator between LR and XGB groups
        ax.axvline(n_lr - 0.5, color="grey", lw=1.0, ls=":", alpha=0.5, zorder=2)
        ax.text(n_lr / 2 - 0.5,        0.005, "LR",  ha="center", va="bottom",
                fontsize=9, color="grey", alpha=0.9)
        ax.text(n_lr + n_xgb / 2 - 0.5, 0.005, "XGB", ha="center", va="bottom",
                fontsize=9, color="grey", alpha=0.9)

        ax.axhline(0, color="black", lw=1.0, ls="--", alpha=0.5, zorder=2)
        ax.yaxis.grid(True, color="#cccccc", lw=0.5, alpha=0.7, zorder=0)
        ax.xaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, fontsize=10)
        ax.set_ylim(0, 0.55)
        ax.set_ylabel("Brier Skill Score" if ax is axes[0] else "", fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    patch_2020 = mpatches.Patch(color=COLOR_2020, alpha=0.85, label="2020")
    patch_2021 = mpatches.Patch(color=COLOR_2021, alpha=0.85, label="2021")
    fig.legend(handles=[patch_2020, patch_2021],
               loc="upper center", ncol=2, fontsize=11,
               frameon=False, bbox_to_anchor=(0.5, 1.03))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "lr_vs_xgb_bss.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2: xgb_calibration.png ───────────────────────────────────────────

def fig_xgb_calibration():
    N_BINS = 10
    fold_colors = {1: COLOR_2020, 2: COLOR_2021}
    fold_labels = {1: "Fold 1 — 2020", 2: "Fold 2 — 2021"}

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(
        "M6 XGBoost Full — Reliability Diagrams",
        fontsize=12, fontweight="bold",
    )

    outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.28)

    for col, (target, title) in enumerate(TARGETS):
        preds = pd.read_parquet(BASE / target / "preds.parquet")
        preds = preds[preds["model_name"] == "model5_xgb"].dropna(subset=["y_true"]).copy()

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col],
            height_ratios=[3, 1], hspace=0.06,
        )
        ax_rel = fig.add_subplot(inner[0])
        ax_his = fig.add_subplot(inner[1], sharex=ax_rel)

        # Perfect calibration diagonal
        ax_rel.plot([0, 1], [0, 1], color="#888888", ls="--", lw=1.2,
                    label="Perfect calibration", zorder=2)

        bins   = np.linspace(0, 1, N_BINS + 1)
        bin_c  = (bins[:-1] + bins[1:]) / 2
        bar_w  = (bins[1] - bins[0]) * 0.42

        for fold_id in [1, 2]:
            sub    = preds[preds["fold_id"] == fold_id]
            y_true = sub["y_true"].astype(int).values
            y_pred = sub["y_pred"].values

            # Binned calibration with Wilson CI
            frac_pos, mean_pred, ci_lo, ci_hi = [], [], [], []
            for b_lo, b_hi in zip(bins[:-1], bins[1:]):
                mask    = (y_pred >= b_lo) & (y_pred < b_hi)
                n_total = mask.sum()
                n_pos   = y_true[mask].sum()
                if n_total < 5:
                    continue
                p   = n_pos / n_total
                lo, hi = wilson_ci(n_pos, n_total)
                frac_pos.append(p)
                mean_pred.append(y_pred[mask].mean())
                ci_lo.append(lo)
                ci_hi.append(hi)

            frac_pos  = np.array(frac_pos)
            mean_pred = np.array(mean_pred)
            ci_lo     = np.array(ci_lo)
            ci_hi     = np.array(ci_hi)

            color = fold_colors[fold_id]
            ax_rel.fill_between(mean_pred, ci_lo, ci_hi,
                                alpha=0.15, color=color, zorder=2)
            ax_rel.plot(mean_pred, frac_pos, color=color, lw=2.0,
                        marker="o", ms=6, label=fold_labels[fold_id], zorder=3)

            # Histogram — offset bars so both folds are visible
            hist, _ = np.histogram(y_pred, bins=bins)
            hist_n  = hist / hist.sum()
            offset  = (fold_id - 1.5) * bar_w
            ax_his.bar(bin_c + offset, hist_n, width=bar_w,
                       color=color, alpha=0.75, edgecolor="white", lw=0.3)

        ax_rel.set_xlim(0, 1)
        ax_rel.set_ylim(0, 1)
        ax_rel.set_ylabel("Fraction of positives", fontsize=11)
        ax_rel.set_title(title, fontsize=11, fontweight="bold")
        ax_rel.yaxis.grid(True, color="#cccccc", lw=0.5, alpha=0.7, zorder=0)
        ax_rel.xaxis.grid(False)
        ax_rel.set_axisbelow(True)
        ax_rel.legend(fontsize=9, loc="upper left", frameon=False)
        ax_rel.spines["left"].set_linewidth(0.8)
        ax_rel.spines["bottom"].set_linewidth(0.8)
        plt.setp(ax_rel.get_xticklabels(), visible=False)

        ax_his.set_xlabel("Mean predicted probability", fontsize=11)
        ax_his.set_ylabel("Fraction\nof samples", fontsize=9)
        ax_his.yaxis.grid(True, color="#cccccc", lw=0.5, alpha=0.7, zorder=0)
        ax_his.xaxis.grid(False)
        ax_his.set_axisbelow(True)
        ax_his.spines["top"].set_visible(False)
        ax_his.spines["right"].set_visible(False)
        ax_his.spines["left"].set_linewidth(0.8)
        ax_his.spines["bottom"].set_linewidth(0.8)

    out = OUT_DIR / "xgb_calibration.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    fig_lr_vs_xgb_bss()
    fig_xgb_calibration()
    print("\nDone.")
