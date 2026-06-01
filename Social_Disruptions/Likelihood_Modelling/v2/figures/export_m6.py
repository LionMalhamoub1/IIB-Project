# Generates M6 (XGBoost full) figures and saves to final_figures/M6/.
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

_HERE    = Path(__file__).resolve().parent
_V2      = _HERE.parent
PROC_DIR = _V2 / "data" / "processed"
EXP_DIR  = PROC_DIR / "expanding_xgb_monthly"
OUT_DIR  = _V2 / "final_figures" / "M6"
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

MODEL_NAME = "model5_xgb"
COLOR      = "#8c564b"  # brown


def fig_m6_static_timelines(target: str) -> None:
    p = PROC_DIR / target / "preds.parquet"
    if not p.exists():
        print(f"No static preds for {target}"); return

    preds = pd.read_parquet(p)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[preds["model_name"] == MODEL_NAME]
    if preds.empty:
        print(f"No M6 predictions in {target}"); return

    countries = [c for c in ILLUSTRATIVE_COUNTRIES if c in preds["country_iso3"].values]
    n = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    fig.suptitle(
        f"M6 XGBoost Full — Predicted Probability Timelines\n"
        f"{TARGET_LABELS[target]} | Static backtest (test folds 2020–2021)",
        fontsize=11, fontweight="bold",
    )

    for ax, iso3 in zip(axes[:, 0], countries):
        c = preds[preds["country_iso3"] == iso3].sort_values("date")
        if c.empty:
            continue
        ax.fill_between(c["date"], c["y_pred"], alpha=0.15, color=COLOR)
        ax.plot(c["date"], c["y_pred"], color=COLOR, lw=1.5, label="M6 XGBoost Full")
        events = c[c["y_true"] == 1]
        if not events.empty:
            ax.vlines(events["date"], 0, 0.12, color="grey", lw=0.8, alpha=0.5, label="Event day")
        boundary = pd.Timestamp("2021-01-01")
        d_min, d_max = c["date"].min(), c["date"].max()
        if d_max >= boundary >= d_min:
            ax.axvline(boundary, color="grey", lw=1, ls=":", alpha=0.7)
            ax.text(boundary, 0.95, " 2021", fontsize=7, color="grey",
                    va="top", transform=ax.get_xaxis_transform())
        ax.set_xlim(d_min, d_max)
        ax.set_title(f"{ILLUSTRATIVE_COUNTRIES.get(iso3, iso3)} ({iso3})",
                     fontsize=9, loc="left", fontweight="bold")
        ax.set_ylabel("P(event)", fontsize=8)
        ax.set_ylim(0, 1.05)
        if ax is axes[0, 0]:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / f"m6_static_timelines_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_m6_expanding_timelines(target: str) -> None:
    p = EXP_DIR / f"preds_{target}.parquet"
    if not p.exists():
        print(f"No expanding preds for {target}"); return

    preds = pd.read_parquet(p)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[preds["model_name"] == MODEL_NAME]
    if preds.empty:
        print(f"No M6 expanding predictions for {target}"); return

    countries = [c for c in ILLUSTRATIVE_COUNTRIES if c in preds["country_iso3"].values]
    retrain_months = sorted(preds["retrain_month"].unique())
    n = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    fig.suptitle(
        f"M6 XGBoost Full — Expanding Window Timelines\n"
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

        # Monthly retrain boundaries (skip text labels — too dense at monthly frequency)
        first = True
        for m in retrain_months:
            rm = pd.Timestamp(m + "-01")
            if d_min <= rm <= d_max:
                ax.axvline(rm, color="#d62728", lw=0.5, ls=":", alpha=0.5,
                           label="Retrain" if first else None)
                first = False

        ax.set_xlim(d_min, d_max)
        ax.set_title(f"{ILLUSTRATIVE_COUNTRIES.get(iso3, iso3)} ({iso3})",
                     fontsize=9, loc="left", fontweight="bold")
        ax.set_ylabel("P(event)", fontsize=8)
        ax.set_ylim(0, 1.05)
        if ax is axes[0, 0]:
            handles = [
                mpatches.Patch(color=COLOR, label="M6 XGBoost Full"),
                plt.Line2D([0], [0], color="#d62728", lw=1, ls=":", label="Retrain boundary"),
                plt.Line2D([0], [0], color="grey", lw=1, alpha=0.5, label="Event day"),
            ]
            ax.legend(handles=handles, fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / f"m6_expanding_timelines_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_m6_expanding_timelines_country(target: str, iso3: str, country_name: str) -> None:
    p = EXP_DIR / f"preds_{target}.parquet"
    if not p.exists():
        print(f"No expanding preds for {target}"); return

    preds = pd.read_parquet(p)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[(preds["model_name"] == MODEL_NAME) & (preds["country_iso3"] == iso3)]
    if preds.empty:
        print(f"No M6 expanding predictions for {iso3} / {target}"); return

    c = preds.sort_values("date")
    retrain_months = sorted(preds["retrain_month"].unique())
    d_min, d_max = c["date"].min(), c["date"].max()

    fig, ax = plt.subplots(figsize=(14, 4))
    fig.suptitle(
        f"M6 XGBoost Full — Expanding Window: {country_name}\n"
        f"{TARGET_LABELS[target]} | Monthly retraining 2020–2021",
        fontsize=11, fontweight="bold",
    )

    ax.fill_between(c["date"], c["y_pred"], alpha=0.15, color="#1f77b4")
    ax.plot(c["date"], c["y_pred"], color="#1f77b4", lw=1.5, label="M6 predicted probability")

    events = c[c["y_true"] == 1]
    if not events.empty:
        ax.vlines(events["date"], 0, 0.12, color="red", lw=0.8, alpha=0.6, label="Event day")

    first = True
    for m in retrain_months:
        rm = pd.Timestamp(m + "-01")
        if d_min <= rm <= d_max:
            ax.axvline(rm, color="#aaaaaa", lw=0.5, ls=":", alpha=0.5,
                       label="Retrain boundary" if first else None)
            first = False

    ax.set_xlim(d_min, d_max)
    ax.set_ylabel("P(event)", fontsize=10)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    out = OUT_DIR / f"m6_expanding_timelines_{iso3.lower()}_{target}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_m6_expanding_performance(target: str) -> None:
    p = EXP_DIR / f"metrics_{target}.csv"
    if not p.exists():
        print(f"No expanding metrics for {target}"); return

    df = pd.read_csv(p)
    df = df[df["model_name"] == MODEL_NAME].copy()
    df = df.sort_values("month")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"M6 XGBoost Full — Expanding Window Monthly Performance\n"
        f"{TARGET_LABELS[target]}, 2020–2021",
        fontsize=11, fontweight="bold",
    )
    x = range(len(df))
    axes[0].plot(x, df["roc_auc"], color=COLOR, lw=1.8, marker="o", ms=5)
    axes[1].plot(x, df["brier_skill_score"], color=COLOR, lw=1.8, marker="o", ms=5)

    tick_idx = [i for i, m in enumerate(df["month"].tolist()) if m.endswith(("-01", "-04", "-07", "-10"))]
    tick_lbl = [df["month"].tolist()[i] for i in tick_idx]
    for ax in axes:
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(tick_lbl, rotation=30, fontsize=8)
        ax.axvline(11.5, color="grey", lw=1.0, ls=":", alpha=0.7)
        ax.text(11.5, ax.get_ylim()[0] + 0.01, " 2021", fontsize=7, color="grey")
        ax.set_xlabel("Month", fontsize=9)

    axes[0].set_title("ROC-AUC", fontsize=10)
    axes[0].set_ylim(0.7, 1.0)
    axes[1].set_title("Brier Skill Score", fontsize=10)
    axes[1].axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    plt.tight_layout()
    out = OUT_DIR / f"m6_expanding_performance_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_m6_shap(target: str) -> None:
    p = PROC_DIR / target / "shap_importance.csv"
    if not p.exists():
        print(f"No shap_importance.csv for {target}"); return

    shap = pd.read_csv(p)
    shap = shap[shap["model_name"] == MODEL_NAME].copy()
    if shap.empty:
        return

    shap = shap.sort_values("mean_abs_shap").tail(25)

    # Clean feature labels
    shap["label"] = (shap["feature"]
                     .str.replace("num__", "").str.replace("remainder__", "")
                     .str.replace("_", " ").str.title())

    fig, ax = plt.subplots(figsize=(9, max(4, len(shap) * 0.42)))
    ax.barh(shap["label"], shap["mean_abs_shap"], color=COLOR,
            edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Mean |SHAP value| (averaged over folds)", fontsize=10)
    ax.set_title(
        f"M6 XGBoost Full — Feature Importance (SHAP)\n"
        f"{TARGET_LABELS[target]} | Top 25 features",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = OUT_DIR / f"m6_shap_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_m6_metrics() -> None:
    rows = []
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
        out.to_csv(OUT_DIR / "m6_metrics.csv", index=False)
        print(f"Saved: {OUT_DIR / 'm6_metrics.csv'}  ({len(out)} rows)")


if __name__ == "__main__":
    import shutil
    print(f"Saving M6 outputs to {OUT_DIR}\n")
    for target in TARGETS:
        fig_m6_static_timelines(target)
        fig_m6_expanding_timelines(target)
        fig_m6_expanding_performance(target)
        fig_m6_shap(target)
    save_m6_metrics()
    shutil.copy2(__file__, OUT_DIR / "export_m6.py")
    print(f"Saved: {OUT_DIR / 'export_m6.py'}")
    print("\nDone.")
