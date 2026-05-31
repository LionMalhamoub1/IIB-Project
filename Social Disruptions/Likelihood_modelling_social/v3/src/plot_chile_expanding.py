"""
plot_chile_expanding.py
=======================
Expanding window predicted probability timelines for Chile,
M7 (XGBoost No Lags) only.

Two panels stacked: Protest (7-day) top, Strike (7-day) bottom.

Output:
  v3/figures/chile_expanding_m7.{png,pdf}
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

_V3      = Path(__file__).resolve().parents[1]
PROC_DIR = _V3 / "data" / "processed"
EXP_XGB  = PROC_DIR / "expanding_xgb_monthly"
FIG_DIR  = _V3 / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["protest_7d", "strike_7d"]
TARGET_LABELS = {"protest_7d": "Protest (7-day)", "strike_7d": "Strike (7-day)"}

LINE_COLOR  = "#2166ac"   # blue
EVENT_COLOR = "#d62728"   # red

STYLE = {
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
}
plt.rcParams.update(STYLE)


def load_chile(target: str) -> pd.DataFrame:
    df = pd.read_parquet(EXP_XGB / f"preds_{target}.parquet")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["country_iso3"] == "CHL") & (df["model_name"] == "model6_xgb_nolag")]
    return df.sort_values("date")


if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)
    fig.suptitle(
        "Chile — M7 XGBoost No Lags, Expanding Window\n"
        "Monthly retraining 2020–2021",
        fontsize=14, fontweight="bold",
    )

    for ax, target in zip(axes, TARGETS):
        df = load_chile(target)
        retrain_months = sorted(pd.to_datetime(df["retrain_month"].unique()))
        d_min, d_max = df["date"].min(), df["date"].max()

        ax.fill_between(df["date"], df["y_pred"], alpha=0.15, color=LINE_COLOR)
        ax.plot(df["date"], df["y_pred"], color=LINE_COLOR, lw=1.8)

        events = df[df["y_true"] == 1]
        if not events.empty:
            ax.vlines(events["date"], 0, 0.08, color=EVENT_COLOR, lw=0.8, alpha=0.7)

        first = True
        for rm in retrain_months:
            if d_min <= rm <= d_max:
                ax.axvline(rm, color="grey", lw=0.7, ls=":", alpha=0.6,
                           label="Retrain boundary" if first else None)
                first = False

        ax.set_xlim(d_min, d_max)
        ax.set_ylim(0, 1.05)
        ax.set_title(TARGET_LABELS[target], fontsize=13, fontweight="bold", loc="left")
        ax.set_ylabel("P(event)", fontsize=13)
        ax.tick_params(axis="both", labelsize=12)

        if ax is axes[0]:
            handles = [
                plt.Line2D([0], [0], color=LINE_COLOR,  lw=2,      label="M7 XGBoost No Lags"),
                plt.Line2D([0], [0], color=EVENT_COLOR, lw=1, alpha=0.7, label="Event day"),
                plt.Line2D([0], [0], color="grey",      lw=1, ls=":", label="Retrain boundary"),
            ]
            ax.legend(handles=handles, fontsize=12, loc="upper right")

    axes[-1].set_xlabel("Date", fontsize=13)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        out = FIG_DIR / f"chile_expanding_m7.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)
    print("Done.")
