"""fao_analyse.py
=================
Generate diagnostic figures from the processed FAO Food Price Index panel.

Run:
    python src/fao_analyse.py

Figures saved to:  FAO/figures/
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
_SRC_DIR      = Path(__file__).resolve().parent
FAO_ROOT      = _SRC_DIR.parent
PROCESSED_DIR = FAO_ROOT / "data" / "processed"
FIGURES_DIR   = FAO_ROOT / "figures"

START_YEAR = 2017
END_YEAR   = 2025
PARQUET    = PROCESSED_DIR / f"fao_food_price_monthly_{START_YEAR}_{END_YEAR}.parquet"

# ─────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────
PALETTE = {
    "fao_food_index":    "#2c3e50",
    "fao_cereals_index": "#e67e22",
    "fao_meat_index":    "#c0392b",
    "fao_dairy_index":   "#2980b9",
    "fao_oils_index":    "#27ae60",
    "fao_sugar_index":   "#8e44ad",
}
LABELS = {
    "fao_food_index":    "Food (Overall)",
    "fao_cereals_index": "Cereals",
    "fao_meat_index":    "Meat",
    "fao_dairy_index":   "Dairy",
    "fao_oils_index":    "Vegetable Oils",
    "fao_sugar_index":   "Sugar",
}
INDEX_COLS = list(PALETTE.keys())
YOY_COLS   = [f"{c}_yoy" for c in INDEX_COLS]

# Key events to annotate
EVENTS = [
    ("2020-03-01", "COVID-19\npandemic onset"),
    ("2021-10-01", "Post-COVID\nsupply crunch"),
    ("2022-02-01", "Russia-Ukraine\nwar"),
    ("2022-06-01", "FFPI peak"),
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved -> %s", path)


def _add_events(ax: plt.Axes, df: pd.DataFrame, y_frac: float = 0.97) -> None:
    """Add vertical lines for key events."""
    ymin, ymax = ax.get_ylim()
    for date_str, label in EVENTS:
        dt = pd.Timestamp(date_str)
        if df["date"].min() <= dt <= df["date"].max():
            ax.axvline(dt, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.text(
                dt, ymin + (ymax - ymin) * y_frac, label,
                fontsize=6, color="grey", ha="left", va="top", rotation=90,
            )


# ─────────────────────────────────────────────
# Figure 1 — All indices over time
# ─────────────────────────────────────────────
def fig_all_indices(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    for col in INDEX_COLS:
        if col not in df.columns:
            continue
        lw = 2.5 if col == "fao_food_index" else 1.3
        ax.plot(df["date"], df[col], label=LABELS[col],
                color=PALETTE[col], linewidth=lw,
                zorder=3 if col == "fao_food_index" else 2)

    ax.axhline(100, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.text(df["date"].min(), 101, "Base (2014-2016 = 100)",
            fontsize=7, color="grey")

    _add_events(ax, df)

    ax.set_title("FAO Food Price Indices (2017–2025)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Index (2014-2016 = 100)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    _save(fig, "01_all_indices.png")


# ─────────────────────────────────────────────
# Figure 2 — YoY % change for each sub-index
# ─────────────────────────────────────────────
def fig_yoy_grid(df: pd.DataFrame) -> None:
    sub_cols = [c for c in INDEX_COLS if c != "fao_food_index"]
    n = len(sub_cols)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    axes = axes.flatten()

    for i, col in enumerate(sub_cols):
        ax = axes[i]
        yoy_col = f"{col}_yoy"
        if yoy_col not in df.columns:
            ax.set_visible(False)
            continue

        vals = df[yoy_col]
        ax.fill_between(df["date"], vals, 0,
                        where=(vals >= 0), color=PALETTE[col], alpha=0.4)
        ax.fill_between(df["date"], vals, 0,
                        where=(vals < 0),  color="#e74c3c",      alpha=0.3)
        ax.plot(df["date"], vals, color=PALETTE[col], linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.7)

        ax.set_title(LABELS[col], fontsize=10, fontweight="bold")
        ax.set_ylabel("YoY %")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.grid(axis="y", alpha=0.3)

    if n < len(axes):
        for j in range(n, len(axes)):
            axes[j].set_visible(False)

    fig.suptitle("FAO Sub-Index Year-on-Year % Change", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, "02_yoy_subindices.png")


# ─────────────────────────────────────────────
# Figure 3 — Correlation heatmap between indices
# ─────────────────────────────────────────────
def fig_correlation(df: pd.DataFrame) -> None:
    available = [c for c in INDEX_COLS if c in df.columns]
    corr = df[available].corr()

    labels = [LABELS[c] for c in available]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(available)):
        for j in range(len(available)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if abs(corr.values[i, j]) < 0.7 else "white")

    ax.set_title("Correlation Between FAO Price Indices (2017–2025)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, "03_correlation_heatmap.png")


# ─────────────────────────────────────────────
# Figure 4 — Overall FFPI with recession shading
#            and rolling volatility (12-month std of YoY)
# ─────────────────────────────────────────────
def fig_volatility(df: pd.DataFrame) -> None:
    if "fao_food_index_yoy" not in df.columns:
        return

    vol = df["fao_food_index_yoy"].rolling(12, min_periods=6).std()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Top: FFPI level
    ax1.plot(df["date"], df["fao_food_index"], color=PALETTE["fao_food_index"],
             linewidth=2, label="Food Price Index")
    ax1.axhline(100, color="black", linewidth=0.7, linestyle=":", alpha=0.5)
    ax1.set_ylabel("Index (2014-2016 = 100)")
    ax1.set_title("FAO Food Price Index — Level & YoY Volatility",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    _add_events(ax1, df)

    # Bottom: rolling volatility of YoY
    ax2.fill_between(df["date"], vol, color="#e67e22", alpha=0.5, label="12-month rolling σ of YoY %")
    ax2.plot(df["date"], vol, color="#e67e22", linewidth=1.2)
    ax2.set_ylabel("Volatility (σ)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())

    plt.tight_layout()
    _save(fig, "04_ffpi_volatility.png")


# ─────────────────────────────────────────────
# Figure 5 — Normalised indices (z-score) fan chart
# ─────────────────────────────────────────────
def fig_normalised(df: pd.DataFrame) -> None:
    available = [c for c in INDEX_COLS if c in df.columns]
    fig, ax = plt.subplots(figsize=(12, 5))

    z_all = pd.DataFrame()
    for col in available:
        mu  = df[col].mean()
        sig = df[col].std()
        if sig > 0:
            z_all[col] = (df[col] - mu) / sig

    if z_all.empty:
        return

    # Shaded band: min/max across sub-indices (excluding overall food)
    sub = [c for c in z_all.columns if c != "fao_food_index"]
    if sub:
        band_min = z_all[sub].min(axis=1)
        band_max = z_all[sub].max(axis=1)
        ax.fill_between(df["date"], band_min, band_max,
                        color="lightgrey", alpha=0.5, label="Sub-index range")

    # Overall FFPI on top
    if "fao_food_index" in z_all:
        ax.plot(df["date"], z_all["fao_food_index"],
                color=PALETTE["fao_food_index"], linewidth=2.5,
                label="Food (Overall)", zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    _add_events(ax, df)

    ax.set_title("FAO Indices — Normalised (z-score, 2017-2025 mean/std)",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Standard deviations from mean")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    _save(fig, "05_normalised_indices.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main() -> None:
    if not PARQUET.exists():
        logger.error("Processed data not found: %s", PARQUET)
        logger.error("Run fao_pipeline.py first.")
        sys.exit(1)

    df = pd.read_parquet(PARQUET)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Loaded %d rows from %s", len(df), PARQUET)
    logger.info("Date range: %s to %s", df["date"].min().date(), df["date"].max().date())
    logger.info("Columns: %s", list(df.columns))

    fig_all_indices(df)
    fig_yoy_grid(df)
    fig_correlation(df)
    fig_volatility(df)
    fig_normalised(df)

    logger.info("All figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
