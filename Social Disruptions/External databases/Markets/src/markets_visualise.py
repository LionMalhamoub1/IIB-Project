"""markets_visualise.py
=======================
Generate diagnostic figures from the processed markets panel.

Run:
    python src/markets_visualise.py

Figures saved to:  Markets/figures/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
MARKETS_ROOT = _SRC_DIR.parent
PROCESSED_DIR = MARKETS_ROOT / "data" / "processed"
FIGURES_DIR = MARKETS_ROOT / "figures"

logger = logging.getLogger(__name__)

# Simple region lookup for colouring scatter plots
REGION: Dict[str, str] = {
    "ARG": "Latin America", "BOL": "Latin America", "BRA": "Latin America",
    "CHL": "Latin America", "COL": "Latin America", "GTM": "Latin America",
    "HND": "Latin America", "MEX": "Latin America", "NIC": "Latin America",
    "PER": "Latin America", "PRY": "Latin America", "URY": "Latin America",
    "VEN": "Latin America",
    "AGO": "Sub-Saharan Africa", "CMR": "Sub-Saharan Africa", "COD": "Sub-Saharan Africa",
    "ETH": "Sub-Saharan Africa", "GHA": "Sub-Saharan Africa", "KEN": "Sub-Saharan Africa",
    "MDG": "Sub-Saharan Africa", "MOZ": "Sub-Saharan Africa", "MWI": "Sub-Saharan Africa",
    "NGA": "Sub-Saharan Africa", "RWA": "Sub-Saharan Africa", "SEN": "Sub-Saharan Africa",
    "TZA": "Sub-Saharan Africa", "UGA": "Sub-Saharan Africa", "ZAF": "Sub-Saharan Africa",
    "ZMB": "Sub-Saharan Africa", "ZWE": "Sub-Saharan Africa",
    "DZA": "MENA", "EGY": "MENA", "MAR": "MENA", "TUN": "MENA", "TUR": "MENA",
    "BGD": "Asia", "IDN": "Asia", "IND": "Asia", "KHM": "Asia", "LKA": "Asia",
    "MMR": "Asia", "NPL": "Asia", "PAK": "Asia", "PHL": "Asia", "THA": "Asia",
    "VNM": "Asia",
    "UKR": "Europe & C. Asia", "UZB": "Europe & C. Asia",
}

REGION_COLOURS = {
    "Latin America":      "#e6194b",
    "Sub-Saharan Africa": "#f58231",
    "MENA":               "#ffe119",
    "Asia":               "#3cb44b",
    "Europe & C. Asia":   "#4363d8",
    "Other":              "#aaaaaa",
}

COMMODITY_PALETTE = {
    "oil_brent_usd": "#1f77b4",
    "wheat":         "#ff7f0e",
    "corn":          "#2ca02c",
    "sugar":         "#d62728",
    "coffee":        "#9467bd",
    "cocoa":         "#8c564b",
    "natgas":        "#e377c2",
    "gold":          "#bcbd22",
}

CRISIS_PERIODS = [
    ("2018-01-01", "2018-12-31", "EM sell-off 2018"),
    ("2020-02-01", "2020-06-30", "COVID-19"),
    ("2022-02-01", "2022-12-31", "Ukraine war"),
]


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_panel() -> pd.DataFrame:
    candidates = sorted(
        PROCESSED_DIR.glob("markets_country_day_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No markets_country_day_*.parquet found in {PROCESSED_DIR}.\n"
            "Run markets_pipeline.py first."
        )
    logger.info("Loading panel: %s", candidates[0].name)
    df = pd.read_parquet(candidates[0])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_summary_stats() -> pd.DataFrame:
    """Load per-country summary_stats.json files into a DataFrame."""
    seen: set = set()
    rows = []
    for p in PROCESSED_DIR.glob("*/*_summary_stats.json"):
        if p in seen:
            continue
        seen.add(p)
        try:
            with open(p, encoding="utf-8") as fh:
                rows.append(json.load(fh))
        except Exception:
            pass
    return pd.DataFrame(rows)


def load_annual_summaries() -> pd.DataFrame:
    """Load per-country annual_summary.csv files into a single DataFrame."""
    frames = []
    for p in PROCESSED_DIR.glob("*/*_annual_summary.csv"):
        iso3 = p.parent.name
        try:
            df = pd.read_csv(p)
            df["country_iso3"] = iso3
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _shade_crises(ax: plt.Axes, alpha: float = 0.08) -> None:
    for start, end, label in CRISIS_PERIODS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color="grey", alpha=alpha, zorder=0)
        ax.text(
            pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2,
            ax.get_ylim()[1] * 0.97,
            label, ha="center", va="top", fontsize=6, color="grey",
        )


# ---------------------------------------------------------------------------
# Figure 1: Global commodity & oil prices (normalised index)
# ---------------------------------------------------------------------------
def fig_commodity_index(panel: pd.DataFrame, figures_dir: Path) -> None:
    commodity_cols = [c for c in COMMODITY_PALETTE if c in panel.columns]
    if not commodity_cols:
        logger.warning("No commodity columns found in panel — skipping commodity index figure.")
        return

    # Take one row per date (global series are identical across countries)
    global_df = (
        panel.drop_duplicates("date")
        .set_index("date")
        .sort_index()[commodity_cols]
    )

    # Normalise: first valid value = 100
    norm = global_df.copy()
    for col in commodity_cols:
        first_valid = norm[col].first_valid_index()
        if first_valid is not None:
            norm[col] = norm[col] / norm[col].loc[first_valid] * 100

    fig, ax = plt.subplots(figsize=(13, 5))
    for col in commodity_cols:
        label = col.replace("_usd", "").replace("_", " ").title()
        colour = COMMODITY_PALETTE.get(col, None)
        norm[col].dropna().plot(ax=ax, label=label, color=colour, linewidth=1.4)

    ax.axhline(100, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    _shade_crises(ax)

    ax.set_title("Global Commodity & Energy Prices (Indexed, first observation = 100)", fontsize=13)
    ax.set_xlabel("")
    ax.set_ylabel("Price index (base = 100)")
    ax.legend(ncol=4, fontsize=9, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
    fig.tight_layout()
    out = figures_dir / "01_commodity_price_index.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out.name)


# ---------------------------------------------------------------------------
# Figure 2: Global risk indicators (VIX, DXY, Gold, US10Y)
# ---------------------------------------------------------------------------
def fig_global_risk(panel: pd.DataFrame, figures_dir: Path) -> None:
    risk_cols = {
        "vix": ("CBOE VIX", "#d62728"),
        "dxy": ("US Dollar Index (DXY)", "#1f77b4"),
        "gold": ("Gold (USD/oz)", "#bcbd22"),
        "yield_us10y": ("US 10Y Yield (%)", "#9467bd"),
    }
    available = [(col, lbl, clr) for col, (lbl, clr) in risk_cols.items() if col in panel.columns]
    if not available:
        logger.warning("No global risk columns found — skipping risk indicators figure.")
        return

    global_df = (
        panel.drop_duplicates("date")
        .set_index("date")
        .sort_index()
    )

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (col, label, colour) in zip(axes, available):
        global_df[col].dropna().plot(ax=ax, color=colour, linewidth=1.2)
        ax.set_ylabel(label, fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}"))
        for start, end, _ in CRISIS_PERIODS:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       color="grey", alpha=0.08, zorder=0)

    axes[0].set_title("Global Risk Environment", fontsize=13)
    axes[-1].set_xlabel("")
    fig.tight_layout()
    out = figures_dir / "02_global_risk_indicators.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out.name)


# ---------------------------------------------------------------------------
# Figure 3: FX depreciation heatmap  (country × year)
# ---------------------------------------------------------------------------
def fig_fx_heatmap(annual: pd.DataFrame, figures_dir: Path) -> None:
    if annual.empty or "fx_depreciation_pct" not in annual.columns:
        logger.warning("No annual summary data — skipping FX heatmap.")
        return

    pivot = annual.pivot_table(
        index="country_iso3", columns="year", values="fx_depreciation_pct"
    )
    pivot = pivot.sort_index()

    # Clip extreme values (VEN, ARG) for visual clarity
    vmax = 100.0
    pivot_clipped = pivot.clip(lower=-vmax, upper=vmax)

    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.28)))
    im = ax.imshow(
        pivot_clipped.values,
        aspect="auto",
        cmap="RdYlGn_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(y) for y in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=8)
    ax.set_title("Annual FX Depreciation vs USD (%, red = weakening local currency)", fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Depreciation % (clipped at ±100%)")

    # Annotate cells with the raw (unclipped) value
    for i, country in enumerate(pivot.index):
        for j, year in enumerate(pivot.columns):
            val = pivot.loc[country, year]
            if pd.notna(val):
                text_colour = "white" if abs(val) > 50 else "black"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=6, color=text_colour)

    fig.tight_layout()
    out = figures_dir / "03_fx_depreciation_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out.name)


# ---------------------------------------------------------------------------
# Figure 4: Crisis days ranking (bar chart)
# ---------------------------------------------------------------------------
def fig_crisis_ranking(stats: pd.DataFrame, figures_dir: Path) -> None:
    if stats.empty or "n_crisis_days_30d" not in stats.columns:
        logger.warning("No summary stats data — skipping crisis ranking figure.")
        return

    df = stats[["country_iso3", "n_crisis_days_30d", "crisis_coverage_pct"]].dropna()
    df = df.sort_values("n_crisis_days_30d", ascending=True)

    colours = [
        REGION_COLOURS.get(REGION.get(iso3, "Other"), "#aaaaaa")
        for iso3 in df["country_iso3"]
    ]

    fig, ax = plt.subplots(figsize=(9, max(6, len(df) * 0.3)))
    bars = ax.barh(df["country_iso3"], df["n_crisis_days_30d"], color=colours, edgecolor="white", linewidth=0.5)

    for bar, coverage in zip(bars, df["crisis_coverage_pct"]):
        ax.text(
            bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
            f"{coverage:.0f}%", va="center", fontsize=7, color="grey",
        )

    ax.set_xlabel("Number of days (30d rolling FX depreciation > 5%)")
    ax.set_title("Currency Crisis Days by Country (2017–2025)", fontsize=12)

    # Legend for regions
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=r) for r, c in REGION_COLOURS.items() if r != "Other"]
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = figures_dir / "04_crisis_days_ranking.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out.name)


# ---------------------------------------------------------------------------
# Figure 5: Volatility vs total depreciation scatter
# ---------------------------------------------------------------------------
def fig_vol_depr_scatter(stats: pd.DataFrame, figures_dir: Path) -> None:
    needed = ["country_iso3", "annualised_vol_pct", "total_depreciation_pct"]
    if stats.empty or not all(c in stats.columns for c in needed):
        logger.warning("Missing columns for scatter plot — skipping.")
        return

    df = stats[needed].dropna()

    fig, ax = plt.subplots(figsize=(9, 7))
    for _, row in df.iterrows():
        iso3 = row["country_iso3"]
        region = REGION.get(iso3, "Other")
        colour = REGION_COLOURS.get(region, "#aaaaaa")
        ax.scatter(row["annualised_vol_pct"], row["total_depreciation_pct"],
                   color=colour, s=60, zorder=3, alpha=0.85)
        ax.annotate(
            iso3,
            (row["annualised_vol_pct"], row["total_depreciation_pct"]),
            textcoords="offset points", xytext=(4, 3),
            fontsize=7, color="dimgrey",
        )

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
    ax.set_xlabel("Annualised FX Volatility (%)")
    ax.set_ylabel("Total FX Depreciation 2017–2025 (%)")
    ax.set_title("FX Volatility vs Total Depreciation by Country", fontsize=12)

    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=r) for r, c in REGION_COLOURS.items() if r != "Other"]
    ax.legend(handles=handles, fontsize=8)

    # Log scale on y if range is very large
    dep_range = df["total_depreciation_pct"].max() - df["total_depreciation_pct"].min()
    if dep_range > 500:
        ax.set_yscale("symlog", linthresh=50)
        ax.set_ylabel("Total FX Depreciation 2017–2025 (%, symlog scale)")

    fig.tight_layout()
    out = figures_dir / "05_volatility_depreciation_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out.name)


# ---------------------------------------------------------------------------
# Figure 6: Regional FX stress distribution (box plots)
# ---------------------------------------------------------------------------
def fig_regional_fx_stress(panel: pd.DataFrame, figures_dir: Path) -> None:
    if "fx_pct_30d" not in panel.columns:
        logger.warning("fx_pct_30d not in panel — skipping regional stress figure.")
        return

    panel = panel.copy()
    panel["region"] = panel["country_iso3"].map(REGION).fillna("Other")

    region_order = [r for r in REGION_COLOURS if r != "Other"]
    region_order = [r for r in region_order if r in panel["region"].unique()]

    data_by_region = [
        panel.loc[panel["region"] == r, "fx_pct_30d"].dropna().values
        for r in region_order
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(
        data_by_region,
        vert=True,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": ".", "markersize": 2, "alpha": 0.3},
        showfliers=True,
        whis=1.5,
    )
    for patch, region in zip(bp["boxes"], region_order):
        patch.set_facecolor(REGION_COLOURS[region])
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(region_order) + 1))
    ax.set_xticklabels(region_order, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.axhline(5, color="red", linewidth=0.8, linestyle=":", alpha=0.5, label="Crisis threshold (5%)")
    ax.set_ylabel("30-day FX depreciation (%)")
    ax.set_title("Distribution of 30-Day FX Depreciation by Region", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = figures_dir / "06_regional_fx_stress.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", out.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate markets diagnostic figures.")
    p.add_argument("--figures-dir", type=Path, default=FIGURES_DIR,
                   help="Output directory for figures (default: Markets/figures/)")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)

    figures_dir: Path = args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Figures will be saved to: %s", figures_dir)

    panel = load_panel()
    stats = load_summary_stats()
    annual = load_annual_summaries()

    logger.info("Panel: %d rows × %d cols | %d countries",
                len(panel), len(panel.columns), panel["country_iso3"].nunique())
    logger.info("Summary stats: %d countries", len(stats))
    logger.info("Annual summaries: %d rows", len(annual))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    })

    fig_commodity_index(panel, figures_dir)
    fig_global_risk(panel, figures_dir)
    fig_fx_heatmap(annual, figures_dir)
    fig_crisis_ranking(stats, figures_dir)
    fig_vol_depr_scatter(stats, figures_dir)
    fig_regional_fx_stress(panel, figures_dir)

    logger.info("Done. %d figures saved to %s", 6, figures_dir)


if __name__ == "__main__":
    main()
