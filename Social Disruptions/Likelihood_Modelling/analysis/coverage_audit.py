from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PANEL_FILE = Path(__file__).resolve().parents[1] / "data" / "interim" / "modelling_panel.parquet"
OUT_DIR    = Path(__file__).resolve().parent / "figures" / "coverage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDICATOR_GROUPS: dict[str, list[str]] = {
    "ACLED lags": [
        "acled_7d_lag", "acled_28d_lag",
        "riot_7d_lag", "riot_28d_lag",
        "violence_7d_lag", "violence_28d_lag",
        "protest_fat_7d_lag", "protest_fat_28d_lag",
    ],
    "Markets": [
        "fx_pct_7d", "fx_pct_30d", "fx_pct_90d",
        "fx_vol_7d", "fx_vol_30d",
        "oil_brent_pct_14d", "oil_brent_pct_30d",
        "yield_us10y",
    ],
    "WDI": [
        "gdp_growth", "gdp_per_capita_growth",
        "inflation_cpi_yoy",
        "unemployment_total", "unemployment_youth",
    ],
    "WGI": [
        "political_stability_est", "voice_accountability_est",
        "government_effectiveness_est", "rule_of_law_est",
    ],
    "Google Trends": [
        "economic_stress_index", "labour_conflict_index",
        "protest_mobilisation_index",
    ],
    "CPI": [
        "food_cpi_inflation", "energy_cpi_inflation",
    ],
    "FAO": [
        "fao_food_index_yoy", "fao_cereals_index_yoy", "fao_oils_index_yoy",
    ],
    "ILOSTAT": [
        "unemployment_sa", "unemployment_rate", "earnings_monthly",
    ],
    "GTA": [
        "gta_harmful_events", "gta_liberalising_events",
        "gta_30d_count", "gta_90d_count",
    ],
}


def coverage_by_country(panel: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for country, grp in panel.groupby("country_iso3"):
        row = {}
        for source, cols in INDICATOR_GROUPS.items():
            present = [c for c in cols if c in grp.columns]
            if not present:
                row[source] = np.nan
            else:
                row[source] = grp[present].notna().any(axis=1).mean()
        rows[country] = row
    return pd.DataFrame(rows).T.sort_index()


def plot_heatmap(coverage: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, max(6, len(coverage) * 0.35)))

    im = ax.imshow(coverage.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    for r in range(coverage.shape[0]):
        for c in range(coverage.shape[1]):
            val = coverage.values[r, c]
            if np.isnan(val):
                ax.text(c, r, "n/a", ha="center", va="center", fontsize=7, color="#888888")
            else:
                txt_color = "black" if 0.2 < val < 0.8 else "white"
                ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                        fontsize=7, color=txt_color)

    ax.set_xticks(range(len(coverage.columns)))
    ax.set_xticklabels(coverage.columns, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(coverage.index)))
    ax.set_yticklabels(coverage.index, fontsize=9)
    ax.set_title("Indicator coverage by country\n(% of days with at least one non-null value in the group)",
                 fontsize=12, pad=12)

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label("Coverage", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "coverage_by_country.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'coverage_by_country.png'}")


def plot_missing_summary(coverage: pd.DataFrame) -> None:
    mean_coverage = coverage.mean(axis=1).sort_values()

    fig, ax = plt.subplots(figsize=(7, max(5, len(coverage) * 0.3)))
    colors = ["#d73027" if v < 0.5 else "#fee08b" if v < 0.8 else "#1a9850"
              for v in mean_coverage.values]
    ax.barh(mean_coverage.index, mean_coverage.values, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0.5, color="#333333", linewidth=0.8, linestyle="--")
    ax.axvline(0.8, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlabel("Mean coverage across all indicator groups")
    ax.set_title("Overall indicator coverage per country", fontsize=12)
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "coverage_summary_by_country.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'coverage_summary_by_country.png'}")


def main() -> None:
    panel = pd.read_parquet(PANEL_FILE)
    panel["date"] = pd.to_datetime(panel["date"])

    coverage = coverage_by_country(panel)

    coverage.to_csv(OUT_DIR / "coverage_by_country.csv")
    print(f"Saved -> {OUT_DIR / 'coverage_by_country.csv'}")

    plot_heatmap(coverage)
    plot_missing_summary(coverage)

    print("\nCoverage summary (mean across indicator groups):")
    print(coverage.mean(axis=1).sort_values().to_string())


if __name__ == "__main__":
    main()
