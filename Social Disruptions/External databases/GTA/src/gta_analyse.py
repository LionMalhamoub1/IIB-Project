from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

GTA_ROOT  = Path(__file__).resolve().parents[1]
DATA_FILE = GTA_ROOT / "data" / "processed" / "gta_country_day_20170101_20251231.parquet"
OUT_DIR   = GTA_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid":       True,
    "grid.alpha":      0.25,
    "grid.linestyle":  ":",
    "figure.dpi":      130,
})


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df


def fig_global_trend(df: pd.DataFrame) -> None:
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))[["gta_harmful_events", "gta_liberalising_events"]]
        .sum()
        .reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(monthly["date"], monthly["gta_harmful_events"],
                    color="#b2182b", alpha=0.7, label="Harmful")
    ax.fill_between(monthly["date"], monthly["gta_liberalising_events"],
                    color="#4393c3", alpha=0.7, label="Liberalising")
    ax.set_ylabel("Total interventions (all countries)")
    ax.set_title("Global trade interventions per month")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gta_global_trend.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'gta_global_trend.png'}")


def fig_top_countries(df: pd.DataFrame, n: int = 30) -> None:
    totals = (
        df.groupby("country_iso3")[["gta_harmful_events", "gta_liberalising_events"]]
        .sum()
        .assign(total=lambda x: x["gta_harmful_events"] + x["gta_liberalising_events"])
        .sort_values("total", ascending=True)
        .tail(n)
    )

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(totals.index, totals["gta_harmful_events"],
            color="#b2182b", alpha=0.8, label="Harmful")
    ax.barh(totals.index, totals["gta_liberalising_events"],
            left=totals["gta_harmful_events"],
            color="#4393c3", alpha=0.8, label="Liberalising")
    ax.set_xlabel("Total interventions (2017–2025)")
    ax.set_title(f"Top {n} countries by trade interventions")
    ax.legend(frameon=False)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gta_top_countries.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'gta_top_countries.png'}")


def fig_year_coverage(df: pd.DataFrame) -> None:
    years = sorted(df["year"].unique())
    active = (
        df[df["gta_policy_events"] > 0]
        .groupby(["country_iso3", "year"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=years, fill_value=0)
    )
    coverage = (active > 0).astype(int)
    coverage = coverage.loc[coverage.sum(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(12, max(8, len(coverage) * 0.18)))
    im = ax.imshow(coverage.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, fontsize=9)
    ax.set_yticks(range(len(coverage)))
    ax.set_yticklabels(coverage.index, fontsize=6)
    ax.set_title("GTA country coverage by year\n(green = interventions recorded, red = none)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gta_year_coverage.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'gta_year_coverage.png'}")


def fig_harmful_vs_liberalising(df: pd.DataFrame) -> None:
    totals = df.groupby("country_iso3")[["gta_harmful_events", "gta_liberalising_events"]].sum()
    totals = totals[(totals > 0).any(axis=1)]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(totals["gta_liberalising_events"], totals["gta_harmful_events"],
               alpha=0.6, color="#4393c3", edgecolors="white", linewidths=0.5, s=50)

    for iso, row in totals[totals["gta_harmful_events"] > 2000].iterrows():
        ax.annotate(iso, (row["gta_liberalising_events"], row["gta_harmful_events"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Liberalising interventions (2017–2025)")
    ax.set_ylabel("Harmful interventions (2017–2025)")
    ax.set_title("Harmful vs liberalising trade interventions by country")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gta_harmful_vs_liberalising.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'gta_harmful_vs_liberalising.png'}")


def main() -> None:
    df = load()
    fig_global_trend(df)
    fig_top_countries(df)
    fig_year_coverage(df)
    fig_harmful_vs_liberalising(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
