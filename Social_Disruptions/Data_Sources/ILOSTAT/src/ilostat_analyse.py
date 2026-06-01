from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ILOSTAT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE    = ILOSTAT_ROOT / "data" / "processed" / "ilostat_country_month_2017_2025.parquet"
OUT_DIR      = ILOSTAT_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDICATORS = ["unemployment_rate", "unemployment_sa"]

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    ":",
    "figure.dpi":        130,
})


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df


def fig_year_coverage(df: pd.DataFrame) -> None:
    years = sorted(df["year"].unique())

    for indicator in INDICATORS:
        coverage = (
            df[df[indicator].notna()]
            .groupby(["country_iso3", "year"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=years, fill_value=0)
        )
        coverage = (coverage > 0).astype(int)
        coverage = coverage.loc[coverage.sum(axis=1).sort_values(ascending=False).index]

        fig, ax = plt.subplots(figsize=(12, max(6, len(coverage) * 0.22)))
        ax.imshow(coverage.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(years)))
        ax.set_xticklabels(years, fontsize=9)
        ax.set_yticks(range(len(coverage)))
        ax.set_yticklabels(coverage.index, fontsize=7)
        ax.set_title(f"{indicator} — country coverage by year\n(green = data present, red = no data)",
                     fontsize=11, pad=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ilostat_coverage_{indicator}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {OUT_DIR / f'ilostat_coverage_{indicator}.png'}")


def fig_missingness_by_country(df: pd.DataFrame) -> None:
    for indicator in INDICATORS:
        coverage = (
            df.groupby("country_iso3")[indicator]
            .apply(lambda s: s.notna().mean())
            .sort_values()
        )
        countries_with_data = coverage[coverage > 0]

        fig, ax = plt.subplots(figsize=(8, max(5, len(countries_with_data) * 0.25)))
        colors = ["#d73027" if v < 0.5 else "#fee08b" if v < 0.8 else "#1a9850"
                  for v in countries_with_data.values]
        ax.barh(countries_with_data.index, countries_with_data.values,
                color=colors, edgecolor="white", height=0.7)
        ax.axvline(0.5, color="#333333", linewidth=0.8, linestyle="--")
        ax.axvline(0.8, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("Fraction of months with data")
        ax.set_title(f"{indicator} — data completeness by country", fontsize=11)
        ax.tick_params(axis="y", labelsize=7)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ilostat_completeness_{indicator}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {OUT_DIR / f'ilostat_completeness_{indicator}.png'}")


def fig_time_series_sample(df: pd.DataFrame, n_countries: int = 12) -> None:
    for indicator in INDICATORS:
        # pick countries with best coverage
        top = (
            df.groupby("country_iso3")[indicator]
            .apply(lambda s: s.notna().mean())
            .sort_values(ascending=False)
            .head(n_countries)
            .index.tolist()
        )

        ncols = 3
        nrows = int(np.ceil(len(top) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3), sharex=False)
        axes_flat = axes.flatten()

        for ax, iso3 in zip(axes_flat, top):
            c = df[df["country_iso3"] == iso3].sort_values("date")
            ax.plot(c["date"], c[indicator], color="#4393c3", linewidth=1.2)
            ax.set_title(iso3, fontsize=10)
            ax.set_ylabel("%", fontsize=8)
            ax.tick_params(axis="x", labelsize=8)
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))

        for ax in axes_flat[len(top):]:
            ax.set_visible(False)

        fig.suptitle(f"{indicator} — sample time series (best-covered countries)",
                     x=0.05, ha="left", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ilostat_timeseries_{indicator}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {OUT_DIR / f'ilostat_timeseries_{indicator}.png'}")


def fig_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(INDICATORS), figsize=(5 * len(INDICATORS), 4))

    for ax, indicator in zip(axes, INDICATORS):
        vals = df[indicator].dropna()
        ax.hist(vals, bins=50, color="#4393c3", edgecolor="white", alpha=0.85)
        ax.axvline(vals.median(), color="#333333", linewidth=1.2, linestyle="--",
                   label=f"Median = {vals.median():.1f}%")
        ax.set_xlabel(indicator)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {indicator}")
        ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "ilostat_distributions.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'ilostat_distributions.png'}")


def main() -> None:
    df = load()
    df = df.drop(columns=[c for c in ["earnings_monthly", "cpi_yoy", "cpi_mom"] if c in df.columns])

    print(f"Loaded: {df.shape[0]} rows | {df['country_iso3'].nunique()} countries")
    print(f"Date range: {df['date'].min().date()} - {df['date'].max().date()}")
    print()

    for indicator in INDICATORS:
        n = df[df[indicator].notna()]["country_iso3"].nunique()
        pct_missing = df[indicator].isna().mean() * 100
        print(f"{indicator}: {n} countries | {pct_missing:.1f}% missing overall")

    print()
    fig_year_coverage(df)
    fig_missingness_by_country(df)
    fig_time_series_sample(df)
    fig_distribution(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
