from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SELECTED_COUNTRIES = ["CHL", "COD", "ZAF", "IDN", "AUS"]

FOCUS_KEYWORD = "protest_mobilisation_index"

HEATMAP_KEYWORD = "protest_mobilisation_index"

SPIKE_THRESHOLD = 2.0

HEATMAP_CLAMP = 4.0

GT_ROOT   = Path(__file__).resolve().parents[1]
DATA_DIR  = GT_ROOT / "data" / "processed"
FIG_DIR   = GT_ROOT / "data" / "figures"

KEYWORD_PALETTE = {
    "economic_stress_index":      "#4e79a7",
    "labour_conflict_index":      "#f28e2b",
    "protest_mobilisation_index": "#e15759",
}


def load_data() -> pd.DataFrame:
    files = sorted(DATA_DIR.glob("google_trends_country_week_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No processed parquet found in {DATA_DIR}.\n"
            "Run google_trends_pipeline.py first."
        )
    path = files[-1]
    print(f"Loading: {path}")
    df = pd.read_parquet(path)
    df["week"] = pd.to_datetime(df["week"])
    print(f"Shape: {df.shape}   Countries: {df['country_iso3'].nunique()}")
    return df


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.show()
    plt.close(fig)


def raw_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if not c.endswith("_zscore") and c not in ("country_iso3", "week")]


def zscore_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_zscore")]


def year_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)


def plot_raw_scores(df: pd.DataFrame) -> None:
    kw_cols = raw_cols(df)
    n = len(SELECTED_COUNTRIES)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, iso3 in zip(axes, SELECTED_COUNTRIES):
        sub = df[df["country_iso3"] == iso3].set_index("week").sort_index()
        for col in kw_cols:
            if col in sub.columns:
                ax.plot(sub.index, sub[col],
                        label=col, linewidth=1.4,
                        color=KEYWORD_PALETTE.get(col))
        ax.set_title(iso3, fontsize=11, fontweight="bold", loc="left")
        ax.set_ylabel("Index value (0–100)")
        ax.set_ylim(0)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        year_axis(ax)

    axes[0].legend(ncol=3, fontsize=8, loc="upper left", framealpha=0.7)
    fig.suptitle("Google Trends — group indices", fontsize=13)
    fig.tight_layout()
    _save(fig, "01_raw_scores")


def plot_anomaly_spikes(df: pd.DataFrame) -> None:
    z_col   = f"{FOCUS_KEYWORD}_zscore"
    raw_col = FOCUS_KEYWORD
    colour  = KEYWORD_PALETTE.get(raw_col, "steelblue")

    n = len(SELECTED_COUNTRIES)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, iso3 in zip(axes, SELECTED_COUNTRIES):
        sub = df[df["country_iso3"] == iso3].set_index("week").sort_index()

        ax.plot(sub.index, sub[raw_col], color=colour, linewidth=1.3, label=raw_col)
        ax.set_ylabel("Index value (0–100)", color=colour)
        ax.tick_params(axis="y", labelcolor=colour)
        ax.set_ylim(0)

        ax2 = ax.twinx()
        ax2.plot(sub.index, sub[z_col], color="grey", linewidth=0.9, alpha=0.6)
        ax2.axhline( SPIKE_THRESHOLD, color="red",  linewidth=0.8, linestyle="--", alpha=0.7)
        ax2.axhline(-SPIKE_THRESHOLD, color="navy", linewidth=0.8, linestyle="--", alpha=0.4)
        ax2.axhline(0,                color="black", linewidth=0.5, alpha=0.3)
        ax2.set_ylabel("Z-score", color="grey")
        ax2.tick_params(axis="y", labelcolor="grey")

        spikes = sub[sub[z_col] >= SPIKE_THRESHOLD]
        for t in spikes.index:
            ax.axvspan(t - pd.Timedelta(days=3), t + pd.Timedelta(days=3),
                       color="red", alpha=0.12)

        ax.set_title(iso3, fontsize=11, fontweight="bold", loc="left")
        ax.grid(axis="x", linewidth=0.4, alpha=0.4)
        ax.spines[["top"]].set_visible(False)
        year_axis(ax)

    fig.suptitle(
        f"Anomaly spikes — '{FOCUS_KEYWORD}'  (threshold = {SPIKE_THRESHOLD}σ)",
        fontsize=13,
    )
    fig.tight_layout()
    _save(fig, "02_anomaly_spikes")


def plot_heatmap(df: pd.DataFrame) -> None:
    if HEATMAP_KEYWORD:
        z_col = f"{HEATMAP_KEYWORD}_zscore"
        pivot = df.pivot_table(index="country_iso3", columns="week",
                               values=z_col, aggfunc="mean")
        title = f"Z-score heatmap — '{HEATMAP_KEYWORD}'"
    else:
        df = df.copy()
        df["_max_z"] = df[zscore_cols(df)].max(axis=1)
        pivot = df.pivot_table(index="country_iso3", columns="week",
                               values="_max_z", aggfunc="mean")
        title = "Z-score heatmap — max across all indices"

    pivot = pivot.clip(lower=0, upper=HEATMAP_CLAMP)
    pivot = pivot.loc[pivot.max(axis=1).sort_values(ascending=False).index]

    pivot.columns = pd.to_datetime(pivot.columns)
    pivot_monthly = pivot.T.resample("ME").mean().T

    fig, ax = plt.subplots(figsize=(18, max(6, len(pivot_monthly) * 0.42)))
    sns.heatmap(
        pivot_monthly,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=HEATMAP_CLAMP,
        linewidths=0,
        xticklabels=False,
        cbar_kws={"label": f"Z-score (clamped at {HEATMAP_CLAMP})"},
    )

    months = pivot_monthly.columns
    year_positions = [i for i, m in enumerate(months) if m.month == 1]
    year_labels    = [str(months[i].year) for i in year_positions]
    ax.set_xticks(year_positions)
    ax.set_xticklabels(year_labels, fontsize=9, rotation=0)
    ax.set_xlabel("Year")
    ax.set_ylabel("")
    ax.set_title(title, fontsize=13, pad=10)
    fig.tight_layout()
    _save(fig, "03_heatmap")


def plot_small_multiples(df: pd.DataFrame) -> None:
    countries = sorted(df["country_iso3"].unique())
    ncols = 5
    nrows = int(np.ceil(len(countries) / ncols))
    colour = KEYWORD_PALETTE.get(FOCUS_KEYWORD, "steelblue")

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 2.2),
        sharex=True, sharey=False,
    )
    axes_flat = axes.flatten()

    for i, iso3 in enumerate(countries):
        ax = axes_flat[i]
        sub = df[df["country_iso3"] == iso3].set_index("week").sort_index()
        ax.fill_between(sub.index, sub[FOCUS_KEYWORD], color=colour, alpha=0.35)
        ax.plot(sub.index, sub[FOCUS_KEYWORD], color=colour, linewidth=0.9)
        ax.set_title(iso3, fontsize=9, fontweight="bold")
        ax.set_ylim(0)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("'%y"))
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    for j in range(len(countries), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(f"'{FOCUS_KEYWORD}' — all countries", fontsize=13)
    fig.tight_layout()
    _save(fig, "04_small_multiples")


if __name__ == "__main__":
    data = load_data()
    plot_raw_scores(data)
    plot_anomaly_spikes(data)
    plot_heatmap(data)
    plot_small_multiples(data)
