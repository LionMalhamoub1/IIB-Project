from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACLED_ROOT = Path(__file__).resolve().parent
RAW_DIR    = ACLED_ROOT / "data" / "raw" / "events"
OUT_DIR    = ACLED_ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_YEAR = 2017
END_YEAR   = 2025
YEARS      = list(range(START_YEAR, END_YEAR + 1))
PROTEST_TYPES = {"Protests"}


def load_all_events() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("iso3=*/year=*.parquet"))
    if not files:
        raise FileNotFoundError(f"No ACLED parquet files found in {RAW_DIR}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def main() -> None:
    print("Loading events...")
    events = load_all_events()
    events["date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["date", "iso3"])
    events["iso3"] = events["iso3"].str.strip().str.upper()
    events["year"] = events["date"].dt.year
    events = events[
        (events["year"] >= START_YEAR) &
        (events["year"] <= END_YEAR) &
        (events["event_type"].isin(PROTEST_TYPES))
    ]

    # events per country per year
    counts = (
        events.groupby(["iso3", "year"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=YEARS, fill_value=0)
    )

    # coverage = 1 if any events recorded that year, 0 otherwise
    coverage = (counts > 0).astype(int)

    # sort by number of years with coverage (descending), then total events
    coverage["years_covered"] = coverage.sum(axis=1)
    coverage = coverage.sort_values("years_covered", ascending=False)
    coverage = coverage.drop(columns="years_covered")

    coverage.to_csv(OUT_DIR / "protest_year_coverage.csv")
    print(f"Saved -> {OUT_DIR / 'protest_year_coverage.csv'}")

    # --- heatmap ---
    fig, ax = plt.subplots(figsize=(12, max(8, len(coverage) * 0.22)))

    im = ax.imshow(coverage.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(YEARS)))
    ax.set_xticklabels(YEARS, fontsize=9)
    ax.set_yticks(range(len(coverage)))
    ax.set_yticklabels(coverage.index, fontsize=7)
    ax.set_title("ACLED protest coverage by country and year\n(green = data present, red = no data)",
                 fontsize=11, pad=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "protest_year_coverage.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved -> {OUT_DIR / 'protest_year_coverage.png'}")

    fully_covered = coverage[coverage.sum(axis=1) == len(YEARS)]
    print(f"\nCountries with protest data in all {len(YEARS)} years: {len(fully_covered)}")
    print(fully_covered.index.tolist())


if __name__ == "__main__":
    main()
