from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from acled_indicators import filter_social_disruption, country_month_panel, admin1_month_panel

ACLED_ROOT = Path(__file__).resolve().parents[0]
RAW_DIR = ACLED_ROOT / "data" / "raw" / "events"
PANELS_DIR = ACLED_ROOT / "panels"
PANELS_DIR.mkdir(parents=True, exist_ok=True)


def load_all_events() -> pd.DataFrame:
    files: List[Path] = sorted(RAW_DIR.glob("iso3=*/year=*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {RAW_DIR}")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--social_only", action="store_true", help="Filter to social event types first.")
    args = ap.parse_args()

    print("\n=== Build ACLED panels ===")
    print(f"RAW_DIR:    {RAW_DIR}")
    print(f"PANELS_DIR: {PANELS_DIR}\n")

    events = load_all_events()
    print(f"Loaded events: {events.shape}")

    if args.social_only:
        events = filter_social_disruption(events)
        print(f"After social filter: {events.shape}")

    country_panel = country_month_panel(events, use_iso3=True, severity="count_plus_fatalities")
    admin1_panel = admin1_month_panel(events)

    cpq = PANELS_DIR / "acled_country_month.parquet"
    ccsv = PANELS_DIR / "acled_country_month.csv"
    apq = PANELS_DIR / "acled_admin1_month.parquet"
    acsv = PANELS_DIR / "acled_admin1_month.csv"

    country_panel.to_parquet(cpq, index=False)
    country_panel.to_csv(ccsv, index=False)
    admin1_panel.to_parquet(apq, index=False)
    admin1_panel.to_csv(acsv, index=False)

    print(f"Saved:\n  {cpq}\n  {apq}\n")


if __name__ == "__main__":
    main()