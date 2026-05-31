"""
Backfill actiongeo_lat / actiongeo_lon into existing data/urls/{date}.csv files.

For each CSV that lacks lat/lon columns, looks up the corresponding enriched
GDELT file and left-joins the coordinates on url_normalized, then overwrites
the CSV in place.
"""
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_DIR       = Path(__file__).resolve().parent
REPO_ROOT      = BASE_DIR.parent
URLS_DIR       = REPO_ROOT / "data" / "urls"
GDELT_DAILY    = REPO_ROOT / "data" / "interim" / "gdelt_event_context_daily"


def _enriched_path(date: str) -> Path:
    year, month, day = date[:4], date[4:6], date[6:8]
    return GDELT_DAILY / year / month / day / f"{date}_event_context_deduped_enriched.csv"


def backfill_one(csv_path: Path) -> str:
    """Return a status string: 'skipped', 'updated', or 'no_enriched'."""
    df = pd.read_csv(csv_path)

    if "actiongeo_lat" in df.columns and "actiongeo_lon" in df.columns:
        return "skipped"

    date = csv_path.stem  # filename without .csv is YYYYMMDD
    enriched = _enriched_path(date)
    if not enriched.exists():
        return "no_enriched"

    geo = pd.read_csv(
        enriched,
        usecols=["url_normalized", "actiongeo_lat", "actiongeo_lon"],
        encoding="utf-8",
        engine="python",
    ).drop_duplicates(subset="url_normalized", keep="first")

    df = df.merge(geo, on="url_normalized", how="left")
    df.to_csv(csv_path, index=False)
    return "updated"


def main():
    csvs = sorted(URLS_DIR.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {URLS_DIR}")
        return

    counts = {"skipped": 0, "updated": 0, "no_enriched": 0}
    for csv_path in tqdm(csvs, desc="Backfilling lat/lon"):
        status = backfill_one(csv_path)
        counts[status] += 1

    print(f"\nDone.")
    print(f"  Already had lat/lon (skipped) : {counts['skipped']}")
    print(f"  Updated with lat/lon          : {counts['updated']}")
    print(f"  No enriched GDELT file found  : {counts['no_enriched']}")


if __name__ == "__main__":
    main()
