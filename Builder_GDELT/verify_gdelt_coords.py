"""
verify_gdelt_coords.py
======================
Validates whether GDELT actiongeo_lat/lon coordinates correspond to the
locations described in news articles, by comparing them against Nominatim-
geocoded versions of the LLM-extracted location_name strings.

Methodology
-----------
1. Load all enriched flood JSONL files (contain LLM-extracted location_name)
2. Load all URL CSVs (contain GDELT actiongeo_lat/lon)
3. Merge on URL to pair location_name with GDELT coords
4. Geocode each unique location_name via Nominatim (OpenStreetMap, free)
5. Compute haversine distance between GDELT coord and geocoded coord
6. Report: % within 50/200/500km, wrong-country rate, null-island hits
7. Save outliers CSV and full results CSV

Usage
-----
    python Builder_GDELT/verify_gdelt_coords.py

Outputs (all in Builder_GDELT/results/coord_verification/)
-----------------------------------------------------------
    geocode_cache.json          - cached Nominatim results (reused on reruns)
    all_results.csv             - every matched row with distance
    outliers.csv                - rows where distance > OUTLIER_KM
    summary.txt                 - printed summary stats
    distance_histogram.png      - distribution of errors
"""

import json
import math
import time
import csv
import os
import glob
import sys
from pathlib import Path
from collections import defaultdict

import requests
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# -- Configuration ------------------------------------------------------------

ROOT = Path(__file__).parent.parent
ENRICHED_DIR   = ROOT / "Builder_GDELT" / "results" / "enriched_floods"
URL_CSV_DIR    = ROOT / "data" / "urls"
OUT_DIR        = ROOT / "Builder_GDELT" / "results" / "coord_verification"
GEOCACHE_FILE  = OUT_DIR / "geocode_cache.json"

OUTLIER_KM = 500        # flag rows farther than this as outliers
NULL_ISLAND_DEG = 1.0   # lat/lon within this of (0,0) = null island
NOMINATIM_DELAY = 1.1   # seconds between Nominatim requests (ToS: max 1/s)
NOMINATIM_UA    = "IIB-Project-CoordVerification/1.0 (cambridge.ac.uk)"

# -- Haversine ----------------------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


# -- Nominatim geocoder -------------------------------------------------------

def geocode_location(location_name: str, cache: dict) -> tuple[float, float] | None:
    """Return (lat, lon) for a location string, using cache to avoid repeat calls."""
    if location_name in cache:
        return cache[location_name]

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location_name, "format": "json", "limit": 1},
            headers={"User-Agent": NOMINATIM_UA},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            lat = float(results[0]["lat"])
            lon = float(results[0]["lon"])
            cache[location_name] = (lat, lon)
            return (lat, lon)
        else:
            cache[location_name] = None
            return None
    except Exception as e:
        print(f"  [geocode error] {location_name!r}: {e}")
        cache[location_name] = None
        return None
    finally:
        time.sleep(NOMINATIM_DELAY)


# -- Data loading -------------------------------------------------------------

def load_enriched_data() -> pd.DataFrame:
    """Load all enriched JSONL files, tagging each record with its source date folder."""
    paths = sorted(ENRICHED_DIR.glob("*/floods_enriched.jsonl"))
    records = []
    for path in tqdm(paths, desc="Loading enriched files", unit="day"):
        date_str = path.parent.name
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        rec["_source_date"] = date_str
                        records.append(rec)
                    except json.JSONDecodeError:
                        pass
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} enriched records from {ENRICHED_DIR}")
    return df


def _read_one_csv(date_str: str, needed: set[str]) -> dict[str, tuple[float, float]]:
    """Read one URL CSV and return only the coords for URLs in `needed`."""
    path = URL_CSV_DIR / f"{date_str}.csv"
    if not path.exists():
        return {}
    try:
        chunk = pd.read_csv(path, usecols=["url_normalized", "actiongeo_lat", "actiongeo_lon"])
        chunk = chunk[chunk["url_normalized"].isin(needed)]
        return {
            row["url_normalized"]: (float(row["actiongeo_lat"]), float(row["actiongeo_lon"]))
            for _, row in chunk.iterrows()
            if pd.notna(row["actiongeo_lat"]) and pd.notna(row["actiongeo_lon"])
        }
    except Exception as e:
        tqdm.write(f"  [csv error] {date_str}.csv: {e}")
        return {}


def load_url_coords(date_to_urls: dict[str, set[str]], max_workers: int = 16) -> dict[str, tuple[float, float]]:
    """
    Load GDELT coords in parallel, reading only the CSVs for dates that
    have records with a location_name (skips all other date CSVs entirely).
    """
    url_to_coords: dict[str, tuple[float, float]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_read_one_csv, date_str, urls): date_str
            for date_str, urls in date_to_urls.items()
        }
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Loading URL CSVs", unit="day"):
            url_to_coords.update(fut.result())
    print(f"Loaded GDELT coords for {len(url_to_coords)} matching URLs "
          f"(from {len(date_to_urls)} date CSVs)")
    return url_to_coords


# -- Main ----------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load enriched data and build a date->{urls} map restricted to records
    #    that have a location_name  -  those are the only ones we'll geocode-verify,
    #    so there's no point loading coords for any other URLs.
    enriched = load_enriched_data()
    has_loc = enriched["location_name"].notna() & (enriched["location_name"].str.strip() != "")
    date_to_urls: dict[str, set[str]] = (
        enriched[has_loc]
        .groupby("_source_date")["url"]
        .apply(lambda s: set(s.dropna()))
        .to_dict()
    )
    date_to_urls = {k: v for k, v in date_to_urls.items() if v}
    print(f"Need URL coords from {len(date_to_urls)} date CSVs "
          f"({len(enriched) - has_loc.sum():,} records without location_name skipped)")
    url_coords = load_url_coords(date_to_urls)

    # 2. Attach GDELT coords to enriched rows
    enriched["gdelt_lat"] = enriched["url"].map(lambda u: url_coords.get(u, (None, None))[0])
    enriched["gdelt_lon"] = enriched["url"].map(lambda u: url_coords.get(u, (None, None))[1])

    # 3. Filter to rows with both a location_name AND GDELT coords
    usable = enriched[
        enriched["location_name"].notna()
        & (enriched["location_name"].str.strip() != "")
        & enriched["gdelt_lat"].notna()
        & enriched["gdelt_lon"].notna()
    ].copy()

    total_enriched   = len(enriched)
    has_location     = int((enriched["location_name"].str.strip() != "").sum())
    has_gdelt_coords = int(enriched["gdelt_lat"].notna().sum())
    n_usable         = len(usable)

    print(f"\nTotal enriched records:          {total_enriched}")
    print(f"With non-empty location_name:    {has_location}")
    print(f"With GDELT coords:               {has_gdelt_coords}")
    print(f"With BOTH (usable for check):    {n_usable}")

    if n_usable == 0:
        print("\nNo rows with both location_name and GDELT coords  -  nothing to verify.")
        print("Tip: run more days through the enrichment pipeline first.")
        return

    # 4. Null-island check (on full dataset with coords, not just usable)
    has_coords = enriched[enriched["gdelt_lat"].notna()].copy()
    null_island_mask = (
        has_coords["gdelt_lat"].abs() < NULL_ISLAND_DEG
    ) & (
        has_coords["gdelt_lon"].abs() < NULL_ISLAND_DEG
    )
    n_null_island = null_island_mask.sum()
    print(f"\nNull-island coords (|lat|<{NULL_ISLAND_DEG}, |lon|<{NULL_ISLAND_DEG}): {n_null_island} / {len(has_coords)} "
          f"({100*n_null_island/max(len(has_coords),1):.1f}%)")

    # 5. Coordinate clustering: find most common coord pairs (country centroid proxy)
    coord_pairs = has_coords[["gdelt_lat", "gdelt_lon"]].round(3)
    coord_counts = coord_pairs.value_counts().head(10)
    print("\nTop 10 most repeated GDELT coordinate pairs (country centroid indicator):")
    print(coord_counts.to_string())

    # 6. Geocode unique location_names via Nominatim
    geocache = {}
    if GEOCACHE_FILE.exists():
        with open(GEOCACHE_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        # Cache entries from geocode_locations.py are 4-element lists:
        # [lat, lon, country_code, bbox_span_km].  Preserve all elements so
        # that if this script re-saves the cache it does not strip the
        # country_code/bbox fields that the pipeline relies on for coord
        # validation and coarse-location screening.
        geocache = {k: tuple(v) if v is not None else None for k, v in raw.items()}
        print(f"\nLoaded {len(geocache)} cached geocodes from {GEOCACHE_FILE.name}")

    unique_locations = [
        loc for loc in usable["location_name"].unique()
        if loc not in geocache
    ]
    print(f"New locations to geocode: {len(unique_locations)}")

    def _save_cache():
        with open(GEOCACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {k: list(v) if v is not None else None for k, v in geocache.items()},
                f, indent=2
            )

    SAVE_EVERY = 50   # flush to disk every N geocodes so interruptions lose minimal work
    if unique_locations:
        for i, loc in enumerate(tqdm(unique_locations, desc="Geocoding (1 req/s)", unit="loc"), 1):
            geocode_location(loc, geocache)
            if i % SAVE_EVERY == 0:
                _save_cache()

        _save_cache()   # final save
        print(f"Cache saved to {GEOCACHE_FILE.name}  ({len(geocache):,} total entries)")

    # 7. Compute distances
    rows = []
    for _, row in usable.iterrows():
        geocoded = geocache.get(row["location_name"])
        if geocoded is None:
            continue
        dist = haversine_km(
            row["gdelt_lat"], row["gdelt_lon"],
            geocoded[0], geocoded[1]
        )
        rows.append({
            "url":            row["url"],
            "location_name":  row["location_name"],
            "gdelt_lat":      row["gdelt_lat"],
            "gdelt_lon":      row["gdelt_lon"],
            "geocoded_lat":   geocoded[0],
            "geocoded_lon":   geocoded[1],
            "distance_km":    round(dist, 1),
            "is_outlier":     dist > OUTLIER_KM,
        })

    if not rows:
        print("No rows could be geocoded. Check Nominatim connectivity.")
        return

    results = pd.DataFrame(rows)

    # 8. Statistics
    n     = len(results)
    d     = results["distance_km"]
    pct   = lambda threshold: f"{100 * (d <= threshold).sum() / n:.1f}%"

    summary_lines = [
        f"GDELT Coordinate Accuracy Report",
        f"=================================",
        f"Rows analysed:                  {n}",
        f"",
        f"Distance to LLM-geocoded location:",
        f"  Median:                       {d.median():.0f} km",
        f"  Mean:                         {d.mean():.0f} km",
        f"  Within  50 km:                {pct(50)}",
        f"  Within 200 km:                {pct(200)}",
        f"  Within 500 km:                {pct(500)}",
        f"  Beyond 500 km (outliers):     {pct(999999)[:-1]} -> {(d > OUTLIER_KM).sum()} rows",
        f"",
        f"Null-island hits (|lat|<{NULL_ISLAND_DEG}, |lon|<{NULL_ISLAND_DEG}):",
        f"  {n_null_island} / {len(has_coords)} rows with any GDELT coord "
        f"({100*n_null_island/max(len(has_coords),1):.1f}%)",
        f"",
        f"Coverage stats:",
        f"  Total enriched records:       {total_enriched}",
        f"  With location_name:           {has_location} ({100*has_location/total_enriched:.1f}%)",
        f"  With GDELT coords:            {has_gdelt_coords} ({100*has_gdelt_coords/total_enriched:.1f}%)",
        f"  Verifiable (both):            {n_usable} ({100*n_usable/total_enriched:.1f}%)",
    ]

    print("\n" + "\n".join(summary_lines))

    summary_path = OUT_DIR / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"\nSummary saved -> {summary_path}")

    # 9. Save CSVs
    all_path = OUT_DIR / "all_results.csv"
    results.to_csv(all_path, index=False)
    print(f"All results -> {all_path}")

    outliers = results[results["is_outlier"]].sort_values("distance_km", ascending=False)
    out_path = OUT_DIR / "outliers.csv"
    outliers.to_csv(out_path, index=False)
    print(f"Outliers ({len(outliers)} rows) -> {out_path}")

    # 10. Distance histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(d.clip(upper=5000), bins=50, edgecolor="white", linewidth=0.4)
    ax.axvline(200, color="orange", linestyle="--", label="200 km")
    ax.axvline(500, color="red",    linestyle="--", label="500 km")
    ax.set_xlabel("Distance from GDELT coord to LLM-geocoded location (km, capped at 5000)")
    ax.set_ylabel("Number of articles")
    ax.set_title("GDELT coordinate accuracy vs. LLM-extracted location")
    ax.legend()
    fig.tight_layout()
    hist_path = OUT_DIR / "distance_histogram.png"
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"Histogram -> {hist_path}")


if __name__ == "__main__":
    main()
