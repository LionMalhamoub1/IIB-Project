# Matches GDELT protest clusters against ACLED per day.
# Geographic match if nearest same-country ACLED event is within GEO_MATCH_KM, else country-level.
# ACLED events with no GDELT match are flagged as missed detections.

from __future__ import annotations

import argparse
import json
import math
import glob
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).resolve().parent
_ACLED   = _HERE.parent / "External_Databases" / "ACLED" / "data" / "raw" / "events"
OUT_DIR  = _HERE / "comparison"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# ACLED event types treated as protests / social disruptions
ACLED_PROTEST_TYPES = {"Protests", "Riots"}

# Days either side of the GDELT cluster date to search in ACLED
DATE_WINDOW = 2

# Kilometre threshold for a "geographic" match (vs country-level)
GEO_MATCH_KM = 100.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _load_acled_for_year(year: int) -> pd.DataFrame:
    """Load all ACLED parquet files for the given year across all countries."""
    pattern = str(_ACLED / f"iso3=*" / f"year={year}.parquet")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        m = re.search(r"iso3=([A-Z]+)", f)
        df = pd.read_parquet(f, columns=[
            "event_id_cnty", "event_date", "event_type", "sub_event_type",
            "country", "admin1", "admin2", "location",
            "latitude", "longitude", "iso3",
        ])
        if m and "iso3" not in df.columns:
            df["iso3"] = m.group(1)
        parts.append(df)
    acled = pd.concat(parts, ignore_index=True)
    acled["event_date"] = pd.to_datetime(acled["event_date"], errors="coerce")
    acled["iso3"] = acled["iso3"].astype(str).str.upper().str.strip()
    acled = acled[acled["event_type"].isin(ACLED_PROTEST_TYPES)].copy()
    return acled


def _load_grouped(path: Path) -> list[dict]:
    clusters = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    clusters.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return clusters


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def _match_clusters_to_acled(
    clusters: list[dict],
    acled: pd.DataFrame,
    file_date: pd.Timestamp,
) -> tuple[list[dict], set]:
    """
    For each GDELT cluster find the best-matching ACLED event.
    Returns (detail_rows, set_of_matched_acled_ids).
    """
    rows = []
    matched_acled_ids: set = set()

    # Pre-filter ACLED to ±DATE_WINDOW around file_date and valid coords
    date_lo = file_date - pd.Timedelta(days=DATE_WINDOW)
    date_hi = file_date + pd.Timedelta(days=DATE_WINDOW)
    acled_window = acled[
        (acled["event_date"] >= date_lo) & (acled["event_date"] <= date_hi)
    ].copy()

    for cl in clusters:
        iso3 = (cl.get("iso3") or "").upper().strip()
        g_lat = cl.get("lat")
        g_lon = cl.get("lon")

        # Filter ACLED to same country
        ac_country = acled_window[acled_window["iso3"] == iso3]

        best_id       = None
        best_dist     = None
        best_date     = None
        best_loc      = None
        best_sub      = None
        best_actor    = None
        match_type    = "no_match"

        if not ac_country.empty:
            if g_lat is not None and g_lon is not None:
                # Compute haversine distance to all ACLED candidates
                ac_valid = ac_country.dropna(subset=["latitude", "longitude"])
                if not ac_valid.empty:
                    dists = ac_valid.apply(
                        lambda r: _haversine_km(g_lat, g_lon, r["latitude"], r["longitude"]),
                        axis=1,
                    )
                    idx_min = dists.idxmin()
                    min_dist = dists[idx_min]
                    best_row = ac_valid.loc[idx_min]
                    best_id   = best_row["event_id_cnty"]
                    best_dist = round(min_dist, 1)
                    best_date = str(best_row["event_date"].date())
                    best_loc  = f"{best_row.get('location', '')} ({best_row.get('admin1', '')})"
                    best_sub  = best_row["sub_event_type"]
                    best_actor = best_row.get("actor1") or ""
                    match_type = "geographic" if min_dist <= GEO_MATCH_KM else "country_only"
            else:
                # No coordinates — country-level match to nearest-date event
                ac_sorted = ac_country.sort_values("event_date")
                if not ac_sorted.empty:
                    best_row  = ac_sorted.iloc[0]
                    best_id   = best_row["event_id_cnty"]
                    best_date = str(best_row["event_date"].date())
                    best_loc  = f"{best_row.get('location', '')} ({best_row.get('admin1', '')})"
                    best_sub  = best_row["sub_event_type"]
                    best_actor = best_row.get("actor1") or ""
                    match_type = "country_only"

        if best_id is not None:
            matched_acled_ids.add(best_id)

        rows.append({
            "file_date":           str(file_date.date()),
            "cluster_id":          cl.get("cluster_id", ""),
            "gdelt_type":          cl.get("disruption_type", ""),
            "gdelt_date":          cl.get("event_date", "")[:10] if cl.get("event_date") else "",
            "gdelt_iso3":          iso3,
            "gdelt_location":      cl.get("location_name", ""),
            "gdelt_n_articles":    cl.get("n_articles", 0),
            "gdelt_confidence":    round(cl.get("confidence_max", 0.0), 3),
            "gdelt_lat":           round(g_lat, 4) if g_lat is not None else None,
            "gdelt_lon":           round(g_lon, 4) if g_lon is not None else None,
            "match_type":          match_type,
            "distance_km":         best_dist,
            "acled_event_id":      best_id,
            "acled_date":          best_date,
            "acled_location":      best_loc,
            "acled_sub_event":     best_sub,
            "acled_actor":         best_actor,
        })

    return rows, matched_acled_ids


def _missed_acled_rows(
    acled: pd.DataFrame,
    file_date: pd.Timestamp,
    matched_ids: set,
) -> list[dict]:
    """Return ACLED events on file_date that were not matched by any GDELT cluster."""
    day_acled = acled[acled["event_date"] == file_date]
    missed = day_acled[~day_acled["event_id_cnty"].isin(matched_ids)]
    rows = []
    for _, r in missed.iterrows():
        rows.append({
            "file_date":       str(file_date.date()),
            "cluster_id":      None,
            "gdelt_type":      None,
            "gdelt_date":      None,
            "gdelt_iso3":      None,
            "gdelt_location":  None,
            "gdelt_n_articles": None,
            "gdelt_confidence": None,
            "gdelt_lat":       None,
            "gdelt_lon":       None,
            "match_type":      "acled_only",
            "distance_km":     None,
            "acled_event_id":  r["event_id_cnty"],
            "acled_date":      str(r["event_date"].date()),
            "acled_location":  f"{r.get('location', '')} ({r.get('admin1', '')})",
            "acled_sub_event": r["sub_event_type"],
            "acled_actor":     r.get("actor1") or "",
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_date(file_date_str: str, acled_cache: dict) -> Optional[dict]:
    grouped_path = _HERE / "grouped" / f"{file_date_str}_grouped.jsonl"
    if not grouped_path.exists():
        print(f"  No grouped file for {file_date_str} — skipping")
        return None

    file_date = pd.Timestamp(
        f"{file_date_str[:4]}-{file_date_str[4:6]}-{file_date_str[6:]}"
    )
    year = file_date.year

    if year not in acled_cache:
        print(f"  Loading ACLED for {year} ...")
        acled_cache[year] = _load_acled_for_year(year)
    acled = acled_cache[year]

    clusters = _load_grouped(grouped_path)
    if not clusters:
        print(f"  {file_date_str}: no clusters")
        return None

    detail_rows, matched_ids = _match_clusters_to_acled(clusters, acled, file_date)
    missed_rows = _missed_acled_rows(acled, file_date, matched_ids)
    all_rows = detail_rows + missed_rows

    # Save per-day detail CSV
    out_path = OUT_DIR / f"{file_date_str}_comparison.csv"
    pd.DataFrame(all_rows).to_csv(out_path, index=False)

    # Summary stats for this day
    n_gdelt      = len(clusters)
    n_geo        = sum(1 for r in detail_rows if r["match_type"] == "geographic")
    n_country    = sum(1 for r in detail_rows if r["match_type"] == "country_only")
    n_no_match   = sum(1 for r in detail_rows if r["match_type"] == "no_match")
    n_acled_only = len(missed_rows)

    # ACLED total on this exact day
    n_acled_day = len(acled[acled["event_date"] == file_date])

    print(
        f"  {file_date_str}: {n_gdelt} GDELT clusters | "
        f"geo={n_geo} country={n_country} no_match={n_no_match} | "
        f"ACLED-only={n_acled_only} / {n_acled_day} ACLED events"
    )

    return {
        "date":             file_date_str,
        "gdelt_clusters":   n_gdelt,
        "geo_match":        n_geo,
        "country_match":    n_country,
        "no_match":         n_no_match,
        "acled_only":       n_acled_only,
        "acled_total_day":  n_acled_day,
        "pct_gdelt_matched": round(100 * (n_geo + n_country) / n_gdelt, 1) if n_gdelt else 0,
        "pct_acled_matched": round(100 * len(matched_ids) / n_acled_day, 1) if n_acled_day else 0,
    }


def _daterange(start: str, end: str) -> list[str]:
    dates = pd.date_range(start=start, end=end, freq="D")
    return [d.strftime("%Y%m%d") for d in dates]


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--date",  help="Single date YYYYMMDD")
    group.add_argument("--start", help="Start date YYYYMMDD (use with --end)")
    parser.add_argument("--end",  help="End date YYYYMMDD")
    args = parser.parse_args()

    if args.date:
        dates = [args.date]
    elif args.start:
        end = args.end or args.start
        dates = _daterange(args.start, end)
    else:
        # Auto-discover from all grouped files
        files = sorted((_HERE / "grouped").glob("*_grouped.jsonl"))
        dates = [f.stem.replace("_grouped", "") for f in files]

    if not dates:
        print("No grouped files found.")
        sys.exit(0)

    print(f"Comparing {len(dates)} date(s): {dates[0]} … {dates[-1]}")

    acled_cache: dict[int, pd.DataFrame] = {}
    summary_rows = []

    for d in dates:
        row = process_date(d, acled_cache)
        if row:
            summary_rows.append(row)

    if summary_rows:
        summary_path = OUT_DIR / "summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
        print(pd.DataFrame(summary_rows).to_string(index=False))
    else:
        print("No results to summarise.")


if __name__ == "__main__":
    main()
