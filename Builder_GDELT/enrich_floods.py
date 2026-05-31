"""
enrich_floods.py
================
Geocode and enrich GDELT flood events with hydro-climate indicators.

Pipeline stages
---------------
1. Geocode   — Attach lat/lon from GDELT URL CSVs (actiongeo_lat/lon),
               then Nominatim for any remaining events with a location_name.
2. CHIRPS    — 3/7/14/30-day rainfall totals + 7-day anomaly.
3. GPM       — 1/3/7-day totals, peak daily and 3-hourly intensity.
4. ERA5      — Soil moisture (day0, 7d, 30d mean), precipitation, runoff.
5. Static    — Population density (WorldPop), JRC surface water occurrence/recurrence.
6. SPI       — 30-day Standardized Precipitation Index.
7. Merge     — Left-join all layers into one final JSONL.

Input
-----
    Builder_GDELT/results/combined/all_consolidated.jsonl
    data/urls/*.csv  (GDELT URL-level CSVs with actiongeo_lat / actiongeo_lon)

Outputs
-------
    Builder_GDELT/outputs/gdelt_floods_geocoded.jsonl
    Builder_GDELT/outputs/gdelt_floods_chirps.jsonl
    Builder_GDELT/outputs/gdelt_floods_gpm.jsonl
    Builder_GDELT/outputs/gdelt_floods_era5.jsonl
    Builder_GDELT/outputs/gdelt_floods_static.jsonl
    Builder_GDELT/outputs/gdelt_floods_spi.jsonl
    Builder_GDELT/outputs/gdelt_floods_enriched.jsonl   ← final product

Usage
-----
    python -m Builder_GDELT.enrich_floods
    python -m Builder_GDELT.enrich_floods --skip-geocode   # if geocoded file exists
    python -m Builder_GDELT.enrich_floods --only-merge     # just re-merge existing layers
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).resolve().parent.parent
INPUT_PATH  = ROOT / "Builder_GDELT" / "results" / "combined" / "all_consolidated.jsonl"
URLS_DIR    = ROOT / "data" / "urls"
OUT_DIR     = ROOT / "Builder_GDELT" / "outputs"

GEE_PROJECT  = "gen-lang-client-0809810190"
BUFFER_KM    = 25.0
MAX_WORKERS  = 10      # parallel GEE threads per enrichment step

GEOCODED_PATH = ROOT / "cache" / "gdelt" / "gdelt_floods_geocoded.jsonl"

STEPS = [
    ("chirps",  OUT_DIR / "gdelt_floods_chirps.jsonl"),
    ("gpm",     OUT_DIR / "gdelt_floods_gpm.jsonl"),
    ("era5",    OUT_DIR / "gdelt_floods_era5.jsonl"),
    ("static",  OUT_DIR / "gdelt_floods_static.jsonl"),
    ("spi",     OUT_DIR / "gdelt_floods_spi.jsonl"),
]

ENRICHMENT_FIELDS = {
    "chirps":  ["chirps_3d_total_mm", "chirps_7d_total_mm", "chirps_14d_total_mm",
                "chirps_30d_total_mm", "chirps_peak_daily_mm", "chirps_7d_baseline_mm",
                "chirps_7d_anom_mm", "chirps_7d_anom_pct"],
    "gpm":     ["gpm_1d_total_mm", "gpm_3d_total_mm", "gpm_7d_total_mm",
                "gpm_peak_daily_mm", "gpm_peak_3h_mm"],
    "era5":    ["era5_soil_moisture_day0", "era5_soil_moisture_7d_mean",
                "era5_soil_moisture_30d_mean", "era5_precip_7d_mm", "era5_runoff_7d_mm"],
    "static":  ["pop_count_25km", "pop_density_km2", "jrc_occurrence_pct", "jrc_recurrence_pct"],
    "spi":     ["spi_30d", "spi_30d_pct"],
}


# ---------------------------------------------------------------------------
# Event key — stable identifier based on sorted URLs
# ---------------------------------------------------------------------------

def _event_key(e: dict) -> str:
    urls = sorted(e.get("urls") or [e.get("url", "")])
    return "|".join(urls)


# ---------------------------------------------------------------------------
# Stage 0: Load base events
# ---------------------------------------------------------------------------

def _load_base_events() -> list[dict]:
    if not INPUT_PATH.exists():
        log.error(f"Input not found: {INPUT_PATH}")
        sys.exit(1)

    all_events: list[dict] = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_events.append(json.loads(line))

    floods = [e for e in all_events if e.get("disruption_type") == "flood"]
    log.info(f"Loaded {len(all_events):,} total events | {len(floods):,} floods")

    # Normalise date field so enrichers see date_start
    for e in floods:
        if "date_start" not in e:
            e["date_start"] = (e.get("event_date") or "")[:10]

    return floods


# ---------------------------------------------------------------------------
# Stage 1: Geocode
# ---------------------------------------------------------------------------

def _build_url_coord_index() -> dict[str, tuple[float, float]]:
    """Read all data/urls/*.csv and return {url_normalized: (lat, lon)}."""
    index: dict[str, tuple[float, float]] = {}
    csv_files = sorted(URLS_DIR.glob("*.csv"))
    if not csv_files:
        log.warning(f"No URL CSVs found in {URLS_DIR}")
        return index
    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                url = row.get("url_normalized", "").strip()
                lat_s = row.get("actiongeo_lat", "").strip()
                lon_s = row.get("actiongeo_lon", "").strip()
                if url and lat_s and lon_s:
                    try:
                        index[url] = (float(lat_s), float(lon_s))
                    except ValueError:
                        pass
    log.info(f"URL->coords index: {len(index):,} entries from {len(csv_files)} CSV files")
    return index


def _nominatim_lookup(query: str) -> Optional[tuple[float, float]]:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    geolocator = Nominatim(user_agent="iib-gdelt-geocoder")
    try:
        location = geolocator.geocode(query, timeout=10)
        if location:
            return (location.latitude, location.longitude)
    except (GeocoderTimedOut, GeocoderServiceError) as exc:
        log.warning(f"Nominatim failed for {repr(query)}: {exc}")
    return None


def geocode_events(events: list[dict]) -> list[dict]:
    """
    Attach lat/lon to events.

    Priority:
      1. Already has lat/lon  → keep, mark geo_source='original'
      2. URL matches GDELT CSV → use actiongeo_lat/lon, geo_source='gdelt_csv'
      3. Has location_name    → Nominatim query, geo_source='nominatim'
      4. Otherwise            → lat/lon remain None, geo_source=None
    """
    url_index = _build_url_coord_index()

    geocoded: list[dict] = []
    n_original = n_csv = n_nominatim = n_none = 0

    # Collect Nominatim targets upfront (unique location names)
    nominatim_cache: dict[str, Optional[tuple[float, float]]] = {}
    need_nominatim: list[dict] = []

    for e in events:
        row = dict(e)
        if row.get("lat") and row.get("lon"):
            row["geo_source"] = row.get("geo_source") or "original"
            n_original += 1
            geocoded.append(row)
            continue

        # Try URL CSV
        matched = False
        for url in (row.get("urls") or []):
            if url in url_index:
                row["lat"], row["lon"] = url_index[url]
                row["geo_source"] = "gdelt_csv"
                n_csv += 1
                matched = True
                break

        if matched:
            geocoded.append(row)
            continue

        # Queue for Nominatim
        loc = (row.get("location_name") or "").strip()
        if loc and loc not in nominatim_cache:
            nominatim_cache[loc] = None  # placeholder
            need_nominatim.append(row)
        elif loc in nominatim_cache:
            need_nominatim.append(row)  # will be resolved below
        else:
            row["geo_source"] = None
            n_none += 1
            geocoded.append(row)

    # Nominatim pass
    if need_nominatim:
        log.info(f"Nominatim geocoding {len(nominatim_cache)} unique location queries "
                 f"({len(need_nominatim)} events, ~1 req/sec)...")
        for i, (loc, _) in enumerate(
            [(k, v) for k, v in nominatim_cache.items() if v is None]
        ):
            result = _nominatim_lookup(loc)
            nominatim_cache[loc] = result
            time.sleep(1.1)
            if (i + 1) % 10 == 0:
                log.info(f"  Nominatim: {i+1}/{len(nominatim_cache)} done")

        for row in need_nominatim:
            loc = (row.get("location_name") or "").strip()
            result = nominatim_cache.get(loc)
            if result:
                row["lat"], row["lon"] = result
                row["geo_source"] = "nominatim"
                n_nominatim += 1
            else:
                row["geo_source"] = None
                n_none += 1
            geocoded.append(row)

    with_coords = sum(1 for e in geocoded if e.get("lat") and e.get("lon"))
    log.info(
        f"Geocoding complete: {n_original} original | {n_csv} gdelt_csv | "
        f"{n_nominatim} nominatim | {n_none} none | "
        f"{with_coords}/{len(geocoded)} have coords"
    )
    return geocoded


# ---------------------------------------------------------------------------
# Stage 7: Merge all enrichment layers
# ---------------------------------------------------------------------------

def _merge(all_events: list[dict]) -> None:
    """
    Left-join all enrichment layers onto the full event base (including
    un-geocoded events which receive null hydro fields).  Strips internal
    _*_processed flags so the final schema is clean.
    """
    merged = {_event_key(e): dict(e) for e in all_events}

    for step_name, step_path in STEPS:
        if not step_path.exists():
            log.warning(f"Step output missing, skipping: {step_path.name}")
            continue
        fields = ENRICHMENT_FIELDS[step_name]
        with step_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = _event_key(row)
                if key in merged:
                    for field in fields:
                        if field in row:
                            merged[key][field] = row[field]

    # Strip internal processing flags
    private_flags = {f"_{name}_processed" for name, _ in STEPS}

    output_path = OUT_DIR / "gdelt_floods_enriched.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for row in merged.values():
            clean = {k: v for k, v in row.items() if k not in private_flags}
            f.write(json.dumps(clean, default=str) + "\n")

    all_fields = [f for fs in ENRICHMENT_FIELDS.values() for f in fs]
    n_enriched = sum(1 for r in merged.values() if any(r.get(f) is not None for f in all_fields))
    n_no_coords = sum(1 for r in merged.values() if not (r.get("lat") and r.get("lon")))
    log.info(f"Written: {output_path}")
    log.info(
        f"  {len(merged):,} total events | {n_enriched:,} with hydro indicators | "
        f"{n_no_coords:,} without coordinates (null hydro fields)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Geocode and enrich GDELT flood events.")
    parser.add_argument("--skip-geocode", action="store_true",
                        help="Skip geocoding and load from existing geocoded file.")
    parser.add_argument("--only-merge", action="store_true",
                        help="Skip geocoding and enrichment; only re-merge existing layers.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load or geocode ---
    if args.only_merge:
        log.info(f"--only-merge: loading from {GEOCODED_PATH}")
        if not GEOCODED_PATH.exists():
            log.error("Geocoded file not found. Run without --only-merge first.")
            sys.exit(1)
        events = [json.loads(l) for l in GEOCODED_PATH.open(encoding="utf-8") if l.strip()]
        log.info(f"Loaded {len(events):,} geocoded events")
        log.info("=== Merging layers ===")
        _merge(events)
        return

    if args.skip_geocode and GEOCODED_PATH.exists():
        log.info(f"--skip-geocode: loading from {GEOCODED_PATH}")
        events = [json.loads(l) for l in GEOCODED_PATH.open(encoding="utf-8") if l.strip()]
        for e in events:
            if not e.get("date_start"):
                e["date_start"] = (e.get("event_date") or "")[:10]
        log.info(f"Loaded {len(events):,} events")
    else:
        log.info("=== Stage 1: Load base events ===")
        events = _load_base_events()

        log.info("=== Stage 2: Geocode ===")
        events = geocode_events(events)

        GEOCODED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with GEOCODED_PATH.open("w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e, default=str) + "\n")
        log.info(f"Geocoded file saved: {GEOCODED_PATH}")

    # Filter to events with coordinates for enrichment
    enrichable = [e for e in events if e.get("lat") and e.get("lon")]
    log.info(f"{len(enrichable):,} events have coordinates and will be enriched")

    if not enrichable:
        log.error("No events with coordinates — cannot run GEE enrichment.")
        sys.exit(1)

    # --- CHIRPS ---
    log.info("=== Stage 3: CHIRPS rainfall ===")
    from Builder_Reference.helper_scripts.enrichment.chirps import enrich_events_with_chirps
    enrich_events_with_chirps(
        events=enrichable, project=GEE_PROJECT, buffer_km=BUFFER_KM,
        output_path=STEPS[0][1], resume=True, max_workers=MAX_WORKERS,
    )

    # --- GPM ---
    log.info("=== Stage 4: GPM IMERG rainfall ===")
    from Builder_Reference.helper_scripts.enrichment.gpm import enrich_events_with_gpm
    enrich_events_with_gpm(
        events=enrichable, project=GEE_PROJECT, buffer_km=BUFFER_KM,
        output_path=STEPS[1][1], resume=True, max_workers=MAX_WORKERS,
    )

    # --- ERA5 ---
    log.info("=== Stage 5: ERA5 soil moisture & runoff ===")
    from Builder_Reference.helper_scripts.enrichment.era5 import enrich_events_with_era5
    enrich_events_with_era5(
        events=enrichable, project=GEE_PROJECT, buffer_km=BUFFER_KM,
        output_path=STEPS[2][1], resume=True, max_workers=MAX_WORKERS,
    )

    # --- Static (WorldPop + JRC) ---
    log.info("=== Stage 6: Population & JRC surface water ===")
    from Builder_Reference.helper_scripts.enrichment.static_features import enrich_events_with_static_features
    enrich_events_with_static_features(
        events=enrichable, project=GEE_PROJECT, buffer_km=BUFFER_KM,
        output_path=STEPS[3][1], resume=True, max_workers=MAX_WORKERS,
    )

    # --- SPI ---
    log.info("=== Stage 7: SPI-30 ===")
    from Builder_Reference.helper_scripts.enrichment.spi import enrich_events_with_spi
    enrich_events_with_spi(
        events=enrichable, project=GEE_PROJECT, buffer_km=BUFFER_KM,
        output_path=STEPS[4][1], resume=True, max_workers=MAX_WORKERS,
    )

    # --- Merge ---
    log.info("=== Stage 8: Merge all layers ===")
    _merge(events)  # all events — un-geocoded ones get null hydro fields

    log.info("Done. Final output: %s", OUT_DIR / "gdelt_floods_enriched.jsonl")


if __name__ == "__main__":
    main()
