"""
Run all hydro-climate enrichments over the consolidated GDELT flood events.

Usage:
    python -m Builder_Reference.helper_scripts.enrichment.run_gdelt_enrichment

Input:
    Builder_GDELT/results/combined/all_consolidated.jsonl
    (lat/lon are already present, passed through from data/urls/ CSVs via
    DisruptionExtractor -> consolidateExtractions)

Filters to disruption_type="flood" in 2017-2020.

Runs all enrichment steps in parallel-friendly order.
Each step output is resumable if interrupted.

Final output:
    cache/gdelt/gdelt_floods_enriched.jsonl
"""

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

GEE_PROJECT = "gen-lang-client-0809810190"
BUFFER_KM   = 25.0
DATE_FROM   = "2017-01-01"
DATE_TO     = "2020-12-31"

ROOT       = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT / "Builder_GDELT" / "results" / "combined" / "all_consolidated.jsonl"
CACHE_DIR  = ROOT / "cache" / "gdelt"

STEPS = [
    ("chirps",  CACHE_DIR / "gdelt_floods_chirps.jsonl"),
    ("gpm",     CACHE_DIR / "gdelt_floods_gpm.jsonl"),
    ("era5",    CACHE_DIR / "gdelt_floods_era5.jsonl"),
    ("static",  CACHE_DIR / "gdelt_floods_static.jsonl"),
    ("spi",     CACHE_DIR / "gdelt_floods_spi.jsonl"),
    ("glofas",  CACHE_DIR / "gdelt_floods_glofas.jsonl"),
]

ENRICHMENT_FIELDS = {
    "chirps":  ["chirps_3d_total_mm","chirps_7d_total_mm","chirps_14d_total_mm",
                "chirps_30d_total_mm","chirps_peak_daily_mm","chirps_7d_baseline_mm",
                "chirps_7d_anom_mm","chirps_7d_anom_pct"],
    "gpm":     ["gpm_1d_total_mm","gpm_3d_total_mm","gpm_7d_total_mm",
                "gpm_peak_daily_mm","gpm_peak_3h_mm"],
    "era5":    ["era5_soil_moisture_day0","era5_soil_moisture_7d_mean",
                "era5_soil_moisture_30d_mean","era5_precip_7d_mm","era5_runoff_7d_mm"],
    "static":  ["pop_count_25km","pop_density_km2","jrc_occurrence_pct","jrc_recurrence_pct"],
    "spi":     ["spi_30d","spi_30d_pct"],
    "glofas":  ["glofas_discharge_m3s","glofas_discharge_7d_max","glofas_discharge_pct_rank"],
}


def _event_key(e: dict) -> str:
    urls = sorted(e.get("urls") or [e.get("url", "")])
    return "|".join(urls)


def _load_events() -> list[dict]:
    if not INPUT_PATH.exists():
        log.error(f"Input not found: {INPUT_PATH}")
        sys.exit(1)

    all_events = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_events.append(json.loads(line))

    events = [
        e for e in all_events
        if e.get("disruption_type") == "flood"
        and DATE_FROM <= (e.get("event_date") or "")[:10] <= DATE_TO
    ]

    has_coords = sum(1 for e in events if e.get("lat") and e.get("lon"))
    log.info(
        f"Loaded {len(all_events):,} total | flood {DATE_FROM}–{DATE_TO}: "
        f"{len(events):,} | with coordinates: {has_coords:,}"
    )

    if has_coords == 0:
        log.error(
            "No events have coordinates. Ensure the data/urls/ CSVs have lat/lon columns "
            "and the database has been rebuilt with the updated DisruptionExtractor."
        )
        sys.exit(1)

    # Normalise date field name to match enricher expectations (event_date -> date_start)
    for e in events:
        if "date_start" not in e:
            e["date_start"] = (e.get("event_date") or "")[:10]

    return events


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading GDELT flood events from {INPUT_PATH}")
    events = _load_events()

    # --- CHIRPS ---
    log.info("=== Step 1/6: CHIRPS rainfall ===")
    from Builder_Reference.helper_scripts.enrichment.chirps import enrich_events_with_chirps
    enrich_events_with_chirps(
        events=events, project=GEE_PROJECT, buffer_km=BUFFER_KM,
        output_path=STEPS[0][1], resume=True,
    )

    # --- GPM ---
    log.info("=== Step 2/6: GPM IMERG rainfall ===")
    from Builder_Reference.helper_scripts.enrichment.gpm import enrich_events_with_gpm
    enrich_events_with_gpm(
        events=events, project=GEE_PROJECT, buffer_km=BUFFER_KM,
        output_path=STEPS[1][1], resume=True,
    )

    # --- ERA5 ---
    log.info("=== Step 3/6: ERA5 soil moisture & runoff ===")
    from Builder_Reference.helper_scripts.enrichment.era5 import enrich_events_with_era5
    enrich_events_with_era5(
        events=events, project=GEE_PROJECT, buffer_km=BUFFER_KM,
        output_path=STEPS[2][1], resume=True,
    )

    # --- Static (WorldPop + JRC) ---
    log.info("=== Step 4/6: Population & JRC surface water ===")
    from Builder_Reference.helper_scripts.enrichment.static_features import enrich_events_with_static_features
    enrich_events_with_static_features(
        events=events, project=GEE_PROJECT, buffer_km=BUFFER_KM,
        output_path=STEPS[3][1], resume=True,
    )

    # --- SPI ---
    log.info("=== Step 5/6: SPI-30 ===")
    from Builder_Reference.helper_scripts.enrichment.spi import enrich_events_with_spi
    enrich_events_with_spi(
        events=events, project=GEE_PROJECT, buffer_km=BUFFER_KM,
        output_path=STEPS[4][1], resume=True,
    )

    # --- GloFAS ---
    if os.environ.get("EWDS_KEY") or os.environ.get("CDSAPI_KEY"):
        log.info("=== Step 6/6: GloFAS river discharge ===")
        from Builder_Reference.helper_scripts.enrichment.glofas import enrich_events_with_glofas
        enrich_events_with_glofas(
            events=events, output_path=STEPS[5][1], resume=True,
        )
    else:
        log.warning("Skipping GloFAS — EWDS_KEY not set.")

    # --- Merge ---
    log.info("=== Merging all steps into final enriched file ===")
    _merge(events)


def _merge(base_events: list[dict]):
    merged = {_event_key(e): dict(e) for e in base_events}

    for step_name, step_path in STEPS:
        if not step_path.exists():
            log.warning(f"Missing step output, skipping: {step_path.name}")
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

    output_path = CACHE_DIR / "gdelt_floods_enriched.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for row in merged.values():
            f.write(json.dumps(row, default=str) + "\n")

    all_fields = [f for fields in ENRICHMENT_FIELDS.values() for f in fields]
    n_any = sum(1 for r in merged.values() if any(r.get(f) is not None for f in all_fields))
    log.info(f"Written to {output_path}")
    log.info(f"  {len(merged):,} events | {n_any:,} with at least one indicator")


if __name__ == "__main__":
    main()
