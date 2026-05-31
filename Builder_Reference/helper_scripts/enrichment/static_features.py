"""
Static GEE feature enrichment: population exposure and JRC surface water recurrence.

Both are time-independent (or year-snapped) so each event requires only one GEE call
per dataset — much faster than the time-varying CHIRPS/GPM/ERA5 extractors.

Indicators extracted per event:

  Population (WorldPop/GP/100m/pop, 2000–2020):
    pop_count_25km       total population within buffer (sum of 100m pixels)
    pop_density_km2      population density (pop_count / buffer area in km²)

  JRC Global Surface Water (JRC/GSW1_4/GlobalSurfaceWater, 1984–2021):
    jrc_occurrence_pct   mean % of months observed as surface water (0–100)
    jrc_recurrence_pct   mean % of wet years with water present (0–100)

  Terrain (USGS/SRTMGL1_003, 30 m DEM):
    terrain_slope_mean   mean slope (degrees) within buffer — flat terrain floods more easily

jrc_occurrence_pct > 5 indicates a location regularly inundated.
jrc_recurrence_pct > 50 indicates a location that floods most years when wet conditions occur.
terrain_slope_mean < 2 degrees indicates flat floodplain terrain.
"""

import ee
import logging
import math

log = logging.getLogger(__name__)

WORLDPOP_COLLECTION = "WorldPop/GP/100m/pop"
WORLDPOP_BAND = "population"
WORLDPOP_SCALE = 100          # native 100 m
WORLDPOP_MIN_YEAR = 2000
WORLDPOP_MAX_YEAR = 2020

JRC_IMAGE = "JRC/GSW1_4/GlobalSurfaceWater"
JRC_SCALE = 30                # native 30 m

SRTM_IMAGE = "USGS/SRTMGL1_003"
SRTM_SCALE = 30               # native 30 m


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_static_features_for_event(
    lat: float,
    lon: float,
    year: int,
    buffer_km: float = 25.0,
) -> dict:
    """
    Extract population and JRC surface water indicators for a single event.

    Makes two GEE calls (one per dataset). The WorldPop year is clamped to
    the available range 2000–2020.

    Returns a dict of indicator_name -> float | None.
    """
    null_result = {
        "pop_count_25km": None,
        "pop_density_km2": None,
        "jrc_occurrence_pct": None,
        "jrc_recurrence_pct": None,
        "terrain_slope_mean": None,
    }

    geom = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000)
    buffer_area_km2 = math.pi * buffer_km ** 2

    # --- WorldPop ---
    wp_year = max(WORLDPOP_MIN_YEAR, min(WORLDPOP_MAX_YEAR, year))
    try:
        wp_img = (
            ee.ImageCollection(WORLDPOP_COLLECTION)
            .filter(ee.Filter.eq("year", wp_year))
            .select(WORLDPOP_BAND)
            .mosaic()   # collection is per-country tiles; mosaic into global image
        )
        wp_val = wp_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geom,
            scale=WORLDPOP_SCALE,
            maxPixels=1e7,
        ).getInfo()
        pop = wp_val.get(WORLDPOP_BAND)
    except Exception as e:
        log.warning(f"WorldPop failed for ({lat},{lon}) {year}: {e}")
        pop = None

    if pop is not None:
        null_result["pop_count_25km"] = round(pop)
        null_result["pop_density_km2"] = round(pop / buffer_area_km2, 1)

    # --- JRC surface water ---
    try:
        jrc = ee.Image(JRC_IMAGE)
        jrc_val = jrc.select(["occurrence", "recurrence"]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=JRC_SCALE,
            maxPixels=1e7,
        ).getInfo()
        null_result["jrc_occurrence_pct"] = (
            round(jrc_val["occurrence"], 1) if jrc_val.get("occurrence") is not None else None
        )
        null_result["jrc_recurrence_pct"] = (
            round(jrc_val["recurrence"], 1) if jrc_val.get("recurrence") is not None else None
        )
    except Exception as e:
        log.warning(f"JRC failed for ({lat},{lon}): {e}")

    # --- SRTM slope ---
    try:
        slope = ee.Terrain.slope(ee.Image(SRTM_IMAGE))
        slope_val = slope.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=SRTM_SCALE,
            maxPixels=1e7,
        ).getInfo()
        null_result["terrain_slope_mean"] = (
            round(slope_val["slope"], 2) if slope_val.get("slope") is not None else None
        )
    except Exception as e:
        log.warning(f"SRTM slope failed for ({lat},{lon}): {e}")

    return null_result


# ---------------------------------------------------------------------------
# Batch enrichment
# ---------------------------------------------------------------------------

def enrich_events_with_static_features(
    events: list[dict],
    project: str,
    buffer_km: float = 25.0,
    output_path=None,
    resume: bool = True,
    max_workers: int = 1,
) -> list[dict]:
    """
    Enrich a list of event dicts with population and JRC surface water indicators.
    Events without lat/lon are passed through unchanged.

    Args:
        max_workers: parallel threads for GEE calls; 1 = sequential (default)
    """
    import json
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    ee.Initialize(project=project)

    processed_keys: set[str] = set()
    if resume and output_path and Path(output_path).exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("_static_processed") is not None:
                        processed_keys.add(_event_key(row))
                except json.JSONDecodeError:
                    pass
        log.info(f"Resuming: {len(processed_keys)} events already processed")

    to_process = [e for e in events if _event_key(e) not in processed_keys]
    log.info(f"{len(to_process)} events to process (workers={max_workers})")

    out_f = None
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out_f = open(output_path, "a", encoding="utf-8")

    write_lock = threading.Lock()
    n_done = n_skipped = n_errors = n_total = 0
    enriched: list[dict] = []

    def _process(event: dict) -> dict:
        nonlocal n_done, n_skipped, n_errors, n_total
        lat = event.get("lat")
        lon = event.get("lon")
        date_start = event.get("date_start")
        row = dict(event)

        if lat and lon and date_start:
            try:
                year = int(str(date_start)[:4])
                indicators = extract_static_features_for_event(
                    float(lat), float(lon), year, buffer_km
                )
                row.update(indicators)
                row["_static_processed"] = True
                outcome = "done"
            except Exception as e:
                log.warning(f"Event {_event_key(event)}: failed — {e}")
                row["_static_processed"] = False
                outcome = "errors"
        else:
            row["_static_processed"] = False
            outcome = "skipped"

        with write_lock:
            if outcome == "done":     n_done    += 1
            elif outcome == "errors": n_errors  += 1
            else:                     n_skipped += 1
            n_total += 1
            enriched.append(row)
            if out_f:
                out_f.write(json.dumps(row, default=str) + "\n")
                out_f.flush()
            if n_total % 100 == 0:
                log.info(
                    f"  {n_total}/{len(to_process)} | enriched={n_done} "
                    f"errors={n_errors} skipped={n_skipped}"
                )
        return row

    try:
        if max_workers <= 1:
            for event in to_process:
                _process(event)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_process, e): e for e in to_process}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        log.error(f"Unhandled worker error: {e}")
    finally:
        if out_f:
            out_f.close()

    log.info(f"Complete. enriched={n_done}, skipped={n_skipped}, errors={n_errors}")
    return enriched


def _event_key(event: dict) -> str:
    return f"{event.get('source','?')}|{event.get('source_id','?')}|{event.get('date_start','?')}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

import json
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

GEE_PROJECT = "gen-lang-client-0809810190"
BUFFER_KM   = 25.0
DATE_FROM   = "2017-01-01"
DATE_TO     = "2021-12-31"

ROOT        = Path(__file__).resolve().parents[3]
INPUT_PATH  = ROOT / "cache" / "floods" / "reference_floods_geocoded.jsonl"
OUTPUT_PATH = ROOT / "cache" / "floods" / "reference_floods_static.jsonl"


def main():
    if not INPUT_PATH.exists():
        log.error(f"Input not found: {INPUT_PATH}")
        sys.exit(1)

    log.info(f"Loading events from {INPUT_PATH}")
    all_events = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            all_events.append(json.loads(line))

    events = [
        e for e in all_events
        if DATE_FROM <= (e.get("date_start") or "")[:10] <= DATE_TO
    ]
    has_loc = sum(1 for e in events if e.get("lat") and e.get("lon"))
    log.info(f"Filtered to {DATE_FROM}-{DATE_TO}: {len(events)} events | with lat/lon: {has_loc}")

    enrich_events_with_static_features(
        events=events,
        project=GEE_PROJECT,
        buffer_km=BUFFER_KM,
        output_path=OUTPUT_PATH,
        resume=True,
        max_workers=15,
    )
    log.info(f"Output written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
