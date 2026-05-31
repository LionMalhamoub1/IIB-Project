"""
ERA5-Land soil moisture enrichment for flood reference events.

For each event with a valid lat/lon and date_start, extracts:
  era5_soil_moisture_day0        volumetric soil water (top 7 cm) on event start date (0–1)
  era5_soil_moisture_7d_mean     mean soil moisture (top 7 cm) over the 7 days before event start
  era5_soil_moisture_30d_mean    mean soil moisture (top 7 cm) over the 30 days before event start
  era5_soil_moisture_deep_day0   volumetric soil water (7–28 cm) on event start date (0–1)
  era5_soil_moisture_deep_7d_mean mean soil moisture (7–28 cm) over the 7 days before event start
  era5_precip_7d_mm              ERA5-Land total precipitation over the 7-day window (mm)
  era5_runoff_7d_mm              ERA5-Land surface runoff over the 7-day window (mm)

Soil moisture is the key indicator for antecedent saturation: even modest rainfall
can trigger major flooding if the soil is already saturated from prior wet conditions.

Data source: ECMWF/ERA5_LAND/DAILY_AGGR on Google Earth Engine.
  - Band volumetric_soil_water_layer_1: top ~7 cm, dimensionless (m³/m³), 0–1
  - Band total_precipitation_sum: daily total (m), converted to mm
  - Band surface_runoff_sum: daily surface runoff (m), converted to mm
  - Resolution: 0.1 degrees (~11 km)
  - Coverage: 1950-present, global
"""

import ee
import logging
from collections import defaultdict
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

ERA5_COLLECTION = "ECMWF/ERA5_LAND/DAILY_AGGR"
ERA5_SCALE = 11132        # native pixel size ~0.1 degrees
ERA5_BANDS = [
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "total_precipitation_sum",
    "surface_runoff_sum",
]


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_era5_for_event(
    lat: float,
    lon: float,
    date_start: str,          # 'YYYY-MM-DD'
    buffer_km: float = 25.0,
) -> dict:
    """
    Extract ERA5-Land indicators for a single event.

    Makes one GEE call: getRegion() over a 30-day window for all three bands.
    All indicators derived client-side from the resulting time series.

    Returns a dict of indicator_name -> float | None.
    """
    null_result = {
        "era5_soil_moisture_day0": None,
        "era5_soil_moisture_7d_mean": None,
        "era5_soil_moisture_30d_mean": None,
        "era5_soil_moisture_deep_day0": None,
        "era5_soil_moisture_deep_7d_mean": None,
        "era5_precip_7d_mm": None,
        "era5_runoff_7d_mm": None,
    }

    dt = datetime.strptime(date_start[:10], "%Y-%m-%d")

    # ERA5-Land daily aggregates are available from 1950 onwards
    if dt.year < 1950:
        return null_result

    geom = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000)

    # Fetch 30 days up to and including the event start date
    win_start = (dt - timedelta(days=29)).strftime("%Y-%m-%d")
    win_end   = (dt + timedelta(days=1)).strftime("%Y-%m-%d")  # exclusive end

    col = (
        ee.ImageCollection(ERA5_COLLECTION)
        .filterDate(win_start, win_end)
        .select(ERA5_BANDS)
    )

    try:
        raw = col.getRegion(geom, scale=ERA5_SCALE).getInfo()
    except Exception as e:
        log.warning(f"getRegion failed for ({lat},{lon}) {date_start}: {e}")
        return null_result

    if not raw or len(raw) < 2:
        return null_result

    header = raw[0]
    try:
        sm1_idx    = header.index("volumetric_soil_water_layer_1")
        sm2_idx    = header.index("volumetric_soil_water_layer_2")
        precip_idx = header.index("total_precipitation_sum")
        runoff_idx = header.index("surface_runoff_sum")
        time_idx   = header.index("time")
    except ValueError:
        return null_result

    # Average over pixels per day
    daily_sm1: dict[str, list[float]] = defaultdict(list)
    daily_sm2: dict[str, list[float]] = defaultdict(list)
    daily_pr:  dict[str, list[float]] = defaultdict(list)
    daily_ro:  dict[str, list[float]] = defaultdict(list)

    for row in raw[1:]:
        day = datetime.utcfromtimestamp(row[time_idx] / 1000).strftime("%Y-%m-%d")
        if row[sm1_idx]    is not None: daily_sm1[day].append(row[sm1_idx])
        if row[sm2_idx]    is not None: daily_sm2[day].append(row[sm2_idx])
        if row[precip_idx] is not None: daily_pr[day].append(row[precip_idx])
        if row[runoff_idx] is not None: daily_ro[day].append(row[runoff_idx])

    if not daily_sm1:
        return null_result

    sm1_mean  = {d: sum(v) / len(v) for d, v in daily_sm1.items()}
    sm2_mean  = {d: sum(v) / len(v) for d, v in daily_sm2.items()}
    pr_mean   = {d: sum(v) / len(v) for d, v in daily_pr.items()}
    ro_mean   = {d: sum(v) / len(v) for d, v in daily_ro.items()}

    sorted_days = sorted(sm1_mean.keys(), reverse=True)  # newest first

    results: dict = {}

    # Surface soil moisture (0-7 cm)
    results["era5_soil_moisture_day0"] = (
        round(sm1_mean[sorted_days[0]], 4) if sorted_days else None
    )
    sm1_7d  = [sm1_mean[d] for d in sorted_days[:7]  if d in sm1_mean]
    sm1_30d = [sm1_mean[d] for d in sorted_days[:30] if d in sm1_mean]
    results["era5_soil_moisture_7d_mean"]  = round(sum(sm1_7d)  / len(sm1_7d),  4) if sm1_7d  else None
    results["era5_soil_moisture_30d_mean"] = round(sum(sm1_30d) / len(sm1_30d), 4) if sm1_30d else None

    # Deep soil moisture (7-28 cm)
    results["era5_soil_moisture_deep_day0"] = (
        round(sm2_mean[sorted_days[0]], 4) if sorted_days and sorted_days[0] in sm2_mean else None
    )
    sm2_7d = [sm2_mean[d] for d in sorted_days[:7] if d in sm2_mean]
    results["era5_soil_moisture_deep_7d_mean"] = round(sum(sm2_7d) / len(sm2_7d), 4) if sm2_7d else None

    # ERA5 precipitation total over 7 days (m -> mm)
    pr_7d = [pr_mean[d] for d in sorted_days[:7] if d in pr_mean]
    results["era5_precip_7d_mm"] = round(sum(pr_7d) * 1000, 2) if pr_7d else None

    # ERA5 surface runoff total over 7 days (m -> mm)
    ro_7d = [ro_mean[d] for d in sorted_days[:7] if d in ro_mean]
    results["era5_runoff_7d_mm"] = round(sum(ro_7d) * 1000, 2) if ro_7d else None

    return results


# ---------------------------------------------------------------------------
# Batch enrichment
# ---------------------------------------------------------------------------

def enrich_events_with_era5(
    events: list[dict],
    project: str,
    buffer_km: float = 25.0,
    output_path=None,
    resume: bool = True,
    max_workers: int = 1,
) -> list[dict]:
    """
    Enrich a list of event dicts with ERA5-Land indicators.

    Events without lat/lon or date_start are passed through unchanged.
    Results written incrementally to output_path (JSONL) if provided.

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
                    if row.get("_era5_processed") is not None:
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
                indicators = extract_era5_for_event(
                    float(lat), float(lon), str(date_start), buffer_km
                )
                row.update(indicators)
                row["_era5_processed"] = True
                outcome = "done"
            except Exception as e:
                log.warning(f"Event {_event_key(event)}: failed — {e}")
                row["_era5_processed"] = False
                outcome = "errors"
        else:
            row["_era5_processed"] = False
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
            if n_total % 50 == 0:
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
OUTPUT_PATH = ROOT / "cache" / "floods" / "reference_floods_era5.jsonl"


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

    enrich_events_with_era5(
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
