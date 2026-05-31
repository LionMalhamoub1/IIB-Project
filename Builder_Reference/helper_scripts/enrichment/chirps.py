"""
CHIRPS rainfall enrichment for flood reference events.

For each event with a valid lat/lon and date_start, extracts:
  chirps_3d_total_mm    total precipitation in the 3 days before event start
  chirps_7d_total_mm    total precipitation in the 7 days before event start
  chirps_14d_total_mm   total precipitation in the 14 days before event start
  chirps_30d_total_mm   total precipitation in the 30 days before event start
  chirps_peak_daily_mm  maximum single-day rainfall in the 30-day window
  chirps_7d_baseline_mm climatological 7-day mean (1991-2020, same calendar window)
  chirps_7d_anom_mm     7-day total minus climatological baseline
  chirps_7d_anom_pct    anomaly as % of baseline

Spatial: values are averaged over a circular buffer (default 25 km radius) around
the event point, using CHIRPS native resolution (~5.5 km).

Data source: UCSB-CHG/CHIRPS/DAILY on Google Earth Engine.
"""

import ee
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
CHIRPS_SCALE = 5566          # native pixel size ~5.5 km
BASELINE_START = "1991-01-01"
BASELINE_END = "2021-01-01"  # exclusive → covers 1991-2020
WINDOWS = [3, 7, 14, 30]


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_chirps_for_event(
    lat: float,
    lon: float,
    date_start: str,          # 'YYYY-MM-DD'
    buffer_km: float = 25.0,
) -> dict:
    """
    Extract CHIRPS indicators for a single event.

    Makes two GEE calls:
      1. getRegion() over 30-day window → daily time series → all window totals
      2. reduceRegion() of DOY-filtered baseline → climatological 7-day mean

    Returns a dict of indicator_name -> float | None.
    """
    dt = datetime.strptime(date_start[:10], "%Y-%m-%d")
    geom = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000)

    null_result = {
        **{f"chirps_{w}d_total_mm": None for w in WINDOWS},
        "chirps_peak_daily_mm": None,
        "chirps_7d_baseline_mm": None,
        "chirps_7d_anom_mm": None,
        "chirps_7d_anom_pct": None,
    }

    # --- Call 1: actual daily values over the 30-day window ---
    win_start = (dt - timedelta(days=30)).strftime("%Y-%m-%d")
    col = (
        ee.ImageCollection(CHIRPS_COLLECTION)
        .filterDate(win_start, date_start)
        .select("precipitation")
    )
    try:
        raw = col.getRegion(geom, scale=CHIRPS_SCALE).getInfo()
    except Exception as e:
        log.warning(f"getRegion failed for ({lat},{lon}) {date_start}: {e}")
        return null_result

    if not raw or len(raw) < 2:
        return null_result

    header = raw[0]
    try:
        precip_idx = header.index("precipitation")
        time_idx = header.index("time")
    except ValueError:
        return null_result

    # Average over pixels per day, then sort descending (most recent first)
    daily_px: dict[str, list[float]] = defaultdict(list)
    for row in raw[1:]:
        val = row[precip_idx]
        if val is not None:
            ts = datetime.utcfromtimestamp(row[time_idx] / 1000).strftime("%Y-%m-%d")
            daily_px[ts].append(val)

    if not daily_px:
        return null_result

    daily_mean: dict[str, float] = {
        d: sum(v) / len(v) for d, v in daily_px.items()
    }
    sorted_days = sorted(daily_mean.keys(), reverse=True)  # newest → oldest

    results: dict = {}
    for w in WINDOWS:
        recent = sorted_days[:w]
        results[f"chirps_{w}d_total_mm"] = round(sum(daily_mean[d] for d in recent), 2)

    results["chirps_peak_daily_mm"] = round(max(daily_mean.values()), 2)

    # --- Call 2: climatological baseline (1991-2020, same day-of-year window) ---
    doy_end = dt.timetuple().tm_yday
    doy_start = max(1, doy_end - 6)   # 7-day window (inclusive both ends)

    baseline_col = (
        ee.ImageCollection(CHIRPS_COLLECTION)
        .filterDate(BASELINE_START, BASELINE_END)
        .filter(ee.Filter.dayOfYear(doy_start, doy_end))
        .select("precipitation")
    )
    try:
        # mean() = mean daily precip across all matching DOYs in 30 years
        # multiply(7) = expected 7-day total under average conditions
        baseline_val = (
            baseline_col
            .mean()
            .multiply(7)
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom,
                scale=CHIRPS_SCALE,
                maxPixels=1e6,
            )
            .getInfo()
        )
    except Exception as e:
        log.warning(f"Baseline reduceRegion failed for ({lat},{lon}) {date_start}: {e}")
        baseline_val = {}

    baseline_7d = baseline_val.get("precipitation")
    results["chirps_7d_baseline_mm"] = (
        round(baseline_7d, 2) if baseline_7d is not None else None
    )

    actual_7d = results.get("chirps_7d_total_mm")
    if actual_7d is not None and baseline_7d is not None and baseline_7d > 0:
        results["chirps_7d_anom_mm"] = round(actual_7d - baseline_7d, 2)
        results["chirps_7d_anom_pct"] = round(
            (actual_7d - baseline_7d) / baseline_7d * 100, 1
        )
    else:
        results["chirps_7d_anom_mm"] = None
        results["chirps_7d_anom_pct"] = None

    return results


# ---------------------------------------------------------------------------
# Batch enrichment
# ---------------------------------------------------------------------------

def enrich_events_with_chirps(
    events: list[dict],
    project: str,
    buffer_km: float = 25.0,
    output_path=None,
    resume: bool = True,
    max_workers: int = 1,
) -> list[dict]:
    """
    Enrich a list of event dicts with CHIRPS indicators.

    Events without lat/lon or date_start are passed through unchanged.
    Results are written incrementally to output_path (JSONL) if provided,
    allowing the run to be interrupted and resumed.

    Args:
        events:      list of event dicts from the combined reference JSONL
        project:     GEE project ID (e.g. 'gen-lang-client-0809810190')
        buffer_km:   spatial buffer radius around event point
        output_path: Path to write enriched JSONL; appended to if resuming
        resume:      skip events already present in output_path
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
                    if row.get("_chirps_processed") is not None:
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
                indicators = extract_chirps_for_event(
                    float(lat), float(lon), str(date_start), buffer_km
                )
                row.update(indicators)
                row["_chirps_processed"] = True
                outcome = "done"
            except Exception as e:
                log.warning(f"Event {_event_key(event)}: failed — {e}")
                row["_chirps_processed"] = False
                outcome = "errors"
        else:
            row["_chirps_processed"] = False
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
    """Stable identifier for an event used for resume tracking."""
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
OUTPUT_PATH = ROOT / "cache" / "floods" / "reference_floods_chirps.jsonl"


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

    enrich_events_with_chirps(
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
