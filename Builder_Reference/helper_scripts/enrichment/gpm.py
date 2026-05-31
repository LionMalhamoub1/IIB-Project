"""
GPM IMERG rainfall intensity enrichment for flood reference events.

For each event with a valid lat/lon and date_start, extracts:
  gpm_1d_total_mm     total precipitation in the 1 day before event start
  gpm_3d_total_mm     total precipitation in the 3 days before event start
  gpm_7d_total_mm     total precipitation in the 7 days before event start
  gpm_peak_daily_mm   maximum single-day rainfall in the 7-day window
  gpm_peak_3h_mm      maximum 3-hour accumulation in the 7-day window

The key advantage over CHIRPS is the 30-minute temporal resolution, which
captures flash-flood-triggering intensity that daily totals miss.

Data source: NASA/GPM_L3/IMERG_V06 on Google Earth Engine.
  - Band: precipitationCal (mm/hr, calibrated with surface gauges)
  - Resolution: 0.1 degrees (~11 km), 30-minute timesteps
  - Coverage: 2000-present, 60°S–60°N

Spatial: values averaged over a circular buffer (default 25 km radius) around
the event point.
"""

import ee
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

log = logging.getLogger(__name__)

GPM_COLLECTION = "NASA/GPM_L3/IMERG_V07"
GPM_BAND = "precipitation"
GPM_SCALE = 11132         # native pixel size ~0.1 degrees (~11 km)
GPM_TIMESTEP_HR = 0.5    # each image covers 30 minutes
GPM_COVERAGE_START = "2000-06-01"   # IMERG V06 starts mid-2000


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_gpm_for_event(
    lat: float,
    lon: float,
    date_start: str,          # 'YYYY-MM-DD'
    buffer_km: float = 25.0,
) -> dict:
    """
    Extract GPM IMERG indicators for a single event.

    Makes one GEE call: getRegion() over a 7-day window at 30-min resolution.
    All indicators are derived client-side from the resulting time series.

    Returns a dict of indicator_name -> float | None.
    """
    null_result = {
        "gpm_1d_total_mm": None,
        "gpm_3d_total_mm": None,
        "gpm_7d_total_mm": None,
        "gpm_peak_daily_mm": None,
        "gpm_peak_3h_mm": None,
    }

    dt = datetime.strptime(date_start[:10], "%Y-%m-%d")

    # GPM only available from mid-2000
    if dt < datetime.strptime(GPM_COVERAGE_START, "%Y-%m-%d"):
        return null_result

    # Coverage is 60S-60N; outside that range return nulls
    if not (-60 <= lat <= 60):
        return null_result

    geom = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000)

    win_start = (dt - timedelta(days=7)).strftime("%Y-%m-%d")
    col = (
        ee.ImageCollection(GPM_COLLECTION)
        .filterDate(win_start, date_start)
        .select(GPM_BAND)
    )

    # getRegion returns one row per (pixel, timestep)
    try:
        raw = col.getRegion(geom, scale=GPM_SCALE).getInfo()
    except Exception as e:
        log.warning(f"getRegion failed for ({lat},{lon}) {date_start}: {e}")
        return null_result

    if not raw or len(raw) < 2:
        return null_result

    header = raw[0]
    try:
        precip_idx = header.index(GPM_BAND)
        time_idx = header.index("time")
    except ValueError:
        return null_result

    # --- Build time series: list of (datetime, mm_per_30min) ---
    # precipitationCal is in mm/hr; multiply by 0.5 to get mm per 30-min step
    # Average over pixels at each timestep
    step_px: dict[int, list[float]] = defaultdict(list)  # time_ms -> [pixel values]
    for row in raw[1:]:
        val = row[precip_idx]
        if val is not None:
            step_px[row[time_idx]].append(val)

    if not step_px:
        return null_result

    # time_ms -> mean mm per 30-min step
    step_mm: dict[int, float] = {
        t: (sum(v) / len(v)) * GPM_TIMESTEP_HR
        for t, v in step_px.items()
    }
    sorted_times = sorted(step_mm.keys(), reverse=True)  # newest first

    # --- Window totals ---
    steps_per_day = int(24 / GPM_TIMESTEP_HR)     # 48
    results: dict = {}
    for days, key in [(1, "gpm_1d_total_mm"), (3, "gpm_3d_total_mm"), (7, "gpm_7d_total_mm")]:
        n_steps = days * steps_per_day
        recent = sorted_times[:n_steps]
        results[key] = round(sum(step_mm[t] for t in recent), 2)

    # --- Peak daily total ---
    daily_mm: dict[str, float] = defaultdict(float)
    for t_ms, mm in step_mm.items():
        day = datetime.utcfromtimestamp(t_ms / 1000).strftime("%Y-%m-%d")
        daily_mm[day] += mm
    results["gpm_peak_daily_mm"] = round(max(daily_mm.values()), 2) if daily_mm else None

    # --- Peak 3-hour accumulation ---
    # Slide a 6-step (3h) window over the sorted time series
    steps_3h = int(3 / GPM_TIMESTEP_HR)   # 6 steps
    all_times_asc = sorted(step_mm.keys())
    peak_3h = 0.0
    for i in range(len(all_times_asc) - steps_3h + 1):
        window_sum = sum(step_mm[all_times_asc[j]] for j in range(i, i + steps_3h))
        if window_sum > peak_3h:
            peak_3h = window_sum
    results["gpm_peak_3h_mm"] = round(peak_3h, 2)

    return results


# ---------------------------------------------------------------------------
# Batch enrichment
# ---------------------------------------------------------------------------

def enrich_events_with_gpm(
    events: list[dict],
    project: str,
    buffer_km: float = 25.0,
    output_path=None,
    resume: bool = True,
    max_workers: int = 1,
) -> list[dict]:
    """
    Enrich a list of event dicts with GPM IMERG indicators.

    Events without lat/lon, before GPM coverage (mid-2000), or outside 60S-60N
    are passed through unchanged. Results written incrementally to output_path.

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
                    if row.get("_gpm_processed") is not None:
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
                indicators = extract_gpm_for_event(
                    float(lat), float(lon), str(date_start), buffer_km
                )
                row.update(indicators)
                row["_gpm_processed"] = True
                outcome = "done"
            except Exception as e:
                log.warning(f"Event {_event_key(event)}: failed — {e}")
                row["_gpm_processed"] = False
                outcome = "errors"
        else:
            row["_gpm_processed"] = False
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
OUTPUT_PATH = ROOT / "cache" / "floods" / "reference_floods_gpm.jsonl"


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

    enrich_events_with_gpm(
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
