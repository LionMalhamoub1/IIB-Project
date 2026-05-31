"""
SPI-30 (30-day Standardized Precipitation Index) enrichment for flood reference events.

SPI measures how anomalous a rainfall accumulation period is relative to the
long-term distribution at that location — more statistically rigorous than a
simple mean anomaly because it accounts for the full distribution shape, including
skewness (rainfall is not normally distributed).

SPI interpretation:
  >= +2.0   Extremely wet
  +1.5      Severely wet
  +1.0      Moderately wet
   0.0      Normal
  -1.0      Moderately dry
  -1.5      Severely dry
  <= -2.0   Extremely dry

Method:
  1. Fetch 40 years of CHIRPS daily data for the same calendar window (±DOY)
     at the event location via a single getRegion() call
  2. Sum per year → 40 historical 30-day totals
  3. Fit a gamma distribution (standard SPI methodology, WMO 2012)
  4. Convert observed 30-day total to SPI via the fitted CDF

Indicators extracted per event:
  spi_30d           SPI for the 30-day window preceding event start
  spi_30d_pct       percentile rank of the 30-day total (0–100)
"""

import ee
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta

import scipy.stats as stats

log = logging.getLogger(__name__)

CHIRPS_COLLECTION = "UCSB-CHG/CHIRPS/DAILY"
CHIRPS_SCALE = 5566
BASELINE_START = "1981-01-01"
BASELINE_END = "2021-01-01"     # exclusive → 1981–2020
SPI_MIN_YEARS = 10              # minimum years of data required to fit distribution


def extract_spi_for_event(
    lat: float,
    lon: float,
    date_start: str,            # 'YYYY-MM-DD'
    buffer_km: float = 25.0,
) -> dict:
    """
    Compute SPI-30 for a single event.

    Makes one GEE call: getRegion() over the 40-year baseline filtered
    to the same 30-day calendar window. All computation is done client-side.

    Returns a dict with spi_30d and spi_30d_pct, or None values on failure.
    """
    null_result = {"spi_30d": None, "spi_30d_pct": None}

    dt = datetime.strptime(date_start[:10], "%Y-%m-%d")
    point = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000)

    # Day-of-year window for the 30-day accumulation period
    doy_end = dt.timetuple().tm_yday
    doy_start = max(1, doy_end - 29)

    # Fetch all baseline images within that DOY window
    baseline_col = (
        ee.ImageCollection(CHIRPS_COLLECTION)
        .filterDate(BASELINE_START, BASELINE_END)
        .filter(ee.Filter.dayOfYear(doy_start, doy_end))
        .select("precipitation")
    )

    try:
        raw = baseline_col.getRegion(point, scale=CHIRPS_SCALE).getInfo()
    except Exception as e:
        log.warning(f"SPI getRegion failed for ({lat},{lon}) {date_start}: {e}")
        return null_result

    if not raw or len(raw) < 2:
        return null_result

    header = raw[0]
    try:
        precip_idx = header.index("precipitation")
        time_idx = header.index("time")
    except ValueError:
        return null_result

    # Average over pixels per timestep, then group by year → annual 30d total
    step_px: dict[tuple, list] = defaultdict(list)
    for row in raw[1:]:
        if row[precip_idx] is not None:
            ts = datetime.utcfromtimestamp(row[time_idx] / 1000)
            step_px[(ts.year, ts.timetuple().tm_yday)].append(row[precip_idx])

    # Mean per day-within-year
    daily_mean: dict[tuple, float] = {
        k: sum(v) / len(v) for k, v in step_px.items()
    }

    # Sum per year
    annual_totals: dict[int, float] = defaultdict(float)
    for (year, doy), val in daily_mean.items():
        annual_totals[year] += val

    if len(annual_totals) < SPI_MIN_YEARS:
        log.warning(f"SPI: only {len(annual_totals)} years of data for ({lat},{lon}), skipping")
        return null_result

    historical = sorted(annual_totals.values())

    # Observed 30-day total: same DOY window, event year
    obs_year = dt.year
    observed = annual_totals.get(obs_year)

    # If event year is outside baseline, fetch it separately
    if observed is None:
        obs_start = (dt - timedelta(days=29)).strftime("%Y-%m-%d")
        obs_end   = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
        obs_col = (
            ee.ImageCollection(CHIRPS_COLLECTION)
            .filterDate(obs_start, obs_end)
            .select("precipitation")
        )
        try:
            obs_raw = obs_col.getRegion(point, scale=CHIRPS_SCALE).getInfo()
            if obs_raw and len(obs_raw) > 1:
                obs_px: dict[str, list] = defaultdict(list)
                for row in obs_raw[1:]:
                    if row[precip_idx] is not None:
                        obs_px[row[time_idx]].append(row[precip_idx])
                observed = sum(
                    sum(v) / len(v) for v in obs_px.values()
                )
        except Exception as e:
            log.warning(f"SPI observed fetch failed: {e}")
            return null_result

    if observed is None:
        return null_result

    # Fit gamma distribution to historical totals (add small offset for zeros)
    historical_arr = [max(x, 0.01) for x in historical]
    try:
        shape, loc, scale = stats.gamma.fit(historical_arr, floc=0)
        cdf = stats.gamma.cdf(max(observed, 0.01), shape, loc=loc, scale=scale)
    except Exception as e:
        log.warning(f"SPI gamma fit failed: {e}")
        return null_result

    # Clamp CDF to avoid infinite SPI from perfect 0 or 1
    cdf = max(1e-6, min(1 - 1e-6, cdf))
    spi = stats.norm.ppf(cdf)

    return {
        "spi_30d": round(float(spi), 3),
        "spi_30d_pct": round(float(cdf * 100), 1),
    }


# ---------------------------------------------------------------------------
# Batch enrichment
# ---------------------------------------------------------------------------

def enrich_events_with_spi(
    events: list[dict],
    project: str,
    buffer_km: float = 25.0,
    output_path=None,
    resume: bool = True,
    max_workers: int = 1,
) -> list[dict]:
    """
    Enrich a list of event dicts with SPI-30. Events without lat/lon are
    passed through unchanged. Results written incrementally to output_path.

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
                    if row.get("_spi_processed") is not None:
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
                indicators = extract_spi_for_event(
                    float(lat), float(lon), str(date_start), buffer_km
                )
                row.update(indicators)
                row["_spi_processed"] = True
                outcome = "done"
            except Exception as e:
                log.warning(f"Event {_event_key(event)}: failed — {e}")
                row["_spi_processed"] = False
                outcome = "errors"
        else:
            row["_spi_processed"] = False
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
OUTPUT_PATH = ROOT / "cache" / "floods" / "reference_floods_spi.jsonl"


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

    enrich_events_with_spi(
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
