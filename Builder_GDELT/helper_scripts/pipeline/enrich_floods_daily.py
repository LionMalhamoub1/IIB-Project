"""
enrich_floods_daily.py
======================
Per-day asynchronous GEE enrichment for GDELT flood events.

All five enrichment layers (CHIRPS, GPM, ERA5, static, SPI) are fetched
concurrently FOR EACH EVENT, collapsing the old 5-step sequential pipeline
into a single parallel pass.

Architecture
------------
    Outer pool : EVENT_WORKERS events processed simultaneously
    Inner pool : 5 threads per event, one per GEE layer
    Max concurrent GEE calls : EVENT_WORKERS * 5

    All five inner futures share a single GEE_LAYER_TIMEOUT deadline.
    Any layer still running after the timeout is abandoned (fields -> None)
    and its thread released to the background  -  the event always returns
    within ~GEE_LAYER_TIMEOUT seconds.

    urllib3 connection pool is patched to pool_maxsize=URLLIB3_POOL_SIZE
    before GEE initialises so persistent connections are reused across
    all concurrent requests without "pool is full" discards.

Called by
---------
    Builder_GDELT/run_enrichment.py  (Stage 3 of the master pipeline, once per day)
    Builder_GDELT/helper_scripts/pipeline/reenrich_failed.py  (Stage 4, retries only)

Inputs
------
    Builder_GDELT/results/daily/YYYYMMDD/by_type/flood/extractions.jsonl

Outputs
-------
    Builder_GDELT/results/enriched_floods/YYYYMMDD/floods_enriched.jsonl
    Builder_GDELT/results/combined/by_type/flood/floods_unconsolidated.jsonl  (appended)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # must be set before any other matplotlib/scipy import; prevents tkinter initialisation in worker threads

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait as futures_wait
from pathlib import Path

log = logging.getLogger(__name__)

GEE_PROJECT        = "gen-lang-client-0809810190"
BUFFER_KM          = 25.0
EVENT_WORKERS      = 8     # events in flight at once -> 8 * 5 = 40 max GEE calls
GEE_LAYER_TIMEOUT  = 120   # seconds for ALL 5 layers per event before giving up
HEARTBEAT_SECS     = 30    # progress log interval during enrichment
URLLIB3_POOL_SIZE  = 60    # persistent connection slots for earthengine.googleapis.com

# Fields used to detect a silently-failed GEE enrichment (all None despite _enriched=True)
HYDRO_FIELDS = [
    "chirps_7d_total_mm", "gpm_7d_total_mm", "era5_soil_moisture_day0",
    "pop_count_25km", "spi_30d",
]

_gee_lock = threading.Lock()


# GEE initialisation

def _ensure_gee() -> None:
    """
    Initialise GEE and patch the urllib3 connection pool.

    Called once per enrichment session (guarded by _gee_lock so concurrent
    calls are safe).  ee.Initialize() is intentionally NOT guarded by a
    'already done' flag  -  if a mid-run auth failure occurs, the next day's
    call re-initialises cleanly.
    """
    with _gee_lock:
        # Patch HTTPSConnectionPool so every new pool's internal queue is
        # sized to URLLIB3_POOL_SIZE instead of the default 10.
        # Only HTTPSConnectionPool is patched (GEE uses HTTPS); patching
        # HTTPConnectionPool too breaks the internal super().__init__ call
        # which passes many positional arguments.
        try:
            import urllib3.connectionpool as _cp
            _orig_https = _cp.HTTPSConnectionPool.__init__
            def _patched_https(self, host, port=None, **kw):
                kw["maxsize"] = max(kw.get("maxsize", 1), URLLIB3_POOL_SIZE)
                _orig_https(self, host, port=port, **kw)
            _cp.HTTPSConnectionPool.__init__ = _patched_https
        except Exception as exc:
            log.warning(f"Could not patch urllib3 pool constructor: {exc}")

        import ee
        try:
            ee.Initialize(project=GEE_PROJECT)
        except Exception as exc:
            log.error(f"GEE initialisation failed: {exc}")
            raise


# Helpers

def _event_key(event: dict) -> str:
    """Stable de-duplication key for a GDELT event (URL-scoped)."""
    return event.get("url") or event.get("source_id") or json.dumps(event, sort_keys=True)


def _resolve_date(event: dict) -> tuple[str, str] | tuple[None, None]:
    """
    Return (date_str, source_label) for the best available date, or (None, None).

    Priority: date_start -> event_date -> publish_date (flagged as fallback).
    """
    d = (event.get("date_start") or event.get("event_date") or "")[:10]
    if d:
        return d, "event_date"
    pub = (event.get("publish_date") or "")[:10]
    if pub:
        return pub, "publish_date_fallback"
    return None, None


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("quota", "rate limit", "resource exhausted", "429"))


def gee_enrichment_failed(event: dict) -> bool:
    """
    True if enrichment was attempted (_enriched=True) but GEE returned
    nothing useful  -  all hydro fields are None.  Used by reenrich_failed.py.
    """
    return bool(event.get("_enriched")) and all(
        event.get(f) is None for f in HYDRO_FIELDS
    )


# Per-event enrichment (all 5 layers in parallel, hard group timeout)

def enrich_single_event(event: dict) -> dict:
    """
    Run all five GEE extractors concurrently for one event.

    Public so reenrich_failed.py can call it directly on individual events.
    All five futures share GEE_LAYER_TIMEOUT seconds as a group; any that
    haven't returned by then are abandoned (fields -> None).
    Accepts an optional pre-resolved 'date_start' key on the event dict.
    """
    from Builder_Reference.helper_scripts.enrichment.chirps import extract_chirps_for_event
    from Builder_Reference.helper_scripts.enrichment.gpm   import extract_gpm_for_event
    from Builder_Reference.helper_scripts.enrichment.era5  import extract_era5_for_event
    from Builder_Reference.helper_scripts.enrichment.static_features import (
        extract_static_features_for_event,
    )
    from Builder_Reference.helper_scripts.enrichment.spi import extract_spi_for_event

    lat = event.get("lat")
    lon = event.get("lon")
    row = dict(event)

    date_start, date_source = _resolve_date(event)

    if not (lat and lon):
        row["_enriched"]           = False
        row["_enrich_skip_reason"] = "no_coords"
        return row

    if not date_start:
        row["_enriched"]           = False
        row["_enrich_skip_reason"] = "no_date"
        return row

    lat, lon = float(lat), float(lon)
    year = int(date_start[:4])
    row["_date_source"] = date_source

    extractors = {
        "chirps": lambda: extract_chirps_for_event(lat, lon, date_start, BUFFER_KM),
        "gpm":    lambda: extract_gpm_for_event(lat, lon, date_start, BUFFER_KM),
        "era5":   lambda: extract_era5_for_event(lat, lon, date_start, BUFFER_KM),
        "static": lambda: extract_static_features_for_event(lat, lon, year, BUFFER_KM),
        "spi":    lambda: extract_spi_for_event(lat, lon, date_start, BUFFER_KM),
    }

    executor = ThreadPoolExecutor(max_workers=5)
    fs = {name: executor.submit(fn) for name, fn in extractors.items()}
    future_to_name = {v: k for k, v in fs.items()}

    done, timed_out = futures_wait(fs.values(), timeout=GEE_LAYER_TIMEOUT)

    for future in done:
        name = future_to_name[future]
        try:
            row.update(future.result())
        except Exception as exc:
            if _is_quota_error(exc):
                log.error(
                    f"[{name}] GEE QUOTA/RATE-LIMIT for {_event_key(event)} — "
                    f"fields will be None. Consider reducing EVENT_WORKERS."
                )
            else:
                log.warning(f"[{name}] {_event_key(event)}: {exc}")

    for future in timed_out:
        name = future_to_name[future]
        log.error(
            f"[{name}] TIMEOUT ({GEE_LAYER_TIMEOUT}s) for {_event_key(event)} "
            f"— layer skipped, thread released to background"
        )
        future.cancel()

    executor.shutdown(wait=False)
    row["_enriched"] = True
    return row


# Heartbeat

def _start_heartbeat(
    yyyymmdd: str,
    counters: list[int],   # [n_enriched, n_gee_failed, n_skipped]
    total: int,
    lock: threading.Lock,
    stop: threading.Event,
) -> threading.Thread:
    def _beat():
        while not stop.wait(HEARTBEAT_SECS):
            with lock:
                n_ok, n_fail, n_skip = counters
            done = n_ok + n_fail + n_skip
            pct  = done / max(total, 1) * 100
            log.info(
                f"[{yyyymmdd}] HEARTBEAT {done}/{total} ({pct:.0f}%) — "
                f"enriched={n_ok} gee_failed={n_fail} skipped(no date/coords)={n_skip}"
            )

    t = threading.Thread(target=_beat, daemon=True, name=f"heartbeat-{yyyymmdd}")
    t.start()
    return t


# Public API

def enrich_day_floods(
    day_dir: Path,
    yyyymmdd: str,
    enriched_root: Path,
    combined_by_type_root: Path,
) -> Path | None:
    """
    Enrich all flood events from a single pipeline day.

    Steps
    -----
    1. Read   day_dir/by_type/flood/extractions.jsonl
    2. Append new events to combined_by_type_root/flood/floods_unconsolidated.jsonl
    3. Enrich via GEE -> enriched_root/YYYYMMDD/floods_enriched.jsonl
       (resume-safe: already-written events are skipped by URL key)

    Counters in logs distinguish three outcomes:
        enriched       -  GEE was called and returned at least some data
        gee_failed     -  GEE was called but all hydro fields came back None
        skipped        -  no coords or no resolvable date; GEE never called

    Returns the enriched output path, or None if no flood events exist.
    Raises nothing  -  errors are caught so the day loop continues.
    """
    flood_path = day_dir / "by_type" / "flood" / "extractions.jsonl"
    if not flood_path.exists():
        log.info(f"[{yyyymmdd}] No flood extractions — skipping enrichment")
        return None

    events: list[dict] = []
    with flood_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not events:
        return None

    log.info(f"[{yyyymmdd}] {len(events)} flood events loaded")

    # Step 1: append new events to combined unconsolidated file
    combined_flood = combined_by_type_root / "flood" / "floods_unconsolidated.jsonl"
    combined_flood.parent.mkdir(parents=True, exist_ok=True)

    existing_keys: set[str] = set()
    if combined_flood.exists():
        with combined_flood.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        existing_keys.add(_event_key(json.loads(line)))
                    except json.JSONDecodeError:
                        pass

    new_events = [e for e in events if _event_key(e) not in existing_keys]
    if new_events:
        with combined_flood.open("a", encoding="utf-8") as fh:
            for e in new_events:
                fh.write(json.dumps(e, default=str) + "\n")
        log.info(f"[{yyyymmdd}] Appended {len(new_events)} floods → {combined_flood}")

    # Step 2: GEE enrichment (resume-safe)
    out_dir  = enriched_root / yyyymmdd
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "floods_enriched.jsonl"

    done_keys: set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        done_keys.add(_event_key(json.loads(line)))
                    except json.JSONDecodeError:
                        pass

    to_process = [e for e in events if _event_key(e) not in done_keys]

    if not to_process:
        log.info(f"[{yyyymmdd}] All {len(events)} events already written — skipping GEE")
        return out_path

    # Split into three buckets so counters are meaningful
    no_coords   = [e for e in to_process if not (e.get("lat") and e.get("lon"))]
    has_coords  = [e for e in to_process if e.get("lat") and e.get("lon")]
    no_date     = [e for e in has_coords if not _resolve_date(e)[0]]
    enrichable  = [e for e in has_coords if _resolve_date(e)[0]]

    n_nominatim = sum(1 for e in enrichable if e.get("geo_source") == "nominatim_location_name")
    n_actiongeo = sum(1 for e in enrichable if e.get("geo_source") != "nominatim_location_name")
    log.info(
        f"[{yyyymmdd}] To process: {len(enrichable)} enrichable "
        f"(nominatim_coords={n_nominatim} actiongeo_coords={n_actiongeo}) | "
        f"{len(no_date)} no-date (skipped) | {len(no_coords)} no-coords (skipped)"
    )

    # Write unenrichable events immediately
    with out_path.open("a", encoding="utf-8") as fh:
        for e in no_coords:
            row = dict(e)
            row["_enriched"] = False
            row["_enrich_skip_reason"] = "no_coords"
            fh.write(json.dumps(row, default=str) + "\n")
        for e in no_date:
            row = dict(e)
            row["_enriched"] = False
            row["_enrich_skip_reason"] = "no_date"
            fh.write(json.dumps(row, default=str) + "\n")

    if not enrichable:
        return out_path

    _ensure_gee()

    # counters[0]=enriched, counters[1]=gee_failed, counters[2]=skipped(shouldn't happen here)
    counters   = [0, 0, 0]
    write_lock = threading.Lock()
    stop_beat  = threading.Event()
    _start_heartbeat(yyyymmdd, counters, len(enrichable), write_lock, stop_beat)

    t_start = time.monotonic()

    try:
        with out_path.open("a", encoding="utf-8") as out_fh:

            def _process(event: dict) -> None:
                result = enrich_single_event(event)
                with write_lock:
                    out_fh.write(json.dumps(result, default=str) + "\n")
                    out_fh.flush()
                    if not result.get("_enriched"):
                        counters[2] += 1   # skip reason set inside enrich_single_event
                    elif gee_enrichment_failed(result):
                        counters[1] += 1   # attempted but GEE returned all None
                    else:
                        counters[0] += 1   # at least some data returned
                    total_done = sum(counters)
                    if total_done % 5 == 0 or total_done == len(enrichable):
                        log.info(
                            f"[{yyyymmdd}] {total_done}/{len(enrichable)} — "
                            f"enriched={counters[0]} gee_failed={counters[1]} "
                            f"skipped={counters[2]}"
                        )

            with ThreadPoolExecutor(max_workers=EVENT_WORKERS) as executor:
                fs = {executor.submit(_process, e): e for e in enrichable}
                for future in as_completed(fs):
                    try:
                        future.result()
                    except Exception as exc:
                        log.error(f"[{yyyymmdd}] Unhandled worker error: {exc}")

    finally:
        stop_beat.set()

    elapsed = time.monotonic() - t_start
    log.info(
        f"[{yyyymmdd}] Enrichment complete in {elapsed:.0f}s — "
        f"enriched={counters[0]} gee_failed={counters[1]} skipped={counters[2]} "
        f"→ {out_path}"
    )
    return out_path
