"""
reenrich_failed.py
==================
Standalone re-enrichment pass for GDELT flood events that previously failed
or were skipped during the main pipeline run.

Targets two categories of fixable failures found in floods_enriched.jsonl:

  1. gee_failed  — _enriched=True but all hydro fields are None.
                   GEE was reached but returned nothing (connection pool
                   exhaustion, transient server error, etc.).  Retrying
                   with the fixed pool size usually recovers these.

  2. no_date     — _enriched=False, _enrich_skip_reason="no_date".
                   Event has coords but no event_date or publish_date.
                   Re-enriched using the YYYYMMDD folder name as a
                   date-of-last-resort (flagged as _date_source="yyyymmdd_fallback").

  no_coords events (_enrich_skip_reason="no_coords") are NOT retried —
  there is no coordinate data to enrich with.

For each retried event the enriched file is rewritten in-place with the
updated record replacing the old one (matched by URL key).  The main
pipeline is not involved and can run concurrently.

Usage
-----
    # Re-enrich all days that have any fixable failures
    python -m Builder_GDELT.helper_scripts.pipeline.reenrich_failed

    # Target specific days only
    python -m Builder_GDELT.helper_scripts.pipeline.reenrich_failed --days 20180101 20180105

    # Dry-run: report failures without re-enriching
    python -m Builder_GDELT.helper_scripts.pipeline.reenrich_failed --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

ROOT          = Path(__file__).resolve().parents[3]   # IIB-Project root
ENRICHED_ROOT = ROOT / "Builder_GDELT" / "results" / "enriched_floods"

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Builder_GDELT.helper_scripts.pipeline.enrich_floods_daily import (
    _ensure_gee,
    _event_key,
    enrich_single_event,
    gee_enrichment_failed,
    EVENT_WORKERS,
)


# ---------------------------------------------------------------------------
# Scan and categorise
# ---------------------------------------------------------------------------

def _scan_day(day_path: Path) -> dict:
    """
    Read floods_enriched.jsonl for one day and return a summary dict:
        {
            "path": Path,
            "yyyymmdd": str,
            "all": [dict, ...],          # every event in the file
            "gee_failed": [dict, ...],   # enriched=True but all hydro None
            "no_date": [dict, ...],      # skip_reason=no_date
        }
    """
    result = {"path": day_path, "yyyymmdd": day_path.name,
              "all": [], "gee_failed": [], "no_date": []}

    jsonl = day_path / "floods_enriched.jsonl"
    result["path"] = jsonl
    if not jsonl.exists():
        return result

    with jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            result["all"].append(e)
            if gee_enrichment_failed(e):
                result["gee_failed"].append(e)
            elif e.get("_enrich_skip_reason") == "no_date":
                result["no_date"].append(e)

    return result


def scan_all_days(enriched_root: Path, days: list[str] | None = None) -> list[dict]:
    """
    Scan enriched_root for all (or selected) day folders.
    Returns a list of _scan_day results that have at least one fixable event.
    """
    if days:
        candidates = [enriched_root / d for d in days if (enriched_root / d).exists()]
    else:
        candidates = sorted(enriched_root.iterdir())

    results = []
    for folder in candidates:
        if not folder.is_dir():
            continue
        summary = _scan_day(folder)
        if summary["gee_failed"] or summary["no_date"]:
            results.append(summary)

    return results


# ---------------------------------------------------------------------------
# Re-enrich a single day's failures
# ---------------------------------------------------------------------------

def _inject_yyyymmdd_date(event: dict, yyyymmdd: str) -> dict:
    """Return a copy of event with date_start set from the folder name."""
    e = dict(event)
    e["date_start"]   = f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"
    e["_date_source"] = "yyyymmdd_fallback"
    # Clear the skip reason so enrich_single_event re-evaluates
    e.pop("_enrich_skip_reason", None)
    e["_enriched"] = False   # will be set properly after re-enrichment
    return e


def reenrich_day(summary: dict) -> tuple[int, int]:
    """
    Re-enrich fixable failures for one day.  Rewrites the enriched JSONL
    in-place with updated records.

    Returns (n_recovered, n_still_failed).
    """
    yyyymmdd   = summary["yyyymmdd"]
    jsonl_path = summary["path"]

    # Build the list of events to retry and map old key → new injected event
    retry_map: dict[str, dict] = {}

    for e in summary["gee_failed"]:
        retry_map[_event_key(e)] = dict(e)   # retry as-is, GEE may work now

    for e in summary["no_date"]:
        injected = _inject_yyyymmdd_date(e, yyyymmdd)
        retry_map[_event_key(e)] = injected   # retry with injected date

    log.info(
        f"[{yyyymmdd}] Retrying {len(retry_map)} events "
        f"({len(summary['gee_failed'])} gee_failed, {len(summary['no_date'])} no_date)"
    )

    _ensure_gee()

    # Run enrichment on all retry candidates concurrently
    results: dict[str, dict] = {}   # key → enriched result
    counters   = [0, 0]             # [n_recovered, n_still_failed]
    write_lock = threading.Lock()
    stop_beat  = threading.Event()
    total      = len(retry_map)

    def _beat():
        while not stop_beat.wait(30):
            with write_lock:
                done = counters[0] + counters[1]
            pct = done / max(total, 1) * 100
            log.info(
                f"[{yyyymmdd}] HEARTBEAT {done}/{total} ({pct:.0f}%) — "
                f"recovered={counters[0]} still_failed={counters[1]}"
            )

    beat_thread = threading.Thread(target=_beat, daemon=True, name=f"heartbeat-reenrich-{yyyymmdd}")
    beat_thread.start()

    def _process(key: str, event: dict) -> None:
        result = enrich_single_event(event)
        with write_lock:
            results[key] = result
            if not gee_enrichment_failed(result) and result.get("_enriched"):
                counters[0] += 1
            else:
                counters[1] += 1

    try:
        with ThreadPoolExecutor(max_workers=EVENT_WORKERS) as executor:
            fs = {executor.submit(_process, k, e): k for k, e in retry_map.items()}
            for future in as_completed(fs):
                try:
                    future.result()
                except Exception as exc:
                    log.error(f"[{yyyymmdd}] Worker error: {exc}")
    finally:
        stop_beat.set()

    # Rewrite the JSONL: replace old records with new results for retried events
    updated: list[dict] = []
    for e in summary["all"]:
        key = _event_key(e)
        updated.append(results.get(key, e))   # use new result if available, else keep old

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for e in updated:
            fh.write(json.dumps(e, default=str) + "\n")

    n_recovered    = sum(1 for k, r in results.items() if not gee_enrichment_failed(r) and r.get("_enriched"))
    n_still_failed = len(results) - n_recovered
    log.info(
        f"[{yyyymmdd}] Done — recovered={n_recovered} still_failed={n_still_failed}"
    )
    return n_recovered, n_still_failed


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Re-enrich failed/skipped flood events without re-running the pipeline."
    )
    parser.add_argument(
        "--days", nargs="+", metavar="YYYYMMDD",
        help="Only process these specific days (default: all days with failures)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report fixable failures without re-enriching."
    )
    args = parser.parse_args()

    if not ENRICHED_ROOT.exists():
        log.error(f"Enriched results root not found: {ENRICHED_ROOT}")
        return

    log.info(f"Scanning {ENRICHED_ROOT} ...")
    summaries = scan_all_days(ENRICHED_ROOT, days=args.days)

    if not summaries:
        log.info("No fixable failures found.")
        return

    total_gee  = sum(len(s["gee_failed"]) for s in summaries)
    total_date = sum(len(s["no_date"])    for s in summaries)
    log.info(
        f"Found {len(summaries)} days with fixable failures: "
        f"{total_gee} gee_failed, {total_date} no_date across all days"
    )

    if args.dry_run:
        for s in summaries:
            log.info(
                f"  {s['yyyymmdd']}: gee_failed={len(s['gee_failed'])} "
                f"no_date={len(s['no_date'])}"
            )
        return

    total_recovered = total_still_failed = 0
    for s in summaries:
        recovered, still_failed = reenrich_day(s)
        total_recovered    += recovered
        total_still_failed += still_failed

    log.info(
        f"\nRe-enrichment complete — "
        f"recovered={total_recovered} still_failed={total_still_failed}"
    )


if __name__ == "__main__":
    main()
