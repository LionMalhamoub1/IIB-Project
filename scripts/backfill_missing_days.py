"""
Backfill stages 2, 3, and 4 for any day in 2017-2021 that is missing
Builder_GDELT/results/daily/{YYYYMMDD}/extractions_consolidated.jsonl.

Run from the project root:
    python backfill_missing_days.py
"""

from datetime import date, timedelta
from pathlib import Path

from Builder_GDELT.pipelineRunner import run_pipeline_date_range as run_stage2
from Builder_GDELT.run_enrichment import _assemble_combined, ENRICHED_ROOT as _ENRICHED_ROOT
from Builder_GDELT.helper_scripts.pipeline.enrich_floods_daily import enrich_day_floods
from Builder_GDELT.helper_scripts.pipeline.reenrich_failed import (
    scan_all_days, reenrich_day, ENRICHED_ROOT,
)

DAILY_RESULTS_DIR_ABS = Path(__file__).resolve().parents[1] / "Builder_GDELT" / "results" / "daily"
ENRICHED_ROOT_ABS     = Path(__file__).resolve().parents[1] / "Builder_GDELT" / "results" / "enriched_floods"
COMBINED_BY_TYPE_ABS  = Path(__file__).resolve().parents[1] / "Builder_GDELT" / "results" / "combined" / "by_type"

DAILY_RESULTS_DIR = Path("Builder_GDELT/results/daily")
START = date(2017, 1, 1)
END   = date(2021, 12, 31)


def find_missing_days(start: date, end: date) -> list[date]:
    missing = []
    d = start
    while d <= end:
        yyyymmdd = d.strftime("%Y%m%d")
        consolidated = DAILY_RESULTS_DIR / yyyymmdd / "extractions_consolidated.jsonl"
        if not consolidated.exists():
            missing.append(d)
        d += timedelta(days=1)
    return missing


def main():
    print("Scanning for missing days...")
    missing = find_missing_days(START, END)

    if not missing:
        print("All days already have stage-2 output. Nothing to do.")
        return

    print(f"Found {len(missing)} missing days.")
    by_year: dict[int, int] = {}
    for d in missing:
        by_year[d.year] = by_year.get(d.year, 0) + 1
    for yr, count in sorted(by_year.items()):
        print(f"  {yr}: {count} missing days")

    # Stage 2 — run day-by-day so already-complete days are never re-touched
    print("\nRunning Stage 2 for missing days...")
    for d in missing:
        run_stage2(d, d, postprocess=False)

    # Stage 3 — GEE enrichment for missing days only (not the full span)
    print("\nRunning Stage 3 (GEE enrichment) for missing days only...")
    ENRICHED_ROOT_ABS.mkdir(parents=True, exist_ok=True)
    COMBINED_BY_TYPE_ABS.mkdir(parents=True, exist_ok=True)
    for d in missing:
        yyyymmdd = d.strftime("%Y%m%d")
        day_dir = DAILY_RESULTS_DIR_ABS / yyyymmdd
        if not day_dir.is_dir():
            continue
        if not (day_dir / "by_type" / "flood" / "extractions.jsonl").exists():
            continue
        enrich_day_floods(
            day_dir=day_dir,
            yyyymmdd=yyyymmdd,
            enriched_root=ENRICHED_ROOT_ABS,
            combined_by_type_root=COMBINED_BY_TYPE_ABS,
        )
    _assemble_combined()  # rebuild combined file once at the end

    # Stage