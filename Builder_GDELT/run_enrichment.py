"""
run_enrichment.py
=================
Stage 3 of the master pipeline: GEE enrichment for GDELT flood events.

This runs AFTER extraction and consolidation (Stage 2) are complete for
the full date range.  Decoupling enrichment from extraction mirrors the
Builder_Reference architecture, where dataset assembly and GEE enrichment
are separate steps.  The key benefit is that extraction (fast: LLM API
calls) no longer stalls waiting for GEE (slow: satellite data queries).

Inputs
------
    Builder_GDELT/results/daily/YYYYMMDD/by_type/flood/extractions.jsonl
    (written per-day by Stage 2 / pipelineRunner.py)

Outputs
-------
    Builder_GDELT/results/enriched_floods/YYYYMMDD/floods_enriched.jsonl
    Builder_GDELT/results/enriched_floods/all_floods_enriched.jsonl
        ↑ this combined file is the input to Builder_Matching

Downstream compatibility
------------------------
The output paths and file formats are identical to what the old inline
enrichment produced, so Builder_Matching, Builder_Combined, and any other
consumers are unaffected.

Resumability
------------
enrich_day_floods() is resume-safe  -  already-written events are skipped
by URL key, so the script can be interrupted and re-run freely.

Usage
-----
    # Enrich all days that have extractions:
    python -m Builder_GDELT.run_enrichment

    # Enrich a specific date range only:
    python -m Builder_GDELT.run_enrichment --date-from 2017-01-01 --date-to 2017-12-31

    # Called programmatically from master_pipeline.py:
    from Builder_GDELT.run_enrichment import run_enrichment
    run_enrichment(date_from=start_date, date_to=end_date)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT               = Path(__file__).resolve().parent.parent
DAILY_RESULTS_DIR  = ROOT / "Builder_GDELT" / "results" / "daily"
ENRICHED_ROOT      = ROOT / "Builder_GDELT" / "results" / "enriched_floods"
COMBINED_BY_TYPE_DIR = ROOT / "Builder_GDELT" / "results" / "combined" / "by_type"


def run_enrichment(
    date_from: date | None = None,
    date_to:   date | None = None,
) -> None:
    """
    Enrich all flood extractions within the given date range (or all days
    if no range is specified).

    Parameters
    ----------
    date_from : date, optional
        First day to enrich (inclusive).  If None, all available days are scanned.
    date_to   : date, optional
        Last day to enrich (inclusive).  If None, all available days are scanned.
    """
    from Builder_GDELT.helper_scripts.pipeline.enrich_floods_daily import enrich_day_floods

    ENRICHED_ROOT.mkdir(parents=True, exist_ok=True)
    COMBINED_BY_TYPE_DIR.mkdir(parents=True, exist_ok=True)

    # Collect candidate day directories
    if date_from is not None and date_to is not None:
        days: list[str] = []
        d = date_from
        while d <= date_to:
            days.append(d.strftime("%Y%m%d"))
            d += timedelta(days=1)
        day_dirs = [DAILY_RESULTS_DIR / yyyymmdd for yyyymmdd in days]
    else:
        day_dirs = sorted(DAILY_RESULTS_DIR.iterdir())

    # Filter to dirs that actually have flood extractions
    enrichable = [
        p for p in day_dirs
        if p.is_dir() and (p / "by_type" / "flood" / "extractions.jsonl").exists()
    ]

    log.info(
        f"Found {len(enrichable)} day(s) with flood extractions "
        f"(scanned {len(day_dirs)} directories)"
    )

    processed = 0
    for day_dir in enrichable:
        yyyymmdd = day_dir.name
        try:
            enrich_day_floods(
                day_dir=day_dir,
                yyyymmdd=yyyymmdd,
                enriched_root=ENRICHED_ROOT,
                combined_by_type_root=COMBINED_BY_TYPE_DIR,
            )
            processed += 1
        except Exception as exc:
            log.error(f"[{yyyymmdd}] Enrichment failed: {exc} — skipping")

    log.info(f"GEE enrichment complete: {processed}/{len(enrichable)} days processed")

    _assemble_combined()


def _assemble_combined() -> None:
    """
    Rebuild all_floods_enriched.jsonl from all per-day enriched files.

    This is the file consumed by Builder_Matching/matching/run_matching.py.
    Assembles ALL enriched days, not just the current run's date range, so
    the combined file always reflects the full historical archive.
    """
    all_enriched_out = ENRICHED_ROOT / "all_floods_enriched.jsonl"
    enriched_day_files = sorted(ENRICHED_ROOT.glob("*/floods_enriched.jsonl"))

    if not enriched_day_files:
        log.warning("No per-day enriched files found  -  combined file not written")
        return

    seen_keys: set[str] = set()
    written = 0
    with all_enriched_out.open("w", encoding="utf-8") as out_fh:
        for day_file in enriched_day_files:
            with day_file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = rec.get("url") or json.dumps(rec, sort_keys=True)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        written += 1

    log.info(f"Combined: {written} enriched flood records → {all_enriched_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: GEE enrichment for GDELT floods.")
    parser.add_argument("--date-from", type=str, default=None,
                        help="Start date YYYY-MM-DD (default: all available days)")
    parser.add_argument("--date-to",   type=str, default=None,
                        help="End date YYYY-MM-DD (default: all available days)")
    args = parser.parse_args()

    df = datetime.strptime(args.date_from, "%Y-%m-%d").date() if args.date_from else None
    dt = datetime.strptime(args.date_to,   "%Y-%m-%d").date() if args.date_to   else None

    run_enrichment(date_from=df, date_to=dt)
