"""
Backfill stages 2 and 3 for any day in 2017-2021 that is missing output.

Stage 2 check: Builder_GDELT/results/daily/YYYYMMDD/extractions_consolidated.jsonl
Stage 3 check: Builder_GDELT/results/enriched_floods/YYYYMMDD/floods_enriched.jsonl
              (only checked for days that actually have flood extractions)

Run from the project root:
    python scripts/backfill_missing_days.py
"""

from datetime import date, timedelta
from pathlib import Path

from Builder_GDELT.pipelineRunner import run_pipeline_date_range as run_stage2
from Builder_GDELT.run_enrichment import _assemble_combined
from Builder_GDELT.helper_scripts.pipeline.enrich_floods_daily import enrich_day_floods

DAILY_RESULTS_DIR_ABS = Path(__file__).resolve().parents[1] / "Builder_GDELT" / "results" / "daily"
ENRICHED_ROOT_ABS     = Path(__file__).resolve().parents[1] / "Builder_GDELT" / "results" / "enriched_floods"
COMBINED_BY_TYPE_ABS  = Path(__file__).resolve().parents[1] / "Builder_GDELT" / "results" / "combined" / "by_type"

START = date(2017, 1, 1)
END   = date(2021, 12, 31)


def _has_stage2(yyyymmdd: str) -> bool:
    return (DAILY_RESULTS_DIR_ABS / yyyymmdd / "extractions_consolidated.jsonl").exists()


def _has_flood_extractions(yyyymmdd: str) -> bool:
    return (DAILY_RESULTS_DIR_ABS / yyyymmdd / "by_type" / "flood" / "extractions.jsonl").exists()


def _has_stage3(yyyymmdd: str) -> bool:
    return (ENRICHED_ROOT_ABS / yyyymmdd / "floods_enriched.jsonl").exists()


def main():
    needs_stage2: list[date] = []
    needs_stage3_only: list[date] = []

    print("Scanning 2017-2021 for incomplete days...")
    d = START
    while d <= END:
        yyyymmdd = d.strftime("%Y%m%d")
        if not _has_stage2(yyyymmdd):
            needs_stage2.append(d)
        elif _has_flood_extractions(yyyymmdd) and not _has_stage3(yyyymmdd):
            # Stage 2 exists but GEE enrichment never completed for this flood day
            needs_stage3_only.append(d)
        d += timedelta(days=1)

    if not needs_stage2 and not needs_stage3_only:
        print("All days have stage-2 and stage-3 output. Nothing to do.")
        return

    if needs_stage2:
        by_year: dict[int, int] = {}
        for d in needs_stage2:
            by_year[d.year] = by_year.get(d.year, 0) + 1
        print(f"\nMissing stage 2: {len(needs_stage2)} days")
        for yr, count in sorted(by_year.items()):
            print(f"  {yr}: {count} days")

    if needs_stage3_only:
        by_year2: dict[int, int] = {}
        for d in needs_stage3_only:
            by_year2[d.year] = by_year2.get(d.year, 0) + 1
        print(f"\nMissing stage 3 only: {len(needs_stage3_only)} days")
        for yr, count in sorted(by_year2.items()):
            print(f"  {yr}: {count} days")

    # Run stage 2 for all days that are missing it
    if needs_stage2:
        print("\nRunning Stage 2 for missing days...")
        for d in needs_stage2:
            run_stage2(d, d, postprocess=False)

    # Stage 3 covers days that just ran stage 2 plus days that already had it but missed GEE
    needs_stage3 = needs_stage2 + needs_stage3_only
    flood_days = [d for d in needs_stage3 if _has_flood_extractions(d.strftime("%Y%m%d"))]

    if flood_days:
        print(f"\nRunning Stage 3 (GEE enrichment) for {len(flood_days)} days with flood extractions...")
        ENRICHED_ROOT_ABS.mkdir(parents=True, exist_ok=True)
        COMBINED_BY_TYPE_ABS.mkdir(parents=True, exist_ok=True)
        for d in flood_days:
            yyyymmdd = d.strftime("%Y%m%d")
            enrich_day_floods(
                day_dir=DAILY_RESULTS_DIR_ABS / yyyymmdd,
                yyyymmdd=yyyymmdd,
                enriched_root=ENRICHED_ROOT_ABS,
                combined_by_type_root=COMBINED_BY_TYPE_ABS,
            )
        _assemble_combined()
    else:
        print("\nNo flood days to enrich.")

    print("\nBackfill complete.")


if __name__ == "__main__":
    main()
