"""
run_backfill.py
===============
Backfills the three missing date ranges in one run:
  1. 2017-01-01 to 2017-12-31
  2. 2020-12-02 to 2020-12-31  (gap left by END_DATE="20201201" in original run)
  3. 2021-01-01 to 2021-12-31

Skips any month whose output file already exists in OUT_DIR so it is safe
to re-run if interrupted.

After all monthly windows are done, rebuilds the global label panel across
the full dataset (2017–2021) and re-runs cluster quality evaluation.
"""

from pathlib import Path
from datetime import datetime, timedelta
from calendar import monthrange
import pandas as pd

from group_articles_global import run_grouping, _daterange
from build_labels import run as run_labels
from build_labels_modelling import run as run_labels_modelling
from evaluate_clusters import run as run_eval

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE   = Path(__file__).resolve().parent
_ROOT   = _HERE.parent.parent.parent

OUT_DIR  = _HERE / "output"
EVAL_DIR = _HERE / "eval"
V1_DIR   = _HERE.parent / "grouped"
RAW_DIR  = _ROOT / "Builder_GDELT" / "results" / "daily"

# ── Settings ──────────────────────────────────────────────────────────────────
RESOLUTION   = 1.0
OVERLAP_DAYS = 14   # must match the value used in the original run

# Three ranges to backfill
BACKFILL_RANGES = [
    ("20170101", "20171231"),
    ("20201202", "20201231"),
    ("20210101", "20211231"),
]

# Full dataset span used for the final label panel + eval
FULL_START = "20170101"
FULL_END   = "20211231"


# ── Helpers ───────────────────────────────────────────────────────────────────
def _month_windows(start: str, end: str, overlap_days: int):
    cur = datetime.strptime(start, "%Y%m%d").replace(day=1)
    fin = datetime.strptime(end,   "%Y%m%d")
    while cur <= fin:
        _, last_day = monthrange(cur.year, cur.month)
        month_start = cur
        month_end   = cur.replace(day=last_day)
        if month_end > fin:
            month_end = fin
        win_start = month_start - timedelta(days=overlap_days)
        win_end   = month_end   + timedelta(days=overlap_days)
        yield (
            win_start.strftime("%Y%m%d"),
            win_end.strftime("%Y%m%d"),
            month_start.strftime("%Y%m%d"),
            month_end.strftime("%Y%m%d"),
        )
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            cur = cur.replace(month=cur.month + 1, day=1)


def _output_exists(lbl_start: str, lbl_end: str) -> bool:
    return (OUT_DIR / f"{lbl_start}_{lbl_end}_grouped.jsonl").exists()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Collect all windows across all backfill ranges, deduplicating by label
    all_windows: list[tuple[str, str, str, str]] = []
    seen_labels: set[str] = set()
    for range_start, range_end in BACKFILL_RANGES:
        for win in _month_windows(range_start, range_end, OVERLAP_DAYS):
            label = f"{win[2]}_{win[3]}"
            if label not in seen_labels:
                all_windows.append(win)
                seen_labels.add(label)

    total     = len(all_windows)
    skipped   = sum(1 for w in all_windows if _output_exists(w[2], w[3]))
    to_run    = total - skipped

    print(f"\nBackfill plan: {total} monthly windows across {len(BACKFILL_RANGES)} ranges")
    print(f"  Already done : {skipped}")
    print(f"  To cluster   : {to_run}\n")

    completed = 0
    for i, (win_start, win_end, lbl_start, lbl_end) in enumerate(all_windows):
        month_label = f"{lbl_start}_{lbl_end}"

        if _output_exists(lbl_start, lbl_end):
            print(f"[{i+1}/{total}] SKIP {month_label} (already exists)")
            continue

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] Clustering {lbl_start} to {lbl_end}"
              f"  (window: {win_start} to {win_end})")
        print(f"{'='*60}\n")

        input_paths = [
            RAW_DIR / d / "extractions.jsonl"
            for d in _daterange(win_start, win_end)
        ]

        run_grouping(
            input_paths, OUT_DIR, month_label,
            resolution = RESOLUTION,
            date_min   = pd.Timestamp(lbl_start),
            date_max   = pd.Timestamp(lbl_end),
        )
        completed += 1

    print(f"\nClustering complete. New months written: {completed}")

    # ── Rebuild global label panel across full 2017-2021 dataset ─────────────
    print(f"\n{'='*60}")
    print(f"Rebuilding global label panel ({FULL_START} to {FULL_END})")
    print(f"{'='*60}\n")
    run_labels(
        grouped_dir = OUT_DIR,
        raw_dir     = RAW_DIR,
        out_dir     = OUT_DIR,
        date_range  = (FULL_START, FULL_END),
        out_stem    = f"labels_{FULL_START}_{FULL_END}",
    )

    # ── Build modelling label panel (protest_7d, strike_7d etc.) ─────────────
    print(f"\n{'='*60}")
    print(f"Building modelling label panel ({FULL_START} to {FULL_END})")
    print(f"{'='*60}\n")
    run_labels_modelling(
        grouped_dir = OUT_DIR,
        raw_dir     = RAW_DIR,
        out_dir     = OUT_DIR,
        date_range  = (FULL_START, FULL_END),
    )

    # ── Re-evaluate cluster quality across full dataset ───────────────────────
    print(f"\n{'='*60}")
    print("Re-evaluating cluster quality (full dataset)")
    print(f"{'='*60}\n")
    run_eval(
        grouped_dir = OUT_DIR,
        compare_dir = V1_DIR if V1_DIR.exists() else None,
        out_dir     = EVAL_DIR,
        date_range  = (FULL_START, FULL_END),
    )

    print(f"\nDone.")
    print(f"  Grouped events : {OUT_DIR}/")
    print(f"  Labels         : {OUT_DIR}/labels_{FULL_START}_{FULL_END}.parquet")
    print(f"  Eval figures   : {EVAL_DIR}/")
