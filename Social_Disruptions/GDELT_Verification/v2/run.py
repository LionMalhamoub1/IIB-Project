"""
run.py
======
Set the date range and run mode below, then run this file.

Two modes
---------
FULL_RUN = False  (default)
    Processes only START_DATE to END_DATE as a single batch.
    Good for testing a short range (e.g. one month).

FULL_RUN = True
    Processes the full date range month by month, with a 2-week overlap
    buffer between months so events spanning a month boundary cluster
    correctly.  Then runs a global movement pass and builds the final
    label panel across all months.
    Use this when running across the full dataset.

Output lands in:
    v2/output/   - per-month grouped events + final labels
    v2/eval/     - quality figures vs v1
"""

from pathlib import Path
from datetime import datetime, timedelta
from calendar import monthrange
import pandas as pd

from group_articles_global import run_grouping, _daterange
from build_labels import run as run_labels
from evaluate_clusters import run as run_eval

# -----------------------------------------------------------------------
# Settings — edit these
# -----------------------------------------------------------------------
START_DATE = "20200501"
END_DATE   = "20201201"

FULL_RUN   = True   # False = single batch, True = month-by-month

# Louvain resolution: higher = smaller, tighter clusters.
# 1.0 is standard. Try 1.5 if clusters are still too large.
RESOLUTION = 1.0

# Overlap buffer added on each side of a monthly window (days).
# Ensures events spanning a month boundary get clustered together.
OVERLAP_DAYS = 14
# -----------------------------------------------------------------------

_HERE  = Path(__file__).resolve().parent
_VER   = _HERE.parent
_SD    = _VER.parent
_ROOT  = _SD.parent

OUT_DIR  = _HERE / "output"
EVAL_DIR = _HERE / "eval"
V1_DIR   = _VER / "grouped"
RAW_DIR  = _ROOT / "Builder_GDELT" / "results" / "daily"


def _month_windows(start: str, end: str, overlap_days: int):
    """Yield (window_start, window_end, label_start, label_end) for each month.

    window_start/end: the buffered range passed to the clusterer (includes
                      overlap so cross-month events cluster correctly).
    label_start/end:  the core month range used to name the output file and
                      filter labels — no overlap, so each event only appears
                      in one month's output.
    """
    cur = datetime.strptime(start, "%Y%m%d").replace(day=1)
    fin = datetime.strptime(end,   "%Y%m%d")

    while cur <= fin:
        # Core month boundaries
        _, last_day = monthrange(cur.year, cur.month)
        month_start = cur
        month_end   = cur.replace(day=last_day)
        if month_end > fin:
            month_end = fin

        # Buffered window for clustering
        win_start = month_start - timedelta(days=overlap_days)
        win_end   = month_end   + timedelta(days=overlap_days)

        yield (
            win_start.strftime("%Y%m%d"),
            win_end.strftime("%Y%m%d"),
            month_start.strftime("%Y%m%d"),
            month_end.strftime("%Y%m%d"),
        )

        # Advance to first day of next month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1, day=1)
        else:
            cur = cur.replace(month=cur.month + 1, day=1)


if __name__ == "__main__":

    if not FULL_RUN:
        # ----------------------------------------------------------------
        # Single batch mode
        # ----------------------------------------------------------------
        range_label = f"{START_DATE}_{END_DATE}"
        date_range  = (START_DATE, END_DATE)
        input_paths = [
            RAW_DIR / d / "extractions.jsonl"
            for d in _daterange(START_DATE, END_DATE)
        ]

        print(f"\n{'='*60}")
        print(f"STEP 1 - Clustering  ({START_DATE} to {END_DATE})")
        print(f"{'='*60}\n")
        run_grouping(
            input_paths, OUT_DIR, range_label,
            resolution = RESOLUTION,
            date_min   = pd.Timestamp(START_DATE),
            date_max   = pd.Timestamp(END_DATE),
        )

        print(f"\n{'='*60}")
        print("STEP 2 - Building label panel")
        print(f"{'='*60}\n")
        run_labels(
            grouped_dir = OUT_DIR,
            raw_dir     = RAW_DIR,
            out_dir     = OUT_DIR,
            date_range  = date_range,
            out_stem    = f"labels_{range_label}",
        )

        print(f"\n{'='*60}")
        print("STEP 3 - Evaluating cluster quality")
        print(f"{'='*60}\n")
        run_eval(
            grouped_dir = OUT_DIR,
            compare_dir = V1_DIR if V1_DIR.exists() else None,
            out_dir     = EVAL_DIR,
            date_range  = date_range,
        )

    else:
        # ----------------------------------------------------------------
        # Full run: month-by-month with overlap buffer
        # ----------------------------------------------------------------
        windows = list(_month_windows(START_DATE, END_DATE, OVERLAP_DAYS))
        print(f"\nFull run: {len(windows)} monthly windows  ({START_DATE} to {END_DATE})")
        print(f"Overlap buffer: {OVERLAP_DAYS} days on each side\n")

        completed = []

        for i, (win_start, win_end, lbl_start, lbl_end) in enumerate(windows):
            month_label = f"{lbl_start}_{lbl_end}"
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(windows)}] Clustering {lbl_start} to {lbl_end}"
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
            completed.append(month_label)

        # ----------------------------------------------------------------
        # Global label panel across all months
        # ----------------------------------------------------------------
        print(f"\n{'='*60}")
        print("Building global label panel across all months")
        print(f"{'='*60}\n")
        run_labels(
            grouped_dir = OUT_DIR,
            raw_dir     = RAW_DIR,
            out_dir     = OUT_DIR,
            date_range  = (START_DATE, END_DATE),
            out_stem    = f"labels_{START_DATE}_{END_DATE}",
        )

        # ----------------------------------------------------------------
        # Evaluation
        # ----------------------------------------------------------------
        print(f"\n{'='*60}")
        print("Evaluating cluster quality")
        print(f"{'='*60}\n")
        run_eval(
            grouped_dir = OUT_DIR,
            compare_dir = V1_DIR if V1_DIR.exists() else None,
            out_dir     = EVAL_DIR,
            date_range  = (START_DATE, END_DATE),
        )

    print(f"\nDone.")
    print(f"  Grouped events : {OUT_DIR}/")
    print(f"  Labels         : {OUT_DIR}/labels_*.parquet")
    print(f"  Eval figures   : {EVAL_DIR}/")
