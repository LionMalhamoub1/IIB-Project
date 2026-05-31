"""
build_labels.py
===============
Converts grouped event clusters into a country-day label panel for use in
likelihood modelling.

The core problem this solves
-----------------------------
In the v1 pipeline, events are extracted at article-publication granularity.
If a strike runs from Jan 5 to Jan 15 and generates 30 news articles across
that period, the clustering output has event_date=Jan5, event_end_date=Jan15.
But there was no step converting that into 11 labelled country-days.  This
script does that conversion.

Labelling logic
---------------
For each canonical event cluster:
  - Every calendar day in [event_date, event_end_date] is labelled as an
    active event day for that country.
  - This correctly treats a multi-day strike as 10 labelled days rather than
    one labelled start-day.

For each country-day, the panel records:
  event_binary       1 if any event is active, 0 otherwise
  n_events           count of distinct events active on this day
  n_articles         total supporting articles across active events
  max_confidence     max confidence_max across active events on this day
  has_movement       1 if any active event belongs to a cross-event movement
  disruption_types   comma-separated set of active disruption types
  gdelt_n_raw        total raw articles extracted by GDELT for this country on
                     this date (loaded from raw extraction files).  Low values
                     flag potential false-negative label days — days where GDELT
                     simply had thin coverage rather than no events occurring.
  coverage_flag      "low" / "medium" / "high" based on gdelt_n_raw thresholds
                     (< LOW_COVERAGE_THRESHOLD → "low", etc.)

Usage
-----
  # From v2 global grouped output
  python build_labels.py --grouped v2/output --raw-dir path/to/daily/extractions
  python build_labels.py --grouped v2/output --out labels.parquet

  # From v1 per-day grouped output (same format, works identically)
  python build_labels.py --grouped verification/grouped

  # Restrict to a date range
  python build_labels.py --grouped v2/output --range 20180101 20180131

Output
------
  labels.parquet    country-day panel with all label columns
  labels.csv        same, CSV format
  label_report.json summary statistics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_VER  = _HERE.parent
_SD   = _VER.parent
_ROOT = _SD.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DEFAULT_GROUPED_DIR = _HERE / "output"
DEFAULT_RAW_DIR     = _ROOT / "Builder_GDELT" / "results" / "daily"
DEFAULT_OUT_DIR     = _HERE / "output"

# Coverage thresholds (raw article count per country-day)
LOW_COVERAGE_THRESHOLD  = 3
HIGH_COVERAGE_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Load grouped clusters
# ---------------------------------------------------------------------------
def load_clusters(grouped_dir: Path, date_range: tuple[str, str] | None = None
                  ) -> list[dict]:
    """Load all clusters from *_grouped.jsonl files in grouped_dir."""
    files = sorted(grouped_dir.glob("*_grouped.jsonl"))
    if date_range:
        start, end = date_range
        # Keep files whose stem starts with a date in the range
        # Works for both YYYYMMDD_grouped and YYYYMMDD_YYYYMMDD_grouped
        files = [
            f for f in files
            if _file_in_range(f.stem, start, end)
        ]

    seen: dict[str, dict] = {}
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = c.get("cluster_id")
                if not cid:
                    continue
                if cid not in seen or c.get("n_articles", 0) > seen[cid].get("n_articles", 0):
                    seen[cid] = c

    clusters = list(seen.values())
    log.info("Loaded %d unique clusters from %d files", len(clusters), len(files))
    return clusters


def _file_in_range(stem: str, start: str, end: str) -> bool:
    """Check if a file stem (YYYYMMDD or YYYYMMDD_YYYYMMDD) overlaps range."""
    parts = stem.replace("_grouped", "").split("_")
    # Extract date-like tokens
    dates = [p for p in parts if p.isdigit() and len(p) == 8]
    if not dates:
        return False
    file_start = dates[0]
    file_end   = dates[-1]
    return file_end >= start and file_start <= end


# ---------------------------------------------------------------------------
# Load movements
# ---------------------------------------------------------------------------
def load_movement_ids(grouped_dir: Path) -> set[str]:
    """Return set of cluster_ids that belong to any movement."""
    movement_ids: set[str] = set()
    for mv_file in grouped_dir.glob("*movements*.jsonl"):
        with mv_file.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    mv = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for cid in mv.get("child_cluster_ids", []):
                    movement_ids.add(cid)
    log.info("Movement membership: %d cluster IDs belong to a movement", len(movement_ids))
    return movement_ids


# ---------------------------------------------------------------------------
# Build GDELT raw coverage index (articles per country-day)
# ---------------------------------------------------------------------------
def build_gdelt_coverage(raw_dir: Path, date_range: tuple[str, str] | None = None
                         ) -> dict[tuple[str, pd.Timestamp], int]:
    """Count total raw GDELT articles per (iso3, date) across daily files.

    This is used to flag country-days with thin GDELT coverage — on those days
    a label of 0 (no event) is less trustworthy because GDELT may simply have
    missed it.  Uses the parent verification _utils for ISO3 resolution.
    """
    if not raw_dir.exists():
        log.warning("Raw dir not found: %s — coverage flags will be omitted.", raw_dir)
        return {}

    sys.path.insert(0, str(_VER))
    from _utils import extract_iso3

    coverage: dict[tuple[str, pd.Timestamp], int] = defaultdict(int)

    date_dirs = sorted(raw_dir.glob("*/extractions.jsonl"))
    if date_range:
        start, end = date_range
        date_dirs = [
            p for p in date_dirs
            if start <= p.parent.name <= end
        ]

    for path in date_dirs:
        date_str = path.parent.name
        try:
            date = pd.Timestamp(datetime.strptime(date_str, "%Y%m%d"))
        except ValueError:
            continue

        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                iso3 = extract_iso3(e)
                if iso3:
                    coverage[(iso3, date)] += 1

    log.info("Coverage index: %d country-day cells", len(coverage))
    return dict(coverage)


# ---------------------------------------------------------------------------
# Build country-day label panel
# ---------------------------------------------------------------------------
def build_label_panel(
    clusters: list[dict],
    movement_ids: set[str],
    coverage: dict[tuple[str, pd.Timestamp], int],
) -> pd.DataFrame:
    """Expand each event cluster across its full date span into country-day rows.

    Key difference from naively labelling only the event_date:
    A cluster with event_date=2018-01-05 and event_end_date=2018-01-12 will
    contribute labels to all 8 days in that range, not just Jan 5.
    """
    # Map: (iso3, date) → list of active event attributes
    day_events: dict[tuple[str, pd.Timestamp], list[dict]] = defaultdict(list)

    for c in clusters:
        iso3 = c.get("iso3")
        if not iso3:
            continue

        try:
            start = pd.Timestamp(c["event_date"])
            end   = pd.Timestamp(c["event_end_date"])
        except (KeyError, ValueError):
            continue

        # Iterate over every day the event spans
        current = start
        while current <= end:
            day_events[(iso3, current)].append({
                "cluster_id":      c["cluster_id"],
                "disruption_type": c.get("disruption_type", "unknown"),
                "n_articles":      c.get("n_articles", 1),
                "confidence_max":  c.get("confidence_max", 0.0),
                "in_movement":     c["cluster_id"] in movement_ids,
            })
            current += pd.Timedelta(days=1)

    # Build rows
    rows = []
    for (iso3, date), events in day_events.items():
        gdelt_n_raw = coverage.get((iso3, date), 0)

        if gdelt_n_raw < LOW_COVERAGE_THRESHOLD:
            cov_flag = "low"
        elif gdelt_n_raw < HIGH_COVERAGE_THRESHOLD:
            cov_flag = "medium"
        else:
            cov_flag = "high"

        rows.append({
            "iso3":              iso3,
            "date":              date,
            "event_binary":      1,
            "n_events":          len(events),
            "n_articles":        sum(e["n_articles"] for e in events),
            "max_confidence":    max(e["confidence_max"] for e in events),
            "has_movement":      int(any(e["in_movement"] for e in events)),
            "disruption_types":  ",".join(sorted(set(e["disruption_type"] for e in events))),
            "gdelt_n_raw":       gdelt_n_raw,
            "coverage_flag":     cov_flag,
        })

    df_events = pd.DataFrame(rows)
    if df_events.empty:
        log.warning("No events produced any labelled country-days.")
        return df_events

    # Determine full country-date grid to fill in zero-event days
    # (only for countries and dates we have GDELT coverage for)
    if coverage:
        cov_df = pd.DataFrame(
            [{"iso3": iso3, "date": date, "gdelt_n_raw": n}
             for (iso3, date), n in coverage.items()],
        )
        # Merge: left join from coverage grid to event labels
        merged = cov_df.merge(df_events, on=["iso3", "date"], how="left",
                              suffixes=("_cov", ""))
        # Fill zero-event days
        merged["event_binary"]  = merged["event_binary"].fillna(0).astype(int)
        merged["n_events"]      = merged["n_events"].fillna(0).astype(int)
        merged["n_articles"]    = merged["n_articles"].fillna(0).astype(int)
        merged["has_movement"]  = merged["has_movement"].fillna(0).astype(int)
        merged["max_confidence"]= merged["max_confidence"].fillna(0.0)
        merged["disruption_types"] = merged["disruption_types"].fillna("")

        # Resolve the gdelt_n_raw column (from coverage grid is authoritative)
        if "gdelt_n_raw_cov" in merged.columns:
            merged["gdelt_n_raw"] = merged["gdelt_n_raw_cov"].fillna(0).astype(int)
            merged.drop(columns=["gdelt_n_raw_cov"], inplace=True)
        else:
            merged["gdelt_n_raw"] = merged["gdelt_n_raw"].fillna(0).astype(int)

        # Recompute coverage flag for zero-event days
        def _flag(n):
            if n < LOW_COVERAGE_THRESHOLD:
                return "low"
            elif n < HIGH_COVERAGE_THRESHOLD:
                return "medium"
            return "high"

        merged["coverage_flag"] = merged["gdelt_n_raw"].apply(_flag)
        df = merged
    else:
        # No coverage data — just return the event-labelled days
        df = df_events

    df = df.sort_values(["iso3", "date"]).reset_index(drop=True)
    log.info(
        "Label panel: %d rows | %d country-days with events | %d countries | "
        "date range %s to %s",
        len(df),
        int(df["event_binary"].sum()),
        df["iso3"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )
    return df


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------
def build_report(df: pd.DataFrame) -> dict:
    total        = len(df)
    n_event_days = int(df["event_binary"].sum())
    n_countries  = df["iso3"].nunique()

    by_type: dict = {}
    for dtype in ["protests", "labour_strike"]:
        mask = df["disruption_types"].str.contains(dtype, na=False)
        by_type[dtype] = int(mask.sum())

    low_cov   = df[df["coverage_flag"] == "low"]
    low_event = int((low_cov["event_binary"] == 1).sum())
    low_zero  = int((low_cov["event_binary"] == 0).sum())

    return {
        "total_country_days":        total,
        "event_days":                n_event_days,
        "zero_days":                 total - n_event_days,
        "event_rate":                round(n_event_days / total, 4) if total else None,
        "countries":                 n_country if (n_country := n_countries) else 0,
        "date_min":                  str(df["date"].min().date()) if not df.empty else None,
        "date_max":                  str(df["date"].max().date()) if not df.empty else None,
        "by_disruption_type":        by_type,
        "movement_days":             int(df["has_movement"].sum()),
        "low_coverage_flag": {
            "total_low_cov_days":    len(low_cov),
            "event_days_low_cov":    low_event,
            "zero_days_low_cov":     low_zero,
            "note": (
                "zero_days_low_cov are unreliable negatives — GDELT had thin "
                "coverage so a 0 label does not confidently mean no event occurred"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(
    grouped_dir: Path,
    raw_dir:     Path,
    out_dir:     Path,
    date_range:  tuple[str, str] | None = None,
    out_stem:    str = "labels",
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)

    clusters     = load_clusters(grouped_dir, date_range)
    movement_ids = load_movement_ids(grouped_dir)
    coverage     = build_gdelt_coverage(raw_dir, date_range)

    df = build_label_panel(clusters, movement_ids, coverage)
    if df.empty:
        log.error("Empty label panel — check grouped_dir contains *_grouped.jsonl files.")
        return df

    out_parquet = out_dir / f"{out_stem}.parquet"
    out_csv     = out_dir / f"{out_stem}.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)
    log.info("Labels written: %s", out_parquet)

    report = build_report(df)
    with (out_dir / f"{out_stem}_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info("Report written: %s", out_dir / f'{out_stem}_report.json')

    # Print key stats
    log.info("=" * 60)
    log.info("LABEL SUMMARY")
    log.info("  Country-days total : %d", report["total_country_days"])
    log.info("  Event days (y=1)   : %d  (%.1f%%)",
             report["event_days"], (report["event_rate"] or 0) * 100)
    log.info("  Countries          : %d", report["countries"])
    log.info("  Low-coverage zeros : %d  (unreliable negatives)",
             report["low_coverage_flag"]["zero_days_low_cov"])
    log.info("=" * 60)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build country-day label panel from grouped event clusters"
    )
    parser.add_argument(
        "--grouped", type=Path, default=DEFAULT_GROUPED_DIR,
        help="Directory containing *_grouped.jsonl files",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW_DIR,
        help="Directory of daily GDELT extraction files (for coverage flags)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--range", nargs=2, metavar=("START", "END"), default=None,
        help="Restrict to date range YYYYMMDD YYYYMMDD",
    )
    parser.add_argument(
        "--stem", type=str, default="labels",
        help="Output filename stem (default: labels)",
    )
    args = parser.parse_args()

    date_range = tuple(args.range) if args.range else None
    run(args.grouped, args.raw_dir, args.out, date_range, args.stem)


if __name__ == "__main__":
    main()
