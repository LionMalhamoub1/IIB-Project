# Builds country-day and country-month label panels from the grouped cluster output.
# Does not re-run clustering. Forward-looking targets (protest_7d etc.) computed here.

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
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
DEFAULT_OUT_DIR     = _HERE / "labels"

# Coverage thresholds (raw article count per country-day)
LOW_COVERAGE_THRESHOLD  = 3
HIGH_COVERAGE_THRESHOLD = 10

# Monthly coverage thresholds
LOW_COVERAGE_MONTHLY  = 30
HIGH_COVERAGE_MONTHLY = 100


# ---------------------------------------------------------------------------
# Load clusters
# ---------------------------------------------------------------------------
def load_clusters(grouped_dir: Path,
                  date_range: tuple[str, str] | None = None) -> list[dict]:
    files = sorted(grouped_dir.glob("*_grouped.jsonl"))
    if date_range:
        start, end = date_range
        files = [f for f in files if _overlaps(f.stem, start, end)]

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
                if cid and (cid not in seen or
                            c.get("n_articles", 0) > seen[cid].get("n_articles", 0)):
                    seen[cid] = c

    log.info("Loaded %d unique clusters from %d files", len(seen), len(files))
    return list(seen.values())


def _overlaps(stem: str, start: str, end: str) -> bool:
    parts = stem.replace("_grouped", "").split("_")
    dates = [p for p in parts if p.isdigit() and len(p) == 8]
    if not dates:
        return False
    return dates[-1] >= start and dates[0] <= end


def load_movement_ids(grouped_dir: Path) -> set[str]:
    ids: set[str] = set()
    for f in grouped_dir.glob("*movements*.jsonl"):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    mv = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for cid in mv.get("child_cluster_ids", []):
                    ids.add(cid)
    log.info("Movement membership: %d cluster IDs", len(ids))
    return ids


# ---------------------------------------------------------------------------
# Build GDELT raw coverage index
# ---------------------------------------------------------------------------
def build_coverage(raw_dir: Path,
                   date_range: tuple[str, str] | None = None
                   ) -> dict[tuple[str, pd.Timestamp], int]:
    if not raw_dir.exists():
        log.warning("Raw dir not found: %s — coverage flags omitted.", raw_dir)
        return {}

    sys.path.insert(0, str(_VER))
    from _utils import extract_iso3

    coverage: dict[tuple[str, pd.Timestamp], int] = defaultdict(int)
    paths = sorted(raw_dir.glob("*/extractions.jsonl"))
    if date_range:
        start, end = date_range
        paths = [p for p in paths if start <= p.parent.name <= end]

    for path in paths:
        try:
            date = pd.Timestamp(path.parent.name)
        except Exception:
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
# Build base country-day event grid
# ---------------------------------------------------------------------------
def build_base_day_grid(
    clusters: list[dict],
    movement_ids: set[str],
) -> pd.DataFrame:
    """
    Expand every cluster across its full date span into country-day rows,
    separated by disruption type (protest vs strike).
    Returns a DataFrame indexed by (iso3, date) with event counts and flags.
    """
    # (iso3, date) -> {protest: [...], strike: [...]}
    day_map: dict[tuple, dict] = defaultdict(lambda: {"protest": [], "strike": []})

    for c in clusters:
        iso3  = c.get("iso3")
        dtype = c.get("disruption_type", "")
        if not iso3 or dtype not in ("protests", "labour_strike"):
            continue

        try:
            start = pd.Timestamp(c["event_date"])
            end   = pd.Timestamp(c["event_end_date"])
        except Exception:
            continue

        key   = "protest" if dtype == "protests" else "strike"
        attrs = {
            "cluster_id":   c["cluster_id"],
            "n_articles":   c.get("n_articles", 1),
            "confidence":   c.get("confidence_max", 0.0),
            "in_movement":  c["cluster_id"] in movement_ids,
        }

        current = start
        while current <= end:
            day_map[(iso3, current)][key].append(attrs)
            current += pd.Timedelta(days=1)

    rows = []
    for (iso3, date), events in day_map.items():
        p = events["protest"]
        s = events["strike"]
        rows.append({
            "iso3":              iso3,
            "date":              date,
            "protest_today":     int(len(p) > 0),
            "strike_today":      int(len(s) > 0),
            "n_protest_events":  len(p),
            "n_strike_events":   len(s),
            "n_articles":        sum(e["n_articles"] for e in p + s),
            "has_movement":      int(any(e["in_movement"] for e in p + s)),
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["iso3", "date", "protest_today", "strike_today",
                 "n_protest_events", "n_strike_events", "n_articles", "has_movement"]
    )
    return df


# ---------------------------------------------------------------------------
# Build country-day panel
# ---------------------------------------------------------------------------
def build_country_day(
    base: pd.DataFrame,
    coverage: dict[tuple[str, pd.Timestamp], int],
) -> pd.DataFrame:
    """
    1. Merge with coverage grid to fill in zero-event days.
    2. Add forward-looking labels: 7-day and 30-day horizons.
    3. Add coverage flags.
    """
    if coverage:
        cov_df = pd.DataFrame(
            [{"iso3": iso3, "date": date, "gdelt_n_raw": n}
             for (iso3, date), n in coverage.items()]
        )
        df = cov_df.merge(base, on=["iso3", "date"], how="left")
        for col in ["protest_today", "strike_today", "n_protest_events",
                    "n_strike_events", "n_articles", "has_movement"]:
            df[col] = df[col].fillna(0).astype(int)
    else:
        df = base.copy()
        df["gdelt_n_raw"] = 0

    df = df.sort_values(["iso3", "date"]).reset_index(drop=True)

    # Coverage flag
    def _flag(n):
        if n < LOW_COVERAGE_THRESHOLD:   return "low"
        if n < HIGH_COVERAGE_THRESHOLD:  return "medium"
        return "high"
    df["coverage_flag"] = df["gdelt_n_raw"].apply(_flag)

    # Forward-looking labels for each country
    log.info("Computing forward-looking labels (7d, 30d)...")
    result_parts = []
    for iso3, grp in df.groupby("iso3"):
        grp = grp.sort_values("date").copy()
        grp = grp.set_index("date")

        for col, horizons in [("protest_today", [7, 30]), ("strike_today", [7, 30])]:
            base_name = col.replace("_today", "")
            for h in horizons:
                # For each date, check if any positive label exists in next h days
                grp[f"{base_name}_{h}d"] = (
                    grp[col]
                    .rolling(f"{h}D", min_periods=1)
                    .max()
                    .shift(-h)           # shift back so it's forward-looking
                    .fillna(0)
                    .astype(int)
                )

        result_parts.append(grp.reset_index())

    df = pd.concat(result_parts, ignore_index=True)

    # Column order
    cols = ["iso3", "date",
            "protest_today", "strike_today",
            "protest_7d", "strike_7d",
            "protest_30d", "strike_30d",
            "n_protest_events", "n_strike_events",
            "n_articles", "has_movement",
            "gdelt_n_raw", "coverage_flag"]
    df = df[[c for c in cols if c in df.columns]]

    log.info(
        "Country-day panel: %d rows | %d countries | %s to %s",
        len(df), df["iso3"].nunique(),
        df["date"].min().date(), df["date"].max().date(),
    )
    return df


# ---------------------------------------------------------------------------
# Build country-month panel
# ---------------------------------------------------------------------------
def build_country_month(
    day_df: pd.DataFrame,
    coverage: dict[tuple[str, pd.Timestamp], int],
) -> pd.DataFrame:
    """
    Aggregate the country-day panel to country-month.
    Adds next-month forward-looking labels.
    """
    df = day_df.copy()
    df["year_month"] = df["date"].dt.to_period("M")

    agg = df.groupby(["iso3", "year_month"]).agg(
        protest_this_month  = ("protest_today",    "max"),
        strike_this_month   = ("strike_today",     "max"),
        n_protest_events    = ("n_protest_events", "sum"),
        n_strike_events     = ("n_strike_events",  "sum"),
        n_protest_days      = ("protest_today",    "sum"),
        n_strike_days       = ("strike_today",     "sum"),
        n_articles          = ("n_articles",       "sum"),
    ).reset_index()

    # Monthly GDELT coverage
    if coverage:
        cov_monthly: dict[tuple, int] = defaultdict(int)
        for (iso3, date), n in coverage.items():
            ym = date.to_period("M")
            cov_monthly[(iso3, ym)] += n
        agg["gdelt_n_raw"] = agg.apply(
            lambda r: cov_monthly.get((r["iso3"], r["year_month"]), 0), axis=1
        )
    else:
        agg["gdelt_n_raw"] = 0

    def _flag_monthly(n):
        if n < LOW_COVERAGE_MONTHLY:   return "low"
        if n < HIGH_COVERAGE_MONTHLY:  return "medium"
        return "high"
    agg["coverage_flag"] = agg["gdelt_n_raw"].apply(_flag_monthly)

    # Next-month forward-looking labels per country
    log.info("Computing next-month forward-looking labels...")
    parts = []
    for iso3, grp in agg.groupby("iso3"):
        grp = grp.sort_values("year_month").copy()
        grp["protest_next_month"] = grp["protest_this_month"].shift(-1).fillna(0).astype(int)
        grp["strike_next_month"]  = grp["strike_this_month"].shift(-1).fillna(0).astype(int)
        parts.append(grp)

    agg = pd.concat(parts, ignore_index=True)

    cols = ["iso3", "year_month",
            "protest_this_month", "strike_this_month",
            "protest_next_month", "strike_next_month",
            "n_protest_events", "n_strike_events",
            "n_protest_days", "n_strike_days",
            "n_articles", "gdelt_n_raw", "coverage_flag"]
    agg = agg[[c for c in cols if c in agg.columns]]
    agg = agg.sort_values(["iso3", "year_month"]).reset_index(drop=True)

    log.info(
        "Country-month panel: %d rows | %d countries | %s to %s",
        len(agg), agg["iso3"].nunique(),
        str(agg["year_month"].min()), str(agg["year_month"].max()),
    )
    return agg


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(day_df: pd.DataFrame, month_df: pd.DataFrame) -> None:
    log.info("=" * 60)
    log.info("LABEL SUMMARY")
    log.info("  --- Country-day ---")
    log.info("  Rows            : %d", len(day_df))
    log.info("  Countries       : %d", day_df["iso3"].nunique())
    log.info("  protest_today=1 : %d  (%.1f%%)", day_df["protest_today"].sum(),
             day_df["protest_today"].mean() * 100)
    log.info("  strike_today=1  : %d  (%.1f%%)", day_df["strike_today"].sum(),
             day_df["strike_today"].mean() * 100)
    log.info("  protest_7d=1    : %d  (%.1f%%)", day_df["protest_7d"].sum(),
             day_df["protest_7d"].mean() * 100)
    log.info("  protest_30d=1   : %d  (%.1f%%)", day_df["protest_30d"].sum(),
             day_df["protest_30d"].mean() * 100)
    log.info("  Low-cov zeros   : %d", ((day_df["protest_today"] == 0) & (day_df["coverage_flag"] == "low")).sum())
    log.info("  --- Country-month ---")
    log.info("  Rows                  : %d", len(month_df))
    log.info("  protest_this_month=1  : %d  (%.1f%%)", month_df["protest_this_month"].sum(),
             month_df["protest_this_month"].mean() * 100)
    log.info("  strike_this_month=1   : %d  (%.1f%%)", month_df["strike_this_month"].sum(),
             month_df["strike_this_month"].mean() * 100)
    log.info("  protest_next_month=1  : %d  (%.1f%%)", month_df["protest_next_month"].sum(),
             month_df["protest_next_month"].mean() * 100)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run(
    grouped_dir: Path,
    raw_dir:     Path,
    out_dir:     Path,
    date_range:  tuple[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    out_dir.mkdir(parents=True, exist_ok=True)

    clusters     = load_clusters(grouped_dir, date_range)
    movement_ids = load_movement_ids(grouped_dir)
    coverage     = build_coverage(raw_dir, date_range)

    base     = build_base_day_grid(clusters, movement_ids)
    day_df   = build_country_day(base, coverage)
    month_df = build_country_month(day_df, coverage)

    # Save
    day_path   = out_dir / "labels_country_day.parquet"
    month_path = out_dir / "labels_country_month.parquet"
    day_df.to_parquet(day_path, index=False)
    month_df.to_parquet(month_path, index=False)
    # Also save month as CSV since period type needs care in parquet readers
    month_df.assign(year_month=month_df["year_month"].astype(str)).to_csv(
        out_dir / "labels_country_month.csv", index=False
    )
    log.info("Saved: %s", day_path)
    log.info("Saved: %s", month_path)

    print_summary(day_df, month_df)
    return day_df, month_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build modelling label panels from clustered GDELT events"
    )
    parser.add_argument("--grouped", type=Path, default=DEFAULT_GROUPED_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out",     type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--range",   nargs=2, metavar=("START", "END"), default=None)
    args = parser.parse_args()

    date_range = tuple(args.range) if args.range else None
    run(args.grouped, args.raw_dir, args.out, date_range)


if __name__ == "__main__":
    main()
