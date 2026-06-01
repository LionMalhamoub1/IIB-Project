"""
consolidate_post_hoc.py
=======================
Post-hoc cross-day consolidation using the SAME criteria as the
existing per-day consolidateExtractions.py, but applied over a
rolling window instead of per-day isolation.

Why this works better than a geo/temporal approach:
  The original per-day algorithm used tight tolerances (1-3 days date
  delta + location token overlap) which were good at identifying genuine
  duplicates without merging different nearby floods. The only problem
  was it ran on each day's batch in isolation, so articles about the
  same flood from day 1 and day 7 were never compared.

  This script sorts all records by date and checks every pair within a
  WINDOW_DAYS sliding window using the exact same criteria. Union-Find
  transitivity means a flood generating articles over 2 weeks can still
  cluster correctly through intermediate articles (day1-day3, day3-day5,
  day5-day7 chain), while the tight per-pair tolerance stops different
  nearby floods from being merged.

SAFE TO RUN: source file is never modified. All output goes to new files.

Reads  : Builder_GDELT/results/enriched_floods/all_floods_enriched.jsonl  (READ ONLY)
Writes : Builder_GDELT/results/enriched_floods/all_floods_consolidated.jsonl  (NEW)
         Builder_GDELT/results/enriched_floods/consolidation_report.json       (NEW)

Usage
-----
    python Builder_GDELT/consolidate_post_hoc.py
    python Builder_GDELT/consolidate_post_hoc.py --window-days 30
    python Builder_GDELT/consolidate_post_hoc.py --date-from 2017-01-01 --date-to 2021-12-31
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging
import multiprocessing
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
INPUT_PATH  = ROOT / "Builder_GDELT" / "results" / "enriched_floods" / "all_floods_enriched.jsonl"
OUT_PATH    = ROOT / "Builder_GDELT" / "results" / "enriched_floods" / "all_floods_consolidated.jsonl"
REPORT_PATH = ROOT / "Builder_GDELT" / "results" / "enriched_floods" / "consolidation_report.json"

# Rolling window: how far ahead to look for potential duplicates.
# The per-pair tolerance (1-3 days) still applies  -  this just limits
# which pairs we bother checking. 30 days allows transitive chains.
DEFAULT_WINDOW_DAYS = 30
DEFAULT_DATE_FROM   = "2017-01-01"
DEFAULT_DATE_TO     = "2021-12-31"
COORD_MERGE_KM      = 50    # max distance for two articles to be considered the same event
JACCARD_FALLBACK    = 0.3   # minimum token Jaccard when coords are missing

# Tolerances copied exactly from consolidateExtractions.py
EVENT_EVENT_TOL   = 1
EVENT_PUBLISH_TOL = 2
PUBLISH_PUBLISH_TOL = 3

HYDRO_FIELDS = [
    "chirps_3d_total_mm", "chirps_7d_total_mm", "chirps_14d_total_mm",
    "chirps_30d_total_mm", "chirps_peak_daily_mm", "chirps_7d_anom_pct",
    "gpm_1d_total_mm", "gpm_7d_total_mm", "gpm_peak_3h_mm",
    "era5_soil_moisture_day0", "era5_soil_moisture_7d_mean",
    "era5_precip_7d_mm", "era5_runoff_7d_mm",
    "pop_density_km2", "jrc_recurrence_pct", "terrain_slope_mean",
    "spi_30d",
]


# -- Union-Find ----------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        # Iterative path compression  -  avoids RecursionError on large datasets
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


# -- Same helpers as consolidateExtractions.py --------------------------------

def _location_tokens(loc: str | None) -> set[str]:
    if not loc:
        return set()
    s = re.sub(r"\(.*?\)", "", loc.lower())
    s = re.sub(r"[^a-z\s]", " ", s)
    return {t for t in s.split() if len(t) > 2}


def _choose_date(r: dict) -> tuple[pd.Timestamp, str] | None:
    for field, src in [("event_date", "event"), ("publish_date", "publish")]:
        v = r.get(f"_ts_{field}")   # pre-parsed timestamp stored on record
        if v is not None and not pd.isna(v):
            return v, src
    return None


def _dates_close(d1: pd.Timestamp, src1: str,
                 d2: pd.Timestamp, src2: str) -> bool:
    delta = abs((d1 - d2).days)
    if src1 == "event" and src2 == "event":
        tol = EVENT_EVENT_TOL
    elif src1 != src2:
        tol = EVENT_PUBLISH_TOL
    else:
        tol = PUBLISH_PUBLISH_TOL
    return delta <= tol


def _same_event(a: dict, b: dict) -> bool:
    """True if two records should be merged  -  identical logic to dedupe_events()."""
    if a.get("disruption_type") != b.get("disruption_type"):
        return False
    da = _choose_date(a)
    db = _choose_date(b)
    if da is None or db is None:
        return False
    if not _dates_close(da[0], da[1], db[0], db[1]):
        return False
    tok_a = _location_tokens(a.get("location_name"))
    tok_b = _location_tokens(b.get("location_name"))
    if not tok_a or not tok_b:
        return False
    return bool(tok_a & tok_b)


# -- Merge a cluster -----------------------------------------------------------

def _hydro_richness(r: dict) -> int:
    return sum(1 for f in HYDRO_FIELDS if r.get(f) is not None)


def _merge_cluster(cluster: list[dict]) -> dict:
    if len(cluster) == 1:
        r = dict(cluster[0])
        r.pop("_ts_event_date",   None)
        r.pop("_ts_publish_date", None)
        r["_cluster_size"] = 1
        r["_consolidated"] = False
        return r

    # Use the most-enriched record as the base (best GEE data)
    anchor = max(cluster, key=_hydro_richness)
    merged = dict(anchor)

    # Earliest event date across cluster
    dates = [str(r.get("event_date") or "")[:10] for r in cluster if r.get("event_date")]
    if dates:
        merged["event_date"] = min(dates)

    # Coordinates: use the single best record's coordinates.
    # Prefer the Nominatim-geocoded record with the longest location_name
    # (longer = more specific place) over any median, which degrades
    # precision by averaging imprecise actiongeo coords in.
    # Fall back to the anchor's coords if no Nominatim exists.
    nom_records = [r for r in cluster
                   if r.get("lat") is not None
                   and r.get("geo_source") == "nominatim_location_name"]
    if nom_records:
        best = max(nom_records, key=lambda r: len(r.get("location_name") or ""))
        merged["lat"]        = best["lat"]
        merged["lon"]        = best["lon"]
        merged["geo_source"] = "nominatim_location_name"
    elif anchor.get("lat") is not None:
        merged["lat"]        = anchor["lat"]
        merged["lon"]        = anchor["lon"]
        merged["geo_source"] = anchor.get("geo_source")

    # URLs: union across cluster
    all_urls: list[str] = []
    for r in cluster:
        urls = r.get("urls") or []
        if isinstance(urls, list):
            all_urls.extend(urls)
        if r.get("url"):
            all_urls.append(r["url"])
    merged["urls"]         = sorted(set(all_urls))
    merged["num_articles"] = sum(r.get("num_articles", 1) for r in cluster)

    # Impact: take max (latest/most complete report, NOT a sum)
    extras = dict(anchor.get("extras") or {})
    for field in ["death_toll", "affected_count", "displaced_count"]:
        vals = []
        for r in cluster:
            v = (r.get("extras") or {}).get(field)
            if v is not None:
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        if vals:
            extras[field] = max(vals)
    merged["extras"] = extras

    # Clean up temp keys
    merged.pop("_ts_event_date",   None)
    merged.pop("_ts_publish_date", None)
    merged["_cluster_size"] = len(cluster)
    merged["_consolidated"] = True
    return merged


# -- Parallel worker (module-level so multiprocessing can pickle it) ----------

def _find_edges_worker(args: tuple) -> list[tuple[int, int]]:
    (i_start, i_end, date_ints, date_srcs, token_sets, dtypes,
     lats, lons, window_days, _TOLS, ee_tol, ep_tol, pp_tol,
     coord_km, jaccard_fallback) = args

    import math as _math
    _R = 6371.0
    edges = []
    n = len(date_ints)

    for i in range(i_start, i_end):
        di = date_ints[i]
        if di == 999999 or not token_sets[i]:
            continue
        si   = date_srcs[i]
        ti   = token_sets[i]
        dti  = dtypes[i]
        lati = lats[i]
        loni = lons[i]
        cutoff = di + window_days

        j_end = bisect.bisect_right(date_ints, cutoff, lo=i + 1)

        for j in range(i + 1, j_end):
            dj = date_ints[j]
            if dj == 999999 or not token_sets[j]:
                continue
            if dtypes[j] != dti:
                continue
            sj  = date_srcs[j]
            tol = _TOLS.get((si, sj), pp_tol)
            if abs(di - dj) > tol:
                continue

            # Token check
            shared = ti & token_sets[j]
            if not shared:
                continue

            latj, lonj = lats[j], lons[j]
            both_have_coords = (lati is not None and latj is not None)

            if both_have_coords:
                # Both signals must agree: tokens match AND within coord_km
                _p1, _p2 = _math.radians(lati), _math.radians(latj)
                _dp = _math.radians(latj - lati)
                _dl = _math.radians(lonj - loni)
                _a  = _math.sin(_dp/2)**2 + _math.cos(_p1)*_math.cos(_p2)*_math.sin(_dl/2)**2
                dist_km = 2 * _R * _math.asin(_math.sqrt(_a))
                if dist_km <= coord_km:
                    edges.append((i, j))
            else:
                # No coords available  -  require stricter token overlap (Jaccard)
                jaccard = len(shared) / len(ti | token_sets[j])
                if jaccard >= jaccard_fallback:
                    edges.append((i, j))

    return edges


# -- Main ----------------------------------------------------------------------

def consolidate(
    window_days: int       = DEFAULT_WINDOW_DAYS,
    date_from:   str | None = DEFAULT_DATE_FROM,
    date_to:     str | None = DEFAULT_DATE_TO,
) -> None:
    log.info(f"Loading {INPUT_PATH.name} ...")
    records: list[dict] = []
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info(f"Loaded {len(records):,} records")

    if date_from or date_to:
        before = len(records)
        records = [
            r for r in records
            if (not date_from or str(r.get("event_date") or "")[:10] >= date_from)
            and (not date_to   or str(r.get("event_date") or "")[:10] <= date_to)
        ]
        log.info(f"Date filter {date_from or '*'} → {date_to or '*'}: {before:,} → {len(records):,}")

    # Pre-parse timestamps once  -  strip tz info so naive/aware subtraction never errors
    for r in records:
        for field in ("event_date", "publish_date"):
            v = r.get(field)
            ts = None
            if v:
                try:
                    ts = pd.Timestamp(v)
                    if ts.tzinfo is not None:
                        ts = ts.tz_convert(None)   # -> tz-naive UTC
                except Exception:
                    ts = None
            r[f"_ts_{field}"] = ts

    # Sort by best available date so window indexing works
    def _sort_key(r: dict) -> str:
        return str(r.get("event_date") or r.get("publish_date") or "9999")

    records.sort(key=_sort_key)
    n = len(records)

    # -- Pre-compute comparison data once (avoids recomputing in inner loop) --
    # Using integer day numbers instead of pandas Timestamps is ~100x faster
    # for per-pair date arithmetic.
    EPOCH = pd.Timestamp("1970-01-01")
    _TOLS = {
        ("event",   "event"):   EVENT_EVENT_TOL,
        ("event",   "publish"): EVENT_PUBLISH_TOL,
        ("publish", "event"):   EVENT_PUBLISH_TOL,
        ("publish", "publish"): PUBLISH_PUBLISH_TOL,
    }

    date_ints: list[int]          = []   # days since epoch, or 999999 if unknown
    date_srcs: list[str]          = []   # "event" | "publish" | ""
    token_sets: list[frozenset]   = []   # pre-computed location tokens
    dtypes: list[str]             = []   # disruption_type

    for r in records:
        ts_e = r.get("_ts_event_date")
        ts_p = r.get("_ts_publish_date")
        if ts_e is not None:
            d_int, d_src = int((ts_e - EPOCH).days), "event"
        elif ts_p is not None:
            d_int, d_src = int((ts_p - EPOCH).days), "publish"
        else:
            d_int, d_src = 999999, ""
        date_ints.append(d_int)
        date_srcs.append(d_src)
        token_sets.append(frozenset(_location_tokens(r.get("location_name"))))
        dtypes.append((r.get("disruption_type") or "").lower().strip())

    lats = [float(r["lat"]) if r.get("lat") is not None else None for r in records]
    lons = [float(r["lon"]) if r.get("lon") is not None else None for r in records]
    n_with_coords = sum(1 for x in lats if x is not None)
    log.info(f"Records with coordinates: {n_with_coords:,} / {n:,}")

    log.info(f"Comparing pairs within {window_days}-day rolling window ...")

    # -- Parallel edge finding -------------------------------------------------
    n_workers = max(1, min(os.cpu_count() or 1, 8))
    chunk_size = max(1, n // n_workers)
    chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]

    log.info(f"  {n_workers} workers, {len(chunks)} chunks")

    worker_args = [
        (i_start, i_end, date_ints, date_srcs, token_sets, dtypes,
         lats, lons, window_days, _TOLS,
         EVENT_EVENT_TOL, EVENT_PUBLISH_TOL, PUBLISH_PUBLISH_TOL,
         COORD_MERGE_KM, JACCARD_FALLBACK)
        for i_start, i_end in chunks
    ]

    all_edges: list[tuple[int, int]] = []
    with multiprocessing.Pool(processes=n_workers) as pool:
        for chunk_edges in tqdm(
            pool.imap_unordered(_find_edges_worker, worker_args),
            total=len(chunks), desc="Rolling window pass", unit="chunk",
        ):
            all_edges.extend(chunk_edges)

    log.info(f"Edges found: {len(all_edges):,}")

    uf = UnionFind(n)
    for i, j in all_edges:
        uf.union(i, j)

    # Collect components
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        components[uf.find(i)].append(i)

    sizes        = [len(v) for v in components.values()]
    n_singletons = sum(1 for s in sizes if s == 1)
    n_merged     = sum(1 for s in sizes if s > 1)

    log.info(f"Merging {len(components):,} clusters ...")
    merged_records: list[dict] = []
    for indices in tqdm(components.values(), desc="Merging clusters", unit="cluster"):
        merged_records.append(_merge_cluster([records[i] for i in sorted(indices)]))

    merged_records.sort(key=lambda r: str(r.get("event_date") or ""))

    log.info(f"Writing {OUT_PATH.name} ...")
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in merged_records:
            f.write(json.dumps(r, default=str) + "\n")

    report = {
        "input_records":    n,
        "output_records":   len(merged_records),
        "records_removed":  n - len(merged_records),
        "reduction_pct":    round(100 * (n - len(merged_records)) / n, 1),
        "singleton_clusters": n_singletons,
        "merged_clusters":    n_merged,
        "articles_saved":     n - len(merged_records),
        "parameters": {"window_days": window_days, "date_from": date_from, "date_to": date_to,
                       "event_event_tol": EVENT_EVENT_TOL,
                       "event_publish_tol": EVENT_PUBLISH_TOL,
                       "publish_publish_tol": PUBLISH_PUBLISH_TOL},
        "cluster_size_distribution": {
            "2":    sum(1 for s in sizes if s == 2),
            "3-5":  sum(1 for s in sizes if 3 <= s <= 5),
            "6-10": sum(1 for s in sizes if 6 <= s <= 10),
            "11+":  sum(1 for s in sizes if s > 10),
        },
    }
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    log.info(f"\n{'='*55}")
    log.info(f"  Input          : {n:,}")
    log.info(f"  Output         : {len(merged_records):,}  ({report['reduction_pct']}% reduction)")
    log.info(f"  Merged clusters: {n_merged:,}")
    log.info(f"  Singletons     : {n_singletons:,}")
    log.info(f"  Output         : {OUT_PATH}")
    log.info(f"{'='*55}")


if __name__ == "__main__":
    multiprocessing.freeze_support()   # required on Windows
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument("--date-from",   type=str, default=DEFAULT_DATE_FROM)
    parser.add_argument("--date-to",     type=str, default=DEFAULT_DATE_TO)
    args = parser.parse_args()
    consolidate(window_days=args.window_days, date_from=args.date_from, date_to=args.date_to)
