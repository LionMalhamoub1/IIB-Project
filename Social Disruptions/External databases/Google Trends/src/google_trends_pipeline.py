from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

import pandas as pd

from google_trends_client import TrendsClient, TrendsClientConfig
from google_trends_transform import add_anomaly_scores, merge_batches, stitch_windows


log = logging.getLogger(__name__)


START_YEAR: int = 2017
END_YEAR:   int = 2025

ECONOMIC_STRESS_TERMS: List[str] = [
    "mortgage help",
    "can't pay mortgage",
    "rent assistance",
    "cost of living",
    "fuel prices",
    "food prices rising",
    "inflation help",
    "energy bills",
]

LABOUR_CONFLICT_TERMS: List[str] = [
    "strike vote",
    "union strike",
    "worker strike",
    "labour dispute",
    "collective bargaining",
]

PROTEST_MOBILISATION_TERMS: List[str] = [
    "protest",
    "demonstration",
    "march protest",
    "general strike",
    "protest near me",
]

SEARCH_GROUPS: Dict[str, List[str]] = {
    "economic_stress":      ECONOMIC_STRESS_TERMS,
    "labour_conflict":      LABOUR_CONFLICT_TERMS,
    "protest_mobilisation": PROTEST_MOBILISATION_TERMS,
}

COUNTRIES_ISO3: List[str] = [
    "CHN",
    "AUS",
    "COD",
    "CHL",
    "ZAF",
    "RUS",
    "IDN",
    "BRA",
    "ARG",
    "BOL",
    "KAZ",
    "CAN",
    "USA",
    "PER",
    "ZMB",
    "ZWE",
    "MAR",
    "MOZ",
    "MDG",
    "MEX",
    "MMR",
    "MNG",
    "TZA",
    "IND",
    "VNM",
    "PHL",
    "GIN",
    "GHA",
    "GAB",
    "TUR",
]

ANCHOR_KEYWORD = "protest"

WINDOW_WEEKS  = 52
OVERLAP_WEEKS = 26

ANOMALY_WINDOW_WEEKS = 12

MAX_KW_PER_REQUEST = 5

MAX_WORKERS        = 5  # country-level concurrency
MAX_WINDOW_WORKERS = 3  # window-level concurrency within each country

USE_CACHE          = True
SLEEP_S            = 6.0
MAX_RETRIES        = 4
RETRY_BACKOFF_BASE = 60.0
HL                 = "en-US"
TZ                 = 0

GT_ROOT     = Path(__file__).resolve().parents[1]
CACHE_DIR   = GT_ROOT / "data" / "raw" / "api_cache"
OUT_PARQUET = GT_ROOT / "data" / "processed" / f"google_trends_country_week_{START_YEAR}_{END_YEAR}.parquet"
OUT_CSV     = GT_ROOT / "data" / "processed" / f"google_trends_country_week_{START_YEAR}_{END_YEAR}.csv"

INDEX_COLS: List[str] = [f"{g}_index" for g in SEARCH_GROUPS]


def slug(keyword: str) -> str:
    return keyword.lower().replace("'", "").replace(" ", "_").replace("-", "_")


def _all_terms() -> List[str]:
    seen: set = set()
    out: List[str] = []
    for terms in SEARCH_GROUPS.values():
        for t in terms:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def build_weekly_windows(start_year: int, end_year: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(f"{start_year}-01-01")
    end   = pd.Timestamp(f"{end_year}-12-31")
    step  = pd.Timedelta(weeks=WINDOW_WEEKS - OVERLAP_WEEKS)
    span  = pd.Timedelta(weeks=WINDOW_WEEKS) - pd.Timedelta(days=1)

    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur <= end:
        win_end = min(cur + span, end)
        windows.append((cur, win_end))
        if win_end >= end:
            break
        cur += step
    return windows


def make_kw_batches(keywords: List[str], anchor: str) -> List[List[str]]:
    non_anchor = [k for k in keywords if k != anchor]
    if not non_anchor:
        return [[anchor]]
    step = MAX_KW_PER_REQUEST - 1
    return [[anchor] + non_anchor[i : i + step] for i in range(0, len(non_anchor), step)]


def _build_group_indices(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for group_name, terms in SEARCH_GROUPS.items():
        term_slugs = [slug(t) for t in terms]
        valid_cols = [
            c for c in term_slugs
            if c in df.columns and df[c].notna().any()
        ]
        missing = [c for c in term_slugs if c not in valid_cols]
        for m in missing:
            log.warning("Group '%s': term '%s' has no data — skipping.", group_name, m)
        if not valid_cols:
            log.warning("Group '%s': no valid term series — column will be NaN.", group_name)
            out[f"{group_name}_index"] = float("nan")
        else:
            out[f"{group_name}_index"] = df[valid_cols].mean(axis=1)
    return out


def _fetch_window(
    iso3: str,
    kw_batches: List[List[str]],
    win_start: pd.Timestamp,
    win_end: pd.Timestamp,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    client = TrendsClient(TrendsClientConfig(
        cache_dir=CACHE_DIR,
        hl=HL,
        tz=TZ,
        sleep_s=SLEEP_S,
        max_retries=MAX_RETRIES,
        retry_backoff_base=RETRY_BACKOFF_BASE,
    ))

    batch_dfs: List[pd.DataFrame] = []
    for kw_batch in kw_batches:
        df = client.fetch_window(iso3, kw_batch, win_start, win_end, use_cache=USE_CACHE)
        if not df.empty:
            batch_dfs.append(df)

    if not batch_dfs:
        log.debug("%s: no data for window %s-%s", iso3, win_start.date(), win_end.date())
        return None

    merged = merge_batches(batch_dfs, anchor=ANCHOR_KEYWORD)
    return (win_start, win_end, merged) if not merged.empty else None


def process_country(
    iso3: str,
    kw_batches: List[List[str]],
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> Optional[pd.DataFrame]:
    window_results: List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = []

    with ThreadPoolExecutor(max_workers=MAX_WINDOW_WORKERS) as win_pool:
        win_futures = {
            win_pool.submit(_fetch_window, iso3, kw_batches, ws, we): (ws, we)
            for ws, we in windows
        }
        with tqdm(total=len(win_futures), desc=f"{iso3} windows", unit="win", leave=False) as pbar:
            for future in as_completed(win_futures):
                result = future.result()
                if result is not None:
                    window_results.append(result)
                pbar.update(1)

    if not window_results:
        log.warning("%s: no data returned in any window — skipping", iso3)
        return None

    stitched = stitch_windows(window_results)
    if stitched.empty:
        return None

    stitched.columns = [slug(c) for c in stitched.columns]
    indices = _build_group_indices(stitched)
    indices.index.name = "week"
    indices = indices.reset_index()
    indices.insert(0, "country_iso3", iso3)

    log.info("%s: %d weeks collected", iso3, len(indices))
    return indices


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    all_terms  = _all_terms()
    kw_batches = make_kw_batches(all_terms, ANCHOR_KEYWORD)
    windows    = build_weekly_windows(START_YEAR, END_YEAR)

    log.info("Countries : %d", len(COUNTRIES_ISO3))
    log.info("Groups    : %s", list(SEARCH_GROUPS.keys()))
    log.info("Terms     : %d  across %d groups", len(all_terms), len(SEARCH_GROUPS))
    log.info("Batches   : %s", kw_batches)
    log.info("Windows   : %d  (%s -> %s)", len(windows), windows[0][0].date(), windows[-1][1].date())
    log.info("Workers   : %d country x %d window = %d max concurrent", MAX_WORKERS, MAX_WINDOW_WORKERS, MAX_WORKERS * MAX_WINDOW_WORKERS)

    all_frames: List[pd.DataFrame] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(process_country, iso3, kw_batches, windows): iso3
            for iso3 in COUNTRIES_ISO3
        }
        with tqdm(total=len(futures), desc="Countries", unit="country") as pbar:
            for future in as_completed(futures):
                iso3 = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    log.error("%s: unhandled exception — %s", iso3, exc)
                    pbar.update(1)
                    continue
                if result is not None:
                    all_frames.append(result)
                pbar.set_postfix_str(iso3)
                pbar.update(1)

    if not all_frames:
        log.error("No data collected — exiting without writing output.")
        return

    combined = (
        pd.concat(all_frames, ignore_index=True)
        .sort_values(["country_iso3", "week"])
        .reset_index(drop=True)
    )
    combined["week"] = pd.to_datetime(combined["week"])

    index_cols_present = [c for c in INDEX_COLS if c in combined.columns]

    scored_parts: List[pd.DataFrame] = []
    for _, group in combined.groupby("country_iso3", sort=False):
        scored_parts.append(
            add_anomaly_scores(
                group.sort_values("week").reset_index(drop=True),
                keyword_cols=index_cols_present,
                window=ANOMALY_WINDOW_WEEKS,
            )
        )

    output = (
        pd.concat(scored_parts, ignore_index=True)
        .sort_values(["country_iso3", "week"])
        .reset_index(drop=True)
    )

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUT_PARQUET, index=False)
    output.to_csv(OUT_CSV, index=False)

    log.info("Saved:\n  %s\n  %s", OUT_PARQUET, OUT_CSV)
    log.info("Shape: %s", output.shape)
    log.info("\nMissingness:\n%s", output.isna().mean().sort_values().to_string())


if __name__ == "__main__":
    main()
