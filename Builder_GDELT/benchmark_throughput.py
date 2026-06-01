"""
benchmark_throughput.py
=======================
Purpose
-------
Measures the throughput of the async extraction pipeline (DisruptionExtractorAsync)
at different LLM API concurrency levels (semaphore sizes: 1, 5, 10, 20).

This generates the data for Figure 5.2 in the dissertation:
    "Throughput of the asynchronous extraction pipeline under different LLM
    concurrency limits  -  articles processed per minute vs. semaphore size."

The same fixed sample of SAMPLE_SIZE URLs is reused across all four runs so
that differences in timing are attributable only to concurrency level, not
workload variation.

Usage
-----
    python benchmark_throughput.py

Output
------
    - A summary table printed to stdout (concurrency -> elapsed -> articles/min)
    - benchmark_results.json written alongside this script for later plotting

Cost estimate (gpt-4o-mini, May 2026)
--------------------------------------
    SAMPLE_SIZE=100 articles × 5 concurrency levels = ~500 API calls
    ≈ $0.15-0.25 (~£0.12-0.20) total depending on article text length.
"""

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import aiohttp
import pandas as pd

# -- Path setup: make helper_scripts/pipeline importable ----------------------
PIPELINE_DIR = Path(__file__).resolve().parent / "helper_scripts" / "pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from DisruptionExtractorAsync import (  # noqa: E402  # type: ignore[import]
    DEFAULT_MODEL,
    HEADERS,
    api_key,
    extract_from_url_async,
)
from openai import AsyncOpenAI  # noqa: E402

# -- Config --------------------------------------------------------------------

# Number of articles to process per concurrency level.
# 100 gives a stable estimate; at concurrency=1 this takes ~5-10 min.
SAMPLE_SIZE = 100

# Semaphore sizes to benchmark (x-axis of Figure 5.2).
# The extra level at 15 helps show where diminishing returns begin.
CONCURRENCY_LEVELS = [1, 5, 10, 15, 20]

# Keep scrape concurrency high so the web-fetch stage is never the bottleneck.
SCRAPE_CONCURRENT = 40

# Fixed seed so every run draws the same sample.
RANDOM_SEED = 42

# Date folder to pull URLs from (any day with a urls_filtered.csv works).
SAMPLE_DATE = "20180115"

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "daily"
OUTPUT_JSON = Path(__file__).resolve().parent / "benchmark_results.json"


# -- Sample loader -------------------------------------------------------------

def load_sample(n: int, seed: int) -> List[Tuple]:
    """Load n URLs from a single day's urls_filtered.csv."""
    csv_path = RESULTS_DIR / SAMPLE_DATE / "urls_filtered.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["url_normalized"])

    has_coords = "actiongeo_lat" in df.columns and "actiongeo_lon" in df.columns

    records: List[Tuple] = []
    for _, row in df.iterrows():
        records.append((
            str(row["url_normalized"]).strip(),
            str(row["top_expert"]).strip() if pd.notna(row.get("top_expert")) else None,
            float(row["top_expert_p"]) if pd.notna(row.get("top_expert_p")) else None,
            float(row["actiongeo_lat"]) if has_coords and pd.notna(row.get("actiongeo_lat")) else None,
            float(row["actiongeo_lon"]) if has_coords and pd.notna(row.get("actiongeo_lon")) else None,
        ))

    random.seed(seed)
    random.shuffle(records)
    sample = records[:n]
    print(f"Loaded {len(sample)} URLs from {SAMPLE_DATE}/urls_filtered.csv  ({len(df)} available)")
    return sample


# -- Core timed runner ---------------------------------------------------------

# Print a milestone line every this many completed articles.
_MILESTONE = 25

async def _run_timed(
    records: List[Tuple],
    api_concurrency: int,
    model: str,
) -> float:
    """
    Run extraction on all records with the given API semaphore size.
    Returns wall-clock elapsed seconds.
    """
    sem_scrape = asyncio.Semaphore(SCRAPE_CONCURRENT)
    sem_api    = asyncio.Semaphore(api_concurrency)
    loop       = asyncio.get_event_loop()
    client     = AsyncOpenAI(api_key=api_key)
    total      = len(records)

    connector = aiohttp.TCPConnector(limit=SCRAPE_CONCURRENT + 10, ssl=False)
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:

        async def _worker(rec: Tuple) -> None:
            url, expert_type, expert_prob, lat, lon = rec
            await extract_from_url_async(
                url=url,
                expert_type=expert_type,
                expert_prob=expert_prob,
                lat=lat,
                lon=lon,
                session=session,
                openai_client=client,
                sem_scrape=sem_scrape,
                sem_api=sem_api,
                loop=loop,
                model=model,
            )

        t_start  = time.perf_counter()
        done     = 0
        tasks    = [asyncio.create_task(_worker(r)) for r in records]

        for coro in asyncio.as_completed(tasks):
            await coro
            done += 1
            elapsed = time.perf_counter() - t_start
            apm     = done / elapsed * 60

            if done % _MILESTONE == 0 or done == total:
                pct = done / total * 100
                bar = ("█" * (done * 20 // total)).ljust(20)
                print(f"    [{bar}] {done:3d}/{total}  ({pct:5.1f}%)  "
                      f"{elapsed:6.1f}s elapsed  |  {apm:5.1f} art/min")

        return time.perf_counter() - t_start


# -- Benchmark runner ----------------------------------------------------------

def run_benchmark(sample: List[Tuple], model: str) -> dict:
    results      = {}
    total_runs   = len(CONCURRENCY_LEVELS)

    for i, level in enumerate(CONCURRENCY_LEVELS, 1):
        print(f"\n{'-' * 50}")
        print(f"  Run {i}/{total_runs}  —  api_concurrency = {level}")
        print(f"{'-' * 50}")

        elapsed = asyncio.run(_run_timed(sample, level, model))
        apm     = len(sample) / elapsed * 60

        results[level] = {
            "elapsed_s":        round(elapsed, 2),
            "articles_per_min": round(apm, 1),
        }
        print(f"\n  Result: {elapsed:.1f}s total  |  {apm:.1f} articles/min\n")

    return results


# -- Entry point ---------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("  Extraction Throughput Benchmark")
    print("=" * 50)
    print(f"  Sample size        : {SAMPLE_SIZE} URLs per level")
    print(f"  Concurrency levels : {CONCURRENCY_LEVELS}")
    print(f"  Total API calls    : ~{SAMPLE_SIZE * len(CONCURRENCY_LEVELS)}")
    print(f"  Model              : {DEFAULT_MODEL}")
    print(f"  Scrape concurrency : {SCRAPE_CONCURRENT} (fixed)")
    print()

    sample = load_sample(SAMPLE_SIZE, RANDOM_SEED)
    results = run_benchmark(sample, DEFAULT_MODEL)

    # Summary table
    print("\n\n" + "=" * 50)
    print("  RESULTS SUMMARY")
    print("=" * 50)
    print(f"  {'Concurrency':>11}  {'Elapsed (s)':>11}  {'Articles/min':>12}")
    print("  " + "-" * 38)
    for level, r in results.items():
        print(f"  {level:>11}  {r['elapsed_s']:>11.1f}  {r['articles_per_min']:>12.1f}")

    payload = {
        "sample_size": SAMPLE_SIZE,
        "model": DEFAULT_MODEL,
        "concurrency_levels": CONCURRENCY_LEVELS,
        "runs": {str(k): v for k, v in results.items()},
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Raw results saved to: {OUTPUT_JSON}")
