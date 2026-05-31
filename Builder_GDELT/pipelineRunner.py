"""
Master automated pipeline — Stage 2: extraction and consolidation only.

Flow (per day):
1) Run extraction from data/urls/YYYYMMDD.csv
2) Enrich publish_date from GDELT dateadded
3) Geocode location_name → lat/lon
4) Split extractions by disruption type
5) Run consolidation (deduplication)
6) Save consolidated file in daily folder

GEE enrichment (formerly step 3 here) has been moved to Stage 3 of the
master pipeline (Builder_GDELT/run_enrichment.py).  Decoupling extraction
from GEE means this stage completes quickly, and enrichment can be run
separately without re-extracting.

Optional:
    Run analytics once over ALL consolidated days (postprocess=True)
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import argparse
import json
import shutil
import pandas as pd


# ---- Project paths ---- #

BASE_DIR = Path("data/urls")
DAILY_URL_DIR = Path("data/urls")
DAILY_RESULTS_DIR = Path("Builder_GDELT/results/daily")
COMBINED_RESULTS_DIR = Path("Builder_GDELT/results/combined")
COMBINED_BY_TYPE_DIR = Path("Builder_GDELT/results/combined/by_type")


# ---- Expert filter ---- #

EXPERT_TYPES = {"flood", "protests", "labour_strike"}


# ---- Import stages ---- #

from .helper_scripts.pipeline.DisruptionExtractorAsync import run_batch
from .helper_scripts.pipeline.enrichPublishDate import enrich_publish_dates
from .helper_scripts.pipeline.geocode_locations import geocode_jsonl_inplace
from .helper_scripts.pipeline.consolidateExtractions import load_extractions, run_consolidation
from .helper_scripts.analysis.debuggerAndMetrics import run_debugger_and_metrics
from .helper_scripts.analysis.DisplayExtractionsPandas import run_display_extractions
from .plots.plotDisruptions import run_plots


# ------------------ UTIL ------------------ #

def split_jsonl_by_type(jsonl_path: Path, skip_types: set[str] | None = None) -> None:
    """Split a .jsonl file into per-type subfolders alongside the source file.

    Args:
        skip_types: disruption_type values to omit from the split output.
                    Floods are skipped from the consolidated split because they
                    are consolidated globally after enrichment, not per-day.
    """
    import json
    skip_types = {t.lower() for t in (skip_types or set())}
    buckets: dict[str, list[str]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            dtype = (rec.get("disruption_type") or "unknown").strip().lower()
            if dtype not in skip_types:
                buckets.setdefault(dtype, []).append(line)

    for dtype, lines in buckets.items():
        out_dir = jsonl_path.parent / "by_type" / dtype
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / jsonl_path.name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[SPLIT] {dtype}: {len(lines)} records -> {out_path}")

def parse_date(s: str) -> datetime.date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


# ------------------ CORE ------------------ #

def run_pipeline_date_range(start_date, end_date, postprocess=False):

    DAILY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_BY_TYPE_DIR.mkdir(parents=True, exist_ok=True)

    consolidated_paths = []
    raw_paths = []   # ← ADDED

    for d in daterange(start_date, end_date):

        yyyymmdd = d.strftime("%Y%m%d")
        url_csv = DAILY_URL_DIR / f"{yyyymmdd}.csv"

        if not url_csv.exists():
            print(f"[SKIP] No URL file for {yyyymmdd}")
            continue

        print(f"\n=== Processing {yyyymmdd} ===")

        day_dir = DAILY_RESULTS_DIR / yyyymmdd
        raw_path = day_dir / "extractions.jsonl"
        consolidated_path = day_dir / "extractions_consolidated.jsonl"

        # ---- 1) Extraction ---- #
        if not raw_path.exists():
            df_urls = pd.read_csv(url_csv)
            df_urls.columns = [c.strip().lower() for c in df_urls.columns]
            df_filtered = df_urls[df_urls["top_expert"].str.strip().isin(EXPERT_TYPES)]
            n_before, n_after = len(df_urls), len(df_filtered)
            print(f"[FILTER] {n_before} URLs -> {n_after} after expert filter ({n_before - n_after} dropped)")
            if df_filtered.empty:
                print(f"[SKIP] No relevant URLs for {yyyymmdd}")
                continue
            filtered_csv = day_dir / "urls_filtered.csv"
            day_dir.mkdir(parents=True, exist_ok=True)
            df_filtered.to_csv(filtered_csv, index=False)
            run_batch(input_csv=str(filtered_csv), input_yyyymmdd=yyyymmdd)

        if not raw_path.exists():
            print(f"[WARN] No extraction output for {yyyymmdd}")
            continue

        raw_paths.append(raw_path)

        # ---- 1b) Enrich publish_date from GDELT dateadded ---- #
        enrich_publish_dates(raw_path, yyyymmdd)

        # ---- 1c) Geocode location_name → improve lat/lon coords ---- #
        print(f"[GEOCODE] Resolving coordinates from LLM location names for {yyyymmdd}...")
        geocode_jsonl_inplace(raw_path)

        # Split AFTER geocoding so by_type files carry improved coords
        split_jsonl_by_type(raw_path)

        # ---- 2) Consolidation ---- #
        if not consolidated_path.exists():
            df_after = run_consolidation(raw_path)
            with open(consolidated_path, "w", encoding="utf-8") as _f:
                for _, _row in df_after.iterrows():
                    _rec = _row.to_dict()
                    for _k in ("event_date", "publish_date"):
                        if isinstance(_rec.get(_k), pd.Timestamp):
                            _rec[_k] = _rec[_k].isoformat()
                        elif _rec.get(_k) is None or (_rec.get(_k) != _rec.get(_k)):
                            _rec[_k] = None
                    _f.write(json.dumps(_rec, ensure_ascii=False) + "\n")
            print(f"[OK] Consolidated saved: {consolidated_path}")

        # Floods are consolidated globally after enrichment — skip per-day flood split
        split_jsonl_by_type(consolidated_path, skip_types={"flood"})

        consolidated_paths.append(consolidated_path)

        # GEE enrichment has moved to Stage 3 (Builder_GDELT/run_enrichment.py)
        # and is no longer run inline here.

    # ------------------ OPTIONAL GLOBAL POSTPROCESS ------------------ #

    if postprocess and consolidated_paths:

        print("\n=== Running global post-processing ===\n")

        # ---- Load AFTER (consolidated) ---- #
        dfs_after = [pd.read_json(p, lines=True) for p in consolidated_paths]
        df_after_all = pd.concat(dfs_after, ignore_index=True)

        combined_raw = COMBINED_RESULTS_DIR / "all_consolidated.jsonl"
        df_after_all.to_json(combined_raw,orient="records",lines=True,force_ascii=False,date_format="iso")
        # ---- Load BEFORE (raw) ---- #
        dfs_before = [pd.read_json(p, lines=True) for p in raw_paths]   # ← ADDED
        df_before_all = pd.concat(dfs_before, ignore_index=True)        # ← ADDED

        # ---- Analytics ---- #
        run_debugger_and_metrics(df_after_all, df_before_all)           # ← MODIFIED
        run_display_extractions(df_after_all, df_before=df_before_all)  # ← MODIFIED
        run_plots(df_after_all, project_root=BASE_DIR)

        print(f"\n[OK] Combined dataset saved to: {combined_raw}")

    print("\nPipeline complete.\n")


# ------------------ CLI ------------------ #

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--range", nargs=2, required=True,
                        metavar=("START", "END"),
                        help="Date range YYYY-MM-DD YYYY-MM-DD")

    parser.add_argument("--postprocess", action="store_true",
                        help="Run global analytics after daily processing")

    args = parser.parse_args()

    start = parse_date(args.range[0])
    end = parse_date(args.range[1])

    run_pipeline_date_range(start, end, postprocess=args.postprocess)


if __name__ == "__main__":
    main()