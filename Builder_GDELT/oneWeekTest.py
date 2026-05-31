from __future__ import annotations

import os
import glob
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import pandas as pd
from tqdm import tqdm

# Import your existing extraction logic
from DisruptionExtractor import extract_from_url_llm_single_pass, DEFAULT_MODEL, MAX_WORKERS


# ================================
# CONFIG
# ================================

YEAR = "2026"
MONTH = "01"

# ================================
# PATH RESOLUTION
# ================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "model_scored_daily",
    YEAR,
    MONTH,
)
OUTPUT_JSONL = os.path.join(BASE_DIR, f"weekly_extractions_{YEAR}{MONTH}.jsonl")
OUTPUT_CSV = os.path.join(BASE_DIR, f"weekly_extractions_{YEAR}{MONTH}.csv")
ERROR_CSV = os.path.join(BASE_DIR, f"weekly_extractions_{YEAR}{MONTH}_errors.csv")


# ================================
# COLLECT ALL 7 CSV FILES
# ================================

def collect_week_csvs() -> List[str]:
    csv_paths = []
    for day in range(1, 8):
        day_str = f"{day:02d}"
        day_folder = os.path.join(DATA_ROOT, day_str)

        pattern = os.path.join(day_folder, "*_interesting_urls_experts_only*.csv")
        matches = glob.glob(pattern)

        if not matches:
            raise FileNotFoundError(f"No CSV found in {day_folder}")

        csv_paths.extend(matches)

    return csv_paths


# ================================
# LOAD URLS FROM ALL FILES
# ================================

def load_all_urls(csv_files: List[str]) -> List[str]:
    urls = []

    for path in csv_files:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        if "url" not in df.columns:
            raise ValueError(f"'url' column missing in {path}")

        file_urls = [
            u.strip()
            for u in df["url"].dropna().astype(str).tolist()
            if u.strip()
        ]

        urls.extend(file_urls)

    # Optional: remove duplicates across days
    urls = list(dict.fromkeys(urls))

    return urls


# ================================
# MAIN RUNNER
# ================================

def main():

    csv_files = collect_week_csvs()
    print(f"Found {len(csv_files)} daily CSV files.")

    urls = load_all_urls(csv_files)
    total = len(urls)

    print(f"Total unique URLs to process: {total}")
    print(f"Using model={DEFAULT_MODEL} with up to {MAX_WORKERS} workers\n")

    if os.path.exists(OUTPUT_JSONL):
        os.remove(OUTPUT_JSONL)

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    def worker(u: str) -> Dict[str, Any]:
        rec = extract_from_url_llm_single_pass(u, model=DEFAULT_MODEL)
        return rec.__dict__

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(worker, u): u for u in urls}

        for fut in tqdm(as_completed(future_to_url), total=total, desc="Extracting"):
            url = future_to_url[fut]

            try:
                data = fut.result()
                results.append(data)

                with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")

            except Exception as e:
                errors.append({"url": url, "error": str(e)})

    if results:
        pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"\nSaved combined database to:")
        print(f"  {OUTPUT_JSONL}")
        print(f"  {OUTPUT_CSV}")

    if errors:
        pd.DataFrame(errors).to_csv(ERROR_CSV, index=False, encoding="utf-8")
        print(f"\nSome URLs failed. See:")
        print(f"  {ERROR_CSV}")
    else:
        print("\nNo errors recorded.")


if __name__ == "__main__":
    main()