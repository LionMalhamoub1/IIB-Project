"""
split_by_classification_agreement.py

For each yyyymmdd/by_type/{type} folder (excluding floods), reads extractions.jsonl
and writes two new files alongside it:
  - extractions_agreed.jsonl    : records where llm_disruption_type == expert_disruption_type
  - extractions_disagreed.jsonl : records where they differ (or one is missing),
                                  with "url_status" as the first field (e.g. 200, 404, "error")
"""

import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "daily"
SKIP_TYPES = {"flood"}
INPUT_FILE = "extractions.jsonl"

REQUEST_TIMEOUT = 10
MAX_WORKERS = 10

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def check_url_status(url: str) -> int | str:
    """Return HTTP status code, or 'error' if the request fails."""
    try:
        resp = requests.head(url, timeout=REQUEST_TIMEOUT, headers=HEADERS, allow_redirects=True)
        # Some servers reject HEAD  -  fall back to GET
        if resp.status_code in (405, 403):
            resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=HEADERS, stream=True)
        return resp.status_code
    except requests.RequestException:
        return "error"


def fetch_statuses(records: list[dict]) -> list[dict]:
    """Add url_status as the first field in each record, fetched concurrently."""
    urls = [r.get("url", "") for r in records]

    statuses: dict[str, int | str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_url = {pool.submit(check_url_status, url): url for url in urls if url}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            statuses[url] = future.result()

    enriched = []
    for record in records:
        url = record.get("url", "")
        status = statuses.get(url, "error") if url else "no_url"
        enriched.append({"url_status": status, **record})

    return enriched


def split_day_type(folder: Path) -> tuple[int, int]:
    """Split a single by_type folder. Returns (agreed_count, disagreed_count)."""
    input_path = folder / INPUT_FILE
    if not input_path.exists():
        return 0, 0

    agreed = []
    disagreed = []

    with input_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping malformed line in {input_path}: {e}")
                continue
            llm = record.get("llm_disruption_type")
            expert = record.get("expert_disruption_type")

            if llm and expert and llm == expert:
                agreed.append(record)
            else:
                disagreed.append(record)

    disagreed = fetch_statuses(disagreed)

    agreed_path = folder / "extractions_agreed.jsonl"
    disagreed_path = folder / "extractions_disagreed.jsonl"

    with agreed_path.open("w", encoding="utf-8") as f:
        for r in agreed:
            f.write(json.dumps(r) + "\n")

    with disagreed_path.open("w", encoding="utf-8") as f:
        for r in disagreed:
            f.write(json.dumps(r) + "\n")

    return len(agreed), len(disagreed)


def run_split(results_root: Path = RESULTS_ROOT):
    if not results_root.exists():
        print(f"Results root not found: {results_root}")
        return

    day_dirs = sorted(d for d in results_root.iterdir() if d.is_dir())
    if not day_dirs:
        print("No daily folders found.")
        return

    total_agreed = 0
    total_disagreed = 0
    processed = 0

    for day_dir in day_dirs:
        by_type_dir = day_dir / "by_type"
        if not by_type_dir.exists():
            continue

        for type_dir in sorted(by_type_dir.iterdir()):
            if not type_dir.is_dir():
                continue
            if type_dir.name in SKIP_TYPES:
                continue

            agreed, disagreed = split_day_type(type_dir)
            if agreed + disagreed == 0:
                continue

            print(f"  {day_dir.name}/{type_dir.name}: {agreed} agreed, {disagreed} disagreed")
            total_agreed += agreed
            total_disagreed += disagreed
            processed += 1

    print(f"\nDone. Processed {processed} type-folders.")
    print(f"Total agreed:    {total_agreed}")
    print(f"Total disagreed: {total_disagreed}")


if __name__ == "__main__":
    run_split()
