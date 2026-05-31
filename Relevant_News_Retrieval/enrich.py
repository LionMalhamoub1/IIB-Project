from __future__ import annotations

import csv
import random
import re
import sys
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue, Empty
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from bs4 import BeautifulSoup
from tqdm import tqdm


# =========================
# CONFIG
# =========================

BASE_DIR = Path("data/interim/gdelt_event_context_daily")
OUTPUT_SUFFIX = "_enriched.csv"

USER_AGENT = "Mozilla/5.0 (compatible; LithiumQRA/1.0)"
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TIMEOUT_S = 20
MAX_RETRIES = 2
BACKOFF_BASE_S = 0.6

SLEEP_BETWEEN_REQ = (0.05, 0.15)
MAX_WORKERS = 40  # tune based on your machine / error rate

MAX_TITLE_CHARS = 300
MAX_DESC_CHARS = 800

# Cache writer flush behavior
CACHE_QUEUE_MAXSIZE = 50_000


# =========================
# HELPERS
# =========================

TRACKING_PARAMS = {
    "gclid", "fbclid", "mc_cid", "mc_eid", "igshid", "spm", "ref", "ref_src"
}

def normalize_url(url: str) -> str:
    """Normalize URLs to improve caching/dedup."""
    url = (url or "").strip()
    if not url:
        return url
    try:
        p = urlparse(url)
        query = [
            (k, v)
            for (k, v) in parse_qsl(p.query, keep_blank_values=True)
            if k.lower() not in TRACKING_PARAMS and not k.lower().startswith("utm_")
        ]
        scheme = p.scheme or "http"
        netloc = (p.netloc or "").lower()
        path = p.path or ""
        return urlunparse((scheme, netloc, path, p.params, urlencode(query, doseq=True), ""))
    except Exception:
        return url

def truncate(s: Optional[str], n: int) -> str:
    s = (s or "").strip()
    return s[:n] if len(s) > n else s

def looks_like_xml(resp: requests.Response) -> bool:
    """Detect XML content for parsing robustness."""
    ct = (resp.headers.get("Content-Type") or "").lower()
    if "xml" in ct or "rss" in ct or "atom" in ct:
        return True
    head = (resp.text or "").lstrip()[:300].lower()
    return (
        head.startswith("<?xml")
        or head.startswith("<rss")
        or head.startswith("<feed")
        or head.startswith("<sitemapindex")
        or head.startswith("<urlset")
    )

def safe_get_meta_description(soup: BeautifulSoup) -> str:
    """Try common description sources in a sensible order."""
    # Standard description
    tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if tag and tag.get("content"):
        return tag.get("content", "")

    # OG description (common)
    tag = soup.find("meta", attrs={"property": re.compile(r"^og:description$", re.I)})
    if tag and tag.get("content"):
        return tag.get("content", "")

    # Twitter description (sometimes best)
    tag = soup.find("meta", attrs={"name": re.compile(r"^twitter:description$", re.I)})
    if tag and tag.get("content"):
        return tag.get("content", "")

    return ""

def read_csv_rows(path: Path) -> List[dict]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv_rows(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# =========================
# THREAD-LOCAL SESSION
# =========================

_thread_local = threading.local()

def get_thread_session() -> requests.Session:
    """One session per thread (requests.Session is NOT thread-safe)."""
    s = getattr(_thread_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers.update(HEADERS)
        adapter = HTTPAdapter(
            pool_connections=MAX_WORKERS,
            pool_maxsize=MAX_WORKERS,
            max_retries=0,
        )
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _thread_local.session = s
    return s


# =========================
# CACHE WRITER THREAD
# =========================

CACHE_FIELDS = ["url_normalized", "title", "meta_description", "http_status", "fetch_error"]

class CacheWriter:
    """Single writer thread to avoid corrupted CSV writes from many workers."""
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.queue: Queue[dict] = Queue(maxsize=CACHE_QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        # push a sentinel
        self.queue.put({"__stop__": "1"})
        self._thread.join(timeout=10)

    def submit(self, row: dict) -> None:
        if not self._started:
            raise RuntimeError("CacheWriter not started")
        self.queue.put(row)

    def _run(self) -> None:
        write_header = not self.cache_path.exists()
        f = open(self.cache_path, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
        if write_header:
            w.writeheader()
            f.flush()

        try:
            while not self._stop.is_set():
                try:
                    item = self.queue.get(timeout=0.5)
                except Empty:
                    continue
                if item.get("__stop__") == "1":
                    break
                w.writerow({k: item.get(k, "") for k in CACHE_FIELDS})
                # Flush periodically (cheap insurance)
                if random.random() < 0.02:
                    f.flush()
        finally:
            f.flush()
            f.close()


# =========================
# FETCH
# =========================

def fetch_title_meta(url: str) -> Tuple[str, str, int, str]:
    """Fetch title + meta description from URL with retries/backoff."""
    session = get_thread_session()

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=TIMEOUT_S, allow_redirects=True)
            status = resp.status_code

            if status != 200 or not resp.text:
                # retry certain transient statuses
                if status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_BASE_S * (attempt + 1))
                    continue
                return "", "", status, f"bad_status:{status}"

            parser = "xml" if looks_like_xml(resp) else "lxml"
            soup = BeautifulSoup(resp.text, parser)

            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string

            desc = safe_get_meta_description(soup)

            return truncate(title, MAX_TITLE_CHARS), truncate(desc, MAX_DESC_CHARS), status, ""

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_BASE_S * (attempt + 1))
                continue
            return "", "", 0, f"exception:{type(e).__name__}"

    return "", "", 0, "failed_after_retries"


# =========================
# PROCESSING
# =========================

def load_cache(cache_path: Path) -> Dict[str, dict]:
    """Load prior cached results for this run-date."""
    cache: Dict[str, dict] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            u = (r.get("url_normalized") or "").strip()
            if u:
                cache[u] = r
    return cache

def load_existing_progress(out_path: Path) -> Dict[str, dict]:
    """Resume: if output exists, reuse successful rows."""
    existing: Dict[str, dict] = {}
    if not out_path.exists():
        return existing
    with open(out_path, "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            u = (r.get("url_normalized") or "").strip()
            if u:
                existing[u] = r
    return existing

def process_row(
    row: dict,
    cache: Dict[str, dict],
    existing: Dict[str, dict],
    cache_writer: CacheWriter,
) -> dict:
    url = (row.get("sourceurl") or "").strip()
    url_norm = normalize_url(url)
    row["url_normalized"] = url_norm

    # Resume: keep already-successful results
    if url_norm in existing and (existing[url_norm].get("http_status") == "200"):
        return existing[url_norm]

    # Non-http
    if not url_norm.startswith("http"):
        row.update({"title": "", "meta_description": "", "http_status": "", "fetch_error": "non_http"})
        return row

    # Cached
    if url_norm in cache:
        c = cache[url_norm]
        row.update(
            {
                "title": c.get("title", ""),
                "meta_description": c.get("meta_description", ""),
                "http_status": str(c.get("http_status", "")),
                "fetch_error": c.get("fetch_error", ""),
            }
        )
        return row

    # gentle jitter to reduce burstiness
    time.sleep(random.uniform(*SLEEP_BETWEEN_REQ))

    title, desc, status, err = fetch_title_meta(url_norm)
    row.update({"title": title, "meta_description": desc, "http_status": str(status), "fetch_error": err})

    # save to cache (writer thread)
    cache_writer.submit(
        {
            "url_normalized": url_norm,
            "title": title,
            "meta_description": desc,
            "http_status": str(status),
            "fetch_error": err,
        }
    )
    return row

def enrich_file(in_path: Path, cache: Dict[str, dict], cache_writer: CacheWriter) -> None:
    # Output next to input, remove "_filtered" from stem (if present), then add suffix.
    out_path = in_path.with_name(in_path.stem.replace("_filtered", "") + OUTPUT_SUFFIX)

    if out_path.exists():
        print(f"Skipping (already enriched): {out_path.name}")
        return

    rows = read_csv_rows(in_path)
    if not rows:
        return

    existing = load_existing_progress(out_path)

    # fieldnames: preserve input order + append new fields (dedup)
    fieldnames = list(rows[0].keys())
    for extra in ["url_normalized", "title", "meta_description", "http_status", "fetch_error"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    start = time.time()

    results: List[dict] = [None] * len(rows)  # stable ordering
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {
            ex.submit(process_row, rows[i], cache, existing, cache_writer): i
            for i in range(len(rows))
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Enriching {in_path.parent.name}/{in_path.name}"):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                # Hard fail on a row shouldn't kill the file
                r = rows[idx]
                r.update({"title": "", "meta_description": "", "http_status": "0", "fetch_error": f"exception:{type(e).__name__}"})
                results[idx] = r

    write_csv_rows(out_path, fieldnames, results)

    elapsed = time.time() - start
    rps = (len(rows) / elapsed) if elapsed > 0 else 0.0
    print(f"\n--- ENRICH REPORT: {in_path.name} ---")
    print(f"Rows: {len(rows)} | Time: {elapsed:.1f}s | Throughput: {rps:.2f} rows/s")
    print(f"Saved: {out_path}\n")

def main(target_date: str) -> None:
    # One cache per date (matches your original pattern)
    cache_path = Path(f"data/interim/_state/url_title_meta_cache_{target_date}.csv")
    cache = load_cache(cache_path)

    # Find inputs recursively
    files = list(BASE_DIR.rglob(f"*{target_date}*_deduped_filtered.csv"))
    if not files:
        print(f"No filtered files found for {target_date} under {BASE_DIR}")
        return

    cache_writer = CacheWriter(cache_path)
    cache_writer.start()

    try:
        for fp in files:
            enrich_file(fp, cache, cache_writer)
            # refresh in-memory cache occasionally so later files benefit immediately
            # (lightweight: reload only if the cache is growing a lot is overkill; this is fine)
            cache.update(load_cache(cache_path))
    finally:
        cache_writer.stop()

if __name__ == "__main__":
    day = input("Enter date (YYYYMMDD): ").strip()
    if not day:
        print("No date provided.")
        sys.exit(1)
    main(day)