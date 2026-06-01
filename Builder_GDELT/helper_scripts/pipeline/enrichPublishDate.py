"""
Enrich extractions.jsonl with publish_date and event_date.

Priority for publish_date (when null):
  1. URL-based date extraction  (e.g. /2024/12/31/article-name/)
  2. GDELT dateadded            (ingest date  -  closest available proxy)

Priority for event_date (when null):
  1. GDELT sqldate              (GDELT's detected event date)

Source of sqldate / dateadded:
    data/urls/{YYYYMMDD}.csv   -  columns url_normalized, ..., sqldate, dateadded
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd


URLS_DIR = Path("data/urls")

# URL date patterns, tried in order (most specific first)
_URL_DATE_PATTERNS = [
    re.compile(r'/(\d{4})/(\d{2})/(\d{2})/'),          # /YYYY/MM/DD/
    re.compile(r'/(\d{4})/(\d{2})/(\d{2})[/_\-]'),     # /YYYY/MM/DD_ or -
    re.compile(r'[/_\-](\d{4})(\d{2})(\d{2})[/_\-]'),  # _YYYYMMDD_
    re.compile(r'[/_\-](\d{4})(\d{2})(\d{2})$'),       # _YYYYMMDD at end
    re.compile(r'/(\d{4})/(\d{2})/'),                   # /YYYY/MM/ (day=01)
]


def _extract_date_from_url(url: str) -> Optional[str]:
    """Try to extract a publish date from a URL path. Returns YYYY-MM-DD or None."""
    if not url:
        return None
    for pat in _URL_DATE_PATTERNS:
        m = pat.search(url)
        if m:
            groups = m.groups()
            yyyy, mm = groups[0], groups[1]
            dd = groups[2] if len(groups) > 2 else "01"
            if 2000 <= int(yyyy) <= 2035 and 1 <= int(mm) <= 12 and 1 <= int(dd) <= 31:
                return f"{yyyy}-{mm}-{dd}"
    return None


def _to_iso_date(val) -> Optional[str]:
    """Convert GDELT sqldate (YYYYMMDD) or dateadded (YYYYMMDDhhmmss) to YYYY-MM-DD."""
    try:
        s = str(int(float(val)))
        if len(s) < 8:
            return None
        return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    except (ValueError, TypeError):
        return None


def _build_url_gdelt_map(yyyymmdd: str) -> dict[str, dict]:
    """Return {url: {sqldate, dateadded}} from data/urls/{YYYYMMDD}.csv."""
    urls_file = URLS_DIR / f"{yyyymmdd}.csv"
    if not urls_file.exists():
        print(f"[enrichPublishDate] URLs file not found: {urls_file}")
        return {}

    df = pd.read_csv(urls_file, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    if "sqldate" not in df.columns or "dateadded" not in df.columns:
        print(f"[enrichPublishDate] sqldate/dateadded columns not found in {urls_file} — skipping GDELT enrichment")
        return {}

    url_col = "url_normalized" if "url_normalized" in df.columns else df.columns[0]

    url_map: dict[str, dict] = {}
    for _, row in df.iterrows():
        url = row.get(url_col)
        if pd.isna(url) or not url:
            continue
        url_map[str(url).strip()] = {
            "sqldate":   _to_iso_date(row.get("sqldate")),
            "dateadded": _to_iso_date(row.get("dateadded")),
        }
    return url_map


def enrich_publish_dates(raw_path: Path, yyyymmdd: str) -> int:
    """
    Fill missing publish_date and event_date in extractions.jsonl.

    publish_date: URL parsing -> GDELT dateadded
    event_date:   GDELT sqldate

    Overwrites extractions.jsonl and extractions.csv in place.
    Returns total number of fields filled across all records.
    """
    url_map = _build_url_gdelt_map(yyyymmdd)

    records: list[dict] = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    pub_from_url = pub_from_gdelt = event_from_gdelt = 0

    for rec in records:
        url = rec.get("url", "")
        gdelt = url_map.get(url, {})

        # --- publish_date ---
        if not rec.get("publish_date"):
            date = _extract_date_from_url(url)
            if date:
                rec["publish_date"] = date
                pub_from_url += 1
            elif gdelt.get("dateadded"):
                rec["publish_date"] = gdelt["dateadded"]
                pub_from_gdelt += 1

        # --- event_date ---
        if not rec.get("event_date"):
            sqldate = gdelt.get("sqldate")
            if sqldate:
                rec["event_date"] = sqldate
                event_from_gdelt += 1

    # Overwrite jsonl
    with open(raw_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    pub_total = pub_from_url + pub_from_gdelt
    total_filled = pub_total + event_from_gdelt
    print(
        f"[enrichPublishDate] {yyyymmdd}: "
        f"publish_date filled {pub_total} (url={pub_from_url}, gdelt_dateadded={pub_from_gdelt}), "
        f"event_date filled {event_from_gdelt} "
        f"-- {total_filled} fields across {len(records)} records."
    )
    return total_filled
