"""
backfill_sqldate_dateadded.py
=============================
One-off script that backfills `sqldate` and `dateadded` columns into every
URL CSV under data/urls/ that is currently missing them.

For each such date it streams the GDELT 15-minute export zips directly from
  http://data.gdeltproject.org/gdeltv2/YYYYMMDDHHMMSS.export.CSV.zip
keeping only rows whose sourceurl (normalised) matches a URL we already have.
Nothing is written to disk beyond the updated URL CSV.

Run from the project root:
    python backfill_sqldate_dateadded.py
"""

from __future__ import annotations

import csv
import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[1]
URLS_DIR = ROOT / "data" / "urls"

GDELT_BASE = "http://data.gdeltproject.org/gdeltv2"

# ---------------------------------------------------------------------------
# URL normalisation  (mirrors enrichv3.py)
# ---------------------------------------------------------------------------
_TRACKING = {"gclid", "fbclid", "mc_cid", "mc_eid", "igshid", "spm", "ref", "ref_src"}

def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return url
    try:
        p = urlparse(url)
        query = [
            (k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
            if k.lower() not in _TRACKING and not k.lower().startswith("utm_")
        ]
        return urlunparse((
            p.scheme or "http",
            (p.netloc or "").lower(),
            p.path or "",
            p.params,
            urlencode(query, doseq=True),
            "",
        ))
    except Exception:
        return url


# ---------------------------------------------------------------------------
# GDELT helpers
# ---------------------------------------------------------------------------

def _15min_urls(date_str: str) -> list[str]:
    """Return all 96 GDELT 15-minute export URLs for a given YYYYMMDD."""
    urls = []
    base = datetime.strptime(date_str, "%Y%m%d")
    t = base
    while t.date() == base.date():
        urls.append(f"{GDELT_BASE}/{t.strftime('%Y%m%d%H%M%S')}.export.CSV.zip")
        t += timedelta(minutes=15)
    return urls


def _fetch_url_lookup(date_str: str, target_urls: set[str]) -> dict[str, tuple[str, str]]:
    """
    Download all 15-min GDELT zips for date_str and return a dict:
        normalised_sourceurl -> (sqldate, dateadded)
    Only rows whose normalised sourceurl is in target_urls are kept.
    """
    lookup: dict[str, tuple[str, str]] = {}
    zip_urls = _15min_urls(date_str)

    for zip_url in tqdm(zip_urls, desc=f"  {date_str}", leave=False):
        try:
            r = requests.get(zip_url, timeout=60, stream=True)
            if r.status_code == 404:
                continue
            r.raise_for_status()

            buf = io.BytesIO(r.content)
            with zipfile.ZipFile(buf) as zf:
                inner = zf.namelist()[0]
                with zf.open(inner) as f_in:
                    reader = csv.reader(
                        io.TextIOWrapper(f_in, encoding="utf-8", errors="replace"),
                        delimiter="\t",
                    )
                    for row in reader:
                        if len(row) < 10:
                            continue
                        sourceurl = row[-1].strip()
                        if not sourceurl:
                            continue
                        norm = normalize_url(sourceurl)
                        if norm not in target_urls:
                            continue
                        if norm in lookup:
                            continue  # keep first occurrence
                        sqldate   = row[1].strip()   if len(row) > 1  else ""
                        dateadded = row[-2].strip()  if len(row) > 1  else ""
                        lookup[norm] = (sqldate, dateadded)

        except Exception as e:
            tqdm.write(f"    [WARN] {zip_url.split('/')[-1]}: {e}")
            continue

    return lookup


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    csv_files = sorted(URLS_DIR.glob("*.csv"))

    # Find files that need backfilling
    to_backfill: list[Path] = []
    for f in csv_files:
        try:
            cols = pd.read_csv(f, nrows=0).columns.tolist()
        except Exception:
            continue
        if "sqldate" not in cols or "dateadded" not in cols:
            to_backfill.append(f)

    print(f"Found {len(to_backfill)} URL CSVs missing sqldate/dateadded.\n")

    for url_csv in to_backfill:
        date_str = url_csv.stem  # filename is YYYYMMDD.csv
        print(f"Processing {date_str} ...")

        df = pd.read_csv(url_csv)
        df.columns = [c.strip().lower() for c in df.columns]

        if "url_normalized" not in df.columns:
            print(f"  [SKIP] no url_normalized column in {url_csv.name}")
            continue

        target_urls = set(df["url_normalized"].dropna().str.strip())
        lookup = _fetch_url_lookup(date_str, target_urls)

        matched = len([u for u in target_urls if u in lookup])
        print(f"  Matched {matched}/{len(target_urls)} URLs from GDELT")

        df["sqldate"]   = df["url_normalized"].map(lambda u: lookup.get(u, ("", ""))[0])
        df["dateadded"] = df["url_normalized"].map(lambda u: lookup.get(u, ("", ""))[1])

        df.to_csv(url_csv, index=False)
        print(f"  Saved: {url_csv.name}\n")

    print("Done.")


if __name__ == "__main__":
    main()
