from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from acled_auth import ACLEDAuth
from acled_client import ACLEDClient, ACLEDClientConfig


ACLED_ROOT = Path(__file__).resolve().parents[0]
COUNTRIES_PATH = ACLED_ROOT / "data" / "reference" / "acled_countries.csv"
RAW_DIR = ACLED_ROOT / "data" / "raw" / "events"
CACHE_DIR = ACLED_ROOT / ".cache"

RAW_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "event_id_cnty", "event_date", "country", "iso3",
    "admin1", "admin2", "location",
    "event_type", "sub_event_type",
    "actor1", "actor2", "assoc_actor_1", "assoc_actor_2",
    "fatalities", "latitude", "longitude",
    "geo_precision", "time_precision", "source_scale",
]


def load_country_iso3_list() -> List[Tuple[str, str]]:
    """Load (country_name, ISO3) pairs from the reference CSV. Handles several ISO3 column name variants."""
    if not COUNTRIES_PATH.exists():
        raise FileNotFoundError(f"Missing: {COUNTRIES_PATH}")

    df = pd.read_csv(COUNTRIES_PATH, sep=",", encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    if "country" not in df.columns:
        raise ValueError(f"{COUNTRIES_PATH} missing 'country' column. Found: {list(df.columns)}")

    iso_col = None
    for c in df.columns:
        if c.lower() in {"iso3", "iso_3", "iso"}:
            iso_col = c
            break
    if iso_col is None:
        if "iso3" in df.columns:
            iso_col = "iso3"
        else:
            for c in df.columns:
                if c.lower() == "iso3":
                    iso_col = c
                    break

    if iso_col is None and "iso3" not in df.columns:
        for c in df.columns:
            if c.strip().lower() == "iso3":
                iso_col = c
                break
    if iso_col is None:
        for c in pd.read_csv(COUNTRIES_PATH, nrows=1, encoding="utf-8-sig").columns:
            if c.strip() == "ISO3":
                iso_col = "ISO3"
                break
    if iso_col is None:
        raise ValueError(f"{COUNTRIES_PATH} missing ISO3 column (e.g. 'ISO3'). Columns: {list(df.columns)}")

    if iso_col == "ISO3" and "iso3" in df.columns:
        iso_col = "iso3"

    out = []
    for _, r in df.iterrows():
        country = str(r["country"]).strip()
        iso3 = str(r[iso_col]).strip().upper()
        if country and country.lower() != "nan" and iso3 and iso3.lower() != "nan":
            out.append((country, iso3))
    if not out:
        raise ValueError("Country list is empty after parsing.")
    return out


def year_windows(start_year: int, end: pd.Timestamp):
    for y in range(start_year, end.year + 1):
        a = pd.Timestamp(f"{y}-01-01")
        b = pd.Timestamp(f"{y}-12-31")
        if b > end:
            b = end
        yield y, a, b


def out_paths(iso3: str, year: int) -> Tuple[Path, Path]:
    iso3 = iso3.upper().strip()
    out_dir = RAW_DIR / f"iso3={iso3}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"year={year}.parquet", out_dir / f"year={year}.csv"


def get_credentials() -> Tuple[str, str]:
    email = os.environ.get("ACLED_EMAIL")
    password = os.environ.get("ACLED_PASSWORD")
    if not email or not password:
        raise RuntimeError(
            "Missing ACLED_EMAIL / ACLED_PASSWORD env vars.\n"
            "PowerShell:\n"
            "  setx ACLED_EMAIL \"you@...\"\n"
            "  setx ACLED_PASSWORD \"...\"\n"
            "Then restart the terminal."
        )
    return email, password


@dataclass(frozen=True)
class Task:
    country: str
    iso3: str
    year: int
    start_date: str
    end_date: str


def build_tasks(start_year: int, end: pd.Timestamp) -> List[Task]:
    tasks: List[Task] = []
    for country, iso3 in load_country_iso3_list():
        for year, a, b in year_windows(start_year, end):
            tasks.append(
                Task(
                    country=country,
                    iso3=iso3,
                    year=year,
                    start_date=a.strftime("%Y-%m-%d"),
                    end_date=b.strftime("%Y-%m-%d"),
                )
            )
    return tasks


def fetch_and_save_one(task: Task, client: ACLEDClient, force: bool = False) -> str:
    pq_path, csv_path = out_paths(task.iso3, task.year)
    if pq_path.exists() and not force:
        return f"[skip] {task.iso3} {task.year} exists"

    events = client.fetch_events(
        countries=[task.country],
        start_date=task.start_date,
        end_date=task.end_date,
        fields=FIELDS,
        use_cache=True,
        force_refresh=False,
        show_progress=False,
    )

    if events.empty:
        return f"[none] {task.iso3} {task.year} (no events)"

    if "iso3" not in events.columns:
        events["iso3"] = task.iso3
    else:
        events["iso3"] = events["iso3"].replace({None: task.iso3}).fillna(task.iso3)
        events["iso3"] = events["iso3"].astype(str).str.upper().str.strip()
        if events["iso3"].eq("").all():
            events["iso3"] = task.iso3

    events.to_parquet(pq_path, index=False)
    events.to_csv(csv_path, index=False)
    return f"[ok] {task.iso3} {task.year} rows={len(events):,}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_year", type=int, default=2017)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="Rewrite files even if already present.")
    args = ap.parse_args()

    print("\n=== ACLED backfill (2017 -> today) ===")
    print(f"ACLED_ROOT:      {ACLED_ROOT}")
    print(f"COUNTRIES_PATH:  {COUNTRIES_PATH}")
    print(f"RAW_DIR:         {RAW_DIR}")
    print(f"CACHE_DIR:       {CACHE_DIR}")
    print(f"start_year={args.start_year} workers={args.workers} force={args.force}\n")

    email, password = get_credentials()
    auth = ACLEDAuth(email=email, password=password)
    client = ACLEDClient(auth, ACLEDClientConfig(cache_dir=str(CACHE_DIR)))

    end = pd.Timestamp(date.today())
    tasks = build_tasks(args.start_year, end)
    print(f"Total tasks (country-year): {len(tasks):,}")

    ok = skip = none = fail = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(fetch_and_save_one, t, client, args.force): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), start=1):
            t = futs[fut]
            try:
                msg = fut.result()
                print(msg)
                if msg.startswith("[ok]"):
                    ok += 1
                elif msg.startswith("[skip]"):
                    skip += 1
                elif msg.startswith("[none]"):
                    none += 1
            except Exception as e:
                fail += 1
                print(f"[FAIL] {t.iso3} {t.year} {t.country} -> {e}")
            if i % 200 == 0 or i == len(tasks):
                print(f"Progress: {i:,}/{len(tasks):,} | ok={ok} skip={skip} none={none} fail={fail}")

    print(f"\nDONE. ok={ok} skip={skip} none={none} fail={fail}")


if __name__ == "__main__":
    main()
