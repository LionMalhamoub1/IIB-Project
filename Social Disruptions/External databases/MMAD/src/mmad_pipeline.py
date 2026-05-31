from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_SRC_DIR   = Path(__file__).resolve().parent
_MMAD_ROOT = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from mmad_client import MMADClient, MMADClientConfig  # noqa: E402

START_YEAR = 2017   
END_YEAR   = 2025
USE_CACHE  = True
LOG_LEVEL  = "INFO"

_CACHE_DIR = _MMAD_ROOT / ".cache"
_RAW_DIR   = _MMAD_ROOT / "data" / "raw"
_PROC_DIR  = _MMAD_ROOT / "data" / "processed"
_REF_DIR   = _MMAD_ROOT / "data" / "reference"

for _d in (_CACHE_DIR, _RAW_DIR, _PROC_DIR, _REF_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


logger = logging.getLogger(__name__)


def load_cow_iso3() -> pd.DataFrame:
    path = _REF_DIR / "cow_iso3.csv"
    if not path.exists():
        raise FileNotFoundError(f"COW→ISO3 reference not found at {path}")
    return pd.read_csv(path)


def add_iso3(df: pd.DataFrame) -> pd.DataFrame:
    mapping = load_cow_iso3()
    merged = df.merge(mapping, on="cowcode", how="left")
    n_unmapped = merged["iso3"].isna().sum()
    if n_unmapped:
        missing = sorted(df.loc[merged["iso3"].isna(), "cowcode"].dropna().astype(int).unique())
        logger.warning("%d rows have no ISO3 mapping. Unmapped cowcodes: %s", n_unmapped, missing)
    return merged


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce", utc=True)

    for col in ("latitude", "longitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("numparticipants", "avg_numparticipants"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "cowcode" in df.columns:
        df["cowcode"] = pd.to_numeric(df["cowcode"], errors="coerce")

    df = add_iso3(df)

    return df


def build_country_month_panel(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "event_date" not in df.columns or "cowcode" not in df.columns:
        return pd.DataFrame(columns=["iso3", "cowcode", "year", "month", "protest_count", "participant_sum"])

    valid = df.dropna(subset=["event_date", "cowcode"]).copy()
    valid["year"]  = valid["event_date"].dt.year.astype(int)
    valid["month"] = valid["event_date"].dt.month.astype(int)

    group_cols = ["iso3", "cowcode"] if "iso3" in valid.columns else ["cowcode"]

    agg = (
        valid.groupby(group_cols + ["year", "month"])
        .agg(
            protest_count   = ("id",              "count"),
            participant_sum = ("numparticipants", "sum"),
        )
        .reset_index()
    )

    agg = agg.sort_values(group_cols + ["year", "month"]).reset_index(drop=True)
    return agg


def save_raw(df: pd.DataFrame) -> None:
    csv_path     = _RAW_DIR / "mmad_reports.csv"
    parquet_path = _RAW_DIR / "mmad_reports.parquet"
    df.to_csv(csv_path,         index=False)
    df.to_parquet(parquet_path, index=False)
    logger.info("Raw saved → %s", _RAW_DIR)


def save_processed(df: pd.DataFrame, stem: str) -> None:
    csv_path     = _PROC_DIR / f"{stem}.csv"
    parquet_path = _PROC_DIR / f"{stem}.parquet"
    df.to_csv(csv_path,         index=False)
    df.to_parquet(parquet_path, index=False)
    logger.info("Processed saved → %s", _PROC_DIR / stem)


def main() -> None:
    setup_logging(LOG_LEVEL)

    logger.info("=" * 60)
    logger.info("MMAD Pipeline")
    logger.info("Year range : %d → %d", START_YEAR, END_YEAR)
    logger.info("Cache      : %s", "enabled" if USE_CACHE else "DISABLED")
    logger.info("=" * 60)

    cfg = MMADClientConfig(cache_dir=_CACHE_DIR)

    with MMADClient(cfg) as client:
        logger.info("Step 1 — Downloading MMAD reports.csv")
        raw = client.fetch_reports(use_cache=USE_CACHE)

    logger.info("Downloaded: %d rows × %d cols", *raw.shape)

    logger.info("Step 2 — Normalising")
    df = normalise(raw)

    if "event_date" in df.columns:
        df = df[
            df["event_date"].dt.year.between(START_YEAR, END_YEAR)
        ].reset_index(drop=True)
        logger.info("After year filter (%d–%d): %d rows", START_YEAR, END_YEAR, len(df))

    logger.info("Step 3 — Saving raw")
    save_raw(df)

    logger.info("Step 4 — Building country-month panel")
    panel = build_country_month_panel(df)
    logger.info("Panel: %d rows × %d cols", *panel.shape)

    logger.info("Step 5 — Saving processed outputs")
    save_processed(df,    "mmad_normalised")
    save_processed(panel, f"mmad_country_month_{START_YEAR}_{END_YEAR}")

    logger.info("=" * 60)
    logger.info(
        "Done.  Raw: %d rows  |  Panel: %d rows × %d cols",
        len(df), len(panel), len(panel.columns),
    )


if __name__ == "__main__":
    main()
