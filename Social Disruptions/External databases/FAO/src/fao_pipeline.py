"""FAO Food Price Index Pipeline

Downloads and processes the FAO Food Price Index (FFPI) — a global benchmark
for international food commodity prices updated monthly.

Unlike the country-level CPI in the inflation pipeline, the FFPI is a single
global series (no country dimension). It is merged into the modelling panel
by date only, broadcasting the same value to all countries.

Indices (base period 2014-2016 = 100):
    fao_food_index    – overall Food Price Index
    fao_meat_index    – Meat
    fao_dairy_index   – Dairy
    fao_cereals_index – Cereals
    fao_oils_index    – Vegetable Oils
    fao_sugar_index   – Sugar

YoY (12-month % change) variants are also computed for each.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import calendar
from datetime import date

import pandas as pd
import requests


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
START_YEAR = 2017
END_YEAR   = 2025

FAO_ROOT      = Path(__file__).resolve().parents[1]
RAW_DIR       = FAO_ROOT / "data" / "raw"
PROCESSED_DIR = FAO_ROOT / "data" / "processed"
RAW_FILE      = RAW_DIR / "fao_food_price_indices.csv"

FAO_BASE_URL = (
    "https://www.fao.org/media/docs/worldfoodsituationlibraries/"
    "default-document-library/food_price_indices_data_csv_{month}.csv"
)

OUTPUT_STEM = f"fao_food_price_monthly_{START_YEAR}_{END_YEAR}"
OUT_PARQUET = PROCESSED_DIR / f"{OUTPUT_STEM}.parquet"
OUT_CSV     = PROCESSED_DIR / f"{OUTPUT_STEM}.csv"

# Flexible column matching: substring (lower) -> output name
COL_MAP = {
    "food price index": "fao_food_index",
    "meat":             "fao_meat_index",
    "dairy":            "fao_dairy_index",
    "cereal":           "fao_cereals_index",
    "oil":              "fao_oils_index",
    "sugar":            "fao_sugar_index",
}


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────
def _candidate_urls() -> list[str]:
    """
    FAO publishes the CSV with a month abbreviation in the filename (e.g. _mar, _feb).
    Try the current month and fall back to the two prior months.
    """
    today = date.today()
    candidates = []
    for delta in range(3):
        month_num = (today.month - delta - 1) % 12 + 1
        month_abbr = calendar.month_abbr[month_num].lower()
        candidates.append(FAO_BASE_URL.format(month=month_abbr))
    return candidates


def download_raw(force: bool = False) -> Path:
    """Download FAO FFPI CSV if not already present, trying recent months."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and not force:
        logger.info("Raw file already present: %s", RAW_FILE)
        return RAW_FILE

    for url in _candidate_urls():
        logger.info("Trying: %s", url)
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            RAW_FILE.write_bytes(resp.content)
            logger.info("Saved: %s  (%d bytes)", RAW_FILE, len(resp.content))
            return RAW_FILE
        except requests.HTTPError as exc:
            logger.warning("Failed (%s) — trying next month.", exc)

    raise RuntimeError(
        "Could not download FAO FFPI CSV for any recent month. "
        "Check https://www.fao.org/worldfoodsituation/foodpricesindex/en/ manually."
    )


# ─────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────
def _parse_raw(path: Path) -> pd.DataFrame:
    """Parse FAO FFPI CSV into a tidy DataFrame with one row per month.

    FAO CSV format (as of 2025):
        Row 0: title  ("FAO Food Price Index")
        Row 1: note   ("2014-2016=100")
        Row 2: header (Date, Food Price Index, Meat, Dairy, Cereals, Oils, Sugar, ...)
        Row 3+: data  (Date in YYYY-MM format)
    """
    df = pd.read_csv(path, skiprows=2, encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(how="all")

    logger.info("Raw columns: %s", list(df.columns))

    # Date column is "YYYY-MM" format
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    if date_col is None:
        raise ValueError(f"Could not find 'Date' column. Found: {list(df.columns)}")

    df["date"] = pd.to_datetime(df[date_col].astype(str).str.strip() + "-01")
    df = df.dropna(subset=["date"])

    # Map FAO column names to standardised output names
    rename: dict[str, str] = {}
    for pattern, out_name in COL_MAP.items():
        matched = next(
            (c for c in df.columns if pattern in c.lower() and c != date_col), None
        )
        if matched:
            rename[matched] = out_name
        else:
            logger.warning("Column matching %r not found — will be absent from output.", pattern)

    df = df.rename(columns=rename)

    index_cols = [c for c in df.columns if c.startswith("fao_")]
    if not index_cols:
        raise ValueError(
            "No FAO index columns could be mapped. "
            f"Check COL_MAP against actual columns: {list(df.columns)}"
        )

    for col in index_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["date"] + index_cols].sort_values("date").reset_index(drop=True)


# ─────────────────────────────────────────────
# Panel construction
# ─────────────────────────────────────────────
def build_panel(path: Path | None = None) -> pd.DataFrame:
    """Load raw FFPI, filter to target date range, compute YoY % changes."""
    if path is None:
        path = RAW_FILE

    df = _parse_raw(path)

    df = df[
        (df["date"] >= pd.Timestamp(f"{START_YEAR}-01-01")) &
        (df["date"] <= pd.Timestamp(f"{END_YEAR}-12-31"))
    ].copy().reset_index(drop=True)

    index_cols = [c for c in df.columns if c.startswith("fao_")]

    # 12-month YoY % change for each index
    for col in index_cols:
        df[f"{col}_yoy"] = df[col].pct_change(periods=12).mul(100).round(4)

    logger.info(
        "Panel built: %d months  |  columns: %s",
        len(df), [c for c in df.columns if c != "date"],
    )
    return df


# ─────────────────────────────────────────────
# Coverage report
# ─────────────────────────────────────────────
def _log_coverage(df: pd.DataFrame) -> None:
    logger.info("-" * 60)
    logger.info("Coverage report")
    logger.info("  Date range : %s to %s",
                df["date"].min().strftime("%Y-%m"),
                df["date"].max().strftime("%Y-%m"))
    logger.info("  Rows       : %d", len(df))
    for col in df.columns:
        if col == "date":
            continue
        n_obs  = df[col].notna().sum()
        n_miss = df[col].isna().sum()
        logger.info(
            "  %-35s  obs=%d  missing=%d (%.1f%%)",
            col, n_obs, n_miss, 100 * n_miss / len(df),
        )
    logger.info("-" * 60)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main() -> None:
    _setup_logging()

    logger.info("=" * 60)
    logger.info("FAO Food Price Index Pipeline")
    logger.info("Series : Food, Meat, Dairy, Cereals, Oils, Sugar  (monthly)")
    logger.info("Period : %d - %d", START_YEAR, END_YEAR)
    logger.info("=" * 60)

    raw_path = download_raw()
    panel    = build_panel(raw_path)
    _log_coverage(panel)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PARQUET, index=False)
    panel.to_csv(OUT_CSV, index=False)

    logger.info("Saved parquet -> %s", OUT_PARQUET)
    logger.info("Saved CSV     -> %s", OUT_CSV)
    logger.info("Done.")


if __name__ == "__main__":
    main()
