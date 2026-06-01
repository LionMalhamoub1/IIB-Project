from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd


START_YEAR = 2017
END_YEAR   = 2025

SERIES = {
    "food_cpi_inflation":   "fcpi_m",
    "energy_cpi_inflation": "ecpi_m",
}

INFL_ROOT    = Path(__file__).resolve().parents[1]
RAW_FILE     = INFL_ROOT / "data" / "raw"  / "Inflation-data.xlsx"
PROCESSED_DIR = INFL_ROOT / "data" / "processed"
OUT_PARQUET  = PROCESSED_DIR / f"cpi_inflation_monthly_{START_YEAR}_{END_YEAR}.parquet"
OUT_CSV      = PROCESSED_DIR / f"cpi_inflation_monthly_{START_YEAR}_{END_YEAR}.csv"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)


def _to_timestamp(col: object) -> pd.Timestamp | None:
    s = str(col).strip()
    if len(s) == 6 and s.isdigit():
        year, month = int(s[:4]), int(s[4:])
        if 1 <= month <= 12:
            return pd.Timestamp(year=year, month=month, day=1)
    return None


def _load_index_wide(xf: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    logger.info("  Reading sheet '%s' ...", sheet)
    raw = xf.parse(sheet, header=0, engine="openpyxl")

    raw = raw.dropna(how="all")

    if "Indicator Type" in raw.columns:
        raw = raw[raw["Indicator Type"].astype(str).str.strip() == "Index"]

    raw = raw.dropna(subset=["Country Code"])
    raw = raw[raw["Country Code"].astype(str).str.match(r"^[A-Za-z]{3}$")]

    date_map: dict[object, pd.Timestamp] = {}
    for col in raw.columns:
        ts = _to_timestamp(col)
        if ts is not None:
            date_map[col] = ts

    if not date_map:
        raise ValueError(f"No date columns detected in sheet '{sheet}'.")

    df = raw[["Country Code"] + list(date_map.keys())].copy()
    df = df.rename(columns={"Country Code": "country_iso3", **date_map})
    df["country_iso3"] = df["country_iso3"].str.strip().str.upper()

    df = df.drop_duplicates(subset=["country_iso3"], keep="first")

    logger.info(
        "    %d countries  |  date range: %s to %s",
        len(df),
        min(date_map.values()).strftime("%Y-%m"),
        max(date_map.values()).strftime("%Y-%m"),
    )
    return df


def _compute_yoy(wide: pd.DataFrame) -> pd.DataFrame:
    date_cols = sorted(
        [c for c in wide.columns if isinstance(c, pd.Timestamp)]
    )

    idx = wide.set_index("country_iso3")[date_cols]

    yoy = idx.pct_change(axis=1, periods=12).mul(100)

    target = [c for c in date_cols if START_YEAR <= c.year <= END_YEAR]
    yoy = yoy[target]

    long = (
        yoy.reset_index()
        .melt(id_vars="country_iso3", var_name="date", value_name="value")
        .dropna(subset=["date"])
        .sort_values(["country_iso3", "date"])
        .reset_index(drop=True)
    )
    return long


def build_panel(path: Path) -> pd.DataFrame:
    logger.info("Opening: %s", path)
    xf = pd.ExcelFile(path, engine="openpyxl")

    series_frames: list[pd.DataFrame] = []
    for col_name, sheet in SERIES.items():
        wide = _load_index_wide(xf, sheet)
        long = _compute_yoy(wide).rename(columns={"value": col_name})
        series_frames.append(long.set_index(["country_iso3", "date"]))

    panel = series_frames[0]
    for sf in series_frames[1:]:
        panel = panel.join(sf, how="outer")

    panel = (
        panel
        .reset_index()
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .sort_values(["country_iso3", "date"])
        .reset_index(drop=True)
    )
    return panel


def _log_coverage(df: pd.DataFrame) -> None:
    logger.info("-" * 60)
    logger.info("Coverage report")

    for col in ["food_cpi_inflation", "energy_cpi_inflation"]:
        if col not in df.columns:
            continue
        n_obs    = df[col].notna().sum()
        n_miss   = df[col].isna().sum()
        n_total  = len(df)
        miss_pct = n_miss / n_total * 100
        n_ctry   = df.loc[df[col].notna(), "country_iso3"].nunique()
        logger.info(
            "  %-26s  countries=%d  obs=%d  missing=%d (%.1f%%)",
            col, n_ctry, n_obs, n_miss, miss_pct,
        )

    logger.info(
        "  Earliest observation : %s",
        df["date"].min().strftime("%Y-%m"),
    )
    logger.info(
        "  Latest observation   : %s",
        df["date"].max().strftime("%Y-%m"),
    )
    logger.info("-" * 60)


def main() -> None:
    _setup_logging()

    logger.info("=" * 60)
    logger.info("World Bank Global Inflation Pipeline")
    logger.info("Series : Food CPI + Energy CPI  (monthly, YoY %%)")
    logger.info("Period : %d - %d", START_YEAR, END_YEAR)
    logger.info("=" * 60)

    if not RAW_FILE.exists():
        logger.error("Raw file not found: %s", RAW_FILE)
        logger.error(
            "Download 'Inflation-data.xlsx' from:\n"
            "  https://www.worldbank.org/en/research/brief/inflation-database\n"
            "and place it in:  %s", RAW_FILE.parent,
        )
        sys.exit(1)

    panel = build_panel(RAW_FILE)
    _log_coverage(panel)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PARQUET, index=False)
    panel.to_csv(OUT_CSV, index=False)

    logger.info("Saved parquet -> %s", OUT_PARQUET)
    logger.info("Saved CSV     -> %s", OUT_CSV)
    logger.info("Done.")


if __name__ == "__main__":
    main()
