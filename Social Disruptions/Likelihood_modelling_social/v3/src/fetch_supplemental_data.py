"""
fetch_supplemental_data.py
==========================
Downloads and saves supplemental data needed for the v2 likelihood model:

  1. Global commodity prices  -> v2/data/interim/commodity_prices_daily.parquet
  2. Gini coefficients        -> v2/data/interim/gini_daily.parquet

Run standalone:
    python src/fetch_supplemental_data.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent   # v2/src
_V2   = _HERE.parent                      # v2
OUT_DIR = _V2 / "data" / "interim"

COMMODITY_FILE = OUT_DIR / "commodity_prices_daily.parquet"
GINI_FILE      = OUT_DIR / "gini_daily.parquet"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
START_DATE = "2016-01-01"
END_DATE   = "2022-12-31"

# 39 panel countries (ISO3)
PANEL_COUNTRIES_ISO3: list[str] = [
    "ARG", "AUS", "BOL", "BRA", "CAN", "CHL", "CHN", "DEU", "ESP", "FRA",
    "GBR", "GRC", "HUN", "IDN", "IND", "IRL", "ITA", "JPN", "KEN", "KOR",
    "LAO", "MAR", "MEX", "MOZ", "MYS", "NAM", "NLD", "NOR", "PER", "PHL",
    "POL", "PRT", "SWE", "THA", "TUR", "USA", "VNM", "ZAF", "ZWE",
]

# ISO3 -> ISO2 mapping for World Bank API
ISO3_TO_ISO2: dict[str, str] = {
    "ARG": "AR", "AUS": "AU", "BOL": "BO", "BRA": "BR", "CAN": "CA",
    "CHL": "CL", "CHN": "CN", "DEU": "DE", "ESP": "ES", "FRA": "FR",
    "GBR": "GB", "GRC": "GR", "HUN": "HU", "IDN": "ID", "IND": "IN",
    "IRL": "IE", "ITA": "IT", "JPN": "JP", "KEN": "KE", "KOR": "KR",
    "LAO": "LA", "MAR": "MA", "MEX": "MX", "MOZ": "MZ", "MYS": "MY",
    "NAM": "NA", "NLD": "NL", "NOR": "NO", "PER": "PE", "PHL": "PH",
    "POL": "PL", "PRT": "PT", "SWE": "SE", "THA": "TH", "TUR": "TR",
    "USA": "US", "VNM": "VN", "ZAF": "ZA", "ZWE": "ZW",
}

# Commodity tickers and their feature prefix
# (ticker, prefix, has_90d_pct)
COMMODITIES: list[tuple[str, str, bool]] = [
    ("HG=F",  "copper",    True),
    ("GC=F",  "gold",      True),
    ("PL=F",  "platinum",  False),
    ("SI=F",  "silver",    False),
    ("PA=F",  "palladium", False),
    ("ALI=F", "aluminum",  False),
    ("NG=F",  "natgas",    False),
]

# Global market sentiment indices — broadcast identically to all 39 countries.
# VIX  : CBOE implied-volatility index; level itself is the key signal (fear gauge).
# DXY  : US Dollar Index; dollar strength pressures EM currencies and commodity prices.
GLOBAL_INDICES_FILE = OUT_DIR / "global_indices_daily.parquet"


# ---------------------------------------------------------------------------
# Part 1: Commodity prices
# ---------------------------------------------------------------------------

def _compute_commodity_features(
    prices: pd.Series,
    prefix: str,
    has_90d: bool,
) -> pd.DataFrame:
    """
    Given a price series (daily, may have gaps), compute pct-change and vol features.
    Returns a DataFrame with columns like {prefix}_pct_30d, {prefix}_vol_30d, etc.
    """
    s = prices.copy()

    # Fill weekend/holiday gaps — forward fill up to 7 days
    s = s.reindex(
        pd.date_range(s.index.min(), s.index.max(), freq="D")
    ).ffill(limit=7)

    log_ret = np.log(s / s.shift(1))

    cols: dict[str, pd.Series] = {}
    cols[f"{prefix}_pct_30d"] = s.pct_change(periods=30) * 100
    if has_90d:
        cols[f"{prefix}_pct_90d"] = s.pct_change(periods=90) * 100
    cols[f"{prefix}_vol_30d"] = log_ret.rolling(30, min_periods=5).std()

    return pd.DataFrame(cols)


def fetch_commodity_prices() -> pd.DataFrame:
    """
    Download commodity futures prices via yfinance and compute rolling features.
    Returns a DataFrame indexed by date with one row per calendar day.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed. Run: pip install yfinance")
        raise

    date_range = pd.date_range(START_DATE, END_DATE, freq="D")
    result     = pd.DataFrame(index=date_range)
    result.index.name = "date"

    for ticker, prefix, has_90d in COMMODITIES:
        log.info("Downloading %s (%s) ...", ticker, prefix)
        try:
            raw = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                log.warning("No data returned for %s — skipping.", ticker)
                continue

            # Handle MultiIndex columns from yfinance
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)

            close = raw["Close"].dropna()
            close.index = pd.to_datetime(close.index)

            feats = _compute_commodity_features(close, prefix, has_90d)

            # Reindex to full date range
            feats = feats.reindex(date_range)

            for col in feats.columns:
                result[col] = feats[col]

            log.info("  -> %s: %d trading days, features: %s",
                     ticker, len(close), list(feats.columns))

        except Exception as exc:
            log.warning("Failed to fetch %s: %s — skipping.", ticker, exc)
            continue

    result = result.reset_index()
    result = result.rename(columns={"index": "date"})
    if "date" not in result.columns:
        result = result.reset_index()

    # Final forward-fill across the date range (up to 7 days)
    feat_cols = [c for c in result.columns if c != "date"]
    result[feat_cols] = result[feat_cols].ffill(limit=7)

    log.info(
        "Commodity features: %d rows x %d feature cols",
        len(result), len(feat_cols),
    )

    missing_pct = result[feat_cols].isna().mean() * 100
    log.info("Missing %% per commodity feature (full range incl. warm-up):\n%s",
             missing_pct[missing_pct > 0].to_string() if (missing_pct > 0).any()
             else "  none")

    return result


# ---------------------------------------------------------------------------
# Part 2: Global market sentiment indices (VIX + DXY)
# ---------------------------------------------------------------------------

def fetch_global_indices() -> pd.DataFrame:
    """
    Download VIX and DXY daily data and compute rolling features.

    VIX (^VIX)
        The CBOE 30-day implied volatility index — the market's 'fear gauge'.
        A high level signals global risk-off conditions that precede EM currency
        stress, capital outflows, and ultimately social unrest.
        Features: vix_level, vix_pct_30d, vix_7d_ma

    DXY (DX-Y.NYB)
        The US Dollar Index — a weighted basket of USD vs major currencies.
        Dollar strength drives up the cost of dollar-denominated imports and
        debt service for EM economies, feeding inflation and social stress.
        Features: dxy_level, dxy_pct_7d, dxy_pct_30d, dxy_vol_30d

    Both are global (one value per date, same for all 39 countries).
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed. Run: pip install yfinance")
        raise

    date_range = pd.date_range(START_DATE, END_DATE, freq="D")
    result = pd.DataFrame(index=date_range)
    result.index.name = "date"

    tickers = {
        "^VIX":     "vix",
        "DX-Y.NYB": "dxy",
    }

    for ticker, prefix in tickers.items():
        log.info("Downloading %s (%s) ...", ticker, prefix)
        try:
            raw = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                log.warning("No data returned for %s — skipping.", ticker)
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.droplevel(1)

            close = raw["Close"].dropna()
            close.index = pd.to_datetime(close.index)

            # Reindex to full daily range and forward-fill weekends/holidays
            s = close.reindex(date_range).ffill(limit=7)

            log_ret = np.log(s / s.shift(1))

            result[f"{prefix}_level"]   = s
            result[f"{prefix}_pct_30d"] = s.pct_change(periods=30) * 100

            if prefix == "vix":
                # 7-day moving average smooths day-to-day noise in the fear gauge
                result[f"{prefix}_7d_ma"] = s.rolling(7, min_periods=3).mean()
            else:
                # DXY: short-term change and realised vol (mirrors FX feature set)
                result[f"{prefix}_pct_7d"]  = s.pct_change(periods=7) * 100
                result[f"{prefix}_vol_30d"] = log_ret.rolling(30, min_periods=5).std()

            log.info("  -> %s: %d trading days", ticker, len(close))

        except Exception as exc:
            log.warning("Failed to fetch %s: %s — skipping.", ticker, exc)
            continue

    result = result.reset_index()

    feat_cols = [c for c in result.columns if c != "date"]

    # Final forward-fill for any trailing gaps
    result[feat_cols] = result[feat_cols].ffill(limit=7)

    # Validation: check missingness within the 2018-2020 panel window
    mask = (result["date"] >= "2018-01-01") & (result["date"] <= "2020-12-31")
    panel_slice = result[mask][feat_cols]
    missing = panel_slice.isna().mean() * 100
    if (missing > 0).any():
        log.warning("Missing %% in 2018-2020 panel window:\n%s",
                    missing[missing > 0].to_string())
    else:
        log.info("Global indices: 0%% missing for 2018-2020 for all features.")

    log.info("Global indices: %d rows x %d feature cols", len(result), len(feat_cols))
    return result


# ---------------------------------------------------------------------------
# Part 3: Gini coefficients (World Bank)
# ---------------------------------------------------------------------------

def _fetch_gini_wb(iso2: str) -> list[tuple[int, float]]:
    """
    Fetch Gini coefficient from World Bank for a single country (ISO2).
    Returns list of (year, gini_value) tuples sorted ascending.
    """
    url = (
        f"https://api.worldbank.org/v2/country/{iso2}"
        f"/indicator/SI.POV.GINI?format=json&mrv=15&per_page=15"
    )
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        if len(data) < 2 or not data[1]:
            return []
        records = []
        for entry in data[1]:
            if entry.get("value") is not None:
                try:
                    year = int(entry["date"])
                    val  = float(entry["value"])
                    records.append((year, val))
                except (ValueError, TypeError):
                    continue
        records.sort(key=lambda x: x[0])
        return records
    except Exception as exc:
        log.warning("World Bank API failed for %s: %s", iso2, exc)
        return []


def fetch_gini_daily() -> pd.DataFrame:
    """
    Fetch Gini coefficients for all 39 panel countries from the World Bank.
    Expands to daily granularity and forward/backward fills within each country.
    Any country with no data at all gets the cross-country mean.
    """
    years = list(range(2016, 2022))  # 2016-2021

    # Build annual Gini table: rows = iso3, cols = years
    annual: dict[str, dict[int, float]] = {}

    for iso3 in PANEL_COUNTRIES_ISO3:
        iso2 = ISO3_TO_ISO2.get(iso3)
        if iso2 is None:
            log.warning("No ISO2 mapping for %s", iso3)
            annual[iso3] = {}
            continue

        records = _fetch_gini_wb(iso2)
        time.sleep(0.15)  # Be polite to the API

        annual[iso3] = {yr: val for yr, val in records if yr in years}

        if records:
            log.info("  %s: %d Gini records found (latest: %d = %.1f)",
                     iso3, len(records),
                     max(yr for yr, _ in records),
                     records[-1][1])
        else:
            log.warning("  %s: NO Gini records found", iso3)

    # Build DataFrame: rows = (country, year)
    rows = []
    for iso3 in PANEL_COUNTRIES_ISO3:
        for yr in years:
            rows.append({
                "country_iso3": iso3,
                "year":         yr,
                "gini_coef":    annual[iso3].get(yr, np.nan),
            })
    ann_df = pd.DataFrame(rows)

    # Forward-fill then backward-fill within each country (annual level)
    ann_df = ann_df.sort_values(["country_iso3", "year"])
    ann_df["gini_coef"] = (
        ann_df.groupby("country_iso3")["gini_coef"]
        .transform(lambda s: s.ffill().bfill())
    )

    # Any country still all-NaN: fill with cross-country mean per year
    year_mean = ann_df.groupby("year")["gini_coef"].transform("mean")
    ann_df["gini_coef"] = ann_df["gini_coef"].fillna(year_mean)

    # Expand to daily
    daily_rows = []
    for _, row in ann_df.iterrows():
        start = pd.Timestamp(f"{int(row['year'])}-01-01")
        end   = pd.Timestamp(f"{int(row['year'])}-12-31")
        dates = pd.date_range(start, end, freq="D")
        for dt in dates:
            daily_rows.append({
                "country_iso3": row["country_iso3"],
                "date":         dt,
                "gini_coef":    row["gini_coef"],
            })

    daily_df = pd.DataFrame(daily_rows)
    daily_df["date"] = pd.to_datetime(daily_df["date"])

    # Validation: check 0% missing for 2018-2020 for all 39 countries
    mask_2018_2020 = (daily_df["date"] >= "2018-01-01") & (daily_df["date"] <= "2020-12-31")
    check = daily_df[mask_2018_2020]
    missing_by_country = check.groupby("country_iso3")["gini_coef"].apply(
        lambda x: x.isna().mean() * 100
    )
    bad = missing_by_country[missing_by_country > 0]
    if bad.empty:
        log.info("Gini validation PASSED: 0%% missing for all 39 countries (2018-2020)")
    else:
        log.warning("Gini validation: countries with missing data:\n%s", bad.to_string())

    log.info(
        "Gini daily: %d rows | %d countries | %s to %s",
        len(daily_df), daily_df["country_iso3"].nunique(),
        daily_df["date"].min().date(), daily_df["date"].max().date(),
    )

    return daily_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Commodity prices ---
    log.info("=" * 60)
    log.info("Fetching commodity prices ...")
    log.info("=" * 60)
    comm_df = fetch_commodity_prices()
    comm_df.to_parquet(COMMODITY_FILE, index=False)
    log.info("Saved -> %s  (%d rows x %d cols)", COMMODITY_FILE,
             len(comm_df), len(comm_df.columns))

    # --- Global market sentiment indices (VIX + DXY) ---
    log.info("=" * 60)
    log.info("Fetching global indices (VIX, DXY) ...")
    log.info("=" * 60)
    idx_df = fetch_global_indices()
    idx_df.to_parquet(GLOBAL_INDICES_FILE, index=False)
    log.info("Saved -> %s  (%d rows x %d cols)", GLOBAL_INDICES_FILE,
             len(idx_df), len(idx_df.columns))

    # --- Gini coefficients ---
    log.info("=" * 60)
    log.info("Fetching Gini coefficients from World Bank ...")
    log.info("=" * 60)
    gini_df = fetch_gini_daily()
    gini_df.to_parquet(GINI_FILE, index=False)
    log.info("Saved -> %s  (%d rows x %d cols)", GINI_FILE,
             len(gini_df), len(gini_df.columns))

    log.info("=" * 60)
    log.info("fetch_supplemental_data.py complete.")
    log.info("=" * 60)


if __name__ == "__main__":
    run()
