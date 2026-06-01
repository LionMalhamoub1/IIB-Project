# Downloads FX data for panel countries that were missing exchange-rate features.

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

_HERE = Path(__file__).resolve().parent
_V2   = _HERE.parent
sys.path.insert(0, str(_HERE))

OUT_FILE = _V2 / "data" / "interim" / "fx_missing_countries.parquet"

# Date range — start a year earlier to allow rolling/pct features to warm up
FETCH_START = "2016-01-01"
FETCH_END   = "2022-12-31"

# Countries missing FX + their currency code for the USD{CCY}=X ticker.
# Eurozone countries all share EUR.  USA is USD (hardcoded to 0).
MISSING_COUNTRIES: dict[str, str] = {
    "AUS": "AUD",  # Australian Dollar
    "CAN": "CAD",  # Canadian Dollar
    "CHN": "CNY",  # Chinese Yuan Renminbi
    "DEU": "EUR",  # Euro
    "ESP": "EUR",  # Euro
    "FRA": "EUR",  # Euro
    "GBR": "GBP",  # British Pound Sterling
    "GRC": "EUR",  # Euro
    "HUN": "HUF",  # Hungarian Forint
    "IRL": "EUR",  # Euro
    "ITA": "EUR",  # Euro
    "JPN": "JPY",  # Japanese Yen
    "KOR": "KRW",  # South Korean Won
    "LAO": "LAK",  # Lao Kip (may be unavailable; fallback: THB proxy)
    "MYS": "MYR",  # Malaysian Ringgit
    "NAM": "ZAR",  # Namibian Dollar (pegged 1:1 to ZAR — use ZAR as proxy)
    "NLD": "EUR",  # Euro
    "NOR": "NOK",  # Norwegian Krone
    "POL": "PLN",  # Polish Zloty
    "PRT": "EUR",  # Euro
    "SWE": "SEK",  # Swedish Krona
    # USA omitted: USD is the base currency (all pct changes = 0 by definition)
}

# Forward-fill limit (trading weekends/holidays)
FFILL_LIMIT = 7

PCT_WINDOWS = [7, 30, 90]
VOL_WINDOWS = [7, 30]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature builders (matches markets_features.py exactly)
# ---------------------------------------------------------------------------

def _log_return(series: pd.Series) -> pd.Series:
    shifted = series.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(series / shifted)
    return pd.Series(lr, index=series.index)


def build_fx_features(raw: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    out["fx_lcu_usd"]    = raw.values
    log_ret              = _log_return(raw)
    out["fx_log_return"] = log_ret.values
    for w in sorted(PCT_WINDOWS):
        out[f"fx_pct_{w}d"] = raw.pct_change(periods=w).mul(100).values
    for w in sorted(VOL_WINDOWS):
        min_p = max(2, w // 2)
        out[f"fx_vol_{w}d"] = log_ret.rolling(w, min_periods=min_p).std().values
    return out


def reindex_to_daily(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    idx = pd.date_range(start=start, end=end, freq="D", name="date")
    return df.reindex(idx).ffill(limit=FFILL_LIMIT)


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def fetch_ticker(ticker: str, retries: int = 2) -> pd.Series | None:
    """Download closing price from Yahoo Finance; return as LCU/USD Series."""
    if not _YF_AVAILABLE:
        log.error("yfinance is not installed. Run: pip install yfinance")
        return None

    for attempt in range(retries + 1):
        try:
            time.sleep(0.4)
            raw = yf.download(
                ticker,
                start=FETCH_START,
                end=FETCH_END,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
            if raw is None or raw.empty:
                log.warning("  [%s] empty download", ticker)
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if "Close" not in raw.columns:
                log.warning("  [%s] no Close column", ticker)
                return None
            s = raw["Close"].dropna()
            s.index = pd.to_datetime(s.index)
            s.index.name = "date"
            log.info("  [%s] %d rows (%s to %s)",
                     ticker, len(s), s.index[0].date(), s.index[-1].date())
            return s
        except Exception as exc:
            if attempt < retries:
                log.warning("  [%s] attempt %d failed: %s — retrying…", ticker, attempt + 1, exc)
                time.sleep(1.0)
            else:
                log.error("  [%s] all attempts failed: %s", ticker, exc)
                return None
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fetch_all() -> pd.DataFrame:
    if not _YF_AVAILABLE:
        log.error("yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    # Cache downloaded series by currency so Eurozone countries share one download
    currency_cache: dict[str, pd.Series | None] = {}

    parts: list[pd.DataFrame] = []

    for iso3, ccy in MISSING_COUNTRIES.items():
        log.info("-" * 50)
        log.info("%s  (%s)  ticker: USD%s=X", iso3, ccy, ccy)

        if ccy not in currency_cache:
            ticker = f"USD{ccy}=X"
            series = fetch_ticker(ticker)
            if series is None and ccy == "LAK":
                # LAO Kip unavailable — use THB as closest regional proxy
                log.warning("  LAK unavailable on Yahoo Finance; using THB as proxy for LAO.")
                series = fetch_ticker("USDTHB=X")
            currency_cache[ccy] = series

        series = currency_cache[ccy]

        if series is None:
            log.warning("  %s: no FX data available — columns will be NaN.", iso3)
            full_idx = pd.date_range(FETCH_START, FETCH_END, freq="D", name="date")
            feats = pd.DataFrame(index=full_idx)
            feats["fx_lcu_usd"]    = np.nan
            feats["fx_log_return"] = np.nan
            for w in sorted(PCT_WINDOWS):
                feats[f"fx_pct_{w}d"] = np.nan
            for w in sorted(VOL_WINDOWS):
                feats[f"fx_vol_{w}d"] = np.nan
        else:
            feats = build_fx_features(series)
            feats = reindex_to_daily(feats, FETCH_START, FETCH_END)

        feats = feats.reset_index()
        feats.insert(0, "country_iso3", iso3)
        parts.append(feats)

    # USD-based economies: set all FX features to 0
    # USA: base currency (USD)
    # ZWE: Zimbabwe was effectively dollarized throughout 2018-2020
    #      (used USD / Bond Notes pegged to USD; ZWL re-introduced 2019 at 1:1)
    full_idx = pd.date_range(FETCH_START, FETCH_END, freq="D", name="date")
    for usd_country in ("USA", "ZWE"):
        log.info("-" * 50)
        log.info("%s  (USD-based)  -- all FX features = 0", usd_country)
        usd_df = pd.DataFrame(index=full_idx)
        usd_df["fx_lcu_usd"]    = 1.0
        usd_df["fx_log_return"] = 0.0
        for w in sorted(PCT_WINDOWS):
            usd_df[f"fx_pct_{w}d"] = 0.0
        for w in sorted(VOL_WINDOWS):
            usd_df[f"fx_vol_{w}d"] = 0.0
        usd_df = usd_df.reset_index()
        usd_df.insert(0, "country_iso3", usd_country)
        parts.append(usd_df)

    df = pd.concat(parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country_iso3", "date"]).reset_index(drop=True)

    log.info("=" * 50)
    log.info("Combined: %d rows | %d countries", len(df), df["country_iso3"].nunique())
    for c in [f"fx_pct_{w}d" for w in PCT_WINDOWS] + [f"fx_vol_{w}d" for w in VOL_WINDOWS]:
        pct_miss = df[c].isna().mean() * 100
        if pct_miss > 0:
            log.info("  %s: %.1f%% missing", c, pct_miss)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="Only print missingness of existing file.")
    args = parser.parse_args()

    if args.check:
        if OUT_FILE.exists():
            df = pd.read_parquet(OUT_FILE)
            log.info("File: %s  (%d rows)", OUT_FILE, len(df))
            for c in df.columns:
                if c not in ("country_iso3", "date"):
                    pct = df[c].isna().mean() * 100
                    if pct > 0:
                        log.info("  %s: %.1f%% missing", c, pct)
        else:
            log.warning("File not found: %s", OUT_FILE)
        return

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_all()
    df.to_parquet(OUT_FILE, index=False)
    log.info("Saved -> %s  (%d rows x %d cols)", OUT_FILE, len(df), len(df.columns))


if __name__ == "__main__":
    main()
