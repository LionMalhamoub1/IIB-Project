"""
Fix lag features so early-January 2017 values come from real 2016 data
instead of being bfilled from the first available 2017 value.

Affected features:
  oil_brent_pct_30d_lag30d/60d/90d  -- computed from BZ=F 2015+
  vix_pct_30d_lag30d/60d            -- computed from global_indices_daily (2016+)
  fx_pct_30d_lag30d/60d             -- computed from extended per-country series (2016+)
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import yfinance as yf

PANEL_PATH   = "Social_Disruptions/Likelihood_Modelling/v2/data/interim/modelling_panel_gdelt.parquet"
GIDX_FILE    = "Social_Disruptions/Likelihood_Modelling/v2/data/interim/global_indices_daily.parquet"
FX_HI_FILE   = "Social_Disruptions/Likelihood_Modelling/v2/data/interim/fx_missing_countries.parquet"

EM_FX: dict[str, str] = {
    "ARG": "USDARS=X", "BOL": "USDBOB=X", "BRA": "USDBRL=X",
    "CHL": "USDCLP=X", "IDN": "USDIDR=X", "IND": "USDINR=X",
    "KEN": "USDKES=X", "MAR": "USDMAD=X", "MEX": "USDMXN=X",
    "MOZ": "USDMZN=X", "PER": "USDPEN=X", "PHL": "USDPHP=X",
    "THA": "USDTHB=X", "TUR": "USDTRY=X", "VNM": "USDVND=X",
    "ZAF": "USDZAR=X",
}
FFILL_LIMIT = 7

print("Loading panel...")
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.sort_values(["country_iso3", "date"]).reset_index(drop=True)

# ── 1. Oil lags ──────────────────────────────────────────────────────────────
print("\n=== Oil lags (BZ=F 2015-2021) ===")
raw = yf.download("BZ=F", start="2015-01-01", end="2021-12-31",
                  auto_adjust=True, progress=False, multi_level_index=False)
oil = raw["Close"].dropna()
oil.index = pd.to_datetime(oil.index)
full_range = pd.date_range("2015-01-01", "2021-12-31", freq="D")
oil = oil.reindex(full_range).ffill(limit=FFILL_LIMIT)

oil_pct30 = oil.pct_change(30) * 100  # full series from 2015

for lag, col in [(30, "oil_brent_pct_30d_lag30d"),
                 (60, "oil_brent_pct_30d_lag60d"),
                 (90, "oil_brent_pct_30d_lag90d")]:
    lagged = oil_pct30.shift(lag)         # shift on full 2015+ series
    lagged_2017 = lagged.loc["2017-01-01":"2021-12-31"]
    panel[col] = panel["date"].map(lagged_2017).values
    n_nan = panel.loc[panel["date"].dt.year.between(2017,2021), col].isna().sum()
    print(f"  {col}: {n_nan} NaN remaining")

# ── 2. VIX lags ──────────────────────────────────────────────────────────────
print("\n=== VIX lags (global_indices_daily 2016-2021) ===")
gidx = pd.read_parquet(GIDX_FILE)
gidx["date"] = pd.to_datetime(gidx["date"])
vix_pct30 = gidx.set_index("date")["vix_pct_30d"].sort_index()  # starts 2016

for lag, col in [(30, "vix_pct_30d_lag30d"), (60, "vix_pct_30d_lag60d")]:
    lagged = vix_pct30.shift(lag)
    lagged_2017 = lagged.loc["2017-01-01":"2021-12-31"]
    panel[col] = panel["date"].map(lagged_2017).values
    n_nan = panel.loc[panel["date"].dt.year.between(2017,2021), col].isna().sum()
    print(f"  {col}: {n_nan} NaN remaining")

# ── 3. FX lags: high-income countries (fx_missing_countries, starts 2016) ────
print("\n=== FX lags: high-income countries ===")
fx_hi = pd.read_parquet(FX_HI_FILE)
fx_hi["date"] = pd.to_datetime(fx_hi["date"])
hi_countries = fx_hi["country_iso3"].unique()

for iso3 in hi_countries:
    series = (fx_hi[fx_hi["country_iso3"] == iso3]
              .set_index("date")["fx_pct_30d"]
              .sort_index())
    mask = panel["country_iso3"] == iso3
    for lag, col in [(30, "fx_pct_30d_lag30d"), (60, "fx_pct_30d_lag60d")]:
        lagged = series.shift(lag)
        lagged_2017 = lagged.loc["2017-01-01":"2021-12-31"]
        panel.loc[mask, col] = panel.loc[mask, "date"].map(lagged_2017).values

nan_hi = panel.loc[panel["country_iso3"].isin(hi_countries) &
                   panel["date"].dt.year.between(2017,2021),
                   ["fx_pct_30d_lag30d","fx_pct_30d_lag60d"]].isna().sum()
print(f"  lag30d NaN: {nan_hi['fx_pct_30d_lag30d']}, lag60d NaN: {nan_hi['fx_pct_30d_lag60d']}")

# ── 4. FX lags: EM countries (re-download 2016 data) ─────────────────────────
print("\n=== FX lags: EM countries (Yahoo Finance 2016+) ===")
currency_cache: dict[str, pd.Series] = {}

for iso3, ticker in EM_FX.items():
    ccy = ticker[3:6]  # e.g. "ARS" from "USDARS=X"

    if ccy not in currency_cache:
        print(f"  Downloading {ticker}...", end=" ")
        time.sleep(0.4)
        try:
            raw = yf.download(ticker, start="2016-01-01", end="2021-12-31",
                              auto_adjust=True, progress=False, multi_level_index=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            s = raw["Close"].dropna()
            s.index = pd.to_datetime(s.index)
            full_range_fx = pd.date_range("2016-01-01", "2021-12-31", freq="D")
            s = s.reindex(full_range_fx).ffill(limit=FFILL_LIMIT)
            pct30 = s.pct_change(30) * 100
            currency_cache[ccy] = pct30
            print("OK")
        except Exception as exc:
            print(f"FAILED ({exc})")
            currency_cache[ccy] = pd.Series(dtype=float)

    pct30 = currency_cache[ccy]
    if pct30.empty:
        continue

    mask = panel["country_iso3"] == iso3
    for lag, col in [(30, "fx_pct_30d_lag30d"), (60, "fx_pct_30d_lag60d")]:
        lagged = pct30.shift(lag)
        lagged_2017 = lagged.loc["2017-01-01":"2021-12-31"]
        panel.loc[mask, col] = panel.loc[mask, "date"].map(lagged_2017).values

nan_em = panel.loc[panel["country_iso3"].isin(list(EM_FX.keys())) &
                   panel["date"].dt.year.between(2017,2021),
                   ["fx_pct_30d_lag30d","fx_pct_30d_lag60d"]].isna().sum()
print(f"  lag30d NaN: {nan_em['fx_pct_30d_lag30d']}, lag60d NaN: {nan_em['fx_pct_30d_lag60d']}")

# ── 5. Final check ────────────────────────────────────────────────────────────
print("\n=== Final NaN check on lag features ===")
lag_cols = ["oil_brent_pct_30d_lag30d", "oil_brent_pct_30d_lag60d", "oil_brent_pct_30d_lag90d",
            "vix_pct_30d_lag30d", "vix_pct_30d_lag60d",
            "fx_pct_30d_lag30d", "fx_pct_30d_lag60d"]
sub = panel[panel["date"].dt.year.between(2017, 2021)]
for col in lag_cols:
    if col in panel.columns:
        n = sub[col].isna().sum()
        status = "OK" if n == 0 else f"STILL {n} NaN"
        print(f"  {col}: {status}")

# Verify early Jan sample is no longer bfilled
print("\n=== Spot check: oil lag on Jan 1, 2017 ===")
jan1 = panel[(panel["date"] == "2017-01-01") & (panel["country_iso3"] == "ARG")]
print(f"  oil_brent_pct_30d (Jan 1):         {jan1['oil_brent_pct_30d'].values[0]:.4f}%")
print(f"  oil_brent_pct_30d_lag30d (Jan 1):  {jan1['oil_brent_pct_30d_lag30d'].values[0]:.4f}%")
print(f"  (lag30 should = oil pct30 from Dec 2, 2016: {oil_pct30.loc['2016-12-02']:.4f}%)")

panel.to_parquet(PANEL_PATH, index=False)
print(f"\nSaved -> {PANEL_PATH}")
