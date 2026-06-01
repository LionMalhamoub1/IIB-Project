"""
Fetch 2016 FX, oil, and yield data to eliminate warmup NaN at the start of
2017 in the modelling panel.

For each affected series:
  1. Download Jan 2016 – Dec 2016 from Yahoo Finance
  2. Concatenate with the 2017+ data already in the panel
  3. Recompute rolling features (pct_change, vol) on the full extended series
  4. Extract 2017-01-01 onward and write back into the panel parquet
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import yfinance as yf

PANEL_PATH = "Social_Disruptions/Likelihood_Modelling/v2/data/interim/modelling_panel_gdelt.parquet"

FETCH_START = "2016-01-01"
FETCH_END   = "2016-12-31"
FFILL_LIMIT = 7

# EM currencies: ISO3 -> Yahoo Finance ticker (USD per LCU convention: USDXXX=X)
EM_FX: dict[str, str] = {
    "ARG": "USDARS=X",
    "BOL": "USDBOB=X",
    "BRA": "USDBRL=X",
    "CHL": "USDCLP=X",
    "IDN": "USDIDR=X",
    "IND": "USDINR=X",
    "KEN": "USDKES=X",
    "MAR": "USDMAD=X",
    "MEX": "USDMXN=X",
    "MOZ": "USDMZN=X",
    "PER": "USDPEN=X",
    "PHL": "USDPHP=X",
    "THA": "USDTHB=X",
    "TUR": "USDTRY=X",
    "VNM": "USDVND=X",
    "ZAF": "USDZAR=X",
}


def fetch_close(ticker: str, start: str, end: str, retries: int = 2) -> pd.Series | None:
    for attempt in range(retries + 1):
        try:
            time.sleep(0.4)
            raw = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False, multi_level_index=False,
            )
            if raw is None or raw.empty:
                return None
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            s = raw["Close"].dropna()
            s.index = pd.to_datetime(s.index)
            return s
        except Exception as exc:
            if attempt < retries:
                time.sleep(1.0)
            else:
                print(f"  [{ticker}] failed: {exc}")
    return None


def reindex_daily(s: pd.Series, start: str, end: str) -> pd.Series:
    idx = pd.date_range(start=start, end=end, freq="D")
    return s.reindex(idx).ffill(limit=FFILL_LIMIT)


def compute_fx_features(s: pd.Series) -> pd.DataFrame:
    log_ret = np.log(s / s.shift(1))
    return pd.DataFrame({
        "fx_lcu_usd":    s,
        "fx_log_return": log_ret,
        "fx_pct_7d":     s.pct_change(7)  * 100,
        "fx_pct_30d":    s.pct_change(30) * 100,
        "fx_pct_90d":    s.pct_change(90) * 100,
        "fx_vol_7d":     log_ret.rolling(7,  min_periods=2).std(),
        "fx_vol_30d":    log_ret.rolling(30, min_periods=5).std(),
    })


print("Loading panel...")
panel = pd.read_parquet(PANEL_PATH)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.sort_values(["country_iso3", "date"]).reset_index(drop=True)

FX_FEAT_COLS = ["fx_lcu_usd", "fx_log_return",
                "fx_pct_7d", "fx_pct_30d", "fx_pct_90d",
                "fx_vol_7d", "fx_vol_30d"]

# ── 1. EM FX features ─────────────────────────────────────────────────────────
print("\n=== Fetching 2016 EM FX data ===")
for iso3, ticker in EM_FX.items():
    print(f"  {iso3} ({ticker})...", end=" ")

    # Get existing 2017+ series from panel (use first non-null to find base level)
    panel_rows = panel[panel["country_iso3"] == iso3].sort_values("date")
    existing_lcu = panel_rows.set_index("date")["fx_lcu_usd"].dropna()

    # Download 2016 data
    s2016 = fetch_close(ticker, FETCH_START, FETCH_END)
    if s2016 is None or s2016.empty:
        print("SKIP (no data)")
        continue

    s2016 = reindex_daily(s2016, FETCH_START, FETCH_END)

    # Combine 2016 with existing 2017+ level series
    full_lcu = pd.concat([s2016, existing_lcu]).sort_index()
    full_lcu = full_lcu[~full_lcu.index.duplicated(keep="last")]

    # Reindex to full 2016-2021 daily range and forward-fill
    full_range = pd.date_range("2016-01-01", "2021-12-31", freq="D")
    full_lcu = full_lcu.reindex(full_range).ffill(limit=FFILL_LIMIT)

    # Recompute features on full series
    feats = compute_fx_features(full_lcu)

    # Extract 2017-01-01 onward and update panel
    feats_2017 = feats.loc["2017-01-01":"2021-12-31"]

    country_mask = panel["country_iso3"] == iso3
    for col in FX_FEAT_COLS:
        if col in panel.columns:
            panel.loc[country_mask, col] = (
                panel.loc[country_mask, "date"]
                .map(feats_2017[col])
                .values
            )

    # Verify the 90-day rolling feature warmup NaN is resolved for this country
    n_nan = panel.loc[country_mask, "fx_pct_90d"].isna().sum()
    print(f"OK (fx_pct_90d NaN remaining: {n_nan})")


# ── 2. Oil (Brent) features ───────────────────────────────────────────────────
print("\n=== Fetching 2016 Brent crude (BZ=F) ===")
oil_2016 = fetch_close("BZ=F", FETCH_START, FETCH_END)
if oil_2016 is None:
    print("  Brent (BZ=F) unavailable, trying WTI (CL=F)...")
    oil_2016 = fetch_close("CL=F", FETCH_START, FETCH_END)

if oil_2016 is not None:
    oil_2016 = reindex_daily(oil_2016, FETCH_START, FETCH_END)

    # Get existing 2017+ oil_brent_usd from panel (global, same for all countries)
    sample_country = panel["country_iso3"].iloc[0]
    existing_oil = (
        panel[panel["country_iso3"] == sample_country]
        .set_index("date")["oil_brent_usd"]
        .dropna()
    )

    full_oil = pd.concat([oil_2016, existing_oil]).sort_index()
    full_oil = full_oil[~full_oil.index.duplicated(keep="last")]
    full_range = pd.date_range("2016-01-01", "2021-12-31", freq="D")
    full_oil = full_oil.reindex(full_range).ffill(limit=FFILL_LIMIT)

    # Compute oil features
    oil_pct_14d = full_oil.pct_change(14) * 100
    oil_pct_30d = full_oil.pct_change(30) * 100

    oil_feats = pd.DataFrame({
        "oil_brent_usd":    full_oil,
        "oil_brent_pct_14d": oil_pct_14d,
        "oil_brent_pct_30d": oil_pct_30d,
    }).loc["2017-01-01":"2021-12-31"]

    for col in ["oil_brent_usd", "oil_brent_pct_14d", "oil_brent_pct_30d"]:
        if col in panel.columns:
            panel[col] = panel["date"].map(oil_feats[col]).values

    n_nan_oil = panel["oil_brent_pct_30d"].isna().sum()
    print(f"  OK (oil_brent_pct_30d NaN remaining: {n_nan_oil})")
else:
    print("  SKIP — no oil data available")


# ── 3. US 10-year yield (^TNX) ────────────────────────────────────────────────
print("\n=== Fetching 2016 US 10Y yield (^TNX) ===")
yield_2016 = fetch_close("^TNX", FETCH_START, FETCH_END)
if yield_2016 is not None:
    yield_2016 = reindex_daily(yield_2016, FETCH_START, FETCH_END)

    sample_country = panel["country_iso3"].iloc[0]
    existing_yield = (
        panel[panel["country_iso3"] == sample_country]
        .set_index("date")["yield_us10y"]
        .dropna()
    )

    full_yield = pd.concat([yield_2016, existing_yield]).sort_index()
    full_yield = full_yield[~full_yield.index.duplicated(keep="last")]
    full_range = pd.date_range("2016-01-01", "2021-12-31", freq="D")
    full_yield = full_yield.reindex(full_range).ffill(limit=FFILL_LIMIT)

    yield_2017 = full_yield.loc["2017-01-01":"2021-12-31"]

    if "yield_us10y" in panel.columns:
        panel["yield_us10y"] = panel["date"].map(yield_2017).values

    n_nan_yield = panel["yield_us10y"].isna().sum()
    print(f"  OK (yield_us10y NaN remaining: {n_nan_yield})")
else:
    print("  SKIP — no yield data available")


# ── 4. Recompute derived features ─────────────────────────────────────────────
print("\n=== Recomputing lag and interaction features ===")

# FX z-score
def expanding_zscore(s, min_obs=3):
    exp = s.expanding(min_periods=min_obs)
    z = (s - exp.mean()) / exp.std().replace(0.0, np.nan)
    return z.ffill().bfill().fillna(0.0)

for src, z in [("fx_pct_30d", "fx_pct_30d_z"), ("fx_vol_30d", "fx_vol_30d_z"),
               ("oil_brent_pct_30d", "oil_brent_pct_30d_z")]:
    if z in panel.columns and src in panel.columns:
        for iso3, grp in panel.groupby("country_iso3"):
            idx = panel["country_iso3"] == iso3
            panel.loc[idx, z] = expanding_zscore(
                panel.loc[idx, src].reset_index(drop=True)
            ).values

# FX lags (per-country shift)
if "fx_pct_30d" in panel.columns:
    parts = []
    for iso3, grp in panel.groupby("country_iso3"):
        grp = grp.sort_values("date").copy()
        grp["fx_pct_30d_lag30d"] = grp["fx_pct_30d"].shift(30)
        grp["fx_pct_30d_lag60d"] = grp["fx_pct_30d"].shift(60)
        parts.append(grp)
    panel = pd.concat(parts, ignore_index=True).sort_values(["country_iso3", "date"])
    # Any remaining NaN at start of series -> fill with 0
    for col in ["fx_pct_30d_lag30d", "fx_pct_30d_lag60d"]:
        panel[col] = panel.groupby("country_iso3")[col].transform(
            lambda s: s.ffill().bfill()
        ).fillna(0.0)

# Oil lags (global shift)
if "oil_brent_pct_30d" in panel.columns:
    # Build date-level lookup then broadcast
    date_oil = panel.drop_duplicates("date").set_index("date")["oil_brent_pct_30d"].sort_index()
    for lag, col in [(30, "oil_brent_pct_30d_lag30d"),
                     (60, "oil_brent_pct_30d_lag60d"),
                     (90, "oil_brent_pct_30d_lag90d")]:
        lagged = date_oil.shift(lag).bfill().fillna(0.0)
        panel[col] = panel["date"].map(lagged).values

# VIX lags (global)
if "vix_pct_30d" in panel.columns:
    date_vix = panel.drop_duplicates("date").set_index("date")["vix_pct_30d"].sort_index()
    for lag, col in [(30, "vix_pct_30d_lag30d"), (60, "vix_pct_30d_lag60d")]:
        lagged = date_vix.shift(lag).bfill().fillna(0.0)
        panel[col] = panel["date"].map(lagged).values

# Interaction terms
if "fx_pct_30d" in panel.columns and "political_stability_est" in panel.columns:
    panel["fx_pct_30d_x_instability"] = (
        panel["fx_pct_30d"] * (1 - panel["political_stability_est"].clip(-3, 3))
    )
if "oil_brent_pct_30d" in panel.columns and "inflation_cpi_yoy" in panel.columns:
    panel["oil_brent_pct_30d_x_inflation"] = (
        panel["oil_brent_pct_30d"] * panel["inflation_cpi_yoy"]
    )
if "oil_brent_pct_30d" in panel.columns:
    if "net_importer" in panel.columns:
        panel["oil_brent_pct_30d_x_net_importer"] = (
            panel["oil_brent_pct_30d"] * panel["net_importer"]
        )
    elif "is_net_oil_importer" in panel.columns:
        panel["oil_brent_pct_30d_x_net_importer"] = (
            panel["oil_brent_pct_30d"] * panel["is_net_oil_importer"]
        )

print("  Done")

# ── 5. Final verification ──────────────────────────────────────────────────────
FEATURES_M1_ADD = ["fx_pct_7d","fx_pct_30d","fx_pct_90d","fx_vol_7d","fx_vol_30d",
                    "oil_brent_pct_14d","oil_brent_pct_30d","yield_us10y",
                    "fx_pct_30d_z","fx_vol_30d_z","oil_brent_pct_30d_z",
                    "copper_pct_30d","copper_pct_90d","copper_vol_30d",
                    "gold_pct_30d","gold_vol_30d","platinum_pct_30d",
                    "silver_pct_30d","natgas_pct_30d",
                    "vix_level","vix_pct_30d","vix_7d_ma",
                    "dxy_level","dxy_pct_30d","dxy_vol_30d",
                    "oil_brent_pct_30d_lag30d","oil_brent_pct_30d_lag60d","oil_brent_pct_30d_lag90d",
                    "vix_pct_30d_lag30d","vix_pct_30d_lag60d",
                    "fx_pct_30d_lag30d","fx_pct_30d_lag60d"]

sub = panel[panel["date"].dt.year.between(2017, 2021)]
avail = [f for f in FEATURES_M1_ADD if f in panel.columns]
miss = sub[avail].isnull().sum()
miss = miss[miss > 0]

print("\n=== M1 Financial/Markets NaN remaining ===")
if miss.empty:
    print("  NONE")
else:
    print(miss)

panel.to_parquet(PANEL_PATH, index=False)
print(f"\nSaved -> {PANEL_PATH}")
