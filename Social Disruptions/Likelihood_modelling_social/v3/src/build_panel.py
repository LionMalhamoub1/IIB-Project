"""
build_panel.py
==============
Merges GDELT-derived event labels with the existing economic feature panel.
Run this before train_backtest.py.

What it does
------------
1. Loads the existing modelling_panel.parquet (39-country feature panel, 2017-2025).
2. Strips ACLED label columns — keeps all economic / market / macro features.
3. Loads GDELT country-day labels from verification/v2/labels/.
4. Restricts both to the 2017-2021 GDELT window.
5. Computes GDELT-based lag features (protest / strike activity in past 7 & 28 days)
   for the persistence (M0) model.
6. Merges features + GDELT labels and saves to v2/data/interim/modelling_panel_gdelt.parquet.

Output columns of interest
---------------------------
  Targets (forward-looking, pre-computed in build_labels_modelling.py):
    protest_7d, protest_30d   -- any protest in next 7 / 30 days
    strike_7d,  strike_30d   -- any strike  in next 7 / 30 days

  Persistence features (new, computed here):
    gdelt_protest_7d_lag      -- protest days in past 7 days  (shifted 1 to avoid leakage)
    gdelt_protest_28d_lag     -- protest days in past 28 days
    gdelt_strike_7d_lag       -- strike  days in past 7 days
    gdelt_strike_28d_lag      -- strike  days in past 28 days

  All other features from the original panel (FX, macro, FAO, GTA, etc.) are
  passed through unchanged.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Supplemental data paths
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE  = Path(__file__).resolve().parent               # v2/src
_V2    = _HERE.parent                                  # v2  (Likelihood_modelling_social/v2)
_MOD   = _V2.parent                                    # Likelihood_modelling_social
_SD    = _MOD.parent                                   # Social Disruptions
_VER   = _SD / "verification"                          # Social Disruptions/verification

GDELT_LABELS_FILE  = _VER / "v2" / "output" / "labels_country_day.parquet"
FEATURE_PANEL_FILE = _MOD / "data" / "interim" / "modelling_panel.parquet"
FX_MISSING_FILE    = _V2 / "data" / "interim" / "fx_missing_countries.parquet"
COMMODITY_FILE     = _V2 / "data" / "interim" / "commodity_prices_daily.parquet"
GLOBAL_INDICES_FILE = _V2 / "data" / "interim" / "global_indices_daily.parquet"
GINI_FILE          = _V2 / "data" / "interim" / "gini_daily.parquet"
OUT_DIR            = _V2 / "data" / "interim"
OUT_FILE           = OUT_DIR / "modelling_panel_gdelt.parquet"

GDELT_START = "2017-01-01"
GDELT_END   = "2021-12-31"

# ---------------------------------------------------------------------------
# Mineral-producer country sets (hardcoded)
# ---------------------------------------------------------------------------
COPPER_PRODUCERS   = {"CHL", "PER", "AUS", "CAN", "CHN", "IDN", "LAO", "NAM", "SWE", "ARG", "BRA"}
GOLD_PRODUCERS     = {"ZAF", "ZWE", "AUS", "BRA", "CAN", "IND", "PER", "USA", "MAR", "KEN", "GRC", "GBR"}
PLATINUM_PRODUCERS = {"ZAF", "ZWE"}        # RUS not in panel
SILVER_PRODUCERS   = {"BOL", "PER", "MEX", "CHL", "AUS", "ARG"}
NICKEL_PRODUCERS   = {"IDN", "PHL", "AUS", "BRA", "NOR", "GRC"}
COAL_PRODUCERS     = {"AUS", "CHN", "IDN", "IND", "USA", "MOZ", "POL", "VNM"}
NATGAS_PRODUCERS   = {"NOR", "AUS", "USA", "MOZ"}
LITHIUM_PRODUCERS  = {"AUS", "CHL", "ARG", "BOL", "CHN", "BRA"}

# Net oil exporters in panel (2017-2021): all other 36 countries treated as importers
NET_OIL_EXPORTERS = {"NOR", "CAN", "USA"}

# Geographic regions for cross-country spillover features
REGIONS: dict[str, set[str]] = {
    "western_europe": {"DEU", "ESP", "FRA", "GBR", "GRC", "HUN", "IRL", "ITA",
                       "NLD", "NOR", "POL", "PRT", "SWE"},
    "east_asia":      {"CHN", "JPN", "KOR", "MYS", "VNM", "THA", "IDN", "PHL", "LAO"},
    "south_asia":     {"IND"},
    "latin_america":  {"ARG", "BOL", "BRA", "CHL", "MEX", "PER"},
    "north_america":  {"CAN", "USA"},
    "africa":         {"KEN", "MAR", "MOZ", "NAM", "ZAF", "ZWE"},
    "mena_turkey":    {"TUR"},
    "oceania":        {"AUS"},
}
COUNTRY_REGION: dict[str, str] = {c: r for r, cs in REGIONS.items() for c in cs}

# ACLED-sourced columns to drop from the feature panel — these are labels, not features
ACLED_LABEL_COLS = [
    "acled_events",
    "riot_events",
    "violence_events",
    "protest_fatalities",
    "acled_7d_lag",
    "acled_28d_lag",
    "riot_7d_lag",
    "riot_28d_lag",
    "violence_7d_lag",
    "violence_28d_lag",
    "protest_fat_7d_lag",
    "protest_fat_28d_lag",
]

# GDELT label columns to carry through to the final panel
GDELT_LABEL_COLS = [
    "protest_today",
    "strike_today",
    "protest_7d",
    "strike_7d",
    "protest_30d",
    "strike_30d",
    "n_protest_events",
    "n_strike_events",
    "n_articles",
    "has_movement",
    "coverage_flag",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rolling_lag(series: pd.Series, days: int) -> pd.Series:
    """
    Sum of `series` over the past `days` calendar days, shifted by 1 day to
    prevent look-ahead leakage.  Requires the series to have a DatetimeIndex.
    """
    return (
        series
        .rolling(f"{days}D", min_periods=1)
        .sum()
        .shift(1)        # use only past information, not today
    )


def compute_gdelt_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds GDELT-based lag features per country.  Operates on the merged
    country-day DataFrame (iso3, date, protest_today, strike_today).
    """
    parts = []
    for iso3, grp in df.groupby("country_iso3"):
        grp = grp.sort_values("date").copy()
        grp = grp.set_index("date")

        for col, label in [("protest_today", "protest"), ("strike_today", "strike")]:
            for days in [7, 28]:
                feat_name = f"gdelt_{label}_{days}d_lag"
                grp[feat_name] = _rolling_lag(grp[col].astype(float), days)

        parts.append(grp.reset_index())

    return pd.concat(parts, ignore_index=True)


def compute_regional_spillover(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds gdelt_protest_region_14d and gdelt_strike_region_14d: sum of protest/strike
    activity in the same geographic region over the past 14 days, excluding the focal
    country.  Shifted 1 day to prevent look-ahead leakage.
    """
    df = df.copy()
    df["_region"] = df["country_iso3"].map(COUNTRY_REGION)

    # Per-country 14-day rolling sum (leakage-safe via _rolling_lag)
    parts = []
    for iso3, grp in df.groupby("country_iso3"):
        grp = grp.sort_values("date").set_index("date").copy()
        grp["_prot14"] = _rolling_lag(grp["protest_today"].astype(float), 14)
        grp["_strk14"] = _rolling_lag(grp["strike_today"].astype(float),  14)
        parts.append(grp.reset_index())
    df = pd.concat(parts, ignore_index=True)

    # Region-date totals across all countries
    reg_totals = (
        df.groupby(["_region", "date"])[["_prot14", "_strk14"]]
        .sum()
        .rename(columns={"_prot14": "_prot14_reg", "_strk14": "_strk14_reg"})
        .reset_index()
    )
    df = df.merge(reg_totals, on=["_region", "date"], how="left")

    # Subtract focal country's own contribution and clip to >=0
    df["gdelt_protest_region_14d"] = (
        df["_prot14_reg"] - df["_prot14"].fillna(0)
    ).clip(lower=0).fillna(0.0)
    df["gdelt_strike_region_14d"] = (
        df["_strk14_reg"] - df["_strk14"].fillna(0)
    ).clip(lower=0).fillna(0.0)

    df = df.drop(columns=["_region", "_prot14", "_strk14",
                           "_prot14_reg", "_strk14_reg"], errors="ignore")
    log.info("Added regional spillover features (14-day window, excl. focal country).")
    return df


def _expanding_zscore(series: pd.Series, min_obs: int = 3) -> pd.Series:
    """Per-country expanding z-score (mirrors utils.expanding_zscore)."""
    exp = series.expanding(min_periods=min_obs)
    mu  = exp.mean()
    sig = exp.std().replace(0.0, np.nan)
    return (series - mu) / sig


def add_supplemental_features(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Adds commodity prices, Gini, recomputed z-scores and interaction terms,
    new engineered features, mineral-producer flags, and commodity x producer
    interaction terms to the merged panel.

    All new columns are guaranteed 0% missing for dates 2017-2021.
    """
    # -----------------------------------------------------------------------
    # a) Commodity prices (global — same for all countries on a given date)
    # -----------------------------------------------------------------------
    if COMMODITY_FILE.exists():
        comm_df = pd.read_parquet(COMMODITY_FILE)
        comm_df["date"] = pd.to_datetime(comm_df["date"])

        # Restrict to panel date range plus a warm-up buffer (2016 onwards)
        merged = merged.merge(comm_df, on="date", how="left")

        comm_feat_cols = [c for c in comm_df.columns if c != "date"]
        # Final forward-fill for any remaining gaps (e.g. end of file)
        merged[comm_feat_cols] = merged[comm_feat_cols].ffill(limit=7)
        log.info("Merged %d commodity feature columns.", len(comm_feat_cols))
    else:
        log.warning(
            "Commodity file not found: %s\n"
            "Run 'python src/fetch_supplemental_data.py' first.",
            COMMODITY_FILE,
        )
        comm_feat_cols = []

    # -----------------------------------------------------------------------
    # a2) Global market sentiment indices (VIX + DXY)
    #     Same-for-all-countries date-level broadcast — zero missingness.
    # -----------------------------------------------------------------------
    if GLOBAL_INDICES_FILE.exists():
        idx_df = pd.read_parquet(GLOBAL_INDICES_FILE)
        idx_df["date"] = pd.to_datetime(idx_df["date"])
        merged = merged.merge(idx_df, on="date", how="left")
        idx_feat_cols = [c for c in idx_df.columns if c != "date"]
        merged[idx_feat_cols] = merged[idx_feat_cols].ffill(limit=7)
        log.info("Merged %d global index feature columns (VIX, DXY).", len(idx_feat_cols))
        comm_feat_cols = comm_feat_cols + idx_feat_cols
    else:
        log.warning(
            "Global indices file not found: %s\n"
            "Run 'python src/fetch_supplemental_data.py' first.",
            GLOBAL_INDICES_FILE,
        )
        idx_feat_cols = []

    # -----------------------------------------------------------------------
    # b) Gini coefficients (country x date)
    # -----------------------------------------------------------------------
    if GINI_FILE.exists():
        gini_df = pd.read_parquet(GINI_FILE)
        gini_df["date"] = pd.to_datetime(gini_df["date"])
        merged = merged.merge(gini_df, on=["country_iso3", "date"], how="left")

        # Fallback: bfill within country, then global mean
        merged["gini_coef"] = (
            merged.groupby("country_iso3")["gini_coef"]
            .transform(lambda s: s.ffill().bfill())
        )
        global_mean_gini = merged["gini_coef"].mean()
        merged["gini_coef"] = merged["gini_coef"].fillna(global_mean_gini)
        log.info("Merged gini_coef. Missing: %.2f%%",
                 merged["gini_coef"].isna().mean() * 100)
    else:
        log.warning(
            "Gini file not found: %s\n"
            "Run 'python src/fetch_supplemental_data.py' first.",
            GINI_FILE,
        )

    # -----------------------------------------------------------------------
    # c) Recompute broken z-scores and interaction terms per country
    # -----------------------------------------------------------------------
    zscore_targets = [
        ("fx_pct_30d",          "fx_pct_30d_z"),
        ("fx_vol_30d",          "fx_vol_30d_z"),
        ("oil_brent_pct_30d",   "oil_brent_pct_30d_z"),
        # Monthly-data z-scores: original pipeline had std=0 from replicated daily values.
        # Recompute from imputed source columns using expanding z-score, then ffill.
        ("inflation_cpi_yoy",    "inflation_cpi_yoy_z"),
        ("unemployment_rate",    "unemployment_rate_z"),
        ("food_cpi_inflation",   "food_cpi_inflation_z"),
        ("energy_cpi_inflation", "energy_cpi_inflation_z"),
    ]

    parts = []
    for iso3, grp in merged.groupby("country_iso3"):
        grp = grp.sort_values("date").copy()
        for src_col, z_col in zscore_targets:
            if src_col in grp.columns:
                grp[z_col] = _expanding_zscore(grp[src_col].reset_index(drop=True), min_obs=3).values
        parts.append(grp)
    merged = pd.concat(parts, ignore_index=True)

    # Fill z-score NaNs: ffill within country (handles warm-up at start),
    # then bfill for countries that start with no data, then global fill with 0
    z_cols = [z for _, z in zscore_targets if z in merged.columns]
    for z_col in z_cols:
        merged[z_col] = (
            merged.groupby("country_iso3")[z_col]
            .transform(lambda s: s.ffill().bfill())
        )
        merged[z_col] = merged[z_col].fillna(0.0)

    # Interaction terms — recompute from scratch
    if "fx_pct_30d" in merged.columns and "political_stability_est" in merged.columns:
        merged["fx_pct_30d_x_instability"] = (
            merged["fx_pct_30d"] * (1.0 - merged["political_stability_est"].clip(-3, 3))
        )
    if "oil_brent_pct_30d" in merged.columns and "inflation_cpi_yoy" in merged.columns:
        # Fill tiny inflation gaps with 0 so the interaction term is always defined
        infl = merged["inflation_cpi_yoy"].fillna(0.0)
        merged["oil_brent_pct_30d_x_inflation"] = merged["oil_brent_pct_30d"] * infl

    log.info("Recomputed z-scores and interaction terms.")

    # -----------------------------------------------------------------------
    # d) New engineered features
    # -----------------------------------------------------------------------

    # covid_period: binary flag, 1 from WHO pandemic declaration 2020-03-11
    merged["covid_period"] = (merged["date"] >= "2020-03-11").astype(int)

    # fx_trend_consistent: sustained depreciation/appreciation signal
    if "fx_pct_7d" in merged.columns and "fx_pct_90d" in merged.columns:
        same_sign = (
            np.sign(merged["fx_pct_7d"].fillna(0)) ==
            np.sign(merged["fx_pct_90d"].fillna(0))
        )
        merged["fx_trend_consistent"] = np.where(
            same_sign & (merged["fx_pct_90d"].fillna(0) > 0),  1,   # sustained depreciation
            np.where(
                same_sign & (merged["fx_pct_90d"].fillna(0) < 0), -1,  # sustained appreciation
                0,
            )
        )
    else:
        merged["fx_trend_consistent"] = 0

    # inflation_accel: change in inflation vs 30 days ago (per country)
    if "inflation_cpi_yoy" in merged.columns:
        parts2 = []
        for iso3, grp in merged.groupby("country_iso3"):
            grp = grp.sort_values("date").copy()
            grp["inflation_accel"] = (
                grp["inflation_cpi_yoy"].diff(30).fillna(0)
            )
            parts2.append(grp)
        merged = pd.concat(parts2, ignore_index=True)
    else:
        merged["inflation_accel"] = 0.0

    log.info("Added engineered features: covid_period, fx_trend_consistent, "
             "inflation_accel.")
    # NOTE: country_protest_baseline and country_strike_baseline are intentionally
    # NOT computed here.  They are computed per-fold in train_backtest.py from
    # training data only, to prevent leakage across walk-forward folds.

    # -----------------------------------------------------------------------
    # e) Binary mineral-producer flags
    # -----------------------------------------------------------------------
    producer_map = {
        "is_copper_producer":   COPPER_PRODUCERS,
        "is_gold_producer":     GOLD_PRODUCERS,
        "is_platinum_producer": PLATINUM_PRODUCERS,
        "is_silver_producer":   SILVER_PRODUCERS,
        "is_nickel_producer":   NICKEL_PRODUCERS,
        "is_coal_producer":     COAL_PRODUCERS,
        "is_natgas_producer":   NATGAS_PRODUCERS,
        "is_lithium_producer":  LITHIUM_PRODUCERS,
    }
    for col, country_set in producer_map.items():
        merged[col] = merged["country_iso3"].isin(country_set).astype(int)

    log.info("Added 8 mineral-producer binary flags.")

    # -----------------------------------------------------------------------
    # f) Commodity x producer interaction terms
    # -----------------------------------------------------------------------
    if "copper_pct_30d" in merged.columns:
        merged["copper_pct_30d_x_copper_prod"]   = (
            merged["copper_pct_30d"] * merged["is_copper_producer"]
        )
    else:
        merged["copper_pct_30d_x_copper_prod"] = 0.0

    if "gold_pct_30d" in merged.columns:
        merged["gold_pct_30d_x_gold_prod"]       = (
            merged["gold_pct_30d"] * merged["is_gold_producer"]
        )
    else:
        merged["gold_pct_30d_x_gold_prod"] = 0.0

    if "platinum_pct_30d" in merged.columns:
        merged["platinum_pct_30d_x_plat_prod"]   = (
            merged["platinum_pct_30d"] * merged["is_platinum_producer"]
        )
    else:
        merged["platinum_pct_30d_x_plat_prod"] = 0.0

    log.info("Added commodity x producer interaction terms.")

    # -----------------------------------------------------------------------
    # g) Economic lag features — delayed transmission of price shocks
    #    Oil/VIX/FAO are global (same value per date); compute on deduplicated
    #    date series then broadcast back via merge.
    # -----------------------------------------------------------------------
    global_lag_specs = []
    if "oil_brent_pct_30d" in merged.columns:
        global_lag_specs += [
            ("oil_brent_pct_30d", 30,  "oil_brent_pct_30d_lag30d"),
            ("oil_brent_pct_30d", 60,  "oil_brent_pct_30d_lag60d"),
            ("oil_brent_pct_30d", 90,  "oil_brent_pct_30d_lag90d"),
        ]
    if "vix_pct_30d" in merged.columns:
        global_lag_specs += [
            ("vix_pct_30d", 30, "vix_pct_30d_lag30d"),
            ("vix_pct_30d", 60, "vix_pct_30d_lag60d"),
        ]
    if "fao_oils_index_yoy" in merged.columns:
        global_lag_specs += [
            ("fao_oils_index_yoy", 30,  "fao_oils_index_yoy_lag1m"),
            ("fao_oils_index_yoy", 90,  "fao_oils_index_yoy_lag3m"),
            ("fao_oils_index_yoy", 180, "fao_oils_index_yoy_lag6m"),
        ]
    if "fao_food_index_yoy" in merged.columns:
        global_lag_specs += [
            ("fao_food_index_yoy", 180, "fao_food_index_yoy_lag6m"),
        ]

    global_lag_new_cols: list[str] = []
    for src_col, lag_d, col_name in global_lag_specs:
        if src_col not in merged.columns or col_name in merged.columns:
            continue
        date_to_val = (
            merged.sort_values("date")
            .drop_duplicates("date")
            .set_index("date")[src_col]
        )
        merged[col_name] = merged["date"].map(date_to_val.shift(lag_d))
        merged[col_name] = (
            merged.groupby("country_iso3")[col_name]
            .transform(lambda s: s.ffill().bfill())
        ).fillna(0.0)
        global_lag_new_cols.append(col_name)
    if global_lag_new_cols:
        log.info("Added %d global economic lag features.", len(global_lag_new_cols))

    # FX lags — per-country (FX rates differ by country)
    fx_lag_new_cols: list[str] = []
    if "fx_pct_30d" in merged.columns:
        fx_lag_days = [30, 60]
        fx_names = [f"fx_pct_30d_lag{d}d" for d in fx_lag_days]
        parts_fx = []
        for iso3, grp in merged.groupby("country_iso3"):
            grp = grp.sort_values("date").copy()
            for d, col_name in zip(fx_lag_days, fx_names):
                grp[col_name] = grp["fx_pct_30d"].shift(d)
            parts_fx.append(grp)
        merged = pd.concat(parts_fx, ignore_index=True)
        for col_name in fx_names:
            merged[col_name] = (
                merged.groupby("country_iso3")[col_name]
                .transform(lambda s: s.ffill().bfill())
            ).fillna(0.0)
        fx_lag_new_cols = fx_names
        log.info("Added %d FX lag features.", len(fx_names))

    # -----------------------------------------------------------------------
    # h) Net oil importer flag and interaction
    # -----------------------------------------------------------------------
    merged["is_net_oil_importer"] = (
        ~merged["country_iso3"].isin(NET_OIL_EXPORTERS)
    ).astype(int)

    if "oil_brent_pct_30d" in merged.columns:
        merged["oil_brent_pct_30d_x_net_importer"] = (
            merged["oil_brent_pct_30d"] * merged["is_net_oil_importer"]
        )
    else:
        merged["oil_brent_pct_30d_x_net_importer"] = 0.0

    log.info("Added net oil importer flag and oil x importer interaction.")

    # -----------------------------------------------------------------------
    # Validation: report missing % for all new columns (2017-2021 only)
    # -----------------------------------------------------------------------
    new_cols = (
        comm_feat_cols
        + ["gini_coef"]
        + [z for _, z in zscore_targets]
        + [
            "covid_period",
            "fx_trend_consistent",
            "inflation_accel",
        ]
        + list(producer_map.keys())
        + [
            "copper_pct_30d_x_copper_prod",
            "gold_pct_30d_x_gold_prod",
            "platinum_pct_30d_x_plat_prod",
        ]
        + global_lag_new_cols
        + fx_lag_new_cols
        + ["is_net_oil_importer", "oil_brent_pct_30d_x_net_importer"]
        + ["gdelt_protest_region_14d", "gdelt_strike_region_14d"]
    )
    new_cols = [c for c in new_cols if c in merged.columns]

    mask_panel = (merged["date"] >= GDELT_START) & (merged["date"] <= GDELT_END)
    panel_check = merged[mask_panel]

    log.info("-" * 50)
    log.info("Supplemental feature validation (2017-2021):")
    any_missing = False
    for col in new_cols:
        pct = panel_check[col].isna().mean() * 100
        if pct > 0:
            log.warning("  MISSING  %-40s  %.2f%%", col, pct)
            any_missing = True
        else:
            log.info("  OK       %-40s  0.00%%", col)
    if not any_missing:
        log.info("All %d supplemental features: 0%% missing for 2017-2021.", len(new_cols))
    log.info("-" * 50)

    return merged


# ---------------------------------------------------------------------------
# Comprehensive feature imputation
# ---------------------------------------------------------------------------

def impute_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill all remaining missing values in model features before z-score recomputation.

    Called after compute_gdelt_lag_features() and before add_supplemental_features(),
    so that source columns are clean when z-scores are (re)computed.

    Strategy by feature group
    -------------------------
    GDELT lag cols          fill 0   (first-row NaN from the shift in _rolling_lag)
    Monthly macro cols      ffill/bfill within country only — countries with no source
                            data at all retain NaN (excluded from that country's feature
                            vector rather than assigned a synthetic cross-country value)
    FAO price cols          ffill/bfill globally on sorted dates (global time series,
                            same value for all countries on a given date)
    GTA count cols          ffill/bfill within country -> 0 fallback
    GTA z-score cols        fill 0   (sparse countries have insufficient variation)
    """
    df = df.copy()

    # 1. GDELT lag features  (shift(1) leaves first row NaN)
    gdelt_lag_cols = [c for c in df.columns if c.startswith("gdelt_") and c.endswith("_lag")]
    for col in gdelt_lag_cols:
        df[col] = df[col].fillna(0.0)
    if gdelt_lag_cols:
        log.info("Imputed %d GDELT lag cols with 0.", len(gdelt_lag_cols))

    # 2. Monthly/annual macro variables (replicated daily within country)
    # ffill/bfill within country bridges interior gaps and the first-year
    # forward-fill lag: WDI/WGI annual indicators have no 2016 prior-year data
    # in the source panel, so 2017 is 100% NaN. bfill() fills from the first
    # available year (2018) which is the closest valid observation.
    # Countries entirely missing a series retain NaN (no cross-country fallback).
    monthly_macro = [
        "inflation_cpi_yoy",
        "food_cpi_inflation",
        "energy_cpi_inflation",
        "unemployment_sa",
        "unemployment_rate",
        "unemployment_total",
        "unemployment_youth",
        # Annual WDI indicators (100% missing in 2017 due to forward-fill lag)
        "gdp_growth",
        "gdp_per_capita_growth",
        # Annual WGI governance indicators (same issue)
        "political_stability_est",
        "voice_accountability_est",
        "government_effectiveness_est",
        "rule_of_law_est",
        "control_of_corruption_est",
        "regulatory_quality_est",
    ]
    filled_macro = []
    for col in [c for c in monthly_macro if c in df.columns]:
        df[col] = df.groupby("country_iso3")[col].transform(
            lambda s: s.ffill().bfill()
        )
        filled_macro.append(col)
    if filled_macro:
        no_data = {
            col: sorted(
                df.loc[df[col].isna(), "country_iso3"].unique().tolist()
            )
            for col in filled_macro
            if df[col].isna().any()
        }
        if no_data:
            for col, countries in no_data.items():
                log.info(
                    "  %s: no source data for %d countries (kept NaN): %s",
                    col, len(countries), countries,
                )
        log.info("Imputed monthly macro cols (ffill/bfill within country, no median fallback): %s", filled_macro)

    # 3. FAO price columns (global — same value for all countries on a given date)
    fao_src_cols = [c for c in df.columns
                    if c.startswith("fao_") and not c.endswith("_z")]
    if fao_src_cols:
        # Sort by date so ffill propagates across countries consistently
        orig_order = df.index
        df = df.sort_values("date")
        # limit=60 covers up to 2 missed months of FAO data
        df[fao_src_cols] = df[fao_src_cols].ffill(limit=60).bfill(limit=60)
        df[fao_src_cols] = df[fao_src_cols].fillna(0.0)
        df = df.sort_values(["country_iso3", "date"]).reset_index(drop=True)
        log.info("Imputed %d FAO source cols (global ffill/bfill).", len(fao_src_cols))

    # 4. GTA count columns (country-specific event counts)
    gta_count_cols = [c for c in df.columns
                      if c.startswith("gta_") and not c.endswith("_z")]
    for col in gta_count_cols:
        df[col] = df.groupby("country_iso3")[col].transform(
            lambda s: s.ffill().bfill()
        )
        df[col] = df[col].fillna(0.0)
    if gta_count_cols:
        log.info("Imputed %d GTA count cols.", len(gta_count_cols))

    # 5. GTA z-score columns (sparse countries have std≈0; fill with 0 = mean)
    gta_z_cols = [c for c in df.columns if c.startswith("gta_") and c.endswith("_z")]
    for col in gta_z_cols:
        df[col] = df[col].fillna(0.0)
    if gta_z_cols:
        log.info("Imputed %d GTA z-score cols with 0.", len(gta_z_cols))

    # Validation: macro cols may legitimately retain NaN (no-data countries);
    # warn only for non-macro cols that should be fully filled.
    non_macro_cols = list(dict.fromkeys(
        gdelt_lag_cols + fao_src_cols + gta_count_cols + gta_z_cols
    ))
    problems = [(c, df[c].isna().mean() * 100)
                for c in non_macro_cols if c in df.columns and df[c].isna().any()]
    if problems:
        log.warning("Unexpected missing values after imputation:")
        for col, pct in problems:
            log.warning("  %s: %.2f%%", col, pct)
    else:
        log.info("All non-macro imputed features: 0%% missing.")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run() -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load feature panel
    # ------------------------------------------------------------------
    if not FEATURE_PANEL_FILE.exists():
        raise FileNotFoundError(f"Feature panel not found: {FEATURE_PANEL_FILE}")

    feat_df = pd.read_parquet(FEATURE_PANEL_FILE)
    feat_df["date"] = pd.to_datetime(feat_df["date"])

    log.info(
        "Feature panel loaded: %d rows | %d countries | %s to %s",
        len(feat_df), feat_df["country_iso3"].nunique(),
        feat_df["date"].min().date(), feat_df["date"].max().date(),
    )

    # Drop ACLED label columns
    drop_cols = [c for c in ACLED_LABEL_COLS if c in feat_df.columns]
    feat_df   = feat_df.drop(columns=drop_cols)
    log.info("Dropped %d ACLED label columns: %s", len(drop_cols), drop_cols)

    # ------------------------------------------------------------------
    # FIX: Fill missing FX data for high-income / Eurozone countries
    # ------------------------------------------------------------------
    # The original markets pipeline only fetched FX for ~17 emerging-market
    # countries.  The 22 high-income countries (AUS, CAN, DEU, GBR, etc.)
    # had 100% NaN FX features.  We supplement with data from
    # fetch_fx_missing.py which downloaded the same USD{CCY}=X Yahoo Finance
    # tickers using the identical feature computation.
    FX_COLS = (
        ["fx_lcu_usd", "fx_log_return"]
        + [f"fx_pct_{w}d" for w in [7, 30, 90]]
        + [f"fx_vol_{w}d" for w in [7, 30]]
    )
    if FX_MISSING_FILE.exists():
        fx_fill = pd.read_parquet(FX_MISSING_FILE)
        fx_fill["date"] = pd.to_datetime(fx_fill["date"])
        # Only use rows for countries that are actually missing FX in the panel
        missing_fx_countries = (
            feat_df.groupby("country_iso3")["fx_pct_30d"]
            .apply(lambda x: x.isna().all())
        )
        missing_fx_countries = set(missing_fx_countries[missing_fx_countries].index)
        fx_fill = fx_fill[fx_fill["country_iso3"].isin(missing_fx_countries)]
        if not fx_fill.empty:
            # Merge FX fill on (country_iso3, date) and update missing values
            feat_df = feat_df.merge(
                fx_fill.rename(columns={c: f"{c}_fill" for c in FX_COLS}),
                on=["country_iso3", "date"],
                how="left",
            )
            for col in FX_COLS:
                if col in feat_df.columns and f"{col}_fill" in feat_df.columns:
                    feat_df[col] = feat_df[col].fillna(feat_df[f"{col}_fill"])
            feat_df = feat_df.drop(columns=[f"{c}_fill" for c in FX_COLS
                                            if f"{c}_fill" in feat_df.columns])
            log.info(
                "Filled FX data for %d countries: %s",
                len(missing_fx_countries), sorted(missing_fx_countries),
            )
    else:
        log.warning(
            "FX fill file not found: %s\n"
            "Run 'python src/fetch_fx_missing.py' to generate it.",
            FX_MISSING_FILE,
        )

    # Restrict to GDELT date window
    feat_df = feat_df[
        (feat_df["date"] >= GDELT_START) &
        (feat_df["date"] <= GDELT_END)
    ].copy()
    log.info("After date filter (2017-2021): %d rows", len(feat_df))

    # ------------------------------------------------------------------
    # 2. Load GDELT labels
    # ------------------------------------------------------------------
    if not GDELT_LABELS_FILE.exists():
        raise FileNotFoundError(
            f"GDELT labels not found: {GDELT_LABELS_FILE}\n"
            "Run build_labels_modelling.py first."
        )

    gdelt_df = pd.read_parquet(GDELT_LABELS_FILE)
    gdelt_df["date"] = pd.to_datetime(gdelt_df["date"])
    gdelt_df = gdelt_df.rename(columns={"iso3": "country_iso3"})

    # Keep only needed GDELT columns
    keep = ["country_iso3", "date"] + [c for c in GDELT_LABEL_COLS if c in gdelt_df.columns]
    gdelt_df = gdelt_df[keep]

    # Restrict to the 39 countries in the feature panel
    panel_countries = set(feat_df["country_iso3"].unique())
    gdelt_df = gdelt_df[gdelt_df["country_iso3"].isin(panel_countries)]

    log.info(
        "GDELT labels: %d rows | %d countries (filtered to panel countries)",
        len(gdelt_df), gdelt_df["country_iso3"].nunique(),
    )

    # ------------------------------------------------------------------
    # 3. Merge
    # ------------------------------------------------------------------
    merged = feat_df.merge(gdelt_df, on=["country_iso3", "date"], how="left")

    # Fill zeros for countries that had GDELT coverage but no events
    # (NaN after left-join means no GDELT data at all for that country-day)
    # NOTE: do NOT fillna the forward-looking label cols here — they are
    # recomputed correctly below after the full date grid is established.
    for col in ["protest_today", "strike_today",
                "n_protest_events", "n_strike_events", "n_articles", "has_movement"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype(int)

    merged["coverage_flag"] = merged["coverage_flag"].fillna("low")

    # ------------------------------------------------------------------
    # FIX: Recompute forward-looking labels from protest_today / strike_today
    # ------------------------------------------------------------------
    # The pre-computed protest_7d / protest_30d etc. in the GDELT labels file
    # were built only on days when GDELT had articles for a country.  Days with
    # no coverage are absent from the labels file and get fillna(0) after the
    # left join.  This creates false-negative labels: a day with no GDELT
    # articles but a protest in the next 7 days gets protest_7d=0 incorrectly.
    #
    # Fix: recompute all forward-looking targets from protest_today (which is
    # correctly 0 for no-coverage days and 1 when events were detected) on the
    # full panel grid.  The same rolling-shift logic used in
    # build_labels_modelling.py is applied here after the complete grid is live.
    log.info("Recomputing forward-looking labels from protest_today / strike_today ...")
    parts = []
    for iso3, grp in merged.groupby("country_iso3"):
        grp = grp.sort_values("date").set_index("date")
        for src_col, horizons in [("protest_today", [7, 30]), ("strike_today", [7, 30])]:
            base = src_col.replace("_today", "")
            for h in horizons:
                grp[f"{base}_{h}d"] = (
                    grp[src_col].astype(float)
                    .rolling(f"{h}D", min_periods=1)
                    .max()
                    .shift(-h)
                    .fillna(0)
                    .astype(int)
                )
        parts.append(grp.reset_index())
    merged = pd.concat(parts, ignore_index=True)
    log.info("Forward-looking labels recomputed.")

    log.info(
        "Merged panel: %d rows | %d countries",
        len(merged), merged["country_iso3"].nunique(),
    )

    # ------------------------------------------------------------------
    # FIX: Broadcast global market indicators to all countries
    # ------------------------------------------------------------------
    # oil_brent_pct_14d, oil_brent_pct_30d, and yield_us10y are GLOBAL
    # indicators (same value on a given date regardless of country), but were
    # only merged for the ~17 countries whose FX data was sourced together.
    # The other 22 countries have 100% NaN for these columns.
    #
    # Fix: build a date-level lookup from the rows that do have the data, then
    # forward-fill it to all countries.
    global_cols = [c for c in ["oil_brent_pct_14d", "oil_brent_pct_30d",
                                "oil_brent_usd", "yield_us10y"]
                   if c in merged.columns]
    if global_cols:
        # Deduplicate by date — all non-NaN rows for a date should have the same value
        global_lookup = (
            merged.dropna(subset=global_cols[:1])   # use first col to filter
            .groupby("date")[global_cols]
            .first()
        )
        if not global_lookup.empty:
            # Merge back on date to fill all countries
            merged = merged.drop(columns=global_cols)
            merged = merged.merge(global_lookup.reset_index(), on="date", how="left")
            log.info(
                "Broadcast global indicators (%s) to all countries — "
                "filled %d previously-missing country-days.",
                global_cols,
                merged[global_cols[0]].notna().sum() - global_lookup.shape[0],
            )

    # ------------------------------------------------------------------
    # 4. Compute GDELT lag features
    # ------------------------------------------------------------------
    log.info("Computing GDELT lag features...")
    merged = compute_gdelt_lag_features(merged)

    lag_cols = [c for c in merged.columns if c.startswith("gdelt_") and c.endswith("_lag")]
    log.info("Added lag features: %s", lag_cols)

    # ------------------------------------------------------------------
    # 4a. Compute regional spillover features
    # ------------------------------------------------------------------
    log.info("Computing regional spillover features...")
    merged = compute_regional_spillover(merged)

    # ------------------------------------------------------------------
    # 4b. Impute all model features before z-score recomputation
    # ------------------------------------------------------------------
    log.info("Imputing model features ...")
    merged = impute_model_features(merged)

    # ------------------------------------------------------------------
    # 4c. Add supplemental features (commodities, Gini, z-scores, flags)
    # ------------------------------------------------------------------
    log.info("Adding supplemental features ...")
    merged = add_supplemental_features(merged)

    # ------------------------------------------------------------------
    # 5. Coverage summary
    # ------------------------------------------------------------------
    gdelt_countries = set(gdelt_df["country_iso3"].unique())
    missing         = panel_countries - gdelt_countries
    if missing:
        log.warning(
            "%d panel countries have NO GDELT label data (all zeros): %s",
            len(missing), sorted(missing),
        )

    flag_counts = merged.groupby("coverage_flag").size()
    log.info("Coverage flag distribution:\n%s", flag_counts.to_string())

    for col in ["protest_7d", "strike_7d", "protest_30d", "strike_30d"]:
        if col in merged.columns:
            rate = merged[col].mean() * 100
            log.info("  %s positive rate: %.1f%%", col, rate)

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    merged = merged.sort_values(["country_iso3", "date"]).reset_index(drop=True)
    merged.to_parquet(OUT_FILE, index=False)
    log.info("Saved -> %s  (%d rows x %d cols)", OUT_FILE, len(merged), len(merged.columns))

    return merged


if __name__ == "__main__":
    run()
