"""Patch all remaining missing values in modelling_panel_gdelt.parquet."""
import pandas as pd
import numpy as np

PANEL_PATH = "Social_Disruptions/Likelihood_Modelling/v2/data/interim/modelling_panel_gdelt.parquet"

panel = pd.read_parquet(PANEL_PATH)
panel['date'] = pd.to_datetime(panel['date'])
panel = panel.sort_values(['country_iso3', 'date']).reset_index(drop=True)

def expanding_zscore(series, min_obs=3):
    exp = series.expanding(min_periods=min_obs)
    mu  = exp.mean()
    sig = exp.std().replace(0.0, np.nan)
    z   = (series - mu) / sig
    return z.ffill().bfill().fillna(0.0)

print("=== Patching missing data ===")

# 1. unemployment_sa (10 EM countries): fill with unemployment_total
mask_sa = panel['unemployment_sa'].isnull()
panel.loc[mask_sa, 'unemployment_sa'] = panel.loc[mask_sa, 'unemployment_total']
print(f"1. unemployment_sa: filled {mask_sa.sum():,} rows from unemployment_total")

# 2. unemployment_rate (LAO, MOZ, NAM): fill with unemployment_total, recompute z-score
mask_ur = panel['unemployment_rate'].isnull()
panel.loc[mask_ur, 'unemployment_rate'] = panel.loc[mask_ur, 'unemployment_total']
ur_countries = panel.loc[mask_ur, 'country_iso3'].unique()
for iso3 in ur_countries:
    idx = panel['country_iso3'] == iso3
    panel.loc[idx, 'unemployment_rate_z'] = expanding_zscore(
        panel.loc[idx, 'unemployment_rate'].reset_index(drop=True)
    ).values
print(f"2. unemployment_rate: filled {mask_ur.sum():,} rows; recomputed z-scores for {list(ur_countries)}")

# 3. food_cpi_inflation: fill fully-absent countries with 0 (z-score already 0)
mask_food = panel['food_cpi_inflation'].isnull()
panel.loc[mask_food, 'food_cpi_inflation'] = 0.0
print(f"3. food_cpi_inflation: filled {mask_food.sum():,} rows with 0.0")

# 4. energy_cpi_inflation: same treatment
mask_energy = panel['energy_cpi_inflation'].isnull()
panel.loc[mask_energy, 'energy_cpi_inflation'] = 0.0
print(f"4. energy_cpi_inflation: filled {mask_energy.sum():,} rows with 0.0")

# 5. FX pct-change warmup (early 2017, 16 EM currencies)
fx_pct_cols = ['fx_pct_7d', 'fx_pct_30d', 'fx_pct_90d', 'fx_pct_30d_lag30d', 'fx_pct_30d_lag60d']
fx_vol_cols = ['fx_vol_7d', 'fx_vol_30d']

for col in fx_pct_cols:
    if col in panel.columns:
        n = panel[col].isnull().sum()
        if n > 0:
            panel[col] = panel.groupby('country_iso3')[col].transform(
                lambda s: s.bfill().fillna(0.0)
            )
            print(f"5. {col}: filled {n:,} rows with 0.0")

for col in fx_vol_cols:
    if col in panel.columns:
        n = panel[col].isnull().sum()
        if n > 0:
            panel[col] = panel.groupby('country_iso3')[col].transform(
                lambda s: s.bfill().fillna(0.0)
            )
            print(f"5. {col}: filled {n:,} rows (bfill then 0)")

# Recompute FX z-scores
for src, z in [('fx_pct_30d', 'fx_pct_30d_z'), ('fx_vol_30d', 'fx_vol_30d_z')]:
    if z in panel.columns:
        for iso3, grp in panel.groupby('country_iso3'):
            idx = panel['country_iso3'] == iso3
            panel.loc[idx, z] = expanding_zscore(
                panel.loc[idx, src].reset_index(drop=True)
            ).values
print("5. fx_pct_30d_z / fx_vol_30d_z: recomputed expanding z-scores")

# 6. Oil/DXY/VIX global pct-change warmup (early 2017)
global_pct_cols = [
    'oil_brent_pct_14d', 'oil_brent_pct_30d',
    'oil_brent_pct_30d_lag30d', 'oil_brent_pct_30d_lag60d', 'oil_brent_pct_30d_lag90d',
    'dxy_pct_30d', 'dxy_vol_30d',
    'vix_pct_30d', 'vix_pct_30d_lag30d', 'vix_pct_30d_lag60d',
]
for col in global_pct_cols:
    if col in panel.columns:
        n = panel[col].isnull().sum()
        if n > 0:
            panel[col] = panel[col].bfill().fillna(0.0)
            print(f"6. {col}: filled {n:,} rows")

# yield_us10y is a level — bfill only (don't zero-fill)
if 'yield_us10y' in panel.columns:
    n = panel['yield_us10y'].isnull().sum()
    if n > 0:
        panel['yield_us10y'] = panel['yield_us10y'].bfill().ffill()
        print(f"6. yield_us10y: filled {n:,} rows (level, bfill)")

# Recompute oil z-score
if 'oil_brent_pct_30d_z' in panel.columns:
    panel['oil_brent_pct_30d_z'] = panel.groupby('country_iso3')['oil_brent_pct_30d'].transform(
        lambda s: expanding_zscore(s.reset_index(drop=True)).values
    )
    print("6. oil_brent_pct_30d_z: recomputed")

# 7. Recompute interaction terms
if 'fx_pct_30d' in panel.columns and 'political_stability_est' in panel.columns:
    panel['fx_pct_30d_x_instability'] = (
        panel['fx_pct_30d'] * (1 - panel['political_stability_est'].clip(-3, 3))
    )
    print("7. fx_pct_30d_x_instability: recomputed")

if 'oil_brent_pct_30d' in panel.columns and 'inflation_cpi_yoy' in panel.columns:
    panel['oil_brent_pct_30d_x_inflation'] = (
        panel['oil_brent_pct_30d'] * panel['inflation_cpi_yoy']
    )
    print("7. oil_brent_pct_30d_x_inflation: recomputed")

# oil x net_importer interaction
if 'oil_brent_pct_30d_x_net_importer' in panel.columns:
    if 'net_importer' in panel.columns:
        panel['oil_brent_pct_30d_x_net_importer'] = (
            panel['oil_brent_pct_30d'] * panel['net_importer']
        )
    else:
        panel['oil_brent_pct_30d_x_net_importer'] = (
            panel['oil_brent_pct_30d_x_net_importer'].fillna(0.0)
        )
    print("7. oil_brent_pct_30d_x_net_importer: recomputed/filled")

# 8. Final check
FEATURES_M0     = ["gdelt_protest_7d_lag","gdelt_protest_28d_lag","gdelt_strike_7d_lag",
                    "gdelt_strike_28d_lag","gdelt_protest_region_14d","gdelt_strike_region_14d"]
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
FEATURES_M2_ADD = ["gdp_growth","gdp_per_capita_growth","inflation_cpi_yoy","inflation_cpi_yoy_z",
                    "unemployment_total","unemployment_youth","unemployment_sa","unemployment_rate",
                    "unemployment_rate_z","political_stability_est","voice_accountability_est",
                    "government_effectiveness_est","rule_of_law_est",
                    "fx_pct_30d_x_instability","oil_brent_pct_30d_x_inflation",
                    "food_cpi_inflation","food_cpi_inflation_z","energy_cpi_inflation","energy_cpi_inflation_z"]
FEATURES_M3_ADD = ["gini_coef","covid_period","fx_trend_consistent","inflation_accel",
                    "copper_pct_30d_x_copper_prod","gold_pct_30d_x_gold_prod",
                    "platinum_pct_30d_x_plat_prod","oil_brent_pct_30d_x_net_importer"]
FEATURES_M4_ADD = ["month_sin","month_cos",
                    "fao_food_index_yoy","fao_cereals_index_yoy","fao_oils_index_yoy",
                    "fao_food_index_yoy_above90","fao_cereals_index_yoy_above90",
                    "fao_cereals_index_yoy_lag1m","fao_cereals_index_yoy_lag3m","fao_cereals_index_yoy_lag6m",
                    "fao_food_index_yoy_lag1m","fao_food_index_yoy_lag3m","fao_food_index_yoy_lag6m",
                    "fao_oils_index_yoy_lag1m","fao_oils_index_yoy_lag3m","fao_oils_index_yoy_lag6m",
                    "fao_cereals_yoy_x_instability","fao_food_yoy_x_youth_unemp",
                    "gta_harmful_events","gta_harmful_events_z",
                    "gta_liberalising_events","gta_liberalising_events_z",
                    "gta_30d_count","gta_30d_count_z","gta_90d_count","gta_90d_count_z"]

ALL = FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD + FEATURES_M4_ADD
avail = [f for f in ALL if f in panel.columns]
sub = panel[panel['date'].dt.year.between(2017, 2021)]
remaining = sub[avail].isnull().sum()
remaining = remaining[remaining > 0]

print(f"\n=== Remaining NaN after patching ({len(avail)} features) ===")
if len(remaining) == 0:
    print("  NONE -- all 91 modelling features are fully populated")
else:
    print(remaining)
    overall = sub[avail].isnull().mean().mean()
    print(f"\nOverall missingness: {100*overall:.3f}%")

panel.to_parquet(PANEL_PATH, index=False)
print(f"\nSaved -> {PANEL_PATH}")
