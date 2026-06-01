# Publication figures for the results chapter (no-lag feature importance).

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_V3      = Path(__file__).resolve().parents[1]
PROC_DIR = _V3 / "data" / "processed"
FIG_DIR  = _V3 / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["protest_7d", "strike_7d"]
TARGET_LABELS = {"protest_7d": "Protest (7-day)", "strike_7d": "Strike (7-day)"}

# ---------------------------------------------------------------------------
# Feature name → readable label mapping
# ---------------------------------------------------------------------------
LABEL_MAP: dict[str, str] = {
    # Governance
    "rule_of_law_est":                   "Rule of Law",
    "political_stability_est":           "Political Stability",
    "government_effectiveness_est":      "Govt. Effectiveness",
    "voice_accountability_est":          "Voice & Accountability",
    # Macro
    "gdp_growth":                        "GDP Growth",
    "gdp_per_capita_growth":             "GDP per Capita Growth",
    "inflation_cpi_yoy":                 "CPI Inflation (YoY)",
    "inflation_cpi_yoy_z":               "CPI Inflation Z-score",
    "inflation_accel":                   "Inflation Acceleration",
    "food_cpi_inflation":                "Food CPI Inflation",
    "food_cpi_inflation_z":              "Food CPI Z-score",
    "energy_cpi_inflation":              "Energy CPI Inflation",
    "energy_cpi_inflation_z":            "Energy CPI Z-score",
    "unemployment_total":                "Unemployment Rate",
    "unemployment_youth":                "Youth Unemployment",
    "unemployment_sa":                   "Unemployment (SA)",
    "unemployment_rate":                 "Unemployment Rate",
    "unemployment_rate_z":               "Unemployment Z-score",
    # Structural
    "gini_coef":                         "Gini Coefficient",
    "covid_period":                      "COVID Period",
    "fx_trend_consistent":               "FX Trend Consistent",
    # Commodity interactions
    "copper_pct_30d_x_copper_prod":      "Copper Price × Production",
    "gold_pct_30d_x_gold_prod":          "Gold Price × Production",
    "platinum_pct_30d_x_plat_prod":      "Platinum Price × Production",
    "oil_brent_pct_30d_x_net_importer":  "Oil Price × Net Importer",
    "fx_pct_30d_x_instability":          "FX Change × Instability",
    "oil_brent_pct_30d_x_inflation":     "Oil Price × Inflation",
    "fao_cereals_yoy_x_instability":     "FAO Cereals × Instability",
    "fao_food_yoy_x_youth_unemp":        "FAO Food × Youth Unemp.",
    # Financial
    "fx_pct_7d":                         "FX Change (7d)",
    "fx_pct_30d":                        "FX Change (30d)",
    "fx_pct_90d":                        "FX Change (90d)",
    "fx_pct_30d_z":                      "FX Change Z-score",
    "fx_pct_30d_lag30d":                 "FX Change (30d, lag 30d)",
    "fx_pct_30d_lag60d":                 "FX Change (30d, lag 60d)",
    "fx_vol_7d":                         "FX Volatility (7d)",
    "fx_vol_30d":                        "FX Volatility (30d)",
    "fx_vol_30d_z":                      "FX Volatility Z-score",
    "oil_brent_pct_14d":                 "Oil Price Change (14d)",
    "oil_brent_pct_30d":                 "Oil Price Change (30d)",
    "oil_brent_pct_30d_z":               "Oil Price Change Z-score",
    "oil_brent_pct_30d_lag30d":          "Oil Price Change (lag 30d)",
    "oil_brent_pct_30d_lag60d":          "Oil Price Change (lag 60d)",
    "oil_brent_pct_30d_lag90d":          "Oil Price Change (lag 90d)",
    "copper_pct_30d":                    "Copper Price Change (30d)",
    "copper_pct_90d":                    "Copper Price Change (90d)",
    "copper_vol_30d":                    "Copper Volatility (30d)",
    "gold_pct_30d":                      "Gold Price Change (30d)",
    "gold_vol_30d":                      "Gold Volatility (30d)",
    "platinum_pct_30d":                  "Platinum Price Change (30d)",
    "silver_pct_30d":                    "Silver Price Change (30d)",
    "natgas_pct_30d":                    "Natural Gas Change (30d)",
    "vix_level":                         "VIX Level",
    "vix_pct_30d":                       "VIX Change (30d)",
    "vix_7d_ma":                         "VIX 7-day MA",
    "vix_pct_30d_lag30d":                "VIX Change (lag 30d)",
    "vix_pct_30d_lag60d":                "VIX Change (lag 60d)",
    "dxy_level":                         "DXY Level",
    "dxy_pct_30d":                       "DXY Change (30d)",
    "dxy_vol_30d":                       "DXY Volatility (30d)",
    "yield_us10y":                       "US 10Y Yield",
    # FAO
    "fao_food_index_yoy":                "FAO Food Index (YoY)",
    "fao_food_index_yoy_above90":        "FAO Food Index >90th pct.",
    "fao_food_index_yoy_lag1m":          "FAO Food Index (lag 1m)",
    "fao_food_index_yoy_lag3m":          "FAO Food Index (lag 3m)",
    "fao_food_index_yoy_lag6m":          "FAO Food Index (lag 6m)",
    "fao_cereals_index_yoy":             "FAO Cereals Index (YoY)",
    "fao_cereals_index_yoy_above90":     "FAO Cereals >90th pct.",
    "fao_cereals_index_yoy_lag1m":       "FAO Cereals Index (lag 1m)",
    "fao_cereals_index_yoy_lag3m":       "FAO Cereals Index (lag 3m)",
    "fao_cereals_index_yoy_lag6m":       "FAO Cereals Index (lag 6m)",
    "fao_oils_index_yoy":                "FAO Oils Index (YoY)",
    "fao_oils_index_yoy_lag1m":          "FAO Oils Index (lag 1m)",
    "fao_oils_index_yoy_lag3m":          "FAO Oils Index (lag 3m)",
    "fao_oils_index_yoy_lag6m":          "FAO Oils Index (lag 6m)",
    # GTA
    "gta_harmful_events":                "GTA Harmful Events",
    "gta_harmful_events_z":              "GTA Harmful Events Z-score",
    "gta_liberalising_events":           "GTA Liberalising Events",
    "gta_liberalising_events_z":         "GTA Liberalising Events Z-score",
    "gta_30d_count":                     "GTA Events (30d)",
    "gta_30d_count_z":                   "GTA Events Z-score (30d)",
    "gta_90d_count":                     "GTA Events (90d)",
    "gta_90d_count_z":                   "GTA Events Z-score (90d)",
    # Temporal
    "month_sin":                         "Month (sin)",
    "month_cos":                         "Month (cos)",
    # Baselines
    "country_protest_baseline":          "Protest Baseline",
    "country_strike_baseline":           "Strike Baseline",
}

GDELT_LAG_PREFIXES = (
    "gdelt_protest_", "gdelt_strike_",
    "gdelt_protest_region_", "gdelt_strike_region_",
)


def clean_feature(raw: str) -> str:
    """Strip sklearn pipeline prefixes and map to readable label."""
    name = raw
    for pfx in ("num__", "fe__"):
        if name.startswith(pfx):
            name = name[len(pfx):]
    return LABEL_MAP.get(name, name.replace("_", " ").title())


def is_gdelt_lag(raw: str) -> bool:
    name = raw.removeprefix("num__").removeprefix("fe__")
    return any(name.startswith(p) for p in GDELT_LAG_PREFIXES)


def is_country_fe(raw: str) -> bool:
    return raw.startswith("fe__country_iso3")


# ---------------------------------------------------------------------------
# Plot style
# ---------------------------------------------------------------------------
STYLE = {
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
}
plt.rcParams.update(STYLE)

POS_COLOR = "#d62728"
NEG_COLOR = "#1f77b4"
BAR_COLOR = "#636363"


# ===========================================================================
# Figure 3 — M3 logistic regression (full, includes GDELT lags)
# ===========================================================================

def load_lr_coefs(target: str, model_name: str, exclude_gdelt: bool = False) -> pd.DataFrame:
    """Generic LR coef loader: average across folds, optionally exclude GDELT lags."""
    path = PROC_DIR / target / "coefs_lr.csv"
    df   = pd.read_csv(path)
    df   = df[df["model_name"] == model_name].copy()
    df   = df[~df["feature"].apply(is_country_fe)]
    if exclude_gdelt:
        df = df[~df["feature"].apply(is_gdelt_lag)]
    avg = (df.groupby("feature")["coefficient"]
             .mean()
             .reset_index()
             .rename(columns={"coefficient": "coef"}))
    avg["abs_coef"] = avg["coef"].abs()
    avg["label"]    = avg["feature"].apply(clean_feature)
    return avg.nlargest(10, "abs_coef").sort_values("abs_coef")


def fig_lr_importance(model_name: str, title: str, fname: str,
                      exclude_gdelt: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, target in zip(axes, TARGETS):
        df = load_lr_coefs(target, model_name, exclude_gdelt)
        colors = [POS_COLOR if c >= 0 else NEG_COLOR for c in df["coef"]]
        ax.barh(df["label"], df["abs_coef"], color=colors, edgecolor="white", height=0.65)
        ax.set_xlabel("Absolute standardised logistic regression coefficient", fontsize=13)
        ax.set_title(TARGET_LABELS[target], fontsize=14, fontweight="bold")
        ax.tick_params(axis="y", labelsize=12)
        ax.tick_params(axis="x", labelsize=12)

    handles = [
        mpatches.Patch(color=POS_COLOR, label="Avg. direction: increases P(event)"),
        mpatches.Patch(color=NEG_COLOR, label="Avg. direction: decreases P(event)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=12, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"{fname}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


# ===========================================================================
# Figure 4 — M6 XGBoost SHAP (full model, includes GDELT lags)
# ===========================================================================

def load_xgb_shap(target: str, model_name: str, exclude_gdelt: bool = False) -> pd.DataFrame:
    """Generic XGBoost SHAP loader."""
    path = PROC_DIR / target / "shap_importance.csv"
    df   = pd.read_csv(path)
    df   = df[df["model_name"] == model_name].copy()
    df   = df[~df["feature"].apply(is_country_fe)]
    if exclude_gdelt:
        df = df[~df["feature"].apply(is_gdelt_lag)]
    df["label"] = df["feature"].apply(clean_feature)
    return df.nlargest(10, "mean_abs_shap").sort_values("mean_abs_shap")


def fig_xgb_shap(model_name: str, title: str, fname: str,
                 exclude_gdelt: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, target in zip(axes, TARGETS):
        df = load_xgb_shap(target, model_name, exclude_gdelt)
        ax.barh(df["label"], df["mean_abs_shap"], color=BAR_COLOR,
                edgecolor="white", height=0.65)
        ax.set_xlabel("Mean absolute SHAP value", fontsize=13)
        ax.set_title(TARGET_LABELS[target], fontsize=14, fontweight="bold")
        ax.tick_params(axis="y", labelsize=12)
        ax.tick_params(axis="x", labelsize=12)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = FIG_DIR / f"{fname}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


# ===========================================================================
# Report top features
# ===========================================================================

def report_top_features() -> None:
    for label, loader, kwargs in [
        ("M3 LR Structural", load_lr_coefs,
         dict(model_name="model3_structural", exclude_gdelt=False)),
        ("M5 LR No-Lag",     load_lr_coefs,
         dict(model_name="model_lr_nolag",    exclude_gdelt=True)),
        ("M6 XGB Full",      load_xgb_shap,
         dict(model_name="model5_xgb",        exclude_gdelt=False)),
        ("M7 XGB No-Lag",    load_xgb_shap,
         dict(model_name="model6_xgb_nolag",  exclude_gdelt=True)),
    ]:
        print("\n=== " + label + " ===")
        for target in TARGETS:
            df = loader(target, **kwargs)
            print("\n  " + target + ":")
            is_lr = "coef" in df.columns
            for _, row in df.sort_values(
                "abs_coef" if is_lr else "mean_abs_shap", ascending=False
            ).iterrows():
                if is_lr:
                    print("    " + row["label"].ljust(45) + "  coef=" + f"{row['coef']:+.4f}")
                else:
                    print("    " + row["label"].ljust(45) +
                          "  mean_abs_shap=" + f"{row['mean_abs_shap']:.4f}" +
                          "  mean_shap=" + f"{row['mean_shap']:+.4f}")


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("Generating M5 logistic regression feature importance figure...")
    fig_lr_importance(
        model_name="model_lr_nolag",
        title=("M5 No-Lag Logistic Regression — Indicator Importance\n"
               "(top 10 features by absolute standardised coefficient, averaged across folds)"),
        fname="m5_logistic_no_lag_feature_importance",
        exclude_gdelt=True,
    )

    print("Generating M7 XGBoost SHAP feature importance figure...")
    fig_xgb_shap(
        model_name="model6_xgb_nolag",
        title=("M7 No-Lag XGBoost — Indicator Importance (SHAP)\n"
               "(top 10 features by mean absolute SHAP value, averaged across folds)"),
        fname="m7_xgboost_no_lag_shap_importance",
        exclude_gdelt=True,
    )

    print("Generating M3 logistic regression feature importance figure...")
    fig_lr_importance(
        model_name="model3_structural",
        title=("M3 Structural Logistic Regression — Indicator Importance\n"
               "(top 10 features by absolute standardised coefficient, averaged across folds)"),
        fname="m3_logistic_structural_feature_importance",
        exclude_gdelt=False,
    )

    print("Generating M6 XGBoost SHAP feature importance figure...")
    fig_xgb_shap(
        model_name="model5_xgb",
        title=("M6 Full XGBoost — Indicator Importance (SHAP)\n"
               "(top 10 features by mean absolute SHAP value, averaged across folds)"),
        fname="m6_xgboost_full_shap_importance",
        exclude_gdelt=False,
    )

    report_top_features()
    print("\nDone.")
