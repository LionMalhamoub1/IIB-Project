# Generates dissertation figures from the GDELT modelling pipeline.

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve as sk_calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve, auc

warnings.filterwarnings("ignore")

_HERE    = Path(__file__).resolve().parent
_V2      = _HERE.parent
PROC_DIR = _V2 / "data" / "processed"
FIG_DIR  = _V2 / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Targets that train_backtest.py actually produces
ACTIVE_TARGETS = ["protest_7d", "strike_7d"]

TARGET_LABELS = {
    "protest_7d":  "Protest (7-day)",
    "strike_7d":   "Strike (7-day)",
}

MODEL_ORDER = [
    "model0_persistence",
    "model1_markets",
    "model2_full",
    "model3_structural",
    "model4_fao",
    "model_lr_nolag",
    "model9_twostage",
    "model5_xgb",
    "model6_xgb_nolag",
    "model7_xgb_nobaseline",
]

MODEL_LABELS = {
    "model0_persistence":      "M0 Persistence",
    "model1_markets":          "M1 + Markets",
    "model2_full":             "M2 + Macro",
    "model3_structural":       "M3 + Structural",
    "model4_fao":              "M4 + FAO/GTA",
    "model_lr_nolag":          "LR No Lags",
    "model9_twostage":         "M9 Two-Stage LR",
    "model5_xgb":              "M5 XGBoost (full)",
    "model6_xgb_nolag":        "M6 XGBoost (no lags)",
    "model7_xgb_nobaseline":   "M7 XGBoost (no baseline)",
}

PALETTE = {
    "protest_7d":  "#2166ac",
    "strike_7d":   "#d73027",
    "strike_30d":  "#f46d43",
}

STYLE = {
    "font.family":        "sans-serif",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "figure.dpi":         150,
}
plt.rcParams.update(STYLE)

# ---------------------------------------------------------------------------
# Feature block definitions (must match train_backtest.py)
# ---------------------------------------------------------------------------

_FEATURES_M0 = [
    "gdelt_protest_7d_lag", "gdelt_protest_28d_lag",
    "gdelt_strike_7d_lag",  "gdelt_strike_28d_lag",
    "gdelt_protest_region_14d", "gdelt_strike_region_14d",
]
_FEATURES_M1 = [
    "fx_pct_7d","fx_pct_30d","fx_pct_90d","fx_vol_7d","fx_vol_30d",
    "oil_brent_pct_14d","oil_brent_pct_30d","yield_us10y",
    "fx_pct_30d_z","fx_vol_30d_z","oil_brent_pct_30d_z",
    "copper_pct_30d","copper_pct_90d","copper_vol_30d",
    "gold_pct_30d","gold_vol_30d","platinum_pct_30d",
    "silver_pct_30d","natgas_pct_30d",
    "vix_level","vix_pct_30d","vix_7d_ma",
    "dxy_level","dxy_pct_30d","dxy_vol_30d",
    "oil_brent_pct_30d_lag30d","oil_brent_pct_30d_lag60d","oil_brent_pct_30d_lag90d",
    "vix_pct_30d_lag30d","vix_pct_30d_lag60d",
    "fx_pct_30d_lag30d","fx_pct_30d_lag60d",
]
_FEATURES_M2 = [
    "gdp_growth","gdp_per_capita_growth",
    "inflation_cpi_yoy","inflation_cpi_yoy_z",
    "unemployment_total","unemployment_youth",
    "unemployment_sa","unemployment_rate","unemployment_rate_z",
    "political_stability_est","voice_accountability_est",
    "government_effectiveness_est","rule_of_law_est",
    "fx_pct_30d_x_instability","oil_brent_pct_30d_x_inflation",
    "food_cpi_inflation","food_cpi_inflation_z",
    "energy_cpi_inflation","energy_cpi_inflation_z",
]
_FEATURES_M3 = [
    "gini_coef","covid_period","fx_trend_consistent","inflation_accel",
    "country_protest_baseline","country_strike_baseline",
    "copper_pct_30d_x_copper_prod","gold_pct_30d_x_gold_prod",
    "platinum_pct_30d_x_plat_prod","oil_brent_pct_30d_x_net_importer",
]
_FEATURES_M4 = [
    "month_sin","month_cos",
    "fao_food_index_yoy","fao_cereals_index_yoy","fao_oils_index_yoy",
    "fao_food_index_yoy_above90","fao_cereals_index_yoy_above90",
    "fao_cereals_index_yoy_lag1m","fao_cereals_index_yoy_lag3m","fao_cereals_index_yoy_lag6m",
    "fao_food_index_yoy_lag1m","fao_food_index_yoy_lag3m","fao_food_index_yoy_lag6m",
    "fao_oils_index_yoy_lag1m","fao_oils_index_yoy_lag3m","fao_oils_index_yoy_lag6m",
    "fao_cereals_yoy_x_instability","fao_food_yoy_x_youth_unemp",
    "gta_harmful_events","gta_harmful_events_z",
    "gta_liberalising_events","gta_liberalising_events_z",
    "gta_30d_count","gta_30d_count_z","gta_90d_count","gta_90d_count_z",
]

BLOCK_MAP: dict[str, str] = {}
for _f in _FEATURES_M0: BLOCK_MAP[_f] = "M0"
for _f in _FEATURES_M1: BLOCK_MAP[_f] = "M1"
for _f in _FEATURES_M2: BLOCK_MAP[_f] = "M2"
for _f in _FEATURES_M3: BLOCK_MAP[_f] = "M3"
for _f in _FEATURES_M4: BLOCK_MAP[_f] = "M4"

BLOCK_COLORS = {
    "M0": "#1f77b4",
    "M1": "#ff7f0e",
    "M2": "#2ca02c",
    "M3": "#d62728",
    "M4": "#9467bd",
    "FE": "#8c564b",
}
BLOCK_LABELS = {
    "M0": "M0: Persistence",
    "M1": "M1: Financial Markets",
    "M2": "M2: Macro / Governance",
    "M3": "M3: Structural",
    "M4": "M4: FAO / GTA",
    "FE": "Country FE",
}

FEATURE_LABELS: dict[str, str] = {
    "gdelt_protest_7d_lag":           "Protests last 7d",
    "gdelt_protest_28d_lag":          "Protests last 28d",
    "gdelt_strike_7d_lag":            "Strikes last 7d",
    "gdelt_strike_28d_lag":           "Strikes last 28d",
    "gdelt_protest_region_14d":       "Regional protests 14d",
    "gdelt_strike_region_14d":        "Regional strikes 14d",
    "fx_pct_7d":                      "FX return (7d)",
    "fx_pct_30d":                     "FX return (30d)",
    "fx_pct_90d":                     "FX return (90d)",
    "fx_vol_7d":                      "FX volatility (7d)",
    "fx_vol_30d":                     "FX volatility (30d)",
    "fx_pct_30d_z":                   "FX return z-score",
    "fx_vol_30d_z":                   "FX vol z-score",
    "oil_brent_pct_14d":              "Oil price chg (14d)",
    "oil_brent_pct_30d":              "Oil price chg (30d)",
    "oil_brent_pct_30d_z":            "Oil price z-score",
    "yield_us10y":                    "US 10Y yield",
    "copper_pct_30d":                 "Copper price (30d)",
    "copper_pct_90d":                 "Copper price (90d)",
    "copper_vol_30d":                 "Copper vol (30d)",
    "gold_pct_30d":                   "Gold price (30d)",
    "gold_vol_30d":                   "Gold vol (30d)",
    "platinum_pct_30d":               "Platinum price (30d)",
    "silver_pct_30d":                 "Silver price (30d)",
    "natgas_pct_30d":                 "Natural gas (30d)",
    "vix_level":                      "VIX level",
    "vix_pct_30d":                    "VIX change (30d)",
    "vix_7d_ma":                      "VIX 7d moving avg",
    "dxy_level":                      "DXY level",
    "dxy_pct_30d":                    "DXY change (30d)",
    "dxy_vol_30d":                    "DXY vol (30d)",
    "oil_brent_pct_30d_lag30d":       "Oil chg lag 30d",
    "oil_brent_pct_30d_lag60d":       "Oil chg lag 60d",
    "oil_brent_pct_30d_lag90d":       "Oil chg lag 90d",
    "vix_pct_30d_lag30d":             "VIX chg lag 30d",
    "vix_pct_30d_lag60d":             "VIX chg lag 60d",
    "fx_pct_30d_lag30d":              "FX return lag 30d",
    "fx_pct_30d_lag60d":              "FX return lag 60d",
    "gdp_growth":                     "GDP growth",
    "gdp_per_capita_growth":          "GDP per capita growth",
    "inflation_cpi_yoy":              "CPI inflation (YoY)",
    "inflation_cpi_yoy_z":            "CPI inflation z-score",
    "unemployment_total":             "Unemployment (total)",
    "unemployment_youth":             "Unemployment (youth)",
    "unemployment_sa":                "Unemployment (SA)",
    "unemployment_rate":              "Unemployment rate",
    "unemployment_rate_z":            "Unemployment rate z",
    "political_stability_est":        "Political stability",
    "voice_accountability_est":       "Voice & accountability",
    "government_effectiveness_est":   "Govt effectiveness",
    "rule_of_law_est":                "Rule of law",
    "fx_pct_30d_x_instability":       "FX x instability",
    "oil_brent_pct_30d_x_inflation":  "Oil x inflation",
    "food_cpi_inflation":             "Food CPI inflation",
    "food_cpi_inflation_z":           "Food CPI z-score",
    "energy_cpi_inflation":           "Energy CPI inflation",
    "energy_cpi_inflation_z":         "Energy CPI z-score",
    "gini_coef":                      "Gini coefficient",
    "covid_period":                   "COVID period",
    "fx_trend_consistent":            "FX trend consistent",
    "inflation_accel":                "Inflation accel.",
    "country_protest_baseline":       "Country protest baseline",
    "country_strike_baseline":        "Country strike baseline",
    "copper_pct_30d_x_copper_prod":   "Copper x producer",
    "gold_pct_30d_x_gold_prod":       "Gold x producer",
    "platinum_pct_30d_x_plat_prod":   "Platinum x producer",
    "oil_brent_pct_30d_x_net_importer": "Oil x net importer",
    "month_sin":                      "Seasonality (sin)",
    "month_cos":                      "Seasonality (cos)",
    "fao_food_index_yoy":             "FAO food (YoY)",
    "fao_cereals_index_yoy":          "FAO cereals (YoY)",
    "fao_oils_index_yoy":             "FAO oils (YoY)",
    "fao_food_index_yoy_above90":     "FAO food > 90th pct",
    "fao_cereals_index_yoy_above90":  "FAO cereals > 90th pct",
    "fao_cereals_index_yoy_lag1m":    "FAO cereals lag 1m",
    "fao_cereals_index_yoy_lag3m":    "FAO cereals lag 3m",
    "fao_cereals_index_yoy_lag6m":    "FAO cereals lag 6m",
    "fao_food_index_yoy_lag1m":       "FAO food lag 1m",
    "fao_food_index_yoy_lag3m":       "FAO food lag 3m",
    "fao_food_index_yoy_lag6m":       "FAO food lag 6m",
    "fao_oils_index_yoy_lag1m":       "FAO oils lag 1m",
    "fao_oils_index_yoy_lag3m":       "FAO oils lag 3m",
    "fao_oils_index_yoy_lag6m":       "FAO oils lag 6m",
    "fao_cereals_yoy_x_instability":  "Cereals x instability",
    "fao_food_yoy_x_youth_unemp":     "Food x youth unemp",
    "gta_harmful_events":             "Trade restrictions",
    "gta_harmful_events_z":           "Trade restrictions z",
    "gta_liberalising_events":        "Trade liberalisation",
    "gta_liberalising_events_z":      "Trade liberalisation z",
    "gta_30d_count":                  "GTA count (30d)",
    "gta_30d_count_z":                "GTA count z (30d)",
    "gta_90d_count":                  "GTA count (90d)",
    "gta_90d_count_z":                "GTA count z (90d)",
}

ILLUSTRATIVE_COUNTRIES = {
    "ARG": "Argentina",
    "CHL": "Chile",
    "BRA": "Brazil",
    "TUR": "Turkiye",
    "KEN": "Kenya",
}


def _clean_feat(raw: str) -> str:
    """Strip sklearn ColumnTransformer prefixes."""
    return raw.replace("num__", "").replace("remainder__", "")


def _feat_label(raw: str) -> str:
    clean = _clean_feat(raw)
    return FEATURE_LABELS.get(clean, clean.replace("_", " ").title())


def _feat_block(raw: str) -> str:
    if raw.startswith("fe__") or "country_iso3" in raw:
        return "FE"
    return BLOCK_MAP.get(_clean_feat(raw), "M4")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_metrics() -> pd.DataFrame:
    perf = PROC_DIR / "model_performance.csv"
    if perf.exists():
        df = pd.read_csv(perf)
        if "target" not in df.columns and "event_type" in df.columns and "horizon" in df.columns:
            df["target"] = df["event_type"] + "_" + df["horizon"].astype(str) + "d"
        return df
    frames = []
    for t in ACTIVE_TARGETS:
        p = PROC_DIR / t / "metrics.csv"
        if p.exists():
            sub = pd.read_csv(p)
            sub["target"] = t
            frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_preds(target: str) -> pd.DataFrame:
    p = PROC_DIR / target / "preds.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_lr_coefs(target: str) -> pd.DataFrame:
    p = PROC_DIR / target / "coefs_lr.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_shap(target: str) -> pd.DataFrame:
    p = PROC_DIR / target / "shap_importance.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def load_xgb_importances(target: str, model_name: str) -> pd.DataFrame:
    p = PROC_DIR / target / "coefs_xgb.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df[df["model_name"] == model_name] if "model_name" in df.columns else df


def find_best_models(metrics: pd.DataFrame,
                     primary_target: str = "protest_7d") -> tuple[str, str]:
    """Return (best_lr, best_xgb) by mean PR-AUC on primary_target."""
    lr_names  = ["model0_persistence","model1_markets","model2_full",
                 "model3_structural","model4_fao"]
    xgb_names = ["model5_xgb","model6_xgb_nolag","model7_xgb_nobaseline"]

    sub = metrics[metrics["target"] == primary_target] if "target" in metrics.columns else metrics
    lr_perf  = sub[sub["model_name"].isin(lr_names)].groupby("model_name")["pr_auc"].mean()
    xgb_perf = sub[sub["model_name"].isin(xgb_names)].groupby("model_name")["pr_auc"].mean()

    best_lr  = lr_perf.idxmax()  if not lr_perf.empty  else "model4_fao"
    best_xgb = xgb_perf.idxmax() if not xgb_perf.empty else "model5_xgb"
    return best_lr, best_xgb


# ---------------------------------------------------------------------------
# Fig 1: Model comparison bar chart
# ---------------------------------------------------------------------------

def fig_model_comparison(metrics: pd.DataFrame) -> None:
    targets  = [t for t in ACTIVE_TARGETS if t in metrics["target"].values]
    models   = [m for m in MODEL_ORDER if m in metrics["model_name"].values]
    n_models = len(models)
    n_targs  = len(targets)
    if not models or not targets:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Model Performance Comparison (mean over test folds 2020-2021)",
                 fontsize=12, fontweight="bold", y=1.01)

    x       = np.arange(n_models)
    width   = 0.22
    offsets = np.linspace(-(n_targs - 1) / 2 * width, (n_targs - 1) / 2 * width, n_targs)

    for ax, metric, ylabel, title in [
        (axes[0], "pr_auc",  "PR-AUC",  "Precision-Recall AUC"),
        (axes[1], "roc_auc", "ROC-AUC", "ROC AUC"),
    ]:
        for i, (target, offset) in enumerate(zip(targets, offsets)):
            sub = metrics[(metrics["target"] == target) &
                          (metrics["model_name"].isin(models))].copy()
            sub["model_name"] = pd.Categorical(sub["model_name"], categories=models, ordered=True)
            vals = sub.groupby("model_name", observed=True)[metric].mean().reindex(models)
            ax.bar(x + offset, vals, width=width * 0.9,
                   color=PALETTE.get(target, "#888"), alpha=0.85,
                   label=TARGET_LABELS.get(target, target))

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models],
                           fontsize=7.5, ha="center", rotation=15)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, pad=8)
        if metric == "roc_auc":
            ax.axhline(0.5, color="grey", lw=1, ls="--", alpha=0.5)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / "01_model_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 2: Precision-Recall curves
# ---------------------------------------------------------------------------

def fig_pr_curves() -> None:
    targets = [t for t in ACTIVE_TARGETS if (PROC_DIR / t / "preds.parquet").exists()]
    if not targets:
        return

    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    fig.suptitle("Precision-Recall Curves (combined test folds 2020-2021)",
                 fontsize=12, fontweight="bold")

    model_colors = {
        "model0_persistence": "#1a9641",
        "model4_fao":         "#fdae61",
        "model5_xgb":         "#d7191c",
        "model6_xgb_nolag":   "#2c7bb6",
    }

    for ax, target in zip(axes[0], targets):
        preds = load_preds(target)
        if preds.empty:
            continue
        ax.axhline(preds["y_true"].mean(), color="grey", ls="--", lw=1, alpha=0.6,
                   label=f"Random ({preds['y_true'].mean():.2f})")
        for mn, color in model_colors.items():
            sub = preds[preds["model_name"] == mn]
            if sub.empty:
                continue
            yt = sub["y_true"].values.astype(float)
            yp = sub["y_pred"].values
            mask = ~(np.isnan(yt) | np.isnan(yp))
            if mask.sum() < 10 or len(np.unique(yt[mask])) < 2:
                continue
            prec, rec, _ = precision_recall_curve(yt[mask], yp[mask])
            pr_auc_val = auc(rec, prec)
            ax.plot(rec, prec, color=color, lw=1.8,
                    label=f"{MODEL_LABELS.get(mn, mn)} ({pr_auc_val:.3f})")
        ax.set_title(TARGET_LABELS.get(target, target), fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=9)
        ax.set_ylabel("Precision", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out = FIG_DIR / "02_pr_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 3: XGBoost feature importance (gain)
# ---------------------------------------------------------------------------

def fig_feature_importance() -> None:
    configs = [
        ("protest_7d", "model5_xgb",      "M5 Full -- Protest 7d",  "#2166ac"),
        ("protest_7d", "model6_xgb_nolag", "M6 No-lag -- Protest 7d","#4dac26"),
        ("strike_7d",  "model5_xgb",       "M5 Full -- Strike 7d",   "#d73027"),
        ("strike_7d",  "model6_xgb_nolag", "M6 No-lag -- Strike 7d", "#f46d43"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("XGBoost Feature Importances (Top 15, gain)", fontsize=13, fontweight="bold")

    for ax, (target, model_name, title, color) in zip(axes.flat, configs):
        imp = load_xgb_importances(target, model_name)
        if imp.empty:
            ax.set_title(title); ax.text(0.5, 0.5, "No data", ha="center",
                                          va="center", transform=ax.transAxes)
            continue
        agg = (imp.groupby("feature")["importance"].mean()
                  .sort_values(ascending=False).head(15).iloc[::-1])
        labels = [_feat_label(f) for f in agg.index]
        bars   = ax.barh(labels, agg.values, color=color, alpha=0.8, edgecolor="white")
        for bar, val in zip(bars, agg.values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=7)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Mean importance (gain)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    out = FIG_DIR / "03_feature_importance_xgb.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 4: Calibration (existing — model4_fao binned reliability)
# ---------------------------------------------------------------------------

def fig_calibration() -> None:
    targets = [t for t in ACTIVE_TARGETS if (PROC_DIR / t / "preds.parquet").exists()]
    n = len(targets)
    if not n:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    fig.suptitle("Calibration: Predicted vs Actual Probability (model4_fao)",
                 fontsize=12, fontweight="bold")

    for ax, target in zip(axes[0], targets):
        preds = load_preds(target)
        if preds.empty:
            continue
        sub = preds[preds["model_name"] == "model4_fao"].dropna(subset=["y_true","y_pred"])
        if sub.empty:
            continue
        yt = sub["y_true"].values.astype(float)
        yp = sub["y_pred"].values
        n_bins  = 10
        bins    = np.linspace(0, 1, n_bins + 1)
        bin_ids = np.clip(np.digitize(yp, bins) - 1, 0, n_bins - 1)
        mp, ma, cnt = [], [], []
        for b in range(n_bins):
            idx = bin_ids == b
            if idx.sum() > 0:
                mp.append(yp[idx].mean()); ma.append(yt[idx].mean()); cnt.append(idx.sum())
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        sc = ax.scatter(mp, ma, c=cnt, cmap="Blues", s=80, zorder=5,
                        edgecolors="navy", linewidths=0.5)
        ax.plot(mp, ma, color="#2166ac", lw=1.5, alpha=0.7)
        plt.colorbar(sc, ax=ax, label="# samples", shrink=0.8)
        ax.set_title(TARGET_LABELS.get(target, target), fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean predicted probability", fontsize=9)
        ax.set_ylabel("Fraction of positives", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    out = FIG_DIR / "04_calibration.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 5: Country prediction timeline (existing)
# ---------------------------------------------------------------------------

def fig_country_timeline() -> None:
    preds = load_preds("protest_7d")
    if preds.empty:
        return
    sub = preds[preds["model_name"] == "model5_xgb"].copy()
    if sub.empty:
        sub = preds[preds["model_name"] == "model4_fao"].copy()

    countries = [c for c in ILLUSTRATIVE_COUNTRIES if c in sub["country_iso3"].values]
    if not countries:
        countries = sub["country_iso3"].value_counts().head(5).index.tolist()

    n = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    fig.suptitle("Predicted Protest Probability vs Actual Events\n(model5_xgb, protest_7d)",
                 fontsize=12, fontweight="bold")

    for ax, iso3 in zip(axes[:, 0], countries):
        c_sub = sub[sub["country_iso3"] == iso3].sort_values("date")
        if c_sub.empty:
            continue
        ax.fill_between(c_sub["date"], c_sub["y_pred"], alpha=0.25, color="#2166ac")
        ax.plot(c_sub["date"], c_sub["y_pred"], color="#2166ac", lw=1.2)
        events = c_sub[c_sub["y_true"] == 1]
        ax.scatter(events["date"], np.ones(len(events)) * 0.05,
                   marker="|", color="#d73027", s=60, lw=1.5, label="Event")
        ax.set_title(f"{ILLUSTRATIVE_COUNTRIES.get(iso3, iso3)} ({iso3})",
                     fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel("P(protest)", fontsize=8)
        ax.set_ylim(0, 1.05)

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "05_country_timeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 6: Lift / cumulative gains curves (existing)
# ---------------------------------------------------------------------------

def fig_lift_curves() -> None:
    targets = [t for t in ACTIVE_TARGETS if (PROC_DIR / t / "preds.parquet").exists()]
    n = len(targets)
    if not n:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    fig.suptitle("Cumulative Gains Curves", fontsize=12, fontweight="bold")

    show_models = {
        "model0_persistence": ("#1a9641", "M0 Persistence"),
        "model5_xgb":         ("#d7191c", "M5 XGBoost (full)"),
        "model6_xgb_nolag":   ("#2c7bb6", "M6 XGBoost (no lags)"),
    }

    for ax, target in zip(axes[0], targets):
        preds = load_preds(target)
        if preds.empty:
            continue
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
        for mn, (color, label) in show_models.items():
            sub = preds[preds["model_name"] == mn].dropna(subset=["y_true","y_pred"])
            if sub.empty:
                continue
            yt = sub["y_true"].values.astype(float)
            yp = sub["y_pred"].values
            if len(np.unique(yt)) < 2:
                continue
            order = np.argsort(yp)[::-1]
            yt_s  = yt[order]
            x_v   = np.arange(1, len(yt_s) + 1) / len(yt_s)
            y_v   = np.cumsum(yt_s) / yt_s.sum()
            ax.plot(x_v, y_v, color=color, lw=2, label=label)
        ax.set_title(TARGET_LABELS.get(target, target), fontsize=11, fontweight="bold")
        ax.set_xlabel("Fraction of country-days flagged", fontsize=9)
        ax.set_ylabel("Fraction of events captured", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / "06_lift_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Fig 7: Model complexity progression
# ---------------------------------------------------------------------------

def fig_model_progression(metrics: pd.DataFrame) -> None:
    lr_models  = ["model0_persistence","model1_markets","model2_full",
                  "model3_structural","model4_fao"]
    xgb_models = ["model5_xgb","model6_xgb_nolag"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Feature Set Progression: PR-AUC as blocks are added",
                 fontsize=12, fontweight="bold")

    for ax, event_type, targ_list in [
        (axes[0], "Protest", ["protest_7d"]),
        (axes[1], "Strike",  ["strike_7d", "strike_30d"]),
    ]:
        x     = np.arange(len(lr_models))
        width = 0.3
        for i, target in enumerate(targ_list):
            sub = metrics[(metrics["target"] == target) &
                          (metrics["model_name"].isin(lr_models))].copy()
            sub["model_name"] = pd.Categorical(sub["model_name"],
                                               categories=lr_models, ordered=True)
            vals = sub.groupby("model_name", observed=True)["pr_auc"].mean().reindex(lr_models)
            offset = (i - (len(targ_list) - 1) / 2) * width
            ax.bar(x + offset, vals, width=width * 0.9,
                   color=PALETTE.get(target, "#888"), alpha=0.85,
                   label=TARGET_LABELS.get(target, target))
        for target in targ_list:
            for mn, ls in [("model5_xgb", "-"), ("model6_xgb_nolag", "--")]:
                sub = metrics[(metrics["target"] == target) &
                              (metrics["model_name"] == mn)]
                if not sub.empty:
                    ax.axhline(sub["pr_auc"].mean(), color=PALETTE.get(target,"#888"),
                               ls=ls, lw=1.2, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in lr_models],
                           fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("PR-AUC", fontsize=10)
        ax.set_title(f"{event_type} -- Feature Progression", fontsize=11)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIG_DIR / "07_model_progression.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Dissertation figure A: LR coefficients coloured by feature block
# ---------------------------------------------------------------------------

def fig_lr_coefficients(best_lr: str, primary_target: str = "protest_7d") -> None:
    coefs = load_lr_coefs(primary_target)
    if coefs.empty:
        print(f"No coefs_lr.csv for {primary_target}")
        return

    coefs = coefs[coefs["model_name"] == best_lr].copy()
    # Exclude country fixed effects
    coefs = coefs[~coefs["feature"].str.startswith("fe__")]
    if coefs.empty:
        print(f"No non-FE coefficients for {best_lr}")
        return

    coefs["feat_clean"] = coefs["feature"].apply(_clean_feat)
    coefs["block"]      = coefs["feature"].apply(_feat_block)
    coefs["label"]      = coefs["feat_clean"].apply(
        lambda n: FEATURE_LABELS.get(n, n.replace("_", " ").title()))

    # Average over folds, sort by absolute magnitude, take top 30
    avg = (coefs.groupby(["feat_clean","block","label"])["coefficient"]
                .mean().reset_index())
    avg["abs_coef"] = avg["coefficient"].abs()
    avg = avg.nlargest(30, "abs_coef").sort_values("abs_coef")

    fig, ax = plt.subplots(figsize=(9, max(6, len(avg) * 0.38)))

    colors = [BLOCK_COLORS.get(b, "#999") for b in avg["block"]]
    ax.barh(avg["label"], avg["coefficient"],
            color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Coefficient (standardised features)", fontsize=10)
    ax.set_title(
        f"Logistic Regression Coefficients\n{best_lr} | {primary_target}",
        fontsize=11, fontweight="bold",
    )

    handles = [mpatches.Patch(color=BLOCK_COLORS[b], label=BLOCK_LABELS[b])
               for b in ["M0","M1","M2","M3","M4"] if b in avg["block"].values]
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / "lr_coefficients.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Dissertation figure B: SHAP values coloured by feature block
# ---------------------------------------------------------------------------

def fig_shap_values(best_xgb: str, primary_target: str = "protest_7d") -> None:
    shap = load_shap(primary_target)
    if shap.empty:
        print(f"No shap_importance.csv for {primary_target}")
        return

    shap = shap[shap["model_name"] == best_xgb].copy()
    if shap.empty:
        print(f"No SHAP data for {best_xgb}")
        return

    shap["feat_clean"] = shap["feature"].apply(_clean_feat)
    shap["block"]      = shap["feature"].apply(_feat_block)
    shap["label"]      = shap["feat_clean"].apply(
        lambda n: FEATURE_LABELS.get(n, n.replace("_", " ").title()))

    # Average over folds, top 30 by mean |SHAP|
    avg = (shap.groupby(["feat_clean","block","label"])["mean_abs_shap"]
               .mean().reset_index())
    avg = avg.nlargest(30, "mean_abs_shap").sort_values("mean_abs_shap")

    fig, ax = plt.subplots(figsize=(9, max(6, len(avg) * 0.38)))

    colors = [BLOCK_COLORS.get(b, "#999") for b in avg["block"]]
    ax.barh(avg["label"], avg["mean_abs_shap"],
            color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_title(
        f"XGBoost SHAP Feature Importance\n{best_xgb} | {primary_target}",
        fontsize=11, fontweight="bold",
    )

    handles = [mpatches.Patch(color=BLOCK_COLORS[b], label=BLOCK_LABELS[b])
               for b in ["M0","M1","M2","M3","M4"] if b in avg["block"].values]
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / "shap_values.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Dissertation figure C: Calibration curves
# ---------------------------------------------------------------------------

def fig_calibration_curve(best_lr: str, best_xgb: str,
                           primary_target: str = "protest_7d") -> None:
    preds = load_preds(primary_target)
    if preds.empty:
        print(f"No predictions for {primary_target}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Calibration Curves -- {primary_target}\n(combined test folds 2020+2021)",
        fontsize=11, fontweight="bold",
    )

    pairs = [
        (axes[0], best_lr,  f"Best LR: {best_lr}",       False),
        (axes[1], best_xgb, f"Best XGBoost: {best_xgb}", True),
    ]
    for ax, model_name, title, show_raw in pairs:
        sub = preds[preds["model_name"] == model_name].dropna(subset=["y_true","y_pred"])
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title, fontsize=9)
            continue

        yt = sub["y_true"].values.astype(int)
        yp = sub["y_pred"].values

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Perfect")

        frac, mean_p = sk_calibration_curve(yt, yp, n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, "s-", color="#3B82F6", lw=2, ms=6, label="Calibrated")

        if show_raw and "y_pred_raw" in sub.columns and sub["y_pred_raw"].notna().any():
            frac_r, mean_r = sk_calibration_curve(
                yt, sub["y_pred_raw"].values, n_bins=10, strategy="quantile")
            ax.plot(mean_r, frac_r, "o--", color="#F59E0B", lw=1.5, ms=5,
                    label="XGB raw")

        # Marginal histogram of predicted probs (pushed to bottom)
        ax2 = ax.twinx()
        ax2.hist(yp, bins=20, color="#93C5FD", alpha=0.25, density=True)
        ax2.set_ylim(0, ax2.get_ylim()[1] * 6)
        ax2.set_ylabel("Density", fontsize=7, color="#93C5FD")
        ax2.tick_params(axis="y", labelcolor="#93C5FD", labelsize=6)

        ax.set_xlabel("Mean predicted probability", fontsize=9)
        ax.set_ylabel("Fraction of positives", fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    out = FIG_DIR / "calibration_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Dissertation figure D: Cumulative gain curves, both folds
# ---------------------------------------------------------------------------

def fig_cumulative_gain(best_lr: str, best_xgb: str,
                        primary_target: str = "protest_7d") -> None:
    preds = load_preds(primary_target)
    if preds.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Cumulative Gain Curves -- {primary_target}",
        fontsize=11, fontweight="bold",
    )

    fold_styles = {1: ("#2166ac", "-",  "Fold 1 (test 2020)"),
                   2: ("#d73027", "--", "Fold 2 (test 2021)")}

    for ax, model_name, title in [
        (axes[0], best_lr,  f"Best LR: {best_lr}"),
        (axes[1], best_xgb, f"Best XGBoost: {best_xgb}"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")

        sub = preds[preds["model_name"] == model_name].dropna(subset=["y_true","y_pred"])
        for fold_id in sorted(sub["fold_id"].unique()):
            fold_sub = sub[sub["fold_id"] == fold_id]
            yt = fold_sub["y_true"].values.astype(float)
            yp = fold_sub["y_pred"].values
            if len(np.unique(yt)) < 2:
                continue
            order = np.argsort(yp)[::-1]
            yt_s  = yt[order]
            x_v   = np.arange(1, len(yt_s) + 1) / len(yt_s)
            y_v   = np.cumsum(yt_s) / yt_s.sum()
            color, ls, lbl = fold_styles.get(int(fold_id), ("#888", "-", f"Fold {fold_id}"))
            ax.plot(x_v, y_v, color=color, lw=2, ls=ls, label=lbl)

        ax.set_xlabel("Fraction of country-days flagged", fontsize=9)
        ax.set_ylabel("Fraction of events captured", fontsize=9)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / "cumulative_gain.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Dissertation figure E: Country probability timelines, both folds
# ---------------------------------------------------------------------------

def fig_country_timelines(best_model: str,
                           primary_target: str = "protest_7d") -> None:
    preds = load_preds(primary_target)
    if preds.empty:
        return

    sub = preds[preds["model_name"] == best_model].copy()
    if sub.empty:
        print(f"No predictions for {best_model}")
        return

    countries = [c for c in ILLUSTRATIVE_COUNTRIES if c in sub["country_iso3"].values][:5]
    if not countries:
        countries = sub["country_iso3"].value_counts().head(5).index.tolist()

    n   = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    fig.suptitle(
        f"Predicted Probability Timelines -- {best_model}\n"
        f"({primary_target}, test folds 2020-2021)",
        fontsize=11, fontweight="bold",
    )

    for ax, iso3 in zip(axes[:, 0], countries):
        c_sub = sub[sub["country_iso3"] == iso3].sort_values("date")
        if c_sub.empty:
            continue

        ax.fill_between(c_sub["date"], c_sub["y_pred"],
                        alpha=0.2, color="#2166ac")
        ax.plot(c_sub["date"], c_sub["y_pred"],
                color="#2166ac", lw=1.5, label="Predicted P(event)")

        events = c_sub[c_sub["y_true"] == 1]
        if not events.empty:
            ax.vlines(events["date"], 0, 0.12,
                      color="#d73027", lw=1.0, alpha=0.6, label="Event day")

        # Fold boundary
        boundary = pd.Timestamp("2021-01-01")
        if c_sub["date"].max() >= boundary >= c_sub["date"].min():
            ax.axvline(boundary, color="grey", lw=1, ls=":", alpha=0.6)
            ax.text(boundary, 0.95, " 2021", fontsize=7, color="grey",
                    va="top", transform=ax.get_xaxis_transform())

        name = ILLUSTRATIVE_COUNTRIES.get(iso3, iso3)
        ax.set_title(f"{name} ({iso3})", fontsize=9, loc="left", fontweight="bold")
        ax.set_ylabel("P(event)", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(c_sub["date"].min(), c_sub["date"].max())
        if ax is axes[0, 0]:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "country_timelines.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Dissertation figure E2: M6 vs M7 country timelines
# ---------------------------------------------------------------------------

def fig_lr_timelines(primary_target: str = "protest_7d") -> None:
    """Overlay LR models M0–M4 per country to show feature-block progression."""
    preds = load_preds(primary_target)
    if preds.empty:
        return

    lr_models = [
        ("model0_persistence",  "M0 Persistence",    "#1f77b4"),
        ("model1_markets",      "M1 + Markets",      "#ff7f0e"),
        ("model2_full",         "M2 + Macro",        "#2ca02c"),
        ("model3_structural",   "M3 + Structural",   "#d62728"),
        ("model4_fao",          "M4 + FAO/GTA",      "#9467bd"),
    ]

    available = {m for m in preds["model_name"].unique()}
    lr_models = [(m, lbl, c) for m, lbl, c in lr_models if m in available]
    if not lr_models:
        print("No LR predictions found")
        return

    ref = preds[preds["model_name"] == lr_models[0][0]]
    countries = [c for c in ILLUSTRATIVE_COUNTRIES if c in ref["country_iso3"].values][:5]
    if not countries:
        countries = ref["country_iso3"].value_counts().head(5).index.tolist()

    n = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    fig.suptitle(
        f"Logistic Regression M0–M4 -- Predicted Probability Timelines\n"
        f"({primary_target}, test folds 2020-2021)",
        fontsize=11, fontweight="bold",
    )

    for ax, iso3 in zip(axes[:, 0], countries):
        for model_name, label, color in lr_models:
            c_sub = (preds[(preds["model_name"] == model_name) &
                           (preds["country_iso3"] == iso3)]
                     .sort_values("date"))
            if c_sub.empty:
                continue
            ax.plot(c_sub["date"], c_sub["y_pred"], color=color, lw=1.3,
                    alpha=0.85, label=label)

        # Event markers from M0 (or first available model)
        ref_sub = (preds[(preds["model_name"] == lr_models[0][0]) &
                         (preds["country_iso3"] == iso3)]
                   .sort_values("date"))
        events = ref_sub[ref_sub["y_true"] == 1]
        if not events.empty:
            ax.vlines(events["date"], 0, 0.12,
                      color="grey", lw=0.8, alpha=0.5, label="Event day")

        boundary = pd.Timestamp("2021-01-01")
        if not ref_sub.empty:
            d_min, d_max = ref_sub["date"].min(), ref_sub["date"].max()
            if d_max >= boundary >= d_min:
                ax.axvline(boundary, color="grey", lw=1, ls=":", alpha=0.6)
                ax.text(boundary, 0.95, " 2021", fontsize=7, color="grey",
                        va="top", transform=ax.get_xaxis_transform())
            ax.set_xlim(d_min, d_max)

        name = ILLUSTRATIVE_COUNTRIES.get(iso3, iso3)
        ax.set_title(f"{name} ({iso3})", fontsize=9, loc="left", fontweight="bold")
        ax.set_ylabel("P(event)", fontsize=8)
        ax.set_ylim(0, 1.05)
        if ax is axes[0, 0]:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "country_timelines_lr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_model67_timelines(primary_target: str = "protest_7d") -> None:
    """Overlay M6 (no lags) and M7 (no baseline) predictions for 5 countries."""
    preds = load_preds(primary_target)
    if preds.empty:
        return

    m6 = preds[preds["model_name"] == "model6_xgb_nolag"].copy()
    m7 = preds[preds["model_name"] == "model7_xgb_nobaseline"].copy()
    if m6.empty and m7.empty:
        print("No predictions for model6 or model7")
        return

    ref = m6 if not m6.empty else m7
    countries = [c for c in ILLUSTRATIVE_COUNTRIES if c in ref["country_iso3"].values][:5]
    if not countries:
        countries = ref["country_iso3"].value_counts().head(5).index.tolist()

    n   = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    fig.suptitle(
        f"M6 (no lags) vs M7 (no baseline) -- Predicted Probability Timelines\n"
        f"({primary_target}, test folds 2020-2021)",
        fontsize=11, fontweight="bold",
    )

    M6_COLOR = "#2166ac"
    M7_COLOR = "#d73027"

    for ax, iso3 in zip(axes[:, 0], countries):
        for df_m, color, label in [
            (m6, M6_COLOR, "M6: no lags"),
            (m7, M7_COLOR, "M7: no baseline"),
        ]:
            if df_m.empty:
                continue
            c_sub = df_m[df_m["country_iso3"] == iso3].sort_values("date")
            if c_sub.empty:
                continue
            ax.fill_between(c_sub["date"], c_sub["y_pred"], alpha=0.12, color=color)
            ax.plot(c_sub["date"], c_sub["y_pred"], color=color, lw=1.5, label=label)

        # Event markers from whichever model has data for this country
        event_src = (m6 if not m6.empty else m7)
        c_events = event_src[event_src["country_iso3"] == iso3].sort_values("date")
        events = c_events[c_events["y_true"] == 1]
        if not events.empty:
            ax.vlines(events["date"], 0, 0.12,
                      color="grey", lw=0.8, alpha=0.5, label="Event day")

        # Fold boundary
        boundary = pd.Timestamp("2021-01-01")
        date_min = c_events["date"].min() if not c_events.empty else pd.Timestamp("2020-01-01")
        date_max = c_events["date"].max() if not c_events.empty else pd.Timestamp("2021-12-31")
        if date_max >= boundary >= date_min:
            ax.axvline(boundary, color="grey", lw=1, ls=":", alpha=0.6)
            ax.text(boundary, 0.95, " 2021", fontsize=7, color="grey",
                    va="top", transform=ax.get_xaxis_transform())

        name = ILLUSTRATIVE_COUNTRIES.get(iso3, iso3)
        ax.set_title(f"{name} ({iso3})", fontsize=9, loc="left", fontweight="bold")
        ax.set_ylabel("P(event)", fontsize=8)
        ax.set_ylim(0, 1.05)
        if not c_events.empty:
            ax.set_xlim(date_min, date_max)
        if ax is axes[0, 0]:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "country_timelines_m6_m7.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Expanding window: monthly ROC-AUC and BSS for selected LR models
# ---------------------------------------------------------------------------

_EXP_DIR = _V2 / "data" / "processed" / "expanding_lr"

_EXP_MODELS = ["model0_persistence", "model4_fao", "model_lr_nolag", "model9_twostage"]
_EXP_COLORS = {
    "model0_persistence": "#1f77b4",
    "model4_fao":         "#9467bd",
    "model_lr_nolag":     "#17becf",
    "model9_twostage":    "#e377c2",
}
_EXP_LABELS = {
    "model0_persistence": "M0 Persistence",
    "model4_fao":         "M4 + FAO/GTA",
    "model_lr_nolag":     "LR No Lags",
    "model9_twostage":    "M9 Two-Stage",
}


def fig_expanding_performance_final(target: str) -> None:
    """Monthly ROC-AUC and BSS for M0, M4, LR No Lags, M9."""
    p = _EXP_DIR / f"metrics_{target}.csv"
    if not p.exists():
        print(f"No expanding metrics for {target}")
        return

    metrics = pd.read_csv(p)
    metrics["month_dt"] = pd.to_datetime(metrics["month"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(
        f"Expanding Window — Monthly Performance ({TARGET_LABELS.get(target, target)})\n"
        "Monthly retraining, 2020–2021",
        fontsize=11, fontweight="bold",
    )

    for model_name in _EXP_MODELS:
        sub = metrics[metrics["model_name"] == model_name].sort_values("month_dt")
        if sub.empty:
            continue
        color = _EXP_COLORS[model_name]
        label = _EXP_LABELS[model_name]
        axes[0].plot(sub["month_dt"], sub["roc_auc"],
                     color=color, lw=1.8, marker="o", ms=4, label=label)
        axes[1].plot(sub["month_dt"], sub["brier_skill_score"],
                     color=color, lw=1.8, marker="o", ms=4, label=label)

    boundary = pd.Timestamp("2021-01-01")
    for ax in axes:
        ax.axvline(boundary, color="grey", lw=1.0, ls=":", alpha=0.7)
        ax.text(boundary, ax.get_ylim()[0] if ax.get_ylim()[0] > -2 else -1.5,
                " 2021", fontsize=7, color="grey")
        ax.set_xlabel("Month", fontsize=9)
        ax.tick_params(axis="x", rotation=30)

    axes[0].set_title("ROC-AUC", fontsize=10)
    axes[0].set_ylim(0.6, 1.0)
    axes[1].set_title("Brier Skill Score", fontsize=10)
    axes[1].axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    out = FIG_DIR / f"expanding_performance_final_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# M9 figures: two-stage LR timelines and feature importance
# ---------------------------------------------------------------------------

def fig_m9_timelines(primary_target: str = "protest_7d") -> None:
    """M9 (two-stage) vs M0 (persistence) probability timelines for 5 countries."""
    preds = load_preds(primary_target)
    if preds.empty:
        return

    m9 = preds[preds["model_name"] == "model9_twostage"].copy()
    m0 = preds[preds["model_name"] == "model0_persistence"].copy()
    if m9.empty:
        print("No predictions for model9_twostage")
        return

    ref = m9 if not m9.empty else m0
    countries = [c for c in ILLUSTRATIVE_COUNTRIES if c in ref["country_iso3"].values][:5]
    if not countries:
        countries = ref["country_iso3"].value_counts().head(5).index.tolist()

    n = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    fig.suptitle(
        f"M0 Persistence vs M9 Two-Stage LR -- Predicted Probability Timelines\n"
        f"({primary_target}, test folds 2020-2021)",
        fontsize=11, fontweight="bold",
    )

    M0_COLOR = "#1f77b4"
    M9_COLOR = "#e377c2"

    for ax, iso3 in zip(axes[:, 0], countries):
        c_events = None
        for df_m, color, label in [
            (m0, M0_COLOR, "M0: Persistence"),
            (m9, M9_COLOR, "M9: Two-Stage"),
        ]:
            if df_m.empty:
                continue
            c_sub = df_m[df_m["country_iso3"] == iso3].sort_values("date")
            if c_sub.empty:
                continue
            ax.fill_between(c_sub["date"], c_sub["y_pred"], alpha=0.10, color=color)
            ax.plot(c_sub["date"], c_sub["y_pred"], color=color, lw=1.5, label=label)
            if c_events is None:
                c_events = c_sub

        if c_events is not None:
            events = c_events[c_events["y_true"] == 1]
            if not events.empty:
                ax.vlines(events["date"], 0, 0.12,
                          color="grey", lw=0.8, alpha=0.5, label="Event day")

            boundary = pd.Timestamp("2021-01-01")
            date_min = c_events["date"].min()
            date_max = c_events["date"].max()
            if date_max >= boundary >= date_min:
                ax.axvline(boundary, color="grey", lw=1, ls=":", alpha=0.6)
                ax.text(boundary, 0.95, " 2021", fontsize=7, color="grey",
                        va="top", transform=ax.get_xaxis_transform())
            ax.set_xlim(date_min, date_max)

        name = ILLUSTRATIVE_COUNTRIES.get(iso3, iso3)
        ax.set_title(f"{name} ({iso3})", fontsize=9, loc="left", fontweight="bold")
        ax.set_ylabel("P(event)", fontsize=8)
        ax.set_ylim(0, 1.05)
        if ax is axes[0, 0]:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / f"m9_timelines_{primary_target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_m9_feature_importance(primary_target: str = "protest_7d") -> None:
    """Side-by-side Stage 1 (structural) and Stage 2 (trigger) coefficients for M9."""
    coefs = load_lr_coefs(primary_target)
    if coefs.empty:
        return

    s1 = coefs[coefs["model_name"] == "model9_stage1"].copy()
    s2 = coefs[coefs["model_name"] == "model9_stage2"].copy()
    if s1.empty and s2.empty:
        print(f"No M9 stage coefs found for {primary_target}")
        return

    # Special colour/label for structural_risk_score in Stage 2
    RISK_COLOR = "#7f7f7f"

    def _prep(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        df = df[~df["feature"].str.startswith("fe__")].copy()
        df["feat_clean"] = df["feature"].apply(_clean_feat)
        df["block"]      = df["feature"].apply(_feat_block)
        df["label"]      = df["feat_clean"].apply(
            lambda n: "Structural Risk Score" if n == "structural_risk_score"
            else FEATURE_LABELS.get(n, n.replace("_", " ").title()))
        avg = (df.groupby(["feat_clean", "block", "label"])["coefficient"]
               .mean().reset_index())
        avg["abs_coef"] = avg["coefficient"].abs()
        return avg.nlargest(top_n, "abs_coef").sort_values("abs_coef")

    avg1 = _prep(s1) if not s1.empty else pd.DataFrame()
    avg2 = _prep(s2) if not s2.empty else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, max(len(avg1), len(avg2)) * 0.38)))
    fig.suptitle(
        f"M9 Two-Stage LR — Feature Importance by Stage\n"
        f"({primary_target}, averaged over test folds)",
        fontsize=11, fontweight="bold",
    )

    for ax, avg, title, stage in [
        (axes[0], avg1, "Stage 1: Structural Risk\n(slow features M2+M3+M4)", "stage1"),
        (axes[1], avg2, "Stage 2: Event Trigger\n(fast features M0+M1 + risk score)", "stage2"),
    ]:
        if avg.empty:
            ax.set_visible(False)
            continue

        colors = []
        for _, row in avg.iterrows():
            if row["feat_clean"] == "structural_risk_score":
                colors.append(RISK_COLOR)
            else:
                colors.append(BLOCK_COLORS.get(row["block"], "#999"))

        ax.barh(avg["label"], avg["coefficient"],
                color=colors, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Coefficient (standardised features)", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")

        # Legend
        blocks_present = avg["block"].unique()
        handles = [mpatches.Patch(color=BLOCK_COLORS[b], label=BLOCK_LABELS[b])
                   for b in ["M0","M1","M2","M3","M4"] if b in blocks_present]
        if stage == "stage2" and any(avg["feat_clean"] == "structural_risk_score"):
            handles.append(mpatches.Patch(color=RISK_COLOR, label="Structural Risk Score"))
        if handles:
            ax.legend(handles=handles, fontsize=7, loc="lower right")

    plt.tight_layout()
    out = FIG_DIR / f"m9_feature_importance_{primary_target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Dissertation figure F: Feature importance text file
# ---------------------------------------------------------------------------

def write_feature_importance_txt(best_lr: str, best_xgb: str,
                                  primary_target: str = "protest_7d") -> None:
    lines = [
        "FEATURE IMPORTANCE -- TOP 10 BY BLOCK",
        f"Target: {primary_target}",
        "=" * 60, "",
    ]

    # LR coefficients
    coefs = load_lr_coefs(primary_target)
    if not coefs.empty:
        coefs = coefs[(coefs["model_name"] == best_lr) &
                      (~coefs["feature"].str.startswith("fe__"))].copy()
        coefs["feat_clean"] = coefs["feature"].apply(_clean_feat)
        coefs["block"]      = coefs["feature"].apply(_feat_block)
        avg = (coefs.groupby(["feat_clean","block"])["coefficient"]
                    .mean().reset_index())
        avg["abs_coef"] = avg["coefficient"].abs()

        lines += [f"LOGISTIC REGRESSION: {best_lr}", "-" * 50]
        for block in ["M0","M1","M2","M3","M4"]:
            block_rows = (avg[avg["block"] == block]
                          .sort_values("abs_coef", ascending=False).head(10))
            if block_rows.empty:
                continue
            lines.append(f"\n  {BLOCK_LABELS[block]}")
            for _, row in block_rows.iterrows():
                lbl = FEATURE_LABELS.get(row["feat_clean"],
                                         row["feat_clean"].replace("_"," ").title())
                lines.append(f"    {lbl:<45}  coef = {row['coefficient']:+.4f}")
        lines.append("")

    # XGBoost SHAP
    shap = load_shap(primary_target)
    if not shap.empty:
        shap = shap[shap["model_name"] == best_xgb].copy()
        shap["feat_clean"] = shap["feature"].apply(_clean_feat)
        shap["block"]      = shap["feature"].apply(_feat_block)
        avg_s = (shap.groupby(["feat_clean","block"])["mean_abs_shap"]
                     .mean().reset_index())

        lines += [f"XGBOOST (SHAP): {best_xgb}", "-" * 50]
        for block in ["M0","M1","M2","M3","M4"]:
            block_rows = (avg_s[avg_s["block"] == block]
                          .sort_values("mean_abs_shap", ascending=False).head(10))
            if block_rows.empty:
                continue
            lines.append(f"\n  {BLOCK_LABELS[block]}")
            for _, row in block_rows.iterrows():
                lbl = FEATURE_LABELS.get(row["feat_clean"],
                                         row["feat_clean"].replace("_"," ").title())
                lines.append(f"    {lbl:<45}  SHAP = {row['mean_abs_shap']:.4f}")

    out = FIG_DIR / "feature_importance.txt"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved: {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading results...")
    metrics = load_metrics()

    if metrics.empty:
        print("No metrics found -- run train_backtest.py first.")
        return

    print(f"Loaded metrics: {metrics['target'].nunique()} targets, "
          f"{metrics['model_name'].nunique()} models")

    best_lr, best_xgb = find_best_models(metrics, primary_target="protest_7d")
    print(f"Best LR:       {best_lr}")
    print(f"Best XGBoost:  {best_xgb}")

    print("\nGenerating standard figures...")
    fig_model_comparison(metrics)
    fig_pr_curves()
    fig_feature_importance()
    fig_calibration()
    fig_country_timeline()
    fig_lift_curves()
    fig_model_progression(metrics)

    print("\nGenerating dissertation figures for best models...")
    fig_lr_coefficients(best_lr)
    fig_shap_values(best_xgb)
    fig_calibration_curve(best_lr, best_xgb)
    fig_cumulative_gain(best_lr, best_xgb)
    fig_country_timelines(best_xgb)
    fig_lr_timelines()
    fig_model67_timelines()
    fig_m9_timelines()
    fig_m9_feature_importance()
    write_feature_importance_txt(best_lr, best_xgb)

    print("\nGenerating expanding window performance figures...")
    for target in ACTIVE_TARGETS:
        fig_expanding_performance_final(target)

    print(f"\nAll figures saved to: {FIG_DIR}/")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"  {f.name}")

    # Copy final selection to final_figures/
    import shutil
    final_dir = _V2 / "final_figures"
    final_dir.mkdir(exist_ok=True)

    finals = [
        "01_model_comparison.png",
        "07_model_progression.png",
        "lr_coefficients.png",
        "shap_values.png",
        "calibration_curve.png",
        "cumulative_gain.png",
        "expanding_performance_final_protest_7d.png",
        "expanding_performance_final_strike_7d.png",
        "country_timelines.png",
        "country_timelines_m6_m7.png",
        "m9_timelines_protest_7d.png",
        "m9_feature_importance_protest_7d.png",
    ]

    print(f"\nCopying {len(finals)} figures to final_figures/...")
    for name in finals:
        src = FIG_DIR / name
        if src.exists():
            shutil.copy2(src, final_dir / name)
            print(f"  Copied: {name}")
        else:
            print(f"  MISSING: {name}")

    # Also save the performance CSV
    perf_src = _V2 / "data" / "processed" / "model_performance_full.csv"
    if perf_src.exists():
        shutil.copy2(perf_src, final_dir / "model_performance_full.csv")
        print(f"  Copied: model_performance_full.csv")
    for f in sorted(FIG_DIR.glob("*.txt")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
