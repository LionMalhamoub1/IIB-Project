"""
export_m4.py
Generates all M4 (FAO + GTA) figures and saves them to final figures/M4/.

M4 = M0 + M1 + M2 + M3 + M4_ADD (FAO food prices, GTA trade, seasonality)

Outputs:
  m4_static_timelines_{protest,strike}_7d.png
  m4_expanding_timelines_{protest,strike}_7d.png
  m4_expanding_performance_{protest,strike}_7d.png
  m4_coefficients_{protest,strike}_7d.png
  m4_metrics.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

_HERE    = Path(__file__).resolve().parent
_V2      = _HERE.parent
PROC_DIR = _V2 / "data" / "processed"
EXP_DIR  = PROC_DIR / "expanding_lr"
OUT_DIR  = _V2 / "final figures" / "M4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = ["protest_7d", "strike_7d"]
TARGET_LABELS = {"protest_7d": "Protest (7-day)", "strike_7d": "Strike (7-day)"}

ILLUSTRATIVE_COUNTRIES = {"ARG": "Argentina", "CHL": "Chile", "BRA": "Brazil",
                           "TUR": "Turkiye",   "KEN": "Kenya"}

STYLE = {
    "font.family": "sans-serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
}
plt.rcParams.update(STYLE)

MODEL_NAME = "model4_fao"
COLOR      = "#9467bd"  # purple


def fig_m4_static_timelines(target: str) -> None:
    p = PROC_DIR / target / "preds.parquet"
    if not p.exists():
        print(f"No static preds for {target}"); return

    preds = pd.read_parquet(p)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[preds["model_name"] == MODEL_NAME]
    if preds.empty:
        print(f"No M4 predictions in {target}"); return

    countries = [c for c in ILLUSTRATIVE_COUNTRIES if c in preds["country_iso3"].values]
    n = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    fig.suptitle(
        f"M4 FAO+GTA — Predicted Probability Timelines\n"
        f"{TARGET_LABELS[target]} | Static backtest (test folds 2020–2021)",
        fontsize=11, fontweight="bold",
    )

    for ax, iso3 in zip(axes[:, 0], countries):
        c = preds[preds["country_iso3"] == iso3].sort_values("date")
        if c.empty:
            continue
        ax.fill_between(c["date"], c["y_pred"], alpha=0.15, color=COLOR)
        ax.plot(c["date"], c["y_pred"], color=COLOR, lw=1.5, label="M4 FAO+GTA")
        events = c[c["y_true"] == 1]
        if not events.empty:
            ax.vlines(events["date"], 0, 0.12, color="grey", lw=0.8, alpha=0.5, label="Event day")
        boundary = pd.Timestamp("2021-01-01")
        d_min, d_max = c["date"].min(), c["date"].max()
        if d_max >= boundary >= d_min:
            ax.axvline(boundary, color="grey", lw=1, ls=":", alpha=0.7)
            ax.text(boundary, 0.95, " 2021", fontsize=7, color="grey",
                    va="top", transform=ax.get_xaxis_transform())
        ax.set_xlim(d_min, d_max)
        ax.set_title(f"{ILLUSTRATIVE_COUNTRIES.get(iso3, iso3)} ({iso3})",
                     fontsize=9, loc="left", fontweight="bold")
        ax.set_ylabel("P(event)", fontsize=8)
        ax.set_ylim(0, 1.05)
        if ax is axes[0, 0]:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / f"m4_static_timelines_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_m4_expanding_timelines(target: str) -> None:
    p = EXP_DIR / f"preds_{target}.parquet"
    if not p.exists():
        print(f"No expanding preds for {target}"); return

    preds = pd.read_parquet(p)
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds[preds["model_name"] == MODEL_NAME]
    if preds.empty:
        print(f"No M4 expanding predictions for {target}"); return

    countries = [c for c in ILLUSTRATIVE_COUNTRIES if c in preds["country_iso3"].values]
    retrain_months = sorted(pd.to_datetime(preds["retrain_month"].unique()))
    n = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    fig.suptitle(
        f"M4 FAO+GTA — Expanding Window Timelines\n"
        f"{TARGET_LABELS[target]} | Monthly retraining 2020–2021",
        fontsize=11, fontweight="bold",
    )

    for ax, iso3 in zip(axes[:, 0], countries):
        c = preds[preds["country_iso3"] == iso3].sort_values("date")
        if c.empty:
            continue
        ax.fill_between(c["date"], c["y_pred"], alpha=0.15, color=COLOR)
        ax.plot(c["date"], c["y_pred"], color=COLOR, lw=1.5)
        events = c[c["y_true"] == 1]
        if not events.empty:
            ax.vlines(events["date"], 0, 0.12, color="grey", lw=0.8, alpha=0.5)
        d_min, d_max = c["date"].min(), c["date"].max()
        first = True
        for rm in retrain_months:
            if d_min <= rm <= d_max:
                ax.axvline(rm, color="#d62728", lw=0.7, ls=":", alpha=0.6,
                           label="Retrain" if first else None)
                first = False
        ax.set_xlim(d_min, d_max)
        ax.set_title(f"{ILLUSTRATIVE_COUNTRIES.get(iso3, iso3)} ({iso3})",
                     fontsize=9, loc="left", fontweight="bold")
        ax.set_ylabel("P(event)", fontsize=8)
        ax.set_ylim(0, 1.05)
        if ax is axes[0, 0]:
            handles = [
                mpatches.Patch(color=COLOR, label="M4 FAO+GTA"),
                plt.Line2D([0], [0], color="#d62728", lw=1, ls=":", label="Retrain boundary"),
                plt.Line2D([0], [0], color="grey", lw=1, alpha=0.5, label="Event day"),
            ]
            ax.legend(handles=handles, fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / f"m4_expanding_timelines_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_m4_expanding_performance(target: str) -> None:
    p = EXP_DIR / f"metrics_{target}.csv"
    if not p.exists():
        print(f"No expanding metrics for {target}"); return

    df = pd.read_csv(p)
    df = df[df["model_name"] == MODEL_NAME].copy()
    df["month_dt"] = pd.to_datetime(df["month"])
    df = df.sort_values("month_dt")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"M4 FAO+GTA — Expanding Window Monthly Performance\n"
        f"{TARGET_LABELS[target]}, 2020–2021",
        fontsize=11, fontweight="bold",
    )
    axes[0].plot(df["month_dt"], df["roc_auc"], color=COLOR, lw=1.8, marker="o", ms=4)
    axes[1].plot(df["month_dt"], df["brier_skill_score"], color=COLOR, lw=1.8, marker="o", ms=4)

    boundary = pd.Timestamp("2021-01-01")
    for ax in axes:
        ax.axvline(boundary, color="grey", lw=1.0, ls=":", alpha=0.7)
        ax.text(boundary, ax.get_ylim()[0] + 0.01, " 2021", fontsize=7, color="grey")
        ax.set_xlabel("Month", fontsize=9)
        ax.tick_params(axis="x", rotation=30)

    axes[0].set_title("ROC-AUC", fontsize=10)
    axes[0].set_ylim(0.7, 1.0)
    axes[1].set_title("Brier Skill Score", fontsize=10)
    axes[1].axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    plt.tight_layout()
    out = OUT_DIR / f"m4_expanding_performance_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


FEATURE_LABELS = {
    # M0
    "gdelt_protest_7d_lag":      "Protests last 7d",
    "gdelt_protest_28d_lag":     "Protests last 28d",
    "gdelt_strike_7d_lag":       "Strikes last 7d",
    "gdelt_strike_28d_lag":      "Strikes last 28d",
    "gdelt_protest_region_14d":  "Regional protests 14d",
    "gdelt_strike_region_14d":   "Regional strikes 14d",
    # M1 FX
    "fx_pct_7d":                 "FX change 7d",
    "fx_pct_30d":                "FX change 30d",
    "fx_pct_90d":                "FX change 90d",
    "fx_vol_7d":                 "FX volatility 7d",
    "fx_vol_30d":                "FX volatility 30d",
    "fx_pct_30d_z":              "FX change 30d (z)",
    "fx_vol_30d_z":              "FX vol 30d (z)",
    "fx_pct_30d_lag30d":         "FX change 30d (lag 30d)",
    "fx_pct_30d_lag60d":         "FX change 30d (lag 60d)",
    # M1 oil
    "oil_brent_pct_14d":         "Oil change 14d",
    "oil_brent_pct_30d":         "Oil change 30d",
    "oil_brent_pct_30d_z":       "Oil change 30d (z)",
    "oil_brent_pct_30d_lag30d":  "Oil change 30d (lag 30d)",
    "oil_brent_pct_30d_lag60d":  "Oil change 30d (lag 60d)",
    "oil_brent_pct_30d_lag90d":  "Oil change 30d (lag 90d)",
    # M1 rates / commodities
    "yield_us10y":               "US 10Y yield",
    "copper_pct_30d":            "Copper change 30d",
    "copper_pct_90d":            "Copper change 90d",
    "copper_vol_30d":            "Copper vol 30d",
    "gold_pct_30d":              "Gold change 30d",
    "gold_vol_30d":              "Gold vol 30d",
    "platinum_pct_30d":          "Platinum change 30d",
    "silver_pct_30d":            "Silver change 30d",
    "natgas_pct_30d":            "Nat gas change 30d",
    # M1 VIX / DXY
    "vix_level":                 "VIX level",
    "vix_pct_30d":               "VIX change 30d",
    "vix_7d_ma":                 "VIX 7d MA",
    "vix_pct_30d_lag30d":        "VIX change 30d (lag 30d)",
    "vix_pct_30d_lag60d":        "VIX change 30d (lag 60d)",
    "dxy_level":                 "DXY level",
    "dxy_pct_30d":               "DXY change 30d",
    "dxy_vol_30d":               "DXY vol 30d",
    # M2 macro / labour
    "gdp_growth":                    "GDP growth",
    "gdp_per_capita_growth":         "GDP per capita growth",
    "inflation_cpi_yoy":             "CPI inflation (YoY)",
    "inflation_cpi_yoy_z":           "CPI inflation (z)",
    "unemployment_total":            "Unemployment total",
    "unemployment_youth":            "Youth unemployment",
    "unemployment_sa":               "Unemployment (SA)",
    "unemployment_rate":             "Unemployment rate",
    "unemployment_rate_z":           "Unemployment rate (z)",
    "food_cpi_inflation":            "Food CPI inflation",
    "food_cpi_inflation_z":          "Food CPI inflation (z)",
    "energy_cpi_inflation":          "Energy CPI inflation",
    "energy_cpi_inflation_z":        "Energy CPI inflation (z)",
    # M2 governance
    "political_stability_est":       "Political stability",
    "voice_accountability_est":      "Voice & accountability",
    "government_effectiveness_est":  "Gov. effectiveness",
    "rule_of_law_est":               "Rule of law",
    # M2 interactions
    "fx_pct_30d_x_instability":          "FX × instability",
    "oil_brent_pct_30d_x_inflation":     "Oil × inflation",
    # M3 structural
    "gini_coef":                         "Gini coefficient",
    "covid_period":                      "COVID period",
    "fx_trend_consistent":               "FX trend consistent",
    "inflation_accel":                   "Inflation accel.",
    "country_protest_baseline":          "Country protest baseline",
    "country_strike_baseline":           "Country strike baseline",
    "copper_pct_30d_x_copper_prod":      "Copper × producer",
    "gold_pct_30d_x_gold_prod":          "Gold × producer",
    "platinum_pct_30d_x_plat_prod":      "Platinum × producer",
    "oil_brent_pct_30d_x_net_importer":  "Oil × net importer",
    # M4 FAO food prices
    "fao_food_index_yoy":                "FAO food (YoY)",
    "fao_cereals_index_yoy":             "FAO cereals (YoY)",
    "fao_oils_index_yoy":                "FAO oils (YoY)",
    "fao_food_index_yoy_above90":        "FAO food >90th pctile",
    "fao_cereals_index_yoy_above90":     "FAO cereals >90th pctile",
    "fao_cereals_index_yoy_lag1m":       "FAO cereals lag 1m",
    "fao_cereals_index_yoy_lag3m":       "FAO cereals lag 3m",
    "fao_cereals_index_yoy_lag6m":       "FAO cereals lag 6m",
    "fao_food_index_yoy_lag1m":          "FAO food lag 1m",
    "fao_food_index_yoy_lag3m":          "FAO food lag 3m",
    "fao_food_index_yoy_lag6m":          "FAO food lag 6m",
    "fao_oils_index_yoy_lag1m":          "FAO oils lag 1m",
    "fao_oils_index_yoy_lag3m":          "FAO oils lag 3m",
    "fao_oils_index_yoy_lag6m":          "FAO oils lag 6m",
    "fao_cereals_yoy_x_instability":     "FAO cereals × instability",
    "fao_food_yoy_x_youth_unemp":        "FAO food × youth unemp.",
    # M4 GTA trade interventions
    "gta_harmful_events":                "GTA harmful events",
    "gta_harmful_events_z":              "GTA harmful events (z)",
    "gta_liberalising_events":           "GTA liberalising events",
    "gta_liberalising_events_z":         "GTA liberalising events (z)",
    "gta_30d_count":                     "GTA 30d count",
    "gta_30d_count_z":                   "GTA 30d count (z)",
    "gta_90d_count":                     "GTA 90d count",
    "gta_90d_count_z":                   "GTA 90d count (z)",
    # M4 seasonality
    "month_sin":                         "Month (sin)",
    "month_cos":                         "Month (cos)",
}


def fig_m4_coefficients(target: str) -> None:
    p = PROC_DIR / target / "coefs_lr.csv"
    if not p.exists():
        print(f"No coefs_lr.csv for {target}"); return

    coefs = pd.read_csv(p)
    coefs = coefs[coefs["model_name"] == MODEL_NAME].copy()
    coefs = coefs[~coefs["feature"].str.startswith("fe__")]
    if coefs.empty:
        return

    coefs["feat_clean"] = coefs["feature"].str.replace("num__", "").str.replace("remainder__", "")
    coefs["label"] = coefs["feat_clean"].map(FEATURE_LABELS).fillna(
        coefs["feat_clean"].str.replace("_", " ").str.title()
    )
    avg = (coefs.groupby(["feat_clean", "label"])["coefficient"]
           .mean().reset_index())
    avg["abs_coef"] = avg["coefficient"].abs()
    avg = avg.sort_values("abs_coef")

    fig, ax = plt.subplots(figsize=(9, max(4, len(avg) * 0.32)))
    ax.barh(avg["label"], avg["coefficient"], color=COLOR, edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Coefficient (standardised features)", fontsize=10)
    ax.set_title(
        f"M4 FAO+GTA — Feature Coefficients\n{TARGET_LABELS[target]} | averaged over folds",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out = OUT_DIR / f"m4_coefficients_{target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_m4_metrics() -> None:
    rows = []
    for target in TARGETS:
        p = PROC_DIR / target / "metrics.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["model_name"] == MODEL_NAME].copy()
        df["evaluation"] = "static"
        df["target"] = target
        rows.append(df[["target", "evaluation", "fold_id",
                         "roc_auc", "pr_auc", "brier", "brier_skill_score", "pos_rate"]])

    for target in TARGETS:
        p = EXP_DIR / f"metrics_{target}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["model_name"] == MODEL_NAME].copy()
        df["evaluation"] = "expanding"
        df["target"] = target
        df.rename(columns={"month": "fold_id"}, inplace=True)
        rows.append(df[["target", "evaluation", "fold_id",
                         "roc_auc", "pr_auc", "brier", "brier_skill_score"]])

    if rows:
        out = pd.concat(rows, ignore_index=True)
        for c in ["roc_auc", "pr_auc", "brier", "brier_skill_score"]:
            out[c] = out[c].round(4)
        out.to_csv(OUT_DIR / "m4_metrics.csv", index=False)
        print(f"Saved: {OUT_DIR / 'm4_metrics.csv'}  ({len(out)} rows)")


if __name__ == "__main__":
    import shutil
    print(f"Saving M4 outputs to {OUT_DIR}\n")
    for target in TARGETS:
        fig_m4_static_timelines(target)
        fig_m4_expanding_timelines(target)
        fig_m4_expanding_performance(target)
        fig_m4_coefficients(target)
    save_m4_metrics()
    shutil.copy2(__file__, OUT_DIR / "export_m4.py")
    print(f"Saved: {OUT_DIR / 'export_m4.py'}")
    print("\nDone.")
