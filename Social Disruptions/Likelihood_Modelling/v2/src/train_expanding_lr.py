# Monthly expanding-window backtest for LR models (M0-M4).
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import compute_metrics, make_target_gdelt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_V2        = _SRC.parent
PANEL_FILE = _V2 / "data" / "interim" / "modelling_panel_gdelt.parquet"
OUT_DIR    = _V2 / "data" / "processed" / "expanding_lr"
FIG_DIR    = _V2 / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
TARGETS            = [("protest", 7), ("strike", 7), ("strike", 30)]
INCLUDE_COUNTRY_FE = True
USE_CLASS_WEIGHT   = True
TRAIN_START        = pd.Timestamp("2017-01-01")
BASE_TRAIN_END     = pd.Timestamp("2019-12-31")
TEST_START         = pd.Timestamp("2020-01-01")
TEST_END           = pd.Timestamp("2021-12-31")

ILLUSTRATIVE_COUNTRIES = {
    "ARG": "Argentina",
    "CHL": "Chile",
    "BRA": "Brazil",
    "TUR": "Turkiye",
    "KEN": "Kenya",
}

# ---------------------------------------------------------------------------
# Feature sets — mirror train_backtest.py exactly
# ---------------------------------------------------------------------------
FEATURES_M0: list[str] = [
    "gdelt_protest_7d_lag", "gdelt_protest_28d_lag",
    "gdelt_strike_7d_lag",  "gdelt_strike_28d_lag",
    "gdelt_protest_region_14d", "gdelt_strike_region_14d",
]
FEATURES_M1_ADD: list[str] = [
    "fx_pct_7d", "fx_pct_30d", "fx_pct_90d", "fx_vol_7d", "fx_vol_30d",
    "oil_brent_pct_14d", "oil_brent_pct_30d", "yield_us10y",
    "fx_pct_30d_z", "fx_vol_30d_z", "oil_brent_pct_30d_z",
    "copper_pct_30d", "copper_pct_90d", "copper_vol_30d",
    "gold_pct_30d", "gold_vol_30d", "platinum_pct_30d",
    "silver_pct_30d", "natgas_pct_30d",
    "vix_level", "vix_pct_30d", "vix_7d_ma",
    "dxy_level", "dxy_pct_30d", "dxy_vol_30d",
    "oil_brent_pct_30d_lag30d", "oil_brent_pct_30d_lag60d", "oil_brent_pct_30d_lag90d",
    "vix_pct_30d_lag30d", "vix_pct_30d_lag60d",
    "fx_pct_30d_lag30d", "fx_pct_30d_lag60d",
]
FEATURES_M2_ADD: list[str] = [
    "gdp_growth", "gdp_per_capita_growth",
    "inflation_cpi_yoy", "inflation_cpi_yoy_z",
    "unemployment_total", "unemployment_youth",
    "unemployment_sa", "unemployment_rate", "unemployment_rate_z",
    "political_stability_est", "voice_accountability_est",
    "government_effectiveness_est", "rule_of_law_est",
    "fx_pct_30d_x_instability", "oil_brent_pct_30d_x_inflation",
    "food_cpi_inflation", "food_cpi_inflation_z",
    "energy_cpi_inflation", "energy_cpi_inflation_z",
]
FEATURES_M3_ADD: list[str] = [
    "gini_coef", "covid_period", "fx_trend_consistent", "inflation_accel",
    "copper_pct_30d_x_copper_prod", "gold_pct_30d_x_gold_prod",
    "platinum_pct_30d_x_plat_prod", "oil_brent_pct_30d_x_net_importer",
]
FEATURES_M4_ADD: list[str] = [
    "month_sin", "month_cos",
    "fao_food_index_yoy", "fao_cereals_index_yoy", "fao_oils_index_yoy",
    "fao_food_index_yoy_above90", "fao_cereals_index_yoy_above90",
    "fao_cereals_index_yoy_lag1m", "fao_cereals_index_yoy_lag3m", "fao_cereals_index_yoy_lag6m",
    "fao_food_index_yoy_lag1m", "fao_food_index_yoy_lag3m", "fao_food_index_yoy_lag6m",
    "fao_oils_index_yoy_lag1m", "fao_oils_index_yoy_lag3m", "fao_oils_index_yoy_lag6m",
    "fao_cereals_yoy_x_instability", "fao_food_yoy_x_youth_unemp",
    "gta_harmful_events", "gta_harmful_events_z",
    "gta_liberalising_events", "gta_liberalising_events_z",
    "gta_30d_count", "gta_30d_count_z", "gta_90d_count", "gta_90d_count_z",
]

FEATURE_SETS: dict[str, list[str]] = {
    "model0_persistence":  FEATURES_M0,
    "model1_markets":      FEATURES_M0 + FEATURES_M1_ADD,
    "model2_full":         FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD,
    "model3_structural":   FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD,
    "model4_fao":          FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD + FEATURES_M4_ADD,
    "model_lr_nolag":      FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD + FEATURES_M4_ADD,
}

# Baseline cols are computed from training data each month — not shifted
BASELINE_COLS = ["country_protest_baseline", "country_strike_baseline"]

MODEL_COLORS = {
    "model0_persistence":  "#1f77b4",
    "model1_markets":      "#ff7f0e",
    "model2_full":         "#2ca02c",
    "model3_structural":   "#d62728",
    "model4_fao":          "#9467bd",
    "model_lr_nolag":      "#17becf",
}
MODEL_LABELS = {
    "model0_persistence":  "M0 Persistence",
    "model1_markets":      "M1 + Markets",
    "model2_full":         "M2 + Macro",
    "model3_structural":   "M3 + Structural",
    "model4_fao":          "M4 + FAO/GTA",
    "model_lr_nolag":      "LR No Lags",
}

STYLE = {
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
}
plt.rcParams.update(STYLE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_lr_pipeline(feature_cols: list[str], include_fe: bool,
                      class_weight) -> Pipeline:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    if include_fe:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, feature_cols),
                ("fe",  OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False), ["country_iso3"]),
            ],
            remainder="drop",
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_pipe, feature_cols)],
            remainder="drop",
        )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=1000,
            class_weight=class_weight,
        )),
    ])


def shift_features_by_horizon(df: pd.DataFrame, feature_cols: list[str],
                               horizon: int) -> pd.DataFrame:
    """Shift feature columns forward by horizon days per country."""
    df = df.copy().sort_values(["country_iso3", "date"]).reset_index(drop=True)
    for col in feature_cols:
        if col in df.columns:
            df[col] = df.groupby("country_iso3")[col].shift(horizon)
    return df


def compute_baselines(train_df: pd.DataFrame) -> pd.DataFrame:
    """Country-level mean protest/strike rate from training data."""
    cols = [c for c in ["protest_today", "strike_today"] if c in train_df.columns]
    if not cols:
        return pd.DataFrame(columns=["country_iso3"] + BASELINE_COLS)
    bl = (train_df.groupby("country_iso3")[cols].mean()
          .rename(columns={"protest_today": "country_protest_baseline",
                            "strike_today":  "country_strike_baseline"})
          .reset_index())
    for col in BASELINE_COLS:
        if col not in bl.columns:
            bl[col] = 0.0
    return bl


def monthly_periods(test_start: pd.Timestamp,
                    test_end: pd.Timestamp) -> list[tuple]:
    """
    Returns list of (train_cutoff, month_start, month_end).
    First period uses BASE_TRAIN_END as train_cutoff (identical to fold 1).
    Subsequent periods expand by one month each time.
    """
    periods = []
    current = test_start
    while current <= test_end:
        month_end    = min(current + pd.offsets.MonthEnd(0), test_end)
        train_cutoff = current - pd.Timedelta(days=1)
        periods.append((train_cutoff, current, month_end))
        current = month_end + pd.Timedelta(days=1)
    return periods


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_expanding_lr(panel: pd.DataFrame) -> None:
    all_panel_feat_cols = sorted(set(
        FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD +
        FEATURES_M3_ADD + FEATURES_M4_ADD
    ))
    # Exclude baseline cols from the horizon shift (computed fresh each month)
    cols_to_shift = [c for c in all_panel_feat_cols
                     if c not in BASELINE_COLS and c in panel.columns]

    periods = monthly_periods(TEST_START, TEST_END)
    log.info("Monthly expanding window: %d periods (2020-2021)", len(periods))

    for event_type, horizon in TARGETS:
        label = f"{event_type}_{horizon}d"
        log.info("=" * 60)
        log.info("TARGET: %s", label)
        log.info("=" * 60)

        # Assign target labels on full panel
        df = panel.copy()
        df["y"] = make_target_gdelt(df, event_type, horizon,
                                    exclude_low_coverage=False)

        # Apply horizon shift once to all non-baseline features
        log.info("Shifting %d features by %d days...", len(cols_to_shift), horizon)
        df = shift_features_by_horizon(df, cols_to_shift, horizon)

        all_preds:   list[pd.DataFrame] = []
        all_metrics: list[dict]         = []

        for train_cutoff, month_start, month_end in periods:
            month_label = month_start.strftime("%Y-%m")

            # Slice train / test from the horizon-shifted panel
            train_df = df[(df["date"] >= TRAIN_START) &
                          (df["date"] <= train_cutoff)].copy()
            test_df  = df[(df["date"] >= month_start) &
                          (df["date"] <= month_end)].copy()

            # Compute country baselines from unshifted training data
            train_orig = panel[(panel["date"] >= TRAIN_START) &
                               (panel["date"] <= train_cutoff)]
            bl = compute_baselines(train_orig)

            for df_split in (train_df, test_df):
                for col in BASELINE_COLS:
                    if col in df_split.columns:
                        df_split.drop(columns=[col], inplace=True)
            train_df = train_df.merge(bl, on="country_iso3", how="left")
            test_df  = test_df.merge(bl,  on="country_iso3", how="left")

            train_labeled = train_df.dropna(subset=["y"])
            test_labeled  = test_df.dropna(subset=["y"])

            if test_labeled.empty:
                continue

            for model_name, feat_list in FEATURE_SETS.items():
                feats = [f for f in feat_list if f in df.columns or f in BASELINE_COLS]
                feats = [f for f in feats if f in train_labeled.columns]
                if not feats:
                    continue

                if (len(train_labeled) < 100 or
                        train_labeled["y"].nunique() < 2):
                    continue

                cw   = "balanced" if USE_CLASS_WEIGHT else None
                pipe = build_lr_pipeline(feats, INCLUDE_COUNTRY_FE, cw)

                input_cols = feats + (["country_iso3"] if INCLUDE_COUNTRY_FE else [])
                try:
                    pipe.fit(train_labeled[input_cols], train_labeled["y"])
                    y_pred = pipe.predict_proba(
                        test_labeled[input_cols])[:, 1]
                except Exception as exc:
                    log.warning("  %s %s FAILED: %s", month_label, model_name, exc)
                    continue

                y_true  = test_labeled["y"].values
                metrics = compute_metrics(y_true, y_pred)

                preds_df = test_labeled[["date", "country_iso3", "y"]].copy()
                preds_df["y_pred"]        = y_pred
                preds_df["model_name"]    = model_name
                preds_df["retrain_month"] = month_label
                preds_df.rename(columns={"y": "y_true"}, inplace=True)
                all_preds.append(preds_df)

                all_metrics.append({
                    "month":      month_label,
                    "model_name": model_name,
                    **metrics,
                })

            n_train = len(train_labeled)
            n_test  = len(test_labeled)
            log.info("  %s | train=%d  test=%d", month_label, n_train, n_test)

        if all_preds:
            out = OUT_DIR / f"preds_{label}.parquet"
            pd.concat(all_preds, ignore_index=True).to_parquet(out, index=False)
            log.info("Saved -> %s", out)

        if all_metrics:
            out = OUT_DIR / f"metrics_{label}.csv"
            pd.DataFrame(all_metrics).to_csv(out, index=False)
            log.info("Saved -> %s", out)


# ---------------------------------------------------------------------------
# Figure A: probability time-series, 5 countries, M0 vs M4
# ---------------------------------------------------------------------------

def fig_expanding_timelines(primary_target: str = "protest_7d") -> None:
    p = OUT_DIR / f"preds_{primary_target}.parquet"
    if not p.exists():
        log.warning("No expanding LR predictions found for %s", primary_target)
        return

    preds = pd.read_parquet(p)
    preds["date"] = pd.to_datetime(preds["date"])

    countries = [c for c in ILLUSTRATIVE_COUNTRIES
                 if c in preds["country_iso3"].values][:5]
    if not countries:
        countries = preds["country_iso3"].value_counts().head(5).index.tolist()

    models_to_show = [
        ("model0_persistence", "M0 Persistence", MODEL_COLORS["model0_persistence"]),
        ("model4_fao",         "M4 + FAO/GTA",   MODEL_COLORS["model4_fao"]),
    ]

    # Monthly retraining boundaries (skip the first — same as base training)
    periods = monthly_periods(TEST_START, TEST_END)
    retrain_marks = [ms for _, ms, _ in periods[1:]]

    n   = len(countries)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n), squeeze=False)
    fig.suptitle(
        f"LR Expanding Window (monthly retraining) -- {primary_target}\n"
        f"M0 Persistence vs M4 Full, 2020-2021",
        fontsize=11, fontweight="bold",
    )

    for ax, iso3 in zip(axes[:, 0], countries):
        for model_name, lbl, color in models_to_show:
            c_sub = (preds[(preds["model_name"] == model_name) &
                           (preds["country_iso3"] == iso3)]
                     .sort_values("date"))
            if c_sub.empty:
                continue
            ax.fill_between(c_sub["date"], c_sub["y_pred"],
                            alpha=0.10, color=color)
            ax.plot(c_sub["date"], c_sub["y_pred"],
                    color=color, lw=1.4, alpha=0.9, label=lbl)

        # Event markers
        ref = (preds[(preds["model_name"] == "model0_persistence") &
                     (preds["country_iso3"] == iso3)]
               .sort_values("date"))
        events = ref[ref["y_true"] == 1]
        if not events.empty:
            ax.vlines(events["date"], 0, 0.10,
                      color="grey", lw=0.7, alpha=0.4, label="Event day")

        # Monthly retraining boundaries and fold marker
        if not ref.empty:
            d_min, d_max = ref["date"].min(), ref["date"].max()
            first_retrain = True
            for rd in retrain_marks:
                if d_min <= rd <= d_max:
                    ax.axvline(rd, color="#d73027", lw=0.8, ls=":",
                               alpha=0.7,
                               label="Model retrain" if first_retrain else None)
                    first_retrain = False

            # 2021 fold boundary
            boundary = pd.Timestamp("2021-01-01")
            if d_min <= boundary <= d_max:
                ax.axvline(boundary, color="black", lw=1.2, ls="--", alpha=0.7)
                ax.text(boundary, 0.97, " 2021", fontsize=7, color="black",
                        va="top", transform=ax.get_xaxis_transform())
            ax.set_xlim(d_min, d_max)

        name = ILLUSTRATIVE_COUNTRIES.get(iso3, iso3)
        ax.set_title(f"{name} ({iso3})", fontsize=9, loc="left", fontweight="bold")
        ax.set_ylabel("P(event)", fontsize=8)
        ax.set_ylim(0, 1.05)
        if ax is axes[0, 0]:
            ax.legend(fontsize=8, loc="upper right")

    axes[-1, 0].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / f"expanding_lr_timelines_{primary_target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Figure B: monthly ROC-AUC and PR-AUC for all LR models
# ---------------------------------------------------------------------------

def fig_expanding_performance(primary_target: str = "protest_7d") -> None:
    p = OUT_DIR / f"metrics_{primary_target}.csv"
    if not p.exists():
        log.warning("No expanding LR metrics found for %s", primary_target)
        return

    metrics = pd.read_csv(p)
    metrics["month_dt"] = pd.to_datetime(metrics["month"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle(
        f"LR Expanding Window -- Monthly Performance ({primary_target})\n"
        "Monthly retraining, 2020-2021",
        fontsize=11, fontweight="bold",
    )

    for model_name, color in MODEL_COLORS.items():
        sub = metrics[metrics["model_name"] == model_name].sort_values("month_dt")
        if sub.empty:
            continue
        lbl = MODEL_LABELS[model_name]
        axes[0].plot(sub["month_dt"], sub["roc_auc"],
                     color=color, lw=1.5, marker="o", ms=3, label=lbl)
        axes[1].plot(sub["month_dt"], sub["pr_auc"],
                     color=color, lw=1.5, marker="o", ms=3, label=lbl)

    # 2021 fold boundary
    boundary = pd.Timestamp("2021-01-01")
    for ax in axes:
        ax.axvline(boundary, color="grey", lw=1.0, ls=":", alpha=0.7)
        ylim = ax.get_ylim()
        ax.text(boundary, ylim[0] + 0.01 * (ylim[1] - ylim[0]),
                " 2021", fontsize=7, color="grey")

    axes[0].set_title("ROC-AUC", fontsize=10)
    axes[1].set_title("PR-AUC",  fontsize=10)
    for ax in axes:
        ax.set_xlabel("Month", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(fontsize=8, loc="lower left")

    plt.tight_layout()
    out = FIG_DIR / f"expanding_lr_performance_{primary_target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("Loading panel...")
    panel = pd.read_parquet(PANEL_FILE)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["country_iso3", "date"]).reset_index(drop=True)
    log.info("Panel: %d rows x %d cols | %d countries",
             len(panel), panel.shape[1], panel["country_iso3"].nunique())

    run_expanding_lr(panel)

    log.info("\nGenerating figures...")
    for target in [t[0] + "_" + str(t[1]) + "d" for t in TARGETS]:
        fig_expanding_timelines(target)
        fig_expanding_performance(target)

    log.info("Done.")


if __name__ == "__main__":
    main()
