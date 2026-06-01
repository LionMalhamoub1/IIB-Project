# Walk-forward backtest using GDELT-derived labels. Two test folds (2020, 2021), M0-M7.

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (
    calibration_summary,
    compute_metrics,
    estimate_labelling_probability,
    estimate_reliable_countries,
    make_target_gdelt,
    make_target_pu,
    make_target_pu_country,
)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Walk-forward backtest window.
# Update LAST_TEST_YEAR to 2022 once GDELT labels for 2021-2022 are ready
# and fetch_fx_missing.py / fetch_supplemental_data.py have been re-run.
FIRST_TEST_YEAR: int = 2020
LAST_TEST_YEAR:  int = 2021

# Targets to run: (event_type, horizon_days)
# protest_30d excluded: true ~94% of the time in well-monitored countries,
# leaving no variation for a classifier to discriminate.
TARGETS: list[tuple[str, int]] = [
    ("protest", 7),
    ("strike",  7),
    ("strike",  30),
]

# Mask zero labels for country-days with very few GDELT articles
EXCLUDE_LOW_COVERAGE: bool = False  # legacy flag — superseded by USE_PU_LEARNING
USE_PU_LEARNING:     bool = False  # treat unreliable-country zeros as unlabelled

# Minimum fraction of days with medium/high GDELT coverage for a country to be
# considered reliably monitored.  Its zero-event days are then trusted as
# genuine negatives.  Countries below this threshold have their zeros masked.
RELIABLE_COUNTRY_THRESHOLD: float = 2.0   # median articles per event — see utils.estimate_reliable_countries

INCLUDE_COUNTRY_FE: bool = True
USE_CLASS_WEIGHT:   bool = True
USE_XGBOOST:        bool = True

XGB_OPTUNA_TRIALS:         int = 40
XGB_EARLY_STOPPING_ROUNDS: int = 20
_MIN_CAL_OBS:              int = 30   # minimum val-set rows required to fit calibration

INCOME_GROUPS: dict[str, set[str]] = {
    "high": {
        "AUS", "CAN", "CHL", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA",
        "JPN", "KOR", "NLD", "NOR", "POL", "PRT", "ESP", "SWE", "GBR", "USA",
    },
    "upper_middle": {
        "ARG", "BRA", "CHN", "MYS", "MEX", "NAM", "PER", "ZAF", "THA", "TUR",
    },
    "lower_middle": {
        # COD removed: not in the 39-country panel
        "BOL", "IND", "IDN", "KEN", "LAO", "MAR", "MOZ", "PHL", "VNM", "ZWE",
    },
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_V2        = _SRC.parent
PANEL_FILE = _V2 / "data" / "interim" / "modelling_panel_gdelt.parquet"
PROC_DIR   = _V2 / "data" / "processed"

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

# M0: GDELT-based autoregressive (persistence) features
FEATURES_M0: list[str] = [
    "gdelt_protest_7d_lag",
    "gdelt_protest_28d_lag",
    "gdelt_strike_7d_lag",
    "gdelt_strike_28d_lag",
    # Cross-country regional spillover (neighboring countries' recent activity)
    "gdelt_protest_region_14d",
    "gdelt_strike_region_14d",
]

# M1: financial market features (including global commodity prices)
FEATURES_M1_ADD: list[str] = [
    "fx_pct_7d",
    "fx_pct_30d",
    "fx_pct_90d",
    "fx_vol_7d",
    "fx_vol_30d",
    "oil_brent_pct_14d",
    "oil_brent_pct_30d",
    "yield_us10y",
    "fx_pct_30d_z",
    "fx_vol_30d_z",
    "oil_brent_pct_30d_z",
    # Global commodity prices
    "copper_pct_30d",
    "copper_pct_90d",
    "copper_vol_30d",
    "gold_pct_30d",
    "gold_vol_30d",
    "platinum_pct_30d",
    "silver_pct_30d",
    "natgas_pct_30d",
    # Global market sentiment (VIX + DXY) — same value for all 39 countries
    "vix_level",       # absolute fear gauge level (spikes precede EM stress)
    "vix_pct_30d",     # 30-day change in fear (rising fear = deteriorating conditions)
    "vix_7d_ma",       # 7-day smoothed VIX (filters day-to-day noise)
    "dxy_level",       # absolute USD strength (high = EM currency/debt pressure)
    "dxy_pct_30d",     # 30-day USD appreciation (EM imports/debt become costlier)
    "dxy_vol_30d",     # USD volatility (uncertainty in global financial conditions)
    # Lagged market signals — delayed transmission of price shocks to unrest
    "oil_brent_pct_30d_lag30d",   # oil price change, as seen 30d ago
    "oil_brent_pct_30d_lag60d",   # oil price change, as seen 60d ago
    "oil_brent_pct_30d_lag90d",   # oil price change, as seen 90d ago
    "vix_pct_30d_lag30d",         # VIX change, lagged 30d
    "vix_pct_30d_lag60d",         # VIX change, lagged 60d
    "fx_pct_30d_lag30d",          # FX depreciation, lagged 30d
    "fx_pct_30d_lag60d",          # FX depreciation, lagged 60d
]

# M2: macro, labour, governance
# Notes on dropped features:
#   - earnings_monthly / earnings_monthly_z : 92% missing, too sparse
#   - *_z variants of annual WDI/WGI indicators (gdp_growth_z, unemployment_*_z,
#     political_stability_est_z, etc.): expanding z-score over 3 years with
#     min_obs=3 means they are NaN for 2018-2019 (the entire training period)
#     and only valid in 2020 (test year) — effectively useless for training
#   - economic_stress_index / labour_conflict_index / protest_mobilisation_index
#     and their _z variants: Google Trends-based, 51% missing, cannot be completed
#     for all 39 countries
FEATURES_M2_ADD: list[str] = [
    "gdp_growth",
    "gdp_per_capita_growth",
    "inflation_cpi_yoy",       "inflation_cpi_yoy_z",
    "unemployment_total",
    "unemployment_youth",
    "unemployment_sa",
    "unemployment_rate",       "unemployment_rate_z",
    "political_stability_est",
    "voice_accountability_est",
    "government_effectiveness_est",
    "rule_of_law_est",
    "fx_pct_30d_x_instability",
    "oil_brent_pct_30d_x_inflation",
    "food_cpi_inflation",      "food_cpi_inflation_z",
    "energy_cpi_inflation",    "energy_cpi_inflation_z",
]

# M3: structural / supplemental features (Gini, COVID, commodity interactions, etc.)
# country_protest_baseline and country_strike_baseline are computed per-fold
# inside run_backtest() from training data only — not baked into the panel.
# Binary producer/importer flags removed — with country FEs active they are
# redundant proxy fixed effects. Interaction terms kept as they capture
# asymmetric commodity price sensitivity.
FEATURES_M3_ADD: list[str] = [
    "gini_coef",
    "covid_period",
    "fx_trend_consistent",
    "inflation_accel",
    "copper_pct_30d_x_copper_prod",
    "gold_pct_30d_x_gold_prod",
    "platinum_pct_30d_x_plat_prod",
    "oil_brent_pct_30d_x_net_importer",
]

# M4: FAO food prices, GTA trade interventions, temporal
FEATURES_M4_ADD: list[str] = [
    "month_sin",
    "month_cos",
    "fao_food_index_yoy",
    "fao_cereals_index_yoy",
    "fao_oils_index_yoy",
    "fao_food_index_yoy_above90",
    "fao_cereals_index_yoy_above90",
    "fao_cereals_index_yoy_lag1m",
    "fao_cereals_index_yoy_lag3m",
    "fao_cereals_index_yoy_lag6m",
    "fao_food_index_yoy_lag1m",
    "fao_food_index_yoy_lag3m",
    "fao_food_index_yoy_lag6m",        # food price 6-month lag
    "fao_oils_index_yoy_lag1m",        # oils price 1-month lag
    "fao_oils_index_yoy_lag3m",        # oils price 3-month lag
    "fao_oils_index_yoy_lag6m",        # oils price 6-month lag
    "fao_cereals_yoy_x_instability",
    "fao_food_yoy_x_youth_unemp",
    "gta_harmful_events",      "gta_harmful_events_z",
    "gta_liberalising_events", "gta_liberalising_events_z",
    "gta_30d_count",           "gta_30d_count_z",
    "gta_90d_count",           "gta_90d_count_z",
]

MODEL_SPECS: dict[str, list[str]] = {
    "model0_persistence":  FEATURES_M0,
    "model1_markets":      FEATURES_M0 + FEATURES_M1_ADD,
    "model2_full":         FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD,
    "model3_structural":   FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD,
    "model4_fao":          FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD + FEATURES_M4_ADD,
    "model_lr_nolag":      FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD + FEATURES_M4_ADD,
}

FEATURES_XGB: list[str] = (
    FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD + FEATURES_M4_ADD
)

# No-lag variant: removes autoregressive GDELT features (pure early-warning model)
FEATURES_XGB_NOLAG: list[str] = (
    FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD + FEATURES_M4_ADD
)

# No-baseline variant: full features minus country_protest/strike_baseline
# Used to isolate how much of the model's behaviour is anchoring to historical rates
_BASELINE_FEATS = {"country_protest_baseline", "country_strike_baseline"}
FEATURES_XGB_NOBASELINE: list[str] = [
    f for f in (
        FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD + FEATURES_M4_ADD
    ) if f not in _BASELINE_FEATS
]

XGB_PARAM_GRID: list[dict[str, Any]] = [
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
     "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
     "reg_lambda": 1.0, "reg_alpha": 0.0},
    {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.03,
     "subsample": 0.8, "colsample_bytree": 0.7, "min_child_weight": 5,
     "reg_lambda": 2.0, "reg_alpha": 0.1},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
     "subsample": 0.7, "colsample_bytree": 0.8, "min_child_weight": 3,
     "reg_lambda": 1.0, "reg_alpha": 0.0},
    {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.02,
     "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 1,
     "reg_lambda": 0.5, "reg_alpha": 0.0},
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Panel loading
# ---------------------------------------------------------------------------

def load_panel() -> pd.DataFrame:
    if not PANEL_FILE.exists():
        raise FileNotFoundError(
            f"GDELT modelling panel not found: {PANEL_FILE}\n"
            "Run build_panel.py first."
        )
    df = pd.read_parquet(PANEL_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country_iso3", "date"]).reset_index(drop=True)
    logger.info(
        "Panel loaded: %d rows x %d cols | %d countries | %s to %s",
        len(df), len(df.columns), df["country_iso3"].nunique(),
        str(df["date"].min().date()), str(df["date"].max().date()),
    )
    return df


def available_features(wanted: list[str], cols: list[str]) -> list[str]:
    present = [f for f in wanted if f in cols]
    missing = [f for f in wanted if f not in cols]
    if missing:
        logger.debug("Features absent from panel (skipped): %s", missing)
    return present


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_lr_pipeline(
    feature_cols: list[str],
    include_fe: bool,
    balanced: bool,
) -> Pipeline:
    cw = "balanced" if balanced else None
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    if include_fe:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, feature_cols),
                ("fe", OneHotEncoder(drop="first", handle_unknown="ignore",
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
            penalty="l2", solver="lbfgs", max_iter=1000, class_weight=cw,
        )),
    ])


def build_xgb_pipeline(
    feature_cols: list[str],
    include_fe: bool,
    pos_neg_ratio: float,
    params: dict[str, Any],
) -> Pipeline:
    xgb = XGBClassifier(
        scale_pos_weight=pos_neg_ratio,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        **params,
    )
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    if include_fe:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, feature_cols),
                ("fe", OneHotEncoder(drop="first", handle_unknown="ignore",
                                     sparse_output=False), ["country_iso3"]),
            ],
            remainder="drop",
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_pipe, feature_cols)],
            remainder="drop",
        )
    return Pipeline([("preprocessor", preprocessor), ("model", xgb)])


def extract_coefs(pipe: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    try:
        names = list(pipe.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        n     = len(pipe.named_steps["model"].coef_[0])
        names = [f"feat_{i}" for i in range(n)]
    coefs = pipe.named_steps["model"].coef_[0]
    return pd.DataFrame({"feature": names, "coefficient": coefs})


# ---------------------------------------------------------------------------
# Probability calibration
# ---------------------------------------------------------------------------

def _fit_calibration(
    pipe:  Pipeline,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> tuple:
    """
    Fit isotonic and sigmoid calibration wrappers on the held-out val set.

    Uses CalibratedClassifierCV with cv='prefit': the XGBoost model inside
    the pipeline is frozen; only the thin calibration layer is fitted.
    Features are preprocessed with the pipeline's own preprocessor.

    Returns
    -------
    (isotonic_cal, sigmoid_cal, best_method, val_briers)
      isotonic_cal / sigmoid_cal : fitted CalibratedClassifierCV or None
      best_method                : "isotonic" | "sigmoid" | "none"
      val_briers                 : {"isotonic": float, "sigmoid": float}
    """
    from sklearn.metrics import brier_score_loss as _brier

    preprocessor = pipe.named_steps["preprocessor"]
    xgb_model    = pipe.named_steps["model"]

    try:
        X_val_proc = preprocessor.transform(X_val)
    except Exception as exc:
        logger.warning("Calibration preprocessing failed: %s", exc)
        return None, None, "none", {"isotonic": np.nan, "sigmoid": np.nan}

    calibrators: dict = {}
    val_briers:  dict = {}

    for method in ("isotonic", "sigmoid"):
        try:
            cal = CalibratedClassifierCV(estimator=xgb_model, cv="prefit", method=method)
            cal.fit(X_val_proc, y_val)
            yp = cal.predict_proba(X_val_proc)[:, 1]
            val_briers[method]  = float(_brier(y_val, yp))
            calibrators[method] = cal
        except Exception as exc:
            logger.warning("Calibration (%s) failed: %s", method, exc)
            calibrators[method] = None
            val_briers[method]  = np.inf

    valid       = {k: v for k, v in val_briers.items() if np.isfinite(v)}
    best_method = min(valid, key=valid.get) if valid else "none"

    logger.info(
        "    Calibration val Brier — isotonic: %.4f  sigmoid: %.4f  → best: %s",
        val_briers.get("isotonic", np.nan),
        val_briers.get("sigmoid",  np.nan),
        best_method,
    )
    return calibrators.get("isotonic"), calibrators.get("sigmoid"), best_method, val_briers


def plot_calibration_figure(preds_all: pd.DataFrame, label: str) -> None:
    """
    Reliability diagram: mean predicted probability vs fraction of positives.

    Plots raw XGBoost output alongside the calibrated output for each of the
    three XGBoost variants (M5 full, M6 no-lag, M7 no-baseline).
    Saved to v2/figures/calibration_{label}.{png,pdf}.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping calibration plot.")
        return

    xgb_variants = [
        m for m in ["model5_xgb", "model6_xgb_nolag", "model7_xgb_nobaseline"]
        if m in preds_all["model_name"].unique()
    ]
    if not xgb_variants:
        return

    n_cols = len(xgb_variants)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.8 * n_cols, 4.2), squeeze=False)
    axes = axes[0]

    name_map = {
        "model5_xgb":           "M5 Full",
        "model6_xgb_nolag":     "M6 No-lag",
        "model7_xgb_nobaseline":"M7 No-baseline",
    }

    for ax, mn in zip(axes, xgb_variants):
        sub = preds_all[preds_all["model_name"] == mn].dropna(subset=["y_true", "y_pred"])
        if len(sub) < 20 or sub["y_true"].nunique() < 2:
            ax.set_visible(False)
            continue

        yt = sub["y_true"].values.astype(int)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1, label="Perfect")

        if "y_pred_raw" in sub.columns and sub["y_pred_raw"].notna().any():
            frac_r, mean_r = calibration_curve(
                yt, sub["y_pred_raw"].values, n_bins=10, strategy="uniform"
            )
            ax.plot(mean_r, frac_r, "o--", color="#F59E0B", lw=1.8, ms=5,
                    label="XGBoost (raw)")

        frac_c, mean_c = calibration_curve(
            yt, sub["y_pred"].values, n_bins=10, strategy="uniform"
        )
        ax.plot(mean_c, frac_c, "s-", color="#3B82F6", lw=2, ms=6,
                label="Calibrated")

        ax.set_xlabel("Mean predicted probability", fontsize=9)
        ax.set_ylabel("Fraction of positives",      fontsize=9)
        ax.set_title(f"{name_map.get(mn, mn)}\n({label})", fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"Calibration reliability diagram — {label}", fontsize=10)
    fig.tight_layout()

    fig_dir = _V2 / "figures"
    fig_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(fig_dir / f"calibration_{label}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Calibration plot saved -> figures/calibration_%s.png", label)


# ---------------------------------------------------------------------------
# XGBoost tuning
# ---------------------------------------------------------------------------

def _tune_xgb_params(
    train_df:    pd.DataFrame,
    feature_cols: list[str],
    val_year:    int,
) -> dict[str, Any]:
    val_mask  = train_df["date"].dt.year == val_year
    sub_train = train_df[~val_mask].dropna(subset=["y"])
    sub_val   = train_df[val_mask].dropna(subset=["y"])

    if sub_train.empty or sub_val.empty or len(sub_val["y"].unique()) < 2:
        logger.warning("Cannot tune XGB: insufficient validation data — using default params.")
        return XGB_PARAM_GRID[0]

    feat  = available_features(feature_cols, list(sub_train.columns))
    imp   = SimpleImputer(strategy="median")
    X_tr  = imp.fit_transform(sub_train[feat])
    y_tr  = sub_train["y"].astype(int).values
    X_val = imp.transform(sub_val[feat])
    y_val = sub_val["y"].astype(int).values

    n_neg = int((y_tr == 0).sum())
    n_pos = int((y_tr == 1).sum())
    ratio = n_neg / max(n_pos, 1)

    if _OPTUNA_AVAILABLE:
        def _objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth":        trial.suggest_int("max_depth", 3, 7),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            }
            clf = XGBClassifier(
                n_estimators=1000,
                scale_pos_weight=ratio,
                eval_metric="aucpr",
                early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                **params,
            )
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return compute_metrics(y_val, clf.predict_proba(X_val)[:, 1])["pr_auc"]

        study = optuna.create_study(direction="maximize")
        study.optimize(_objective, n_trials=XGB_OPTUNA_TRIALS, show_progress_bar=False)
        best = dict(study.best_params)

        best_clf = XGBClassifier(
            n_estimators=1000,
            scale_pos_weight=ratio,
            eval_metric="aucpr",
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **best,
        )
        best_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        best["n_estimators"] = best_clf.best_iteration + 1

        logger.info(
            "Optuna XGB (%d trials): best PR-AUC=%.4f  n_est=%d",
            XGB_OPTUNA_TRIALS, study.best_value, best["n_estimators"],
        )
        return best

    # Fallback grid search
    logger.warning("optuna not installed — using grid search for XGB tuning.")
    best_score, best_params = -np.inf, XGB_PARAM_GRID[0]
    for params in XGB_PARAM_GRID:
        try:
            pipe = build_xgb_pipeline(feat, False, ratio, params)
            pipe.fit(sub_train[feat], y_tr)
            m = compute_metrics(y_val, pipe.predict_proba(sub_val[feat])[:, 1])
            if m["pr_auc"] > best_score:
                best_score, best_params = m["pr_auc"], params
        except Exception as exc:
            logger.warning("XGB param config failed: %s", exc)
    logger.info("Grid XGB tuning: best PR-AUC=%.4f", best_score)
    return best_params


# ---------------------------------------------------------------------------
# Fold runners
# ---------------------------------------------------------------------------

def run_lr_fold(
    train_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    feature_cols: list[str],
    event_type:   str,
    horizon:      int,
    fold_id:      int,
    model_name:   str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feat = available_features(feature_cols, list(train_df.columns))
    if not feat:
        logger.warning("No valid features for %s fold=%d.", model_name, fold_id)
        return pd.DataFrame(), pd.DataFrame(), {}

    train_clean = train_df.dropna(subset=["y"])
    test_clean  = test_df.dropna(subset=["y"])
    if train_clean.empty or test_clean.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    X_train = train_clean[feat + (["country_iso3"] if INCLUDE_COUNTRY_FE else [])]
    y_train = train_clean["y"].astype(int).values
    X_test  = test_clean[feat  + (["country_iso3"] if INCLUDE_COUNTRY_FE else [])]
    y_true  = test_clean["y"].astype(int).values

    pipe   = build_lr_pipeline(feat, INCLUDE_COUNTRY_FE, USE_CLASS_WEIGHT)
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    m = compute_metrics(y_true, y_prob)
    m.update({
        "fold_id":    fold_id,
        "event_type": event_type,
        "horizon":    horizon,
        "model_name": model_name,
        "n_train":    len(train_clean),
        "n_test":     len(test_clean),
        "pos_rate":   float(y_true.mean()),
    })

    preds_df = pd.DataFrame({
        "country_iso3": test_clean["country_iso3"].values,
        "date":         test_clean["date"].values,
        "y_true":       y_true,
        "y_pred":       y_prob,
        "fold_id":      fold_id,
        "event_type":   event_type,
        "horizon":      horizon,
        "model_name":   model_name,
    })

    coefs_df = extract_coefs(pipe, feat).assign(
        fold_id=fold_id, event_type=event_type,
        horizon=horizon, model_name=model_name,
    )

    return preds_df, coefs_df, m


def run_xgb_fold(
    train_df:     pd.DataFrame,
    test_df:      pd.DataFrame,
    feature_cols: list[str],
    event_type:   str,
    horizon:      int,
    fold_id:      int,
    tuned_params: dict[str, Any],
    model_name:   str = "model5_xgb",
    val_year:     int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Train XGBoost, optionally calibrate, and evaluate on the test fold.

    Calibration workflow (when val_year is provided)
    -------------------------------------------------
    The training data is split into two parts that mirror _tune_xgb_params:
      train_sub  (years < val_year)  — XGBoost is fitted here
      val_sub    (year == val_year)  — calibration layer is fitted here

    Both isotonic regression and Platt scaling (sigmoid) are tried on the
    val_sub Brier score; the better method is applied to the test set.
    If the val split is too small (_MIN_CAL_OBS) calibration is skipped and
    raw XGBoost probabilities are used.

    The returned preds_df always contains both y_pred (calibrated when
    available, else raw) and y_pred_raw (uncalibrated XGBoost output).
    The metrics dict carries roc_auc_raw / pr_auc_raw / brier_raw for the
    before-calibration comparison.
    """
    feat = available_features(feature_cols, list(train_df.columns))
    if not feat:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    # Split training data: sub-train for XGBoost, val for calibration.
    # Mirrors the split used in _tune_xgb_params so we never fit the
    # calibration layer on data the base model was trained on.
    if val_year is not None:
        val_mask  = train_df["date"].dt.year == val_year
        train_sub = train_df[~val_mask]
        val_sub   = train_df[val_mask]
    else:
        train_sub = train_df
        val_sub   = pd.DataFrame()

    fe_cols     = ["country_iso3"] if INCLUDE_COUNTRY_FE else []
    train_clean = train_sub.dropna(subset=["y"])
    test_clean  = test_df.dropna(subset=["y"])
    if train_clean.empty or test_clean.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    X_train = train_clean[feat + fe_cols]
    y_train = train_clean["y"].astype(int).values
    X_test  = test_clean[feat  + fe_cols]
    y_true  = test_clean["y"].astype(int).values

    n_neg, n_pos  = int((y_train == 0).sum()), int((y_train == 1).sum())
    pos_neg_ratio = n_neg / max(n_pos, 1)

    pipe = build_xgb_pipeline(feat, INCLUDE_COUNTRY_FE, pos_neg_ratio, tuned_params)
    pipe.fit(X_train, y_train)

    # Raw (uncalibrated) test predictions
    y_prob_raw  = pipe.predict_proba(X_test)[:, 1]
    raw_metrics = compute_metrics(y_true, y_prob_raw)

    # Probability calibration on validation split
    y_prob      = y_prob_raw
    best_method = "none"
    val_briers  = {"isotonic": np.nan, "sigmoid": np.nan}

    val_clean = val_sub.dropna(subset=["y"]) if not val_sub.empty else pd.DataFrame()
    if (not val_clean.empty
            and len(val_clean) >= _MIN_CAL_OBS
            and val_clean["y"].nunique() >= 2):
        iso_cal, sig_cal, best_method, val_briers = _fit_calibration(
            pipe, val_clean[feat + fe_cols], val_clean["y"].astype(int).values,
        )
        best_cal = iso_cal if best_method == "isotonic" else sig_cal
        if best_cal is not None:
            X_test_proc = pipe.named_steps["preprocessor"].transform(X_test)
            y_prob      = best_cal.predict_proba(X_test_proc)[:, 1]
    elif val_year is not None:
        logger.warning(
            "Calibration skipped for %s fold %d: val has %d obs or no class variance.",
            model_name, fold_id, len(val_clean),
        )

    m = compute_metrics(y_true, y_prob)
    m.update({
        "fold_id":            fold_id,
        "event_type":         event_type,
        "horizon":            horizon,
        "model_name":         model_name,
        "n_train":            len(train_clean),
        "n_test":             len(test_clean),
        "pos_rate":           float(y_true.mean()),
        # Before/after calibration
        "roc_auc_raw":        raw_metrics["roc_auc"],
        "pr_auc_raw":         raw_metrics["pr_auc"],
        "brier_raw":          raw_metrics["brier"],
        "brier_val_isotonic": val_briers["isotonic"],
        "brier_val_sigmoid":  val_briers["sigmoid"],
        "cal_method":         best_method,
    })

    preds_df = pd.DataFrame({
        "country_iso3": test_clean["country_iso3"].values,
        "date":         test_clean["date"].values,
        "y_true":       y_true,
        "y_pred":       y_prob,       # calibrated (or raw if calibration skipped)
        "y_pred_raw":   y_prob_raw,   # always the raw XGBoost output
        "fold_id":      fold_id,
        "event_type":   event_type,
        "horizon":      horizon,
        "model_name":   model_name,
    })

    try:
        feat_names = list(pipe.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        feat_names = [f"feat_{i}" for i in range(
            len(pipe.named_steps["model"].feature_importances_)
        )]

    imps_df = pd.DataFrame({
        "feature":    feat_names,
        "importance": pipe.named_steps["model"].feature_importances_,
        "fold_id":    fold_id,
        "event_type": event_type,
        "horizon":    horizon,
        "model_name": model_name,
    })

    # SHAP values on test set
    shap_df = pd.DataFrame()
    if _SHAP_AVAILABLE:
        try:
            xgb_model    = pipe.named_steps["model"]
            preprocessor = pipe.named_steps["preprocessor"]
            X_test_proc  = preprocessor.transform(X_test)
            explainer    = shap.TreeExplainer(xgb_model)
            shap_vals    = explainer.shap_values(X_test_proc)
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_df = pd.DataFrame({
                "feature":       feat_names,
                "mean_abs_shap": mean_abs_shap,
                "fold_id":       fold_id,
                "event_type":    event_type,
                "horizon":       horizon,
                "model_name":    model_name,
            })
        except Exception as exc:
            logger.warning("SHAP computation failed for %s fold %d: %s", model_name, fold_id, exc)

    return preds_df, imps_df, shap_df, m


# ---------------------------------------------------------------------------
# PU learning helpers
# ---------------------------------------------------------------------------

def _apply_pu_correction(preds_df: pd.DataFrame, c: float) -> pd.DataFrame:
    """
    Apply Elkan-Noto probability correction: P_true = clip(P_observed / c, 0, 1).

    If c=1 (no correction needed), returns preds_df unchanged.
    Stores both raw and corrected probabilities.
    """
    if c >= 1.0 or not USE_PU_LEARNING:
        # Don't overwrite y_pred_raw if already populated by calibration
        if "y_pred_raw" not in preds_df.columns:
            preds_df["y_pred_raw"] = preds_df["y_pred"]
        return preds_df
    preds_df = preds_df.copy()
    preds_df["y_pred_raw"] = preds_df["y_pred"]
    preds_df["y_pred"]     = np.clip(preds_df["y_pred"] / c, 0.0, 1.0)
    return preds_df


def _recompute_metrics(preds_df: pd.DataFrame, m: dict) -> dict:
    """Recompute ROC-AUC, PR-AUC and Brier from (possibly corrected) y_pred."""
    from utils import compute_metrics
    yt = preds_df["y_true"].values.astype(float)
    yp = preds_df["y_pred"].values
    mask = ~(np.isnan(yt) | np.isnan(yp))
    updated = compute_metrics(yt[mask], yp[mask])
    m = dict(m)
    m.update(updated)
    m["pos_rate"] = float(yt[mask].mean()) if mask.sum() > 0 else 0.0
    return m


# ---------------------------------------------------------------------------
# Per-fold country baselines
# ---------------------------------------------------------------------------

def compute_country_baselines(train_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute country-level protest/strike baseline rates from training data only.

    Returns a DataFrame with columns [country_iso3, country_protest_baseline,
    country_strike_baseline] — one row per country.  Applied per fold so the
    baseline reflects only information available before the test year.
    """
    cols = [c for c in ["protest_today", "strike_today"] if c in train_df.columns]
    if not cols:
        return pd.DataFrame(columns=["country_iso3",
                                     "country_protest_baseline",
                                     "country_strike_baseline"])
    baseline = (
        train_df.groupby("country_iso3")[cols]
        .mean()
        .rename(columns={
            "protest_today": "country_protest_baseline",
            "strike_today":  "country_strike_baseline",
        })
        .reset_index()
    )
    # Ensure both columns exist even if source col was absent
    for col in ("country_protest_baseline", "country_strike_baseline"):
        if col not in baseline.columns:
            baseline[col] = 0.0
    return baseline


# ---------------------------------------------------------------------------
# Leakage-safe feature shifting
# ---------------------------------------------------------------------------

def shift_features_by_horizon(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
) -> pd.DataFrame:
    """
    Shift all feature columns forward by `horizon` days per country.

    For a forecast horizon of h days, the target at row t covers events in
    [t+1, t+h].  Any feature measured at t may reflect activity at t that
    is correlated with the target window (e.g. FX rates react to protests
    happening today, which may persist into tomorrow).

    After shifting by h, the feature values at row t come from row t-h,
    i.e. economic conditions h days before the prediction date.  This
    guarantees a clean separation: features are from t-h or earlier, and
    the target window starts at t+1.

    The first `horizon` rows per country become NaN and are dropped when
    the model calls dropna(subset=["y"]).
    """
    avail = [c for c in feature_cols if c in df.columns]
    parts = []
    for iso3, grp in df.groupby("country_iso3", sort=False):
        grp = grp.sort_values("date").copy()
        grp[avail] = grp[avail].shift(horizon)
        parts.append(grp)
    result = pd.concat(parts, ignore_index=True)
    logger.debug(
        "Features shifted by %d days — first %d rows per country become NaN.",
        horizon, horizon,
    )
    return result


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest(panel: pd.DataFrame) -> None:
    logger.info("=" * 60)
    logger.info("GDELT backtest: %d targets", len(TARGETS))
    logger.info("Test years: %d - %d", FIRST_TEST_YEAR, LAST_TEST_YEAR)
    logger.info("PU learning: %s  (country-level, threshold=%.0f%%)",
                USE_PU_LEARNING, RELIABLE_COUNTRY_THRESHOLD * 100)
    logger.info("Feature leakage protection: shift features by forecast horizon")
    logger.info("=" * 60)

    all_global_metrics: list[dict] = []  # accumulates across all targets

    # All feature columns that may need shifting
    all_feature_cols = list(dict.fromkeys(
        FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD + FEATURES_M4_ADD
    ))

    # Identify reliable countries from the full panel (coverage doesn't change)
    if USE_PU_LEARNING:
        reliable_countries = estimate_reliable_countries(panel, RELIABLE_COUNTRY_THRESHOLD)
        unreliable = sorted(set(panel["country_iso3"].unique()) - reliable_countries)
        logger.info(
            "Reliable countries (%d): %s",
            len(reliable_countries), sorted(reliable_countries),
        )
        logger.info(
            "Unreliable countries (%d, zeros masked): %s",
            len(unreliable), unreliable,
        )
    else:
        reliable_countries = set(panel["country_iso3"].unique())

    for event_type, horizon in TARGETS:
        label    = f"{event_type}_{horizon}d"
        out_dir  = PROC_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n%s", "=" * 60)
        logger.info("TARGET: %s  (horizon=%dd)", label, horizon)
        logger.info("%s", "=" * 60)

        # Assign target — country-level PU masks unreliable-country zeros
        df = panel.copy()
        if USE_PU_LEARNING:
            df["y"] = make_target_pu_country(df, event_type, horizon, reliable_countries)
        else:
            df["y"] = make_target_gdelt(df, event_type, horizon, EXCLUDE_LOW_COVERAGE)

        # Shift all features by horizon to ensure no information from [t, t+h]
        # contaminates the features used to predict events in [t+1, t+h]
        logger.info("Shifting features by %d days to prevent leakage...", horizon)
        df = shift_features_by_horizon(df, all_feature_cols, horizon)

        n_labeled   = df["y"].notna().sum()
        n_pos       = (df["y"] == 1).sum()
        n_neg       = (df["y"] == 0).sum()
        n_unlabelled = df["y"].isna().sum()
        logger.info(
            "Labels: %d labeled (%d pos / %d neg) | %d unlabelled (PU-masked)",
            n_labeled, n_pos, n_neg, n_unlabelled,
        )
        pos_rate = df["y"].mean()
        logger.info("Positive rate (labeled rows only): %.1f%%", 100 * pos_rate)

        all_preds:   list[pd.DataFrame] = []
        all_coefs:   list[pd.DataFrame] = []
        all_metrics: list[dict]         = []
        all_imps:    list[pd.DataFrame] = []
        all_shaps:   list[pd.DataFrame] = []

        for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
            train_df = df[df["date"].dt.year <  test_year].copy()
            test_df  = df[df["date"].dt.year == test_year].copy()
            fold_id  = test_year - FIRST_TEST_YEAR + 1

            # Per-fold country baselines — computed from training data only.
            # Drop any pre-baked panel versions first to avoid merge conflicts.
            baselines = compute_country_baselines(train_df)
            for bl_col in ("country_protest_baseline", "country_strike_baseline"):
                train_df = train_df.drop(columns=[bl_col], errors="ignore")
                test_df  = test_df.drop(columns=[bl_col], errors="ignore")
            train_df = train_df.merge(baselines, on="country_iso3", how="left")
            test_df  = test_df.merge(baselines, on="country_iso3", how="left")
            for bl_col in ("country_protest_baseline", "country_strike_baseline"):
                train_df[bl_col] = train_df[bl_col].fillna(0.0)
                test_df[bl_col]  = test_df[bl_col].fillna(0.0)

            # Estimate labelling probability c from training data only
            # c = P(GDELT detects | event happened), used to correct predicted probs
            if USE_PU_LEARNING:
                c = estimate_labelling_probability(
                    panel[panel["date"].dt.year < test_year],
                    event_type, horizon,
                )
                logger.info(
                    "  Fold %d | c=%.3f (GDELT detects %.0f%% of real events in train)",
                    fold_id, c, c * 100,
                )
            else:
                c = 1.0

            logger.info(
                "  Fold %d | train <=%d (%d rows)  test %d (%d rows)",
                fold_id, test_year - 1, len(train_df), test_year, len(test_df),
            )

            # Logistic regression models
            for model_name, feature_list in MODEL_SPECS.items():
                preds_df, coefs_df, m = run_lr_fold(
                    train_df, test_df, feature_list,
                    event_type, horizon, fold_id, model_name,
                )
                if preds_df.empty:
                    continue
                preds_df = _apply_pu_correction(preds_df, c)
                m = _recompute_metrics(preds_df, m)
                all_preds.append(preds_df)
                all_coefs.append(coefs_df)
                all_metrics.append(m)
                logger.info(
                    "    %-30s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f",
                    model_name, m["roc_auc"], m["pr_auc"], m["brier"],
                )

            # XGBoost models
            if USE_XGBOOST and _XGBOOST_AVAILABLE:
                val_year = test_year - 1

                # M5: full feature set including GDELT lags
                tuned_p = _tune_xgb_params(train_df, FEATURES_XGB, val_year)
                px, ix, sx, mx = run_xgb_fold(
                    train_df, test_df, FEATURES_XGB,
                    event_type, horizon, fold_id, tuned_p,
                    model_name="model5_xgb", val_year=val_year,
                )
                if not px.empty:
                    px = _apply_pu_correction(px, c)
                    mx = _recompute_metrics(px, mx)
                    all_preds.append(px)
                    all_imps.append(ix)
                    if not sx.empty:
                        all_shaps.append(sx)
                    all_metrics.append(mx)
                    logger.info(
                        "    %-30s  ROC-AUC=%.3f  PR-AUC=%.3f  "
                        "Brier=%.4f (raw=%.4f, cal=%s)",
                        "model5_xgb", mx["roc_auc"], mx["pr_auc"], mx["brier"],
                        mx.get("brier_raw", mx["brier"]), mx.get("cal_method", "—"),
                    )

                # M6: no GDELT lag features (early-warning only)
                tuned_p_nl = _tune_xgb_params(train_df, FEATURES_XGB_NOLAG, val_year)
                px6, ix6, sx6, mx6 = run_xgb_fold(
                    train_df, test_df, FEATURES_XGB_NOLAG,
                    event_type, horizon, fold_id, tuned_p_nl,
                    model_name="model6_xgb_nolag", val_year=val_year,
                )
                if not px6.empty:
                    px6 = _apply_pu_correction(px6, c)
                    mx6 = _recompute_metrics(px6, mx6)
                    all_preds.append(px6)
                    all_imps.append(ix6)
                    if not sx6.empty:
                        all_shaps.append(sx6)
                    all_metrics.append(mx6)
                    logger.info(
                        "    %-30s  ROC-AUC=%.3f  PR-AUC=%.3f  "
                        "Brier=%.4f (raw=%.4f, cal=%s)",
                        "model6_xgb_nolag", mx6["roc_auc"], mx6["pr_auc"], mx6["brier"],
                        mx6.get("brier_raw", mx6["brier"]), mx6.get("cal_method", "—"),
                    )

                # M7: no country baselines (to see how much anchoring they cause)
                tuned_p_nb = _tune_xgb_params(train_df, FEATURES_XGB_NOBASELINE, val_year)
                px7, ix7, sx7, mx7 = run_xgb_fold(
                    train_df, test_df, FEATURES_XGB_NOBASELINE,
                    event_type, horizon, fold_id, tuned_p_nb,
                    model_name="model7_xgb_nobaseline", val_year=val_year,
                )
                if not px7.empty:
                    px7 = _apply_pu_correction(px7, c)
                    mx7 = _recompute_metrics(px7, mx7)
                    all_preds.append(px7)
                    all_imps.append(ix7)
                    if not sx7.empty:
                        all_shaps.append(sx7)
                    all_metrics.append(mx7)
                    logger.info(
                        "    %-30s  ROC-AUC=%.3f  PR-AUC=%.3f  "
                        "Brier=%.4f (raw=%.4f, cal=%s)",
                        "model7_xgb_nobaseline", mx7["roc_auc"], mx7["pr_auc"], mx7["brier"],
                        mx7.get("brier_raw", mx7["brier"]), mx7.get("cal_method", "—"),
                    )

                # Income-group stratified XGBoost
                for feat_list, mn in [
                    (FEATURES_XGB,       "model5_income_group"),
                    (FEATURES_XGB_NOLAG, "model6_income_group_nolag"),
                ]:
                    group_preds, group_imps = [], []
                    for group_name, group_countries in INCOME_GROUPS.items():
                        tr_g = train_df[train_df["country_iso3"].isin(group_countries)]
                        te_g = test_df[test_df["country_iso3"].isin(group_countries)]
                        if tr_g.empty or te_g.empty:
                            continue
                        tuned_g = _tune_xgb_params(tr_g, feat_list, val_year)
                        pg, ig, _, _ = run_xgb_fold(
                            tr_g, te_g, feat_list,
                            event_type, horizon, fold_id, tuned_g,
                            model_name=mn, val_year=val_year,
                        )
                        if not pg.empty:
                            group_preds.append(pg)
                            group_imps.append(ig)

                    if group_preds:
                        combined = pd.concat(group_preds, ignore_index=True)
                        combined = _apply_pu_correction(combined, c)
                        yt_all = combined["y_true"].values.astype(float)
                        yp_all = combined["y_pred"].values
                        mask   = ~(np.isnan(yt_all) | np.isnan(yp_all))
                        mg_all = compute_metrics(yt_all[mask], yp_all[mask])
                        mg_all.update({
                            "fold_id":    fold_id,
                            "event_type": event_type,
                            "horizon":    horizon,
                            "model_name": mn,
                            "n_train":    len(train_df),
                            "n_test":     len(combined),
                            "pos_rate":   float(yt_all[mask].mean()),
                        })
                        all_preds.append(combined)
                        if group_imps:
                            all_imps.append(pd.concat(group_imps, ignore_index=True))
                        all_metrics.append(mg_all)
                        logger.info(
                            "    %-30s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f",
                            mn, mg_all["roc_auc"], mg_all["pr_auc"], mg_all["brier"],
                        )

        if not all_preds:
            logger.warning("No predictions for target=%s", label)
            continue

        preds_all  = pd.concat(all_preds,  ignore_index=True)
        coefs_all  = pd.concat(all_coefs,  ignore_index=True) if all_coefs else pd.DataFrame()
        metrics_df = pd.DataFrame(all_metrics)

        logger.info("\n  Overall metrics (%s):", label)
        for mn, grp in preds_all.groupby("model_name"):
            ov = compute_metrics(grp["y_true"].values, grp["y_pred"].values)
            logger.info(
                "  %-30s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f",
                mn, ov["roc_auc"], ov["pr_auc"], ov["brier"],
            )

        # Calibration check on model4_fao (LR binned reliability)
        m4_preds = preds_all[preds_all["model_name"] == "model4_fao"]
        if not m4_preds.empty:
            cal = calibration_summary(m4_preds["y_true"].values, m4_preds["y_pred"].values)
            logger.info("\n  Calibration (model4_fao, %s):\n%s", label, cal.to_string(index=False))

        # XGBoost calibration before/after comparison table
        _cal_keys = ("model_name", "fold_id", "event_type", "horizon",
                     "roc_auc_raw", "pr_auc_raw", "brier_raw",
                     "roc_auc",     "pr_auc",     "brier",
                     "brier_val_isotonic", "brier_val_sigmoid", "cal_method")
        cal_rows = [
            {k: m_i[k] for k in _cal_keys if k in m_i}
            for m_i in all_metrics
            if "cal_method" in m_i
        ]
        if cal_rows:
            cal_df = pd.DataFrame(cal_rows)
            cal_df.to_csv(out_dir / "calibration_comparison.csv", index=False)
            logger.info("\n  Calibration comparison (XGBoost models):")
            for _, row in cal_df.iterrows():
                logger.info(
                    "    %-30s  fold=%d  Brier: raw=%.4f → cal=%.4f  method=%s",
                    row.get("model_name", ""), int(row.get("fold_id", 0)),
                    row.get("brier_raw", np.nan), row.get("brier", np.nan),
                    row.get("cal_method", "—"),
                )

        # Accumulate into global performance table
        for m_i in all_metrics:
            row = dict(m_i)
            row["target"]    = label
            row["test_year"] = int(row.get("fold_id", 1)) + FIRST_TEST_YEAR - 1
            all_global_metrics.append(row)

        # Save predictions, metrics, coefficients
        preds_all.to_parquet(out_dir / "preds.parquet", index=False)
        metrics_df.to_csv(out_dir / "metrics.csv", index=False)
        if not coefs_all.empty:
            coefs_all.to_csv(out_dir / "coefs_lr.csv", index=False)

        imps_df = pd.DataFrame()
        if all_imps:
            imps_df = pd.concat(all_imps, ignore_index=True)
            imps_df.to_csv(out_dir / "coefs_xgb.csv", index=False)

        if all_shaps:
            shap_all = pd.concat(all_shaps, ignore_index=True)
            shap_summary = (
                shap_all.groupby(["model_name", "feature"])["mean_abs_shap"]
                .mean()
                .reset_index()
                .sort_values(["model_name", "mean_abs_shap"], ascending=[True, False])
            )
            shap_summary.to_csv(out_dir / "shap_importance.csv", index=False)
            top_shap = (
                shap_summary[shap_summary["model_name"] == "model5_xgb"]
                .head(15)
            )
            if not top_shap.empty:
                logger.info("\n  Top 15 SHAP features (model5_xgb, %s):", label)
                for _, row in top_shap.iterrows():
                    logger.info("    %-45s  %.4f", row["feature"], row["mean_abs_shap"])

        # Per-country performance breakdown
        country_rows = []
        for country, grp in preds_all.groupby("country_iso3"):
            yt = grp["y_true"].values.astype(float)
            yp = grp["y_pred"].values
            mask = ~(np.isnan(yt) | np.isnan(yp))
            if mask.sum() < 10 or len(np.unique(yt[mask])) < 2:
                continue
            m_c = compute_metrics(yt[mask], yp[mask])
            m_c["country_iso3"] = country
            m_c["n_obs"]        = int(mask.sum())
            m_c["pos_rate"]     = float(yt[mask].mean())
            country_rows.append(m_c)
        if country_rows:
            country_df = (
                pd.DataFrame(country_rows)
                .sort_values("roc_auc", ascending=False)
                .reset_index(drop=True)
            )
            country_df.to_csv(out_dir / "country_metrics.csv", index=False)
            logger.info("\n  Per-country ROC-AUC (%s, all models combined):", label)
            for _, row in country_df.iterrows():
                logger.info(
                    "    %-6s  ROC-AUC=%.3f  PR-AUC=%.3f  n=%d  pos=%.0f%%",
                    row["country_iso3"], row["roc_auc"], row["pr_auc"],
                    int(row["n_obs"]), row["pos_rate"] * 100,
                )

        # Feature importance summary (XGBoost average across folds/models)
        if not imps_df.empty and "feature" in imps_df.columns:
            imp_summary = (
                imps_df.groupby("feature")["importance"]
                .mean()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )
            imp_summary.to_csv(out_dir / "feature_importance_summary.csv", index=False)
            logger.info("\n  Top 15 XGBoost features (%s):", label)
            for _, row in imp_summary.iterrows():
                logger.info("    %-45s  %.4f", row["feature"], row["importance"])

        # Reliability diagram (XGBoost raw vs calibrated)
        plot_calibration_figure(preds_all, label)

        logger.info("Saved results -> %s", out_dir)

    # Save unified model_performance.csv across all targets and folds
    if all_global_metrics:
        perf_df = pd.DataFrame(all_global_metrics)
        keep = [
            "target", "test_year", "model_name", "event_type", "horizon", "fold_id",
            "roc_auc", "pr_auc", "brier", "brier_skill_score",
            "pos_rate", "n_train", "n_test",
            "roc_auc_raw", "pr_auc_raw", "brier_raw", "cal_method",
        ]
        out_cols = [c for c in keep if c in perf_df.columns]
        perf_df[out_cols].to_csv(PROC_DIR / "model_performance.csv", index=False)
        logger.info("Saved model_performance.csv -> %s", PROC_DIR / "model_performance.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _setup_logging()

    panel = load_panel()
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    if USE_XGBOOST and not _XGBOOST_AVAILABLE:
        logger.warning("xgboost not installed — skipping XGBoost models.")

    run_backtest(panel)


if __name__ == "__main__":
    main()
