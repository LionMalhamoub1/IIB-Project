# Chile-only version of the v3 static backtest.

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
from sklearn.preprocessing import StandardScaler

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

_SRC   = Path(__file__).resolve().parent
_V4    = _SRC.parent
_V3_SRC = _SRC.parent.parent / "v3" / "src"
if str(_V3_SRC) not in sys.path:
    sys.path.insert(0, str(_V3_SRC))

from utils import compute_metrics, vif_filter

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

COUNTRY_FILTER: str = "CHL"

FIRST_TEST_YEAR: int = 2020
LAST_TEST_YEAR:  int = 2021

TARGETS: list[tuple[str, int]] = [
    ("protest", 7),
    ("strike",  7),
]

USE_PU_LEARNING:    bool = False
INCLUDE_COUNTRY_FE: bool = False   # only one country
USE_CLASS_WEIGHT:   bool = True
USE_XGBOOST:        bool = True
APPLY_VIF_FILTER:   bool = False   # constant features break VIF for single country
VIF_THRESHOLD:      float = 10.0

XGB_OPTUNA_TRIALS:         int = 20
XGB_EARLY_STOPPING_ROUNDS: int = 20
_MIN_CAL_OBS:              int = 20   # lower threshold — smaller test set

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PANEL_FILE = _V4.parent / "v3" / "data" / "interim" / "modelling_panel_gdelt.parquet"
PROC_DIR   = _V4 / "data" / "processed"

# ---------------------------------------------------------------------------
# Feature sets  (identical to v3)
# ---------------------------------------------------------------------------

FEATURES_M0: list[str] = [
    "gdelt_protest_7d_lag", "gdelt_protest_28d_lag",
    "gdelt_strike_7d_lag",  "gdelt_strike_28d_lag",
    "gdelt_protest_region_14d", "gdelt_strike_region_14d",
]

FEATURES_M1_ADD: list[str] = [
    "fx_pct_7d", "fx_pct_30d", "fx_pct_90d",
    "fx_vol_7d", "fx_vol_30d",
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
    "gini_coef", "covid_period",
    "fx_trend_consistent", "inflation_accel",
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
    "gta_30d_count", "gta_30d_count_z",
    "gta_90d_count", "gta_90d_count_z",
]

MODEL_SPECS: dict[str, list[str]] = {
    "model0_persistence": FEATURES_M0,
    "model1_markets":     FEATURES_M0 + FEATURES_M1_ADD,
    "model2_full":        FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD,
    "model3_structural":  FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD,
    "model4_fao":         FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD + FEATURES_M4_ADD,
    "model_lr_nolag":     FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD + FEATURES_M4_ADD,
}

FEATURES_XGB: list[str] = (
    FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD + FEATURES_M4_ADD
)
FEATURES_XGB_NOLAG: list[str] = (
    FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M3_ADD + FEATURES_M4_ADD
)

XGB_PARAM_GRID: list[dict[str, Any]] = [
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05,
     "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
     "reg_lambda": 1.0, "reg_alpha": 0.0},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
     "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3,
     "reg_lambda": 1.0, "reg_alpha": 0.0},
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
        raise FileNotFoundError(f"Panel not found: {PANEL_FILE}")
    df = pd.read_parquet(PANEL_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["country_iso3"] == COUNTRY_FILTER].copy()
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(
        "Chile panel: %d rows x %d cols | %s to %s",
        len(df), len(df.columns),
        str(df["date"].min().date()), str(df["date"].max().date()),
    )
    return df


def available_features(wanted: list[str], cols: list[str]) -> list[str]:
    return [f for f in wanted if f in cols]


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_lr_pipeline(feature_cols: list[str], balanced: bool) -> Pipeline:
    cw = "balanced" if balanced else None
    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), feature_cols)],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(penalty="l2", solver="lbfgs",
                                     max_iter=2000, class_weight=cw)),
    ])


def build_xgb_pipeline(feature_cols: list[str], pos_neg_ratio: float,
                        params: dict[str, Any]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[("num", SimpleImputer(strategy="median"), feature_cols)],
        remainder="drop",
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            scale_pos_weight=pos_neg_ratio, eval_metric="logloss",
            random_state=42, n_jobs=-1, **params,
        )),
    ])


def extract_coefs(pipe: Pipeline) -> pd.DataFrame:
    try:
        names = list(pipe.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        n = len(pipe.named_steps["model"].coef_[0])
        names = [f"feat_{i}" for i in range(n)]
    return pd.DataFrame({"feature": names,
                         "coefficient": pipe.named_steps["model"].coef_[0]})


# ---------------------------------------------------------------------------
# XGBoost tuning
# ---------------------------------------------------------------------------

def _tune_xgb_params(train_df: pd.DataFrame, feature_cols: list[str],
                     val_year: int) -> dict[str, Any]:
    val_mask  = train_df["date"].dt.year == val_year
    sub_train = train_df[~val_mask].dropna(subset=["y"])
    sub_val   = train_df[val_mask].dropna(subset=["y"])

    if sub_train.empty or sub_val.empty or len(sub_val["y"].unique()) < 2:
        logger.warning("Cannot tune XGB — using default params.")
        return XGB_PARAM_GRID[0]

    feat  = available_features(feature_cols, list(sub_train.columns))
    imp   = SimpleImputer(strategy="median")
    X_tr  = imp.fit_transform(sub_train[feat])
    y_tr  = sub_train["y"].astype(int).values
    X_val = imp.transform(sub_val[feat])
    y_val = sub_val["y"].astype(int).values
    ratio = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)

    if _OPTUNA_AVAILABLE:
        def _objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth":        trial.suggest_int("max_depth", 2, 5),
                "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "reg_lambda":       trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
                "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
            }
            clf = XGBClassifier(
                n_estimators=500, scale_pos_weight=ratio,
                eval_metric="aucpr",
                early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                random_state=42, n_jobs=-1, verbosity=0, **params,
            )
            clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return compute_metrics(y_val, clf.predict_proba(X_val)[:, 1])["pr_auc"]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(_objective, n_trials=XGB_OPTUNA_TRIALS, show_progress_bar=False)
        best = dict(study.best_params)
        best_clf = XGBClassifier(
            n_estimators=500, scale_pos_weight=ratio,
            eval_metric="aucpr",
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
            random_state=42, n_jobs=-1, verbosity=0, **best,
        )
        best_clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        best["n_estimators"] = best_clf.best_iteration + 1
        logger.info("Optuna XGB (%d trials): PR-AUC=%.4f  n_est=%d",
                    XGB_OPTUNA_TRIALS, study.best_value, best["n_estimators"])
        return best

    best_score, best_params = -np.inf, XGB_PARAM_GRID[0]
    for params in XGB_PARAM_GRID:
        try:
            imp2 = SimpleImputer(strategy="median")
            X_tr2 = imp2.fit_transform(sub_train[feat])
            clf = XGBClassifier(scale_pos_weight=ratio, random_state=42, **params)
            clf.fit(X_tr2, y_tr)
            sc = compute_metrics(y_val, clf.predict_proba(imp2.transform(sub_val[feat]))[:, 1])["pr_auc"]
            if sc > best_score:
                best_score, best_params = sc, params
        except Exception:
            pass
    return best_params


# ---------------------------------------------------------------------------
# Fold runners
# ---------------------------------------------------------------------------

def run_lr_fold(train_df, test_df, feature_cols, event_type, horizon,
                fold_id, model_name):
    feat = available_features(feature_cols, list(train_df.columns))
    if not feat:
        return pd.DataFrame(), pd.DataFrame(), {}
    train_clean = train_df.dropna(subset=["y"])
    test_clean  = test_df.dropna(subset=["y"])
    if train_clean.empty or test_clean.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    pipe   = build_lr_pipeline(feat, USE_CLASS_WEIGHT)
    pipe.fit(train_clean[feat], train_clean["y"].astype(int).values)
    y_true = test_clean["y"].astype(int).values
    y_prob = pipe.predict_proba(test_clean[feat])[:, 1]

    m = compute_metrics(y_true, y_prob)
    m.update({"fold_id": fold_id, "event_type": event_type, "horizon": horizon,
              "model_name": model_name, "n_train": len(train_clean),
              "n_test": len(test_clean), "pos_rate": float(y_true.mean())})

    preds_df = pd.DataFrame({
        "country_iso3": test_clean["country_iso3"].values,
        "date":  test_clean["date"].values,
        "y_true": y_true, "y_pred": y_prob,
        "fold_id": fold_id, "event_type": event_type,
        "horizon": horizon, "model_name": model_name,
    })
    coefs_df = extract_coefs(pipe).assign(
        fold_id=fold_id, event_type=event_type,
        horizon=horizon, model_name=model_name,
    )
    return preds_df, coefs_df, m


def run_xgb_fold(train_df, test_df, feature_cols, event_type, horizon,
                 fold_id, tuned_params, model_name="model5_xgb", val_year=None):
    feat = available_features(feature_cols, list(train_df.columns))
    if not feat:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    if val_year is not None:
        val_mask  = train_df["date"].dt.year == val_year
        train_sub = train_df[~val_mask]
        val_sub   = train_df[val_mask]
    else:
        train_sub, val_sub = train_df, pd.DataFrame()

    train_clean = train_sub.dropna(subset=["y"])
    test_clean  = test_df.dropna(subset=["y"])
    if train_clean.empty or test_clean.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    y_train      = train_clean["y"].astype(int).values
    y_true       = test_clean["y"].astype(int).values
    pos_neg_ratio = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)

    pipe        = build_xgb_pipeline(feat, pos_neg_ratio, tuned_params)
    pipe.fit(train_clean[feat], y_train)
    y_prob_raw  = pipe.predict_proba(test_clean[feat])[:, 1]
    raw_metrics = compute_metrics(y_true, y_prob_raw)

    y_prob, best_method = y_prob_raw, "none"
    val_briers = {"isotonic": np.nan, "sigmoid": np.nan}
    val_clean  = val_sub.dropna(subset=["y"]) if not val_sub.empty else pd.DataFrame()

    if (not val_clean.empty and len(val_clean) >= _MIN_CAL_OBS
            and val_clean["y"].nunique() >= 2):
        xgb_model    = pipe.named_steps["model"]
        preprocessor = pipe.named_steps["preprocessor"]
        X_val_proc   = preprocessor.transform(val_clean[feat])

        best_brier, best_cal = np.inf, None
        for method in ("isotonic", "sigmoid"):
            try:
                cal = CalibratedClassifierCV(estimator=xgb_model,
                                             cv="prefit", method=method)
                cal.fit(X_val_proc, val_clean["y"].astype(int).values)
                yp = cal.predict_proba(X_val_proc)[:, 1]
                from sklearn.metrics import brier_score_loss
                b = brier_score_loss(val_clean["y"].astype(int).values, yp)
                val_briers[method] = float(b)
                if b < best_brier:
                    best_brier, best_cal, best_method = b, cal, method
            except Exception:
                pass
        if best_cal is not None:
            X_test_proc = preprocessor.transform(test_clean[feat])
            y_prob = best_cal.predict_proba(X_test_proc)[:, 1]

    m = compute_metrics(y_true, y_prob)
    m.update({"fold_id": fold_id, "event_type": event_type, "horizon": horizon,
              "model_name": model_name, "n_train": len(train_clean),
              "n_test": len(test_clean), "pos_rate": float(y_true.mean()),
              "roc_auc_raw": raw_metrics["roc_auc"],
              "pr_auc_raw":  raw_metrics["pr_auc"],
              "brier_raw":   raw_metrics["brier"],
              "brier_val_isotonic": val_briers["isotonic"],
              "brier_val_sigmoid":  val_briers["sigmoid"],
              "cal_method":  best_method})

    preds_df = pd.DataFrame({
        "country_iso3": test_clean["country_iso3"].values,
        "date":      test_clean["date"].values,
        "y_true":    y_true, "y_pred": y_prob, "y_pred_raw": y_prob_raw,
        "fold_id":   fold_id, "event_type": event_type,
        "horizon":   horizon, "model_name": model_name,
    })

    try:
        feat_names = list(pipe.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        feat_names = [f"feat_{i}" for i in
                      range(len(pipe.named_steps["model"].feature_importances_))]

    imps_df = pd.DataFrame({
        "feature":    feat_names,
        "importance": pipe.named_steps["model"].feature_importances_,
        "fold_id": fold_id, "event_type": event_type,
        "horizon": horizon, "model_name": model_name,
    })

    shap_df = pd.DataFrame()
    if _SHAP_AVAILABLE:
        try:
            explainer    = shap.TreeExplainer(pipe.named_steps["model"])
            X_test_proc  = pipe.named_steps["preprocessor"].transform(test_clean[feat])
            shap_vals    = explainer.shap_values(X_test_proc)
            shap_df = pd.DataFrame({
                "feature":       feat_names,
                "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
                "mean_shap":     shap_vals.mean(axis=0),
                "fold_id": fold_id, "event_type": event_type,
                "horizon": horizon, "model_name": model_name,
            })
        except Exception as exc:
            logger.warning("SHAP failed for %s fold %d: %s", model_name, fold_id, exc)

    return preds_df, imps_df, shap_df, m


# ---------------------------------------------------------------------------
# Country baselines
# ---------------------------------------------------------------------------

def compute_country_baselines(train_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["protest_today", "strike_today"] if c in train_df.columns]
    if not cols:
        return pd.DataFrame(columns=["country_iso3",
                                     "country_protest_baseline",
                                     "country_strike_baseline"])
    baseline = (train_df.groupby("country_iso3")[cols]
                .mean()
                .rename(columns={"protest_today": "country_protest_baseline",
                                  "strike_today":  "country_strike_baseline"})
                .reset_index())
    for col in ("country_protest_baseline", "country_strike_baseline"):
        if col not in baseline.columns:
            baseline[col] = 0.0
    return baseline


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def run_backtest() -> None:
    _setup_logging()
    logger.info("=" * 60)
    logger.info("CHILE backtest  (country=%s)", COUNTRY_FILTER)
    logger.info("=" * 60)

    df = load_panel()

    for event_type, horizon in TARGETS:
        label     = f"{event_type}_{horizon}d"
        out_dir   = PROC_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n%s\nTARGET: %s\n%s", "=" * 60, label, "=" * 60)

        # Build target column
        target_col = f"{event_type}_today"
        if target_col not in df.columns:
            logger.error("Column %s not found — skipping.", target_col)
            continue

        df["y"] = (
            df.groupby("country_iso3")[target_col]
            .transform(lambda s: s.shift(-horizon).rolling(horizon, min_periods=1).max())
        )

        all_preds:   list[pd.DataFrame] = []
        all_coefs:   list[pd.DataFrame] = []
        all_metrics: list[dict]         = []
        all_imps:    list[pd.DataFrame] = []
        all_shaps:   list[pd.DataFrame] = []

        for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
            train_df = df[df["date"].dt.year < test_year].copy()
            test_df  = df[df["date"].dt.year == test_year].copy()
            fold_id  = test_year - FIRST_TEST_YEAR + 1
            val_year = test_year - 1

            logger.info("  Fold %d | train <=  %d (%d rows)  test %d (%d rows)",
                        fold_id, test_year - 1, len(train_df.dropna(subset=["y"])),
                        test_year, len(test_df.dropna(subset=["y"])))

            # Attach country baselines
            baselines = compute_country_baselines(train_df)
            for tmp_df in (train_df, test_df):
                for col in ("country_protest_baseline", "country_strike_baseline"):
                    if col in baselines.columns:
                        val = baselines.loc[baselines["country_iso3"] == COUNTRY_FILTER, col]
                        tmp_df[col] = float(val.values[0]) if len(val) else 0.0

            # --- LR models ---
            for model_name, feat_list in MODEL_SPECS.items():
                px, cx, mx = run_lr_fold(
                    train_df, test_df, feat_list,
                    event_type, horizon, fold_id, model_name,
                )
                if not px.empty:
                    all_preds.append(px)
                    all_coefs.append(cx)
                    all_metrics.append(mx)
                    logger.info("    %-30s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.4f",
                                model_name, mx["roc_auc"], mx["pr_auc"], mx["brier"])

            if not USE_XGBOOST or not _XGBOOST_AVAILABLE:
                continue

            # Tune XGB once per fold
            tuned_p = _tune_xgb_params(train_df, FEATURES_XGB, val_year)

            # M6 — XGB Full
            px6, ix6, sx6, mx6 = run_xgb_fold(
                train_df, test_df, FEATURES_XGB,
                event_type, horizon, fold_id, tuned_p,
                model_name="model5_xgb", val_year=val_year,
            )
            if not px6.empty:
                all_preds.append(px6); all_imps.append(ix6)
                if not sx6.empty: all_shaps.append(sx6)
                all_metrics.append(mx6)
                logger.info("    %-30s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.4f (cal=%s)",
                            "model5_xgb", mx6["roc_auc"], mx6["pr_auc"], mx6["brier"],
                            mx6.get("cal_method", "—"))

            # M7 — XGB No Lags (re-tune)
            tuned_p_nl = _tune_xgb_params(train_df, FEATURES_XGB_NOLAG, val_year)
            px7, ix7, sx7, mx7 = run_xgb_fold(
                train_df, test_df, FEATURES_XGB_NOLAG,
                event_type, horizon, fold_id, tuned_p_nl,
                model_name="model6_xgb_nolag", val_year=val_year,
            )
            if not px7.empty:
                all_preds.append(px7); all_imps.append(ix7)
                if not sx7.empty: all_shaps.append(sx7)
                all_metrics.append(mx7)
                logger.info("    %-30s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.4f (cal=%s)",
                            "model6_xgb_nolag", mx7["roc_auc"], mx7["pr_auc"], mx7["brier"],
                            mx7.get("cal_method", "—"))

        # --- Save outputs ---
        if not all_preds:
            logger.warning("No predictions generated for %s.", label)
            continue

        preds_df = pd.concat(all_preds, ignore_index=True)
        preds_df.to_parquet(out_dir / "preds.parquet", index=False)

        metrics_df = pd.DataFrame(all_metrics)
        metrics_df.to_csv(out_dir / "metrics.csv", index=False)

        if all_coefs:
            pd.concat(all_coefs, ignore_index=True).to_csv(
                out_dir / "coefs_lr.csv", index=False)

        if all_imps:
            pd.concat(all_imps, ignore_index=True).to_csv(
                out_dir / "coefs_xgb.csv", index=False)

        # Per-country metrics (just Chile)
        country_rows = []
        for (mn, fi), grp in preds_df.groupby(["model_name", "fold_id"]):
            sub = grp.dropna(subset=["y_true", "y_pred"])
            if len(sub) < 5 or sub["y_true"].nunique() < 2:
                continue
            cm = compute_metrics(sub["y_true"].astype(int).values, sub["y_pred"].values)
            cm.update({"country_iso3": COUNTRY_FILTER, "model_name": mn,
                       "fold_id": fi, "n": len(sub),
                       "pos_rate": float(sub["y_true"].mean())})
            country_rows.append(cm)
        if country_rows:
            pd.DataFrame(country_rows).to_csv(out_dir / "country_metrics.csv", index=False)

        # Calibration comparison
        cal_rows = []
        for row in all_metrics:
            if "brier_raw" in row and pd.notna(row.get("brier_raw")):
                cal_rows.append({
                    "model_name": row["model_name"], "fold_id": row["fold_id"],
                    "brier_raw": row["brier_raw"], "brier_calibrated": row["brier"],
                    "cal_method": row.get("cal_method", "none"),
                    "brier_val_isotonic": row.get("brier_val_isotonic"),
                    "brier_val_sigmoid":  row.get("brier_val_sigmoid"),
                })
        if cal_rows:
            pd.DataFrame(cal_rows).to_csv(out_dir / "calibration_comparison.csv", index=False)

        # SHAP summary
        if all_shaps:
            shap_all = pd.concat(all_shaps, ignore_index=True)
            shap_summary = (
                shap_all.groupby(["model_name", "feature"])[["mean_abs_shap", "mean_shap"]]
                .mean().reset_index()
                .sort_values(["model_name", "mean_abs_shap"], ascending=[True, False])
            )
            shap_summary.to_csv(out_dir / "shap_importance.csv", index=False)

        logger.info("Saved outputs to %s", out_dir)

    logger.info("\nChile backtest complete.")


if __name__ == "__main__":
    run_backtest()
