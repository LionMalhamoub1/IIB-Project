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

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sklearn.impute import SimpleImputer as _SimpleImputer

from utils import (  # noqa: E402
    calibration_summary,
    compute_metrics,
    make_target,
    make_target_elevated,
)


HORIZONS: list[int]      = [7, 30]
FIRST_TEST_YEAR: int     = 2020
LAST_TEST_YEAR: int      = 2024
INCLUDE_COUNTRY_FE: bool = False
USE_CLASS_WEIGHT: bool   = True
USE_XGBOOST: bool        = True

XGB_OPTUNA_TRIALS: int = 40
XGB_EARLY_STOPPING_ROUNDS: int = 20

ELEVATED_MULTIPLIER: float = 1.5
ELEVATED_MIN_EVENTS: int   = 3
ELEVATED_BASELINE_WINDOW: int = 90

INCOME_GROUPS: dict[str, set[str]] = {
    "high": {
        "AUS", "CAN", "CHL", "CZE", "FRA", "DEU", "GRC", "HUN", "IRL", "ITA",
        "JPN", "KOR", "NLD", "NOR", "POL", "PRT", "ESP", "SWE", "GBR", "USA",
    },
    "upper_middle": {
        "ARG", "BRA", "CHN", "MYS", "MEX", "NAM", "PER", "ZAF", "THA", "TUR",
    },
    "lower_middle": {
        "BOL", "COD", "IND", "IDN", "KEN", "LAO", "MAR", "MOZ", "PHL", "VNM", "ZWE",
    },
}


MOD_ROOT   = _SRC.parent
PANEL_FILE = MOD_ROOT / "data" / "interim"   / "modelling_panel.parquet"
PROC_DIR   = MOD_ROOT / "data" / "processed"


FEATURES_M0: list[str] = [
    "acled_7d_lag",
    "acled_28d_lag",
    "riot_7d_lag",
    "riot_28d_lag",
    "violence_7d_lag",
    "violence_28d_lag",
    "protest_fat_7d_lag",
    "protest_fat_28d_lag",
]

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
]

FEATURES_M2_ADD: list[str] = [
    # WDI — annual (z-score + absolute)
    "gdp_growth",              "gdp_growth_z",
    "gdp_per_capita_growth",   "gdp_per_capita_growth_z",
    "inflation_cpi_yoy",       "inflation_cpi_yoy_z",
    "unemployment_total",      "unemployment_total_z",
    "unemployment_youth",      "unemployment_youth_z",
    # ILOSTAT — monthly (z-score + absolute)
    "unemployment_sa",         "unemployment_sa_z",
    "unemployment_rate",       "unemployment_rate_z",
    "earnings_monthly",        "earnings_monthly_z",
    # WGI — annual (z-score + absolute)
    "political_stability_est",        "political_stability_est_z",
    "voice_accountability_est",       "voice_accountability_est_z",
    "government_effectiveness_est",   "government_effectiveness_est_z",
    "rule_of_law_est",                "rule_of_law_est_z",
    # Interactions
    "fx_pct_30d_x_instability",
    "oil_brent_pct_30d_x_inflation",
    # Google Trends (z-score + absolute)
    "economic_stress_index",   "economic_stress_index_z",
    "labour_conflict_index",   "labour_conflict_index_z",
    "protest_mobilisation_index", "protest_mobilisation_index_z",
    # CPI (z-score + absolute)
    "food_cpi_inflation",      "food_cpi_inflation_z",
    "energy_cpi_inflation",    "energy_cpi_inflation_z",
]

FEATURES_M4_ADD: list[str] = [
    # Temporal
    "month_sin",
    "month_cos",
    # FAO raw YoY
    "fao_food_index_yoy",
    "fao_cereals_index_yoy",
    "fao_oils_index_yoy",
    # FAO threshold indicators
    "fao_food_index_yoy_above90",
    "fao_cereals_index_yoy_above90",
    # FAO lagged YoY (literature-motivated lead times)
    "fao_cereals_index_yoy_lag1m",
    "fao_cereals_index_yoy_lag3m",
    "fao_cereals_index_yoy_lag6m",
    "fao_food_index_yoy_lag1m",
    "fao_food_index_yoy_lag3m",
    # Literature-motivated interactions
    "fao_cereals_yoy_x_instability",
    "fao_food_yoy_x_youth_unemp",
    # GTA trade interventions (z-score + absolute)
    "gta_harmful_events",      "gta_harmful_events_z",
    "gta_liberalising_events", "gta_liberalising_events_z",
    "gta_30d_count",           "gta_30d_count_z",
    "gta_90d_count",           "gta_90d_count_z",
]

MODEL_SPECS: dict[str, list[str]] = {
    "model0_persistence": FEATURES_M0,
    "model1_markets":     FEATURES_M0 + FEATURES_M1_ADD,
    "model2_full":        FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD,
    "model4_fao":         FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M4_ADD,
}

FEATURES_XGB: list[str] = (
    FEATURES_M0 + FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M4_ADD
)

# M6: same feature set but without ACLED lag columns (early-warning ablation)
FEATURES_XGB_NOLAG: list[str] = (
    FEATURES_M1_ADD + FEATURES_M2_ADD + FEATURES_M4_ADD
)

# XGBoost parameter grid — best config selected on last year of each training fold
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


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)


def load_panel() -> pd.DataFrame:
    if not PANEL_FILE.exists():
        raise FileNotFoundError(
            f"Modelling panel not found: {PANEL_FILE}\n"
            "Run build_panel_country_day.py first."
        )
    df = pd.read_parquet(PANEL_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country_iso3", "date"]).reset_index(drop=True)
    logger.info(
        "Panel loaded: %d rows x %d cols | %d countries | %s to %s.",
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
        names: list[str] = list(
            pipe.named_steps["preprocessor"].get_feature_names_out()
        )
    except Exception:
        n     = len(pipe.named_steps["model"].coef_[0])
        names = [f"feat_{i}" for i in range(n)]

    coefs = pipe.named_steps["model"].coef_[0]
    return pd.DataFrame({"feature": names, "coefficient": coefs})


def _tune_xgb_params(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    val_year: int,
) -> dict[str, Any]:
    val_mask  = train_df["date"].dt.year == val_year
    sub_train = train_df[~val_mask].dropna(subset=["y"])
    sub_val   = train_df[val_mask].dropna(subset=["y"])

    if sub_train.empty or sub_val.empty or len(sub_val["y"].unique()) < 2:
        logger.warning("Cannot tune XGB: insufficient validation data — using default params.")
        return XGB_PARAM_GRID[0]

    feat  = available_features(feature_cols, list(sub_train.columns))
    imp   = _SimpleImputer(strategy="median")
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

        # Re-fit with best params to retrieve the optimal n_estimators from early stopping
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
            "Optuna XGB (%d trials): best PR-AUC=%.4f  n_est=%d  params=%s",
            XGB_OPTUNA_TRIALS, study.best_value, best["n_estimators"],
            {k: v for k, v in best.items() if k != "n_estimators"},
        )
        return best

    # Fallback: grid search over XGB_PARAM_GRID
    logger.warning("optuna not installed — using grid search for XGB tuning.")
    best_score  = -np.inf
    best_params = XGB_PARAM_GRID[0]
    for params in XGB_PARAM_GRID:
        try:
            pipe = build_xgb_pipeline(feat, False, ratio, params)
            pipe.fit(sub_train[feat], y_tr)
            m = compute_metrics(y_val, pipe.predict_proba(sub_val[feat])[:, 1])
            if m["pr_auc"] > best_score:
                best_score  = m["pr_auc"]
                best_params = params
        except Exception as exc:
            logger.warning("XGB param config failed: %s", exc)
    logger.info("Grid XGB tuning: best PR-AUC=%.4f", best_score)
    return best_params


def run_lr_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    fold_id: int,
    model_name: str,
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
        "horizon":      horizon,
        "model_name":   model_name,
    })

    coefs_df = extract_coefs(pipe, feat).assign(
        fold_id=fold_id, horizon=horizon, model_name=model_name,
    )

    return preds_df, coefs_df, m


def run_xgb_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    fold_id: int,
    tuned_params: dict[str, Any],
    model_name: str = "model5_xgboost_tuned",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feat = available_features(feature_cols, list(train_df.columns))
    if not feat:
        logger.warning("No valid features for XGBoost fold=%d.", fold_id)
        return pd.DataFrame(), pd.DataFrame(), {}

    train_clean = train_df.dropna(subset=["y"])
    test_clean  = test_df.dropna(subset=["y"])

    if train_clean.empty or test_clean.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    X_train = train_clean[feat + (["country_iso3"] if INCLUDE_COUNTRY_FE else [])]
    y_train = train_clean["y"].astype(int).values
    X_test  = test_clean[feat  + (["country_iso3"] if INCLUDE_COUNTRY_FE else [])]
    y_true  = test_clean["y"].astype(int).values

    n_neg         = int((y_train == 0).sum())
    n_pos         = int((y_train == 1).sum())
    pos_neg_ratio = n_neg / max(n_pos, 1)

    pipe   = build_xgb_pipeline(feat, INCLUDE_COUNTRY_FE, pos_neg_ratio, tuned_params)
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    m = compute_metrics(y_true, y_prob)
    m.update({
        "fold_id":    fold_id,
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
        "horizon":    horizon,
        "model_name": model_name,
    })

    return preds_df, imps_df, m


def run_backtest(panel: pd.DataFrame, target_col: str, label: str) -> None:
    logger.info("=" * 60)
    logger.info("Backtest: target=%s", label)
    logger.info("Horizons     : %s days", HORIZONS)
    logger.info("Test years   : %d – %d", FIRST_TEST_YEAR, LAST_TEST_YEAR)
    logger.info("=" * 60)

    out_dir = PROC_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    for horizon in HORIZONS:
        logger.info("-" * 60)
        logger.info("H = %d days", horizon)

        if target_col == "original":
            panel["y"] = make_target(panel, horizon)
        else:
            panel["y"] = make_target_elevated(
                panel, horizon,
                multiplier=ELEVATED_MULTIPLIER,
                min_events=ELEVATED_MIN_EVENTS,
                baseline_window=ELEVATED_BASELINE_WINDOW,
            )

        pos_rate_overall = panel["y"].mean()
        logger.info("Target positive rate: %.1f%%", 100 * pos_rate_overall)

        all_preds: list[pd.DataFrame]  = []
        all_coefs: list[pd.DataFrame]  = []
        all_metrics: list[dict]        = []
        all_imps: list[pd.DataFrame]   = []

        for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
            train_df = panel[panel["date"].dt.year <  test_year].copy()
            test_df  = panel[panel["date"].dt.year == test_year].copy()
            fold_id  = test_year - FIRST_TEST_YEAR + 1

            logger.info(
                "  Fold %d | train <=%d (%d rows)  test %d (%d rows)",
                fold_id, test_year - 1, len(train_df), test_year, len(test_df),
            )

            for model_name, feature_list in MODEL_SPECS.items():
                preds_df, coefs_df, m = run_lr_fold(
                    train_df, test_df, feature_list, horizon, fold_id, model_name,
                )
                if preds_df.empty:
                    continue
                all_preds.append(preds_df)
                all_coefs.append(coefs_df)
                all_metrics.append(m)
                logger.info(
                    "    %-28s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f",
                    model_name, m["roc_auc"], m["pr_auc"], m["brier"],
                )

            if USE_XGBOOST and _XGBOOST_AVAILABLE:
                val_year = test_year - 1

                # M5: full feature set including ACLED lags
                tuned_p    = _tune_xgb_params(train_df, FEATURES_XGB, val_year)
                px, ix, mx = run_xgb_fold(
                    train_df, test_df, FEATURES_XGB, horizon, fold_id, tuned_p,
                    model_name="model5_xgboost_tuned",
                )
                if not px.empty:
                    all_preds.append(px)
                    all_imps.append(ix)
                    all_metrics.append(mx)
                    logger.info(
                        "    %-28s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f",
                        "model5_xgboost_tuned", mx["roc_auc"], mx["pr_auc"], mx["brier"],
                    )

                # M6: no ACLED lag features (early-warning ablation)
                tuned_p_nl    = _tune_xgb_params(train_df, FEATURES_XGB_NOLAG, val_year)
                px6, ix6, mx6 = run_xgb_fold(
                    train_df, test_df, FEATURES_XGB_NOLAG, horizon, fold_id, tuned_p_nl,
                    model_name="model6_xgb_nolag",
                )
                if not px6.empty:
                    all_preds.append(px6)
                    all_imps.append(ix6)
                    all_metrics.append(mx6)
                    logger.info(
                        "    %-28s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f",
                        "model6_xgb_nolag", mx6["roc_auc"], mx6["pr_auc"], mx6["brier"],
                    )

                # M5 / M6 by income group
                for feature_cols, model_name in [
                    (FEATURES_XGB,       "model5_income_group"),
                    (FEATURES_XGB_NOLAG, "model6_income_group_nolag"),
                ]:
                    group_preds: list[pd.DataFrame] = []
                    group_imps:  list[pd.DataFrame] = []

                    for group_name, group_countries in INCOME_GROUPS.items():
                        tr_g = train_df[train_df["country_iso3"].isin(group_countries)]
                        te_g = test_df[test_df["country_iso3"].isin(group_countries)]
                        if tr_g.empty or te_g.empty:
                            continue
                        tuned_g = _tune_xgb_params(tr_g, feature_cols, val_year)
                        pg, ig, mg = run_xgb_fold(
                            tr_g, te_g, feature_cols, horizon, fold_id, tuned_g,
                            model_name=model_name,
                        )
                        if not pg.empty:
                            group_preds.append(pg)
                            group_imps.append(ig)

                    if group_preds:
                        combined_preds = pd.concat(group_preds, ignore_index=True)
                        combined_imps  = pd.concat(group_imps,  ignore_index=True)
                        all_preds.append(combined_preds)
                        all_imps.append(combined_imps)
                        yt_all = combined_preds["y_true"].values.astype(float)
                        yp_all = combined_preds["y_pred"].values
                        mask   = ~(np.isnan(yt_all) | np.isnan(yp_all))
                        mg_all = compute_metrics(yt_all[mask], yp_all[mask])
                        mg_all.update({
                            "fold_id":    fold_id,
                            "horizon":    horizon,
                            "model_name": model_name,
                            "n_train":    sum(len(tr_g) for tr_g in [
                                train_df[train_df["country_iso3"].isin(c)]
                                for c in INCOME_GROUPS.values()
                            ]),
                            "n_test":     len(combined_preds),
                            "pos_rate":   float(yt_all[mask].mean()),
                        })
                        all_metrics.append(mg_all)
                        logger.info(
                            "    %-28s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f",
                            model_name, mg_all["roc_auc"], mg_all["pr_auc"], mg_all["brier"],
                        )

        if not all_preds:
            logger.warning("No predictions for H=%d target=%s.", horizon, label)
            continue

        preds_all  = pd.concat(all_preds,  ignore_index=True)
        coefs_all  = pd.concat(all_coefs,  ignore_index=True) if all_coefs else pd.DataFrame()
        metrics_df = pd.DataFrame(all_metrics)

        logger.info("\n  Overall metrics (all folds, H=%d, target=%s):", horizon, label)
        for mn, grp in preds_all.groupby("model_name"):
            ov = compute_metrics(grp["y_true"].values, grp["y_pred"].values)
            logger.info(
                "  %-28s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f",
                mn, ov["roc_auc"], ov["pr_auc"], ov["brier"],
            )

        m4_preds = preds_all[preds_all["model_name"] == "model4_fao"]
        if not m4_preds.empty:
            cal = calibration_summary(m4_preds["y_true"].values, m4_preds["y_pred"].values)
            logger.info(
                "\n  Calibration (model4_fao, H=%d, target=%s):\n%s",
                horizon, label, cal.to_string(index=False),
            )

        preds_all.to_parquet(out_dir / f"preds_h{horizon}.parquet", index=False)
        metrics_df.to_csv(out_dir / f"metrics_h{horizon}.csv", index=False)
        if not coefs_all.empty:
            coefs_all.to_csv(out_dir / f"coefs_h{horizon}.csv", index=False)
        if all_imps:
            pd.concat(all_imps, ignore_index=True).to_csv(
                out_dir / f"coefs_xgb_h{horizon}.csv", index=False
            )

        logger.info("Saved results -> %s", out_dir)


def main() -> None:
    _setup_logging()

    panel = load_panel()
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    if not _XGBOOST_AVAILABLE and USE_XGBOOST:
        logger.warning("xgboost not installed — skipping XGBoost models.")

    run_backtest(panel.copy(), "original",  "target_original")
    run_backtest(panel.copy(), "elevated",  "target_elevated")


if __name__ == "__main__":
    main()
