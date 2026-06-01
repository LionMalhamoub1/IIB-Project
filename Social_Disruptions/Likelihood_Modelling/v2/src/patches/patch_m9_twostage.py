# Adds model9_twostage to static backtest outputs.
# Stage 1: structural risk (M2+M3+M4). Stage 2: fast triggers (M0+M1) + risk score.
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Import from existing scripts
# ---------------------------------------------------------------------------

def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / f"{name}.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

log.info("Loading train_backtest...")
tb = _load("train_backtest")

from utils import compute_metrics

MODEL_NAME = "model9_twostage"

# Stage 1: pure structural environment (no GDELT, no FX/VIX)
SLOW_FEATS = tb.FEATURES_M2_ADD + tb.FEATURES_M3_ADD + tb.FEATURES_M4_ADD

# Stage 2: fast triggers + structural risk score
FAST_FEATS = tb.FEATURES_M0 + tb.FEATURES_M1_ADD
RISK_SCORE = "structural_risk_score"


def _available(feats: list[str], cols: list[str]) -> list[str]:
    return [f for f in feats if f in cols]


def run_twostage_fold(
    train_df:  pd.DataFrame,
    test_df:   pd.DataFrame,
    event_type: str,
    horizon:   int,
    fold_id:   int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns (preds_df, coefs_stage1_df, coefs_stage2_df, metrics_dict).
    coefs are saved separately under model_name='model9_stage1' / 'model9_stage2'
    so they appear as distinct rows in coefs_lr.csv.
    """
    train_clean = train_df.dropna(subset=["y"])
    test_clean  = test_df.dropna(subset=["y"])
    if train_clean.empty or test_clean.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    cols = list(train_clean.columns)
    slow = _available(SLOW_FEATS, cols)
    fast = _available(FAST_FEATS, cols)

    if not slow or not fast:
        log.warning("Fold %d: missing slow or fast features.", fold_id)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    y_train = train_clean["y"].astype(int).values
    y_true  = test_clean["y"].astype(int).values

    # ------------------------------------------------------------------
    # Stage 1: structural risk — no country FE
    # ------------------------------------------------------------------
    stage1_pipe = tb.build_lr_pipeline(slow, False, tb.USE_CLASS_WEIGHT)

    # Out-of-fold predictions on training data (5-fold stratified CV)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stage1_train_scores = cross_val_predict(
        stage1_pipe, train_clean[slow], y_train,
        cv=cv, method="predict_proba",
    )[:, 1]

    # Refit on full training data for test predictions
    stage1_pipe.fit(train_clean[slow], y_train)
    stage1_test_scores = stage1_pipe.predict_proba(test_clean[slow])[:, 1]

    # Stage 1 coefficients
    coefs1 = _extract_coefs(stage1_pipe, slow, include_fe=False,
                             event_type=event_type, horizon=horizon,
                             fold_id=fold_id, model_name="model9_stage1")

    # ------------------------------------------------------------------
    # Stage 2: trigger model — fast features + structural_risk_score
    # ------------------------------------------------------------------
    stage2_feats = fast + [RISK_SCORE]

    train_s2 = train_clean[fast + (["country_iso3"] if tb.INCLUDE_COUNTRY_FE else [])].copy()
    test_s2  = test_clean[fast  + (["country_iso3"] if tb.INCLUDE_COUNTRY_FE else [])].copy()
    train_s2[RISK_SCORE] = stage1_train_scores
    test_s2[RISK_SCORE]  = stage1_test_scores

    stage2_pipe = tb.build_lr_pipeline(stage2_feats, tb.INCLUDE_COUNTRY_FE, tb.USE_CLASS_WEIGHT)
    stage2_pipe.fit(train_s2, y_train)
    y_pred = stage2_pipe.predict_proba(test_s2)[:, 1]

    coefs2 = _extract_coefs(stage2_pipe, stage2_feats,
                             include_fe=tb.INCLUDE_COUNTRY_FE,
                             event_type=event_type, horizon=horizon,
                             fold_id=fold_id, model_name="model9_stage2")

    # ------------------------------------------------------------------
    # Predictions and metrics
    # ------------------------------------------------------------------
    m = compute_metrics(y_true, y_pred)
    m.update({
        "fold_id":    fold_id,
        "event_type": event_type,
        "horizon":    horizon,
        "model_name": MODEL_NAME,
        "n_train":    len(train_clean),
        "n_test":     len(test_clean),
        "pos_rate":   float(y_true.mean()),
    })

    preds_df = test_clean[["date", "country_iso3", "y"]].copy()
    preds_df["y_pred"]     = y_pred
    preds_df["y_true"]     = preds_df["y"]
    preds_df["model_name"] = MODEL_NAME
    preds_df["fold_id"]    = fold_id
    preds_df["event_type"] = event_type
    preds_df["horizon"]    = horizon
    preds_df = preds_df.drop(columns=["y"])

    return preds_df, coefs1, coefs2, m


def _extract_coefs(pipe, feature_cols: list[str], include_fe: bool,
                   event_type: str, horizon: int, fold_id: int,
                   model_name: str) -> pd.DataFrame:
    """Extract LR coefficients from a fitted pipeline."""
    try:
        lr        = pipe.named_steps["model"]
        pre       = pipe.named_steps["preprocessor"]
        num_feats = feature_cols[:]

        if include_fe:
            try:
                fe_enc   = pre.named_transformers_["fe"]
                fe_names = list(fe_enc.get_feature_names_out(["country_iso3"]))
            except Exception:
                fe_names = []
        else:
            fe_names = []

        all_names = num_feats + fe_names
        coefs = lr.coef_[0]

        rows = []
        for name, coef in zip(all_names, coefs):
            if not name.startswith("country_iso3_"):
                rows.append({
                    "feature":    name,
                    "coefficient": coef,
                    "fold_id":    fold_id,
                    "event_type": event_type,
                    "horizon":    horizon,
                    "model_name": model_name,
                })
        return pd.DataFrame(rows)
    except Exception as exc:
        log.warning("Could not extract coefs for %s: %s", model_name, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main patch
# ---------------------------------------------------------------------------

def patch_m9() -> None:
    log.info("=" * 60)
    log.info("Patching %s (two-stage LR)", MODEL_NAME)
    log.info("Slow feats: %d  |  Fast feats: %d", len(SLOW_FEATS), len(FAST_FEATS))
    log.info("Stage 1 CV: 5-fold stratified (no country FE)")
    log.info("Stage 2: fast + risk_score (with country FE)")
    log.info("=" * 60)

    panel = pd.read_parquet(tb.PANEL_FILE)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["country_iso3", "date"]).reset_index(drop=True)
    log.info("Panel: %d rows x %d cols | %d countries",
             len(panel), panel.shape[1], panel["country_iso3"].nunique())

    all_feature_cols = list(dict.fromkeys(
        tb.FEATURES_M0 + tb.FEATURES_M1_ADD + tb.FEATURES_M2_ADD +
        tb.FEATURES_M3_ADD + tb.FEATURES_M4_ADD
    ))

    new_global: list[dict] = []

    for event_type, horizon in tb.TARGETS:
        label   = f"{event_type}_{horizon}d"
        out_dir = tb.PROC_DIR / label
        log.info("Target: %s", label)

        df = panel.copy()
        df["y"] = tb.make_target_gdelt(df, event_type, horizon, tb.EXCLUDE_LOW_COVERAGE)
        df = tb.shift_features_by_horizon(df, all_feature_cols, horizon)

        new_preds:  list[pd.DataFrame] = []
        new_coefs1: list[pd.DataFrame] = []
        new_coefs2: list[pd.DataFrame] = []
        new_metrics: list[dict]        = []

        for test_year in range(tb.FIRST_TEST_YEAR, tb.LAST_TEST_YEAR + 1):
            train_df = df[df["date"].dt.year <  test_year].copy()
            test_df  = df[df["date"].dt.year == test_year].copy()
            fold_id  = test_year - tb.FIRST_TEST_YEAR + 1

            # Per-fold country baselines
            baselines = tb.compute_country_baselines(train_df)
            for bl_col in ("country_protest_baseline", "country_strike_baseline"):
                train_df = train_df.drop(columns=[bl_col], errors="ignore")
                test_df  = test_df.drop(columns=[bl_col], errors="ignore")
            train_df = train_df.merge(baselines, on="country_iso3", how="left")
            test_df  = test_df.merge(baselines, on="country_iso3", how="left")
            for bl_col in ("country_protest_baseline", "country_strike_baseline"):
                train_df[bl_col] = train_df[bl_col].fillna(0.0)
                test_df[bl_col]  = test_df[bl_col].fillna(0.0)

            preds_df, coefs1, coefs2, m = run_twostage_fold(
                train_df, test_df, event_type, horizon, fold_id,
            )
            if preds_df.empty:
                continue

            new_preds.append(preds_df)
            if not coefs1.empty:
                new_coefs1.append(coefs1)
            if not coefs2.empty:
                new_coefs2.append(coefs2)
            new_metrics.append(m)
            new_global.append({**m, "target": label})

            log.info("  Fold %d | %-26s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f  BSS=%.3f",
                     fold_id, MODEL_NAME,
                     m["roc_auc"], m["pr_auc"], m["brier"],
                     m.get("brier_skill_score", float("nan")))

        # Append to existing files
        if new_preds:
            p = out_dir / "preds.parquet"
            existing = pd.read_parquet(p)
            existing = existing[existing["model_name"] != MODEL_NAME]
            pd.concat([existing, *new_preds], ignore_index=True).to_parquet(p, index=False)

        if new_coefs1 or new_coefs2:
            p = out_dir / "coefs_lr.csv"
            existing = pd.read_csv(p)
            existing = existing[~existing["model_name"].isin(["model9_stage1", "model9_stage2"])]
            new_coefs = pd.concat(new_coefs1 + new_coefs2, ignore_index=True)
            pd.concat([existing, new_coefs], ignore_index=True).to_csv(p, index=False)

        if new_metrics:
            p = out_dir / "metrics.csv"
            existing = pd.read_csv(p)
            existing = existing[existing["model_name"] != MODEL_NAME]
            pd.concat([existing, pd.DataFrame(new_metrics)], ignore_index=True).to_csv(p, index=False)
            log.info("  Saved -> %s", p)

    if new_global:
        p = tb.PROC_DIR / "model_performance.csv"
        existing = pd.read_csv(p)
        if "model_name" in existing.columns:
            existing = existing[existing["model_name"] != MODEL_NAME]
        pd.concat([existing, pd.DataFrame(new_global)], ignore_index=True).to_csv(p, index=False)
        log.info("Updated model_performance.csv")


if __name__ == "__main__":
    patch_m9()
    log.info("Done. Run make_figures.py to regenerate figures.")
