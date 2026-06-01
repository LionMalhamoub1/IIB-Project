# Adds model9_twostage to the monthly expanding-LR outputs.
import importlib.util
import logging
import sys
from pathlib import Path

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


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / f"{name}.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


log.info("Loading train_expanding_lr...")
elr = _load("train_expanding_lr")

from utils import compute_metrics

MODEL_NAME = "model9_twostage"
SLOW_FEATS = elr.FEATURES_M2_ADD + elr.FEATURES_M3_ADD + elr.FEATURES_M4_ADD
FAST_FEATS = elr.FEATURES_M0 + elr.FEATURES_M1_ADD
RISK_SCORE = "structural_risk_score"


def patch_m9_expanding() -> None:
    log.info("=" * 60)
    log.info("Expanding M9 two-stage patch")
    log.info("Slow feats: %d  |  Fast feats: %d", len(SLOW_FEATS), len(FAST_FEATS))
    log.info("=" * 60)

    panel = pd.read_parquet(elr.PANEL_FILE)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["country_iso3", "date"]).reset_index(drop=True)
    log.info("Panel: %d rows x %d cols | %d countries",
             len(panel), panel.shape[1], panel["country_iso3"].nunique())

    all_panel_feat_cols = sorted(set(
        elr.FEATURES_M0 + elr.FEATURES_M1_ADD + elr.FEATURES_M2_ADD +
        elr.FEATURES_M3_ADD + elr.FEATURES_M4_ADD
    ))
    cols_to_shift = [c for c in all_panel_feat_cols
                     if c not in elr.BASELINE_COLS and c in panel.columns]

    periods = elr.monthly_periods(elr.TEST_START, elr.TEST_END)
    log.info("Monthly expanding window: %d periods", len(periods))

    for event_type, horizon in elr.TARGETS:
        label = f"{event_type}_{horizon}d"
        log.info("Target: %s", label)

        df = panel.copy()
        df["y"] = elr.make_target_gdelt(df, event_type, horizon,
                                        exclude_low_coverage=False)
        df = elr.shift_features_by_horizon(df, cols_to_shift, horizon)

        new_preds:   list[pd.DataFrame] = []
        new_metrics: list[dict]         = []

        for train_cutoff, month_start, month_end in periods:
            month_label = month_start.strftime("%Y-%m")

            train_df = df[(df["date"] >= elr.TRAIN_START) &
                          (df["date"] <= train_cutoff)].copy()
            test_df  = df[(df["date"] >= month_start) &
                          (df["date"] <= month_end)].copy()

            train_orig = panel[(panel["date"] >= elr.TRAIN_START) &
                               (panel["date"] <= train_cutoff)]
            bl = elr.compute_baselines(train_orig)

            for df_split in (train_df, test_df):
                for col in elr.BASELINE_COLS:
                    if col in df_split.columns:
                        df_split.drop(columns=[col], inplace=True)
            train_df = train_df.merge(bl, on="country_iso3", how="left")
            test_df  = test_df.merge(bl,  on="country_iso3", how="left")

            train_labeled = train_df.dropna(subset=["y"])
            test_labeled  = test_df.dropna(subset=["y"])

            if test_labeled.empty or len(train_labeled) < 100:
                continue
            if train_labeled["y"].nunique() < 2:
                continue

            slow = [f for f in SLOW_FEATS if f in train_labeled.columns]
            fast = [f for f in FAST_FEATS if f in train_labeled.columns]
            if not slow or not fast:
                continue

            y_train = train_labeled["y"].astype(int).values

            # Stage 1: structural risk — no country FE, 5-fold CV
            stage1 = elr.build_lr_pipeline(slow, False,
                                            "balanced" if elr.USE_CLASS_WEIGHT else None)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            try:
                stage1_train_scores = cross_val_predict(
                    stage1, train_labeled[slow], y_train,
                    cv=cv, method="predict_proba",
                )[:, 1]
                stage1.fit(train_labeled[slow], y_train)
                stage1_test_scores = stage1.predict_proba(test_labeled[slow])[:, 1]
            except Exception as exc:
                log.warning("  %s Stage1 FAILED: %s", month_label, exc)
                continue

            # Stage 2: fast features + risk score, with country FE
            stage2_feats = fast + [RISK_SCORE]
            input_cols   = stage2_feats + (["country_iso3"] if elr.INCLUDE_COUNTRY_FE else [])

            train_s2 = train_labeled[fast + (["country_iso3"] if elr.INCLUDE_COUNTRY_FE else [])].copy()
            test_s2  = test_labeled[fast  + (["country_iso3"] if elr.INCLUDE_COUNTRY_FE else [])].copy()
            train_s2[RISK_SCORE] = stage1_train_scores
            test_s2[RISK_SCORE]  = stage1_test_scores

            stage2 = elr.build_lr_pipeline(stage2_feats, elr.INCLUDE_COUNTRY_FE,
                                            "balanced" if elr.USE_CLASS_WEIGHT else None)
            try:
                stage2.fit(train_s2[input_cols], y_train)
                y_pred = stage2.predict_proba(test_s2[input_cols])[:, 1]
            except Exception as exc:
                log.warning("  %s Stage2 FAILED: %s", month_label, exc)
                continue

            y_true  = test_labeled["y"].values
            metrics = compute_metrics(y_true, y_pred)

            preds_df = test_labeled[["date", "country_iso3", "y"]].copy()
            preds_df["y_pred"]        = y_pred
            preds_df["model_name"]    = MODEL_NAME
            preds_df["retrain_month"] = month_label
            preds_df.rename(columns={"y": "y_true"}, inplace=True)
            new_preds.append(preds_df)

            new_metrics.append({
                "month":      month_label,
                "model_name": MODEL_NAME,
                **metrics,
            })
            log.info("  %s | train=%d  test=%d  ROC-AUC=%.3f  BSS=%.3f",
                     month_label, len(train_labeled), len(test_labeled),
                     metrics.get("roc_auc", float("nan")),
                     metrics.get("brier_skill_score", float("nan")))

        if new_preds:
            p = elr.OUT_DIR / f"preds_{label}.parquet"
            existing = pd.read_parquet(p)
            existing = existing[existing["model_name"] != MODEL_NAME]
            pd.concat([existing, *new_preds], ignore_index=True).to_parquet(p, index=False)
            log.info("Saved preds -> %s", p)

        if new_metrics:
            p = elr.OUT_DIR / f"metrics_{label}.csv"
            existing = pd.read_csv(p)
            existing = existing[existing["model_name"] != MODEL_NAME]
            pd.concat([existing, pd.DataFrame(new_metrics)], ignore_index=True).to_csv(p, index=False)
            log.info("Saved metrics -> %s", p)


if __name__ == "__main__":
    patch_m9_expanding()
    log.info("Done.")
