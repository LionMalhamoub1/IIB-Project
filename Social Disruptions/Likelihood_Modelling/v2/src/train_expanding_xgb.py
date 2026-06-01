# Quarterly expanding-window evaluation for XGBoost models.
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


log.info("Loading train_backtest...")
tb = _load("train_backtest")

from utils import compute_metrics

OUT_DIR = tb.PROC_DIR / "expanding_xgb"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [("protest", 7), ("strike", 7)]

MODELS = {
    "model5_xgb":       tb.FEATURES_XGB,
    "model6_xgb_nolag": tb.FEATURES_XGB_NOLAG,
}

# Fixed params — avoids Optuna overhead per quarter
FIXED_PARAMS = tb.XGB_PARAM_GRID[0]

TRAIN_START = pd.Timestamp("2017-01-01")
QUARTERS = [
    # (train_cutoff, quarter_start, quarter_end, label)
    (pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-31"), "2020-Q1"),
    (pd.Timestamp("2020-03-31"), pd.Timestamp("2020-04-01"), pd.Timestamp("2020-06-30"), "2020-Q2"),
    (pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-01"), pd.Timestamp("2020-09-30"), "2020-Q3"),
    (pd.Timestamp("2020-09-30"), pd.Timestamp("2020-10-01"), pd.Timestamp("2020-12-31"), "2020-Q4"),
    (pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01"), pd.Timestamp("2021-03-31"), "2021-Q1"),
    (pd.Timestamp("2021-03-31"), pd.Timestamp("2021-04-01"), pd.Timestamp("2021-06-30"), "2021-Q2"),
    (pd.Timestamp("2021-06-30"), pd.Timestamp("2021-07-01"), pd.Timestamp("2021-09-30"), "2021-Q3"),
    (pd.Timestamp("2021-09-30"), pd.Timestamp("2021-10-01"), pd.Timestamp("2021-12-31"), "2021-Q4"),
]

BASELINE_COLS = ["country_protest_baseline", "country_strike_baseline"]


def run_expanding_xgb() -> None:
    log.info("=" * 60)
    log.info("Quarterly expanding XGB: %s", list(MODELS.keys()))
    log.info("=" * 60)

    panel = pd.read_parquet(tb.PANEL_FILE)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["country_iso3", "date"]).reset_index(drop=True)
    log.info("Panel: %d rows x %d cols | %d countries",
             len(panel), panel.shape[1], panel["country_iso3"].nunique())

    all_feat_cols = list(dict.fromkeys(
        tb.FEATURES_M0 + tb.FEATURES_M1_ADD + tb.FEATURES_M2_ADD +
        tb.FEATURES_M3_ADD + tb.FEATURES_M4_ADD
    ))
    cols_to_shift = [c for c in all_feat_cols
                     if c not in BASELINE_COLS and c in panel.columns]

    for event_type, horizon in TARGETS:
        label = f"{event_type}_{horizon}d"
        log.info("Target: %s", label)

        df = panel.copy()
        df["y"] = tb.make_target_gdelt(df, event_type, horizon, tb.EXCLUDE_LOW_COVERAGE)
        df = tb.shift_features_by_horizon(df, cols_to_shift, horizon)

        for model_name, feat_cols in MODELS.items():
            log.info("  Model: %s  (%d features)", model_name, len(feat_cols))

            new_preds:   list[pd.DataFrame] = []
            new_metrics: list[dict]         = []

            for train_cutoff, q_start, q_end, q_label in QUARTERS:

                train_df = df[(df["date"] >= TRAIN_START) &
                              (df["date"] <= train_cutoff)].copy()
                test_df  = df[(df["date"] >= q_start) &
                              (df["date"] <= q_end)].copy()

                # Compute baselines from training data (not used as features
                # after removal from FEATURES_M3_ADD, but pipeline expects them)
                train_orig = panel[(panel["date"] >= TRAIN_START) &
                                   (panel["date"] <= train_cutoff)]
                bl = (train_orig.groupby("country_iso3")[
                          [c for c in ["protest_today", "strike_today"]
                           if c in train_orig.columns]]
                      .mean()
                      .rename(columns={"protest_today": "country_protest_baseline",
                                       "strike_today":  "country_strike_baseline"})
                      .reset_index())
                for df_split in (train_df, test_df):
                    for col in BASELINE_COLS:
                        if col in df_split.columns:
                            df_split.drop(columns=[col], inplace=True)
                train_df = train_df.merge(bl, on="country_iso3", how="left")
                test_df  = test_df.merge(bl,  on="country_iso3", how="left")

                train_labeled = train_df.dropna(subset=["y"])
                test_labeled  = test_df.dropna(subset=["y"])

                if test_labeled.empty or len(train_labeled) < 200:
                    log.warning("  %s: skipping (train=%d, test=%d)",
                                q_label, len(train_labeled), len(test_labeled))
                    continue
                if train_labeled["y"].nunique() < 2:
                    continue

                feats = tb.available_features(feat_cols, list(train_labeled.columns))
                if not feats:
                    continue

                fe_cols = ["country_iso3"] if tb.INCLUDE_COUNTRY_FE else []
                n_neg = int((train_labeled["y"] == 0).sum())
                n_pos = int((train_labeled["y"] == 1).sum())
                pos_neg_ratio = n_neg / max(n_pos, 1)

                pipe = tb.build_xgb_pipeline(feats, tb.INCLUDE_COUNTRY_FE,
                                             pos_neg_ratio, FIXED_PARAMS)
                try:
                    pipe.fit(train_labeled[feats + fe_cols],
                             train_labeled["y"].astype(int))
                    y_pred = pipe.predict_proba(
                        test_labeled[feats + fe_cols])[:, 1]
                except Exception as exc:
                    log.warning("  %s %s FAILED: %s", q_label, model_name, exc)
                    continue

                y_true  = test_labeled["y"].astype(int).values
                metrics = compute_metrics(y_true, y_pred)

                preds_df = test_labeled[["date", "country_iso3", "y"]].copy()
                preds_df["y_pred"]         = y_pred
                preds_df["model_name"]     = model_name
                preds_df["retrain_quarter"] = q_label
                preds_df.rename(columns={"y": "y_true"}, inplace=True)
                new_preds.append(preds_df)

                new_metrics.append({
                    "quarter":    q_label,
                    "model_name": model_name,
                    **metrics,
                })
                log.info("  %s | %s | train=%d  test=%d  ROC-AUC=%.3f  BSS=%.3f",
                         q_label, model_name,
                         len(train_labeled), len(test_labeled),
                         metrics.get("roc_auc", float("nan")),
                         metrics.get("brier_skill_score", float("nan")))

            if new_preds:
                p = OUT_DIR / f"preds_{label}.parquet"
                if p.exists():
                    ex = pd.read_parquet(p)
                    ex = ex[ex["model_name"] != model_name]
                    pd.concat([ex, *new_preds], ignore_index=True).to_parquet(p, index=False)
                else:
                    pd.concat(new_preds, ignore_index=True).to_parquet(p, index=False)
                log.info("  Saved preds -> %s", p)

            if new_metrics:
                p = OUT_DIR / f"metrics_{label}.csv"
                if p.exists():
                    ex = pd.read_csv(p)
                    ex = ex[ex["model_name"] != model_name]
                    pd.concat([ex, pd.DataFrame(new_metrics)], ignore_index=True).to_csv(p, index=False)
                else:
                    pd.DataFrame(new_metrics).to_csv(p, index=False)
                log.info("  Saved metrics -> %s", p)


if __name__ == "__main__":
    run_expanding_xgb()
    log.info("Done.")
