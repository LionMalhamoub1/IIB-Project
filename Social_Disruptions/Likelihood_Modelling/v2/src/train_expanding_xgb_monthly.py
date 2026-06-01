# Monthly expanding-window evaluation for XGBoost models.
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

QEXP_DIR = tb.PROC_DIR / "expanding_xgb"
OUT_DIR  = tb.PROC_DIR / "expanding_xgb_monthly"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [("protest", 7), ("strike", 7)]

MODELS = {
    "model5_xgb":       tb.FEATURES_XGB,
    "model6_xgb_nolag": tb.FEATURES_XGB_NOLAG,
}

FIXED_PARAMS  = tb.XGB_PARAM_GRID[0]
TRAIN_START   = pd.Timestamp("2017-01-01")
BASELINE_COLS = ["country_protest_baseline", "country_strike_baseline"]

# All 24 monthly periods: (train_cutoff, month_start, month_end, label)
MONTHS = [
    (pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31"), "2020-01"),
    (pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-29"), "2020-02"),
    (pd.Timestamp("2020-02-29"), pd.Timestamp("2020-03-01"), pd.Timestamp("2020-03-31"), "2020-03"),
    (pd.Timestamp("2020-03-31"), pd.Timestamp("2020-04-01"), pd.Timestamp("2020-04-30"), "2020-04"),
    (pd.Timestamp("2020-04-30"), pd.Timestamp("2020-05-01"), pd.Timestamp("2020-05-31"), "2020-05"),
    (pd.Timestamp("2020-05-31"), pd.Timestamp("2020-06-01"), pd.Timestamp("2020-06-30"), "2020-06"),
    (pd.Timestamp("2020-06-30"), pd.Timestamp("2020-07-01"), pd.Timestamp("2020-07-31"), "2020-07"),
    (pd.Timestamp("2020-07-31"), pd.Timestamp("2020-08-01"), pd.Timestamp("2020-08-31"), "2020-08"),
    (pd.Timestamp("2020-08-31"), pd.Timestamp("2020-09-01"), pd.Timestamp("2020-09-30"), "2020-09"),
    (pd.Timestamp("2020-09-30"), pd.Timestamp("2020-10-01"), pd.Timestamp("2020-10-31"), "2020-10"),
    (pd.Timestamp("2020-10-31"), pd.Timestamp("2020-11-01"), pd.Timestamp("2020-11-30"), "2020-11"),
    (pd.Timestamp("2020-11-30"), pd.Timestamp("2020-12-01"), pd.Timestamp("2020-12-31"), "2020-12"),
    (pd.Timestamp("2020-12-31"), pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-31"), "2021-01"),
    (pd.Timestamp("2021-01-31"), pd.Timestamp("2021-02-01"), pd.Timestamp("2021-02-28"), "2021-02"),
    (pd.Timestamp("2021-02-28"), pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-31"), "2021-03"),
    (pd.Timestamp("2021-03-31"), pd.Timestamp("2021-04-01"), pd.Timestamp("2021-04-30"), "2021-04"),
    (pd.Timestamp("2021-04-30"), pd.Timestamp("2021-05-01"), pd.Timestamp("2021-05-31"), "2021-05"),
    (pd.Timestamp("2021-05-31"), pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-30"), "2021-06"),
    (pd.Timestamp("2021-06-30"), pd.Timestamp("2021-07-01"), pd.Timestamp("2021-07-31"), "2021-07"),
    (pd.Timestamp("2021-07-31"), pd.Timestamp("2021-08-01"), pd.Timestamp("2021-08-31"), "2021-08"),
    (pd.Timestamp("2021-08-31"), pd.Timestamp("2021-09-01"), pd.Timestamp("2021-09-30"), "2021-09"),
    (pd.Timestamp("2021-09-30"), pd.Timestamp("2021-10-01"), pd.Timestamp("2021-10-31"), "2021-10"),
    (pd.Timestamp("2021-10-31"), pd.Timestamp("2021-11-01"), pd.Timestamp("2021-11-30"), "2021-11"),
    (pd.Timestamp("2021-11-30"), pd.Timestamp("2021-12-01"), pd.Timestamp("2021-12-31"), "2021-12"),
]

# Training cutoffs shared with the quarterly run — reuse those predictions
QUARTERLY_CUTOFFS = {
    pd.Timestamp("2019-12-31"),  # Jan 2020
    pd.Timestamp("2020-03-31"),  # Apr 2020
    pd.Timestamp("2020-06-30"),  # Jul 2020
    pd.Timestamp("2020-09-30"),  # Oct 2020
    pd.Timestamp("2020-12-31"),  # Jan 2021
    pd.Timestamp("2021-03-31"),  # Apr 2021
    pd.Timestamp("2021-06-30"),  # Jul 2021
    pd.Timestamp("2021-09-30"),  # Oct 2021
}


def run_expanding_xgb_monthly() -> None:
    log.info("=" * 60)
    log.info("Monthly expanding XGB  (reusing quarterly preds for 8 months)")
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

        # Load quarterly preds once per target for reuse
        q_path = QEXP_DIR / f"preds_{label}.parquet"
        if q_path.exists():
            q_preds = pd.read_parquet(q_path)
            q_preds["date"] = pd.to_datetime(q_preds["date"])
            log.info("  Loaded quarterly preds: %d rows", len(q_preds))
        else:
            q_preds = pd.DataFrame()
            log.warning("  No quarterly preds found at %s", q_path)

        for model_name, feat_cols in MODELS.items():
            log.info("  Model: %s  (%d features)", model_name, len(feat_cols))

            all_preds:   list[pd.DataFrame] = []
            all_metrics: list[dict]         = []

            for train_cutoff, m_start, m_end, m_label in MONTHS:

                # ── Reuse quarterly predictions ──────────────────────────────
                if train_cutoff in QUARTERLY_CUTOFFS and not q_preds.empty:
                    q_slice = q_preds[
                        (q_preds["model_name"] == model_name) &
                        (q_preds["date"] >= m_start) &
                        (q_preds["date"] <= m_end)
                    ].copy()

                    if q_slice.empty:
                        log.warning("  %s | %s: no quarterly preds for this month, retraining",
                                    m_label, model_name)
                        # Fall through to retrain below
                    else:
                        q_slice["retrain_month"] = m_label
                        q_slice = q_slice.drop(columns=["retrain_quarter"], errors="ignore")
                        metrics = compute_metrics(
                            q_slice["y_true"].astype(int).values,
                            q_slice["y_pred"].values,
                        )
                        all_preds.append(q_slice)
                        all_metrics.append({"month": m_label, "model_name": model_name, **metrics})
                        log.info("  %s | %s | REUSED quarterly (n=%d)  ROC-AUC=%.3f  BSS=%.3f",
                                 m_label, model_name, len(q_slice),
                                 metrics.get("roc_auc", float("nan")),
                                 metrics.get("brier_skill_score", float("nan")))
                        continue

                # ── New retrain ───────────────────────────────────────────────
                train_df = df[(df["date"] >= TRAIN_START) &
                              (df["date"] <= train_cutoff)].copy()
                test_df  = df[(df["date"] >= m_start) &
                              (df["date"] <= m_end)].copy()

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
                    log.warning("  %s | %s: skipping (train=%d, test=%d)",
                                m_label, model_name,
                                len(train_labeled), len(test_labeled))
                    continue
                if train_labeled["y"].nunique() < 2:
                    continue

                feats   = tb.available_features(feat_cols, list(train_labeled.columns))
                if not feats:
                    continue

                fe_cols       = ["country_iso3"] if tb.INCLUDE_COUNTRY_FE else []
                n_neg         = int((train_labeled["y"] == 0).sum())
                n_pos         = int((train_labeled["y"] == 1).sum())
                pos_neg_ratio = n_neg / max(n_pos, 1)

                pipe = tb.build_xgb_pipeline(feats, tb.INCLUDE_COUNTRY_FE,
                                             pos_neg_ratio, FIXED_PARAMS)
                try:
                    pipe.fit(train_labeled[feats + fe_cols],
                             train_labeled["y"].astype(int))
                    y_pred = pipe.predict_proba(
                        test_labeled[feats + fe_cols])[:, 1]
                except Exception as exc:
                    log.warning("  %s | %s FAILED: %s", m_label, model_name, exc)
                    continue

                y_true  = test_labeled["y"].astype(int).values
                metrics = compute_metrics(y_true, y_pred)

                preds_df = test_labeled[["date", "country_iso3", "y"]].copy()
                preds_df["y_pred"]         = y_pred
                preds_df["model_name"]     = model_name
                preds_df["retrain_month"]  = m_label
                preds_df.rename(columns={"y": "y_true"}, inplace=True)
                all_preds.append(preds_df)
                all_metrics.append({"month": m_label, "model_name": model_name, **metrics})
                log.info("  %s | %s | train=%d  test=%d  ROC-AUC=%.3f  BSS=%.3f",
                         m_label, model_name,
                         len(train_labeled), len(test_labeled),
                         metrics.get("roc_auc", float("nan")),
                         metrics.get("brier_skill_score", float("nan")))

            if all_preds:
                p = OUT_DIR / f"preds_{label}.parquet"
                if p.exists():
                    ex = pd.read_parquet(p)
                    ex = ex[ex["model_name"] != model_name]
                    pd.concat([ex, *all_preds], ignore_index=True).to_parquet(p, index=False)
                else:
                    pd.concat(all_preds, ignore_index=True).to_parquet(p, index=False)
                log.info("  Saved preds -> %s", p)

            if all_metrics:
                p = OUT_DIR / f"metrics_{label}.csv"
                if p.exists():
                    ex = pd.read_csv(p)
                    ex = ex[ex["model_name"] != model_name]
                    pd.concat([ex, pd.DataFrame(all_metrics)], ignore_index=True).to_csv(p, index=False)
                else:
                    pd.DataFrame(all_metrics).to_csv(p, index=False)
                log.info("  Saved metrics -> %s", p)


if __name__ == "__main__":
    run_expanding_xgb_monthly()
    log.info("Done.")
