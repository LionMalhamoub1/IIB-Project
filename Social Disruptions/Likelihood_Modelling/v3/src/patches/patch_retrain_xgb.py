# Retrains model5_xgb and model6_xgb_nolag after removing country baselines.
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

MODELS = {
    "model5_xgb":       tb.FEATURES_XGB,
    "model6_xgb_nolag": tb.FEATURES_XGB_NOLAG,
}
TARGETS = [("protest", 7), ("strike", 7)]


def patch_xgb_backtest() -> None:
    log.info("=" * 60)
    log.info("XGB static backtest retrain: %s", list(MODELS.keys()))
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

    for event_type, horizon in TARGETS:
        label   = f"{event_type}_{horizon}d"
        out_dir = tb.PROC_DIR / label
        log.info("Target: %s", label)

        df = panel.copy()
        df["y"] = tb.make_target_gdelt(df, event_type, horizon, tb.EXCLUDE_LOW_COVERAGE)
        df = tb.shift_features_by_horizon(df, all_feature_cols, horizon)

        for model_name, feat_cols in MODELS.items():
            log.info("  Model: %s  (%d features)", model_name, len(feat_cols))

            new_preds:  list[pd.DataFrame] = []
            new_imps:   list[pd.DataFrame] = []
            new_shaps:  list[pd.DataFrame] = []
            new_metrics: list[dict]        = []

            for test_year in range(tb.FIRST_TEST_YEAR, tb.LAST_TEST_YEAR + 1):
                val_year = test_year - 1
                train_df = df[df["date"].dt.year <  test_year].copy()
                test_df  = df[df["date"].dt.year == test_year].copy()
                fold_id  = test_year - tb.FIRST_TEST_YEAR + 1

                baselines = tb.compute_country_baselines(train_df)
                for bl_col in ("country_protest_baseline", "country_strike_baseline"):
                    train_df = train_df.drop(columns=[bl_col], errors="ignore")
                    test_df  = test_df.drop(columns=[bl_col], errors="ignore")
                train_df = train_df.merge(baselines, on="country_iso3", how="left")
                test_df  = test_df.merge(baselines, on="country_iso3", how="left")

                log.info("    Fold %d | Tuning XGB (Optuna)...", fold_id)
                tuned_p = tb._tune_xgb_params(train_df, feat_cols, val_year)

                px, ix, sx, mx = tb.run_xgb_fold(
                    train_df, test_df, feat_cols,
                    event_type, horizon, fold_id, tuned_p,
                    model_name=model_name, val_year=val_year,
                )
                if px.empty:
                    continue

                px = tb._apply_pu_correction(px, 1.0)
                mx = tb._recompute_metrics(px, mx)

                new_preds.append(px)
                if not ix.empty:
                    new_imps.append(ix)
                if not sx.empty:
                    new_shaps.append(sx)
                new_metrics.append(mx)

                log.info("    Fold %d | ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.4f  cal=%s",
                         fold_id, mx["roc_auc"], mx["pr_auc"], mx["brier"],
                         mx.get("cal_method", "—"))

            if new_preds:
                p = out_dir / "preds.parquet"
                ex = pd.read_parquet(p)
                ex = ex[ex["model_name"] != model_name]
                pd.concat([ex, *new_preds], ignore_index=True).to_parquet(p, index=False)
                log.info("    Saved preds -> %s", p)

            if new_metrics:
                p = out_dir / "metrics.csv"
                ex = pd.read_csv(p)
                ex = ex[ex["model_name"] != model_name]
                pd.concat([ex, pd.DataFrame(new_metrics)], ignore_index=True).to_csv(p, index=False)
                log.info("    Saved metrics -> %s", p)

            if new_shaps:
                shap_all = pd.concat(new_shaps, ignore_index=True)
                shap_summary = (
                    shap_all.groupby(["model_name", "feature"])["mean_abs_shap"]
                    .mean().reset_index()
                    .sort_values(["model_name", "mean_abs_shap"], ascending=[True, False])
                )
                p = out_dir / "shap_importance.csv"
                ex = pd.read_csv(p) if p.exists() else pd.DataFrame()
                if not ex.empty:
                    ex = ex[ex["model_name"] != model_name]
                pd.concat([ex, shap_summary], ignore_index=True).to_csv(p, index=False)
                log.info("    Saved SHAP -> %s", p)

            if new_imps:
                imp_all = pd.concat(new_imps, ignore_index=True)
                imp_summary = (
                    imp_all.groupby("feature")["importance"]
                    .mean().reset_index()
                    .sort_values("importance", ascending=False)
                )
                imp_summary["model_name"] = model_name
                p = out_dir / "feature_importance_summary.csv"
                ex = pd.read_csv(p) if p.exists() else pd.DataFrame()
                if not ex.empty and "model_name" in ex.columns:
                    ex = ex[ex["model_name"] != model_name]
                pd.concat([ex, imp_summary], ignore_index=True).to_csv(p, index=False)
                log.info("    Saved importance -> %s", p)


if __name__ == "__main__":
    patch_xgb_backtest()
    log.info("Done.")
