# Adds model_lr_nolag (M1+M2+M3+M4, no GDELT lags) to existing backtest and expanding-LR outputs.
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

# ---------------------------------------------------------------------------
# Import helpers from existing scripts without running their main()
# ---------------------------------------------------------------------------

def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / f"{name}.py")
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

log.info("Loading train_backtest...")
tb  = _load("train_backtest")
log.info("Loading train_expanding_lr...")
elr = _load("train_expanding_lr")

MODEL_NAME  = "model_lr_nolag"
NOLAG_FEATS = tb.FEATURES_M1_ADD + tb.FEATURES_M2_ADD + tb.FEATURES_M3_ADD + tb.FEATURES_M4_ADD

# ---------------------------------------------------------------------------
# Part 1: static backtest patch
# ---------------------------------------------------------------------------

def patch_backtest() -> None:
    log.info("=" * 60)
    log.info("PART 1: Static backtest — appending %s", MODEL_NAME)
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

        new_preds:   list[pd.DataFrame] = []
        new_coefs:   list[pd.DataFrame] = []
        new_metrics: list[dict]         = []

        for test_year in range(tb.FIRST_TEST_YEAR, tb.LAST_TEST_YEAR + 1):
            train_df = df[df["date"].dt.year <  test_year].copy()
            test_df  = df[df["date"].dt.year == test_year].copy()
            fold_id  = test_year - tb.FIRST_TEST_YEAR + 1

            # Per-fold baselines from training data only
            baselines = tb.compute_country_baselines(train_df)
            for bl_col in ("country_protest_baseline", "country_strike_baseline"):
                train_df = train_df.drop(columns=[bl_col], errors="ignore")
                test_df  = test_df.drop(columns=[bl_col], errors="ignore")
            train_df = train_df.merge(baselines, on="country_iso3", how="left")
            test_df  = test_df.merge(baselines, on="country_iso3", how="left")
            for bl_col in ("country_protest_baseline", "country_strike_baseline"):
                train_df[bl_col] = train_df[bl_col].fillna(0.0)
                test_df[bl_col]  = test_df[bl_col].fillna(0.0)

            preds_df, coefs_df, m = tb.run_lr_fold(
                train_df, test_df, NOLAG_FEATS,
                event_type, horizon, fold_id, MODEL_NAME,
            )
            if preds_df.empty:
                continue

            preds_df = tb._apply_pu_correction(preds_df, 1.0)
            m = tb._recompute_metrics(preds_df, m)

            new_preds.append(preds_df)
            new_coefs.append(coefs_df)
            new_metrics.append(m)
            new_global.append({**m, "target": label})

            log.info("  Fold %d | %-26s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f",
                     fold_id, MODEL_NAME, m["roc_auc"], m["pr_auc"], m["brier"])

        # Append to existing parquet / csv files
        if new_preds:
            preds_path = out_dir / "preds.parquet"
            existing   = pd.read_parquet(preds_path)
            # Remove any stale model_lr_nolag rows before appending
            existing   = existing[existing["model_name"] != MODEL_NAME]
            updated    = pd.concat([existing, *new_preds], ignore_index=True)
            updated.to_parquet(preds_path, index=False)
            log.info("  Saved preds -> %s", preds_path)

        if new_coefs:
            coefs_path = out_dir / "coefs_lr.csv"
            existing   = pd.read_csv(coefs_path)
            existing   = existing[existing["model_name"] != MODEL_NAME]
            updated    = pd.concat([existing, *new_coefs], ignore_index=True)
            updated.to_csv(coefs_path, index=False)
            log.info("  Saved coefs -> %s", coefs_path)

        if new_metrics:
            metrics_path = out_dir / "metrics.csv"
            existing     = pd.read_csv(metrics_path)
            existing     = existing[existing["model_name"] != MODEL_NAME]
            new_df       = pd.DataFrame(new_metrics)
            updated      = pd.concat([existing, new_df], ignore_index=True)
            updated.to_csv(metrics_path, index=False)
            log.info("  Saved metrics -> %s", metrics_path)

    # Update global model_performance.csv
    if new_global:
        perf_path = tb.PROC_DIR / "model_performance.csv"
        existing  = pd.read_csv(perf_path)
        existing  = existing[existing.get("model_name", existing.columns[0]) != MODEL_NAME
                              if "model_name" in existing.columns else slice(None)]
        new_perf  = pd.DataFrame(new_global)
        updated   = pd.concat([existing, new_perf], ignore_index=True)
        updated.to_csv(perf_path, index=False)
        log.info("Updated model_performance.csv")


# ---------------------------------------------------------------------------
# Part 2: expanding LR patch
# ---------------------------------------------------------------------------

def patch_expanding_lr() -> None:
    log.info("=" * 60)
    log.info("PART 2: Expanding LR — appending %s", MODEL_NAME)
    log.info("=" * 60)

    panel = pd.read_parquet(elr.PANEL_FILE)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["country_iso3", "date"]).reset_index(drop=True)

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

            feats = [f for f in NOLAG_FEATS
                     if f in train_labeled.columns]
            if not feats:
                continue

            cw   = "balanced" if elr.USE_CLASS_WEIGHT else None
            pipe = elr.build_lr_pipeline(feats, elr.INCLUDE_COUNTRY_FE, cw)

            input_cols = feats + (["country_iso3"] if elr.INCLUDE_COUNTRY_FE else [])
            try:
                pipe.fit(train_labeled[input_cols], train_labeled["y"])
                y_pred = pipe.predict_proba(test_labeled[input_cols])[:, 1]
            except Exception as exc:
                log.warning("  %s %s FAILED: %s", month_label, MODEL_NAME, exc)
                continue

            y_true  = test_labeled["y"].values
            metrics = elr.compute_metrics(y_true, y_pred)

            preds_df = test_labeled[["date", "country_iso3", "y"]].copy()
            preds_df["y_pred"]        = y_pred
            preds_df["model_name"]    = MODEL_NAME
            preds_df["retrain_month"] = month_label
            preds_df.rename(columns={"y": "y_true"}, inplace=True)
            new_preds.append(preds_df)

            new_metrics.append({
                "month": month_label,
                "model_name": MODEL_NAME,
                **metrics,
            })
            log.info("  %s | train=%d  test=%d  ROC-AUC=%.3f",
                     month_label, len(train_labeled), len(test_labeled),
                     metrics.get("roc_auc", float("nan")))

        if new_preds:
            preds_path = elr.OUT_DIR / f"preds_{label}.parquet"
            existing   = pd.read_parquet(preds_path)
            existing   = existing[existing["model_name"] != MODEL_NAME]
            updated    = pd.concat([existing, *new_preds], ignore_index=True)
            updated.to_parquet(preds_path, index=False)
            log.info("Saved preds -> %s", preds_path)

        if new_metrics:
            metrics_path = elr.OUT_DIR / f"metrics_{label}.csv"
            existing     = pd.read_csv(metrics_path)
            existing     = existing[existing["model_name"] != MODEL_NAME]
            updated      = pd.concat([existing, pd.DataFrame(new_metrics)], ignore_index=True)
            updated.to_csv(metrics_path, index=False)
            log.info("Saved metrics -> %s", metrics_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    patch_backtest()
    patch_expanding_lr()
    log.info("Done. Run make_figures.py to regenerate figures.")
