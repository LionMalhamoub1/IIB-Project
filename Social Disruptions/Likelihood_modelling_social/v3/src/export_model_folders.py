"""
export_model_folders.py
=======================
Reorganises v3 static backtest outputs from target-level folders
into individual per-model folders (M0-M7).

Input:  v3/data/processed/{protest_7d,strike_7d}/
Output: v3/data/processed/{model_name}/
"""
import shutil
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "data" / "processed"

TARGETS = ["protest_7d", "strike_7d"]

MODEL_MAP = {
    "M0": "model0_persistence",
    "M1": "model1_markets",
    "M2": "model2_full",
    "M3": "model3_structural",
    "M4": "model4_fao",
    "M5": "model_lr_nolag",
    "M6": "model5_xgb",
    "M7": "model6_xgb_nolag",
}

LR_MODELS  = {"model0_persistence", "model1_markets", "model2_full",
              "model3_structural",  "model4_fao",    "model_lr_nolag"}
XGB_MODELS = {"model5_xgb", "model6_xgb_nolag"}


def load_concat(filename: str, col_filter: str | None = None,
                value: str | None = None) -> pd.DataFrame:
    """Load a CSV from all target folders and concatenate, adding a 'target' column."""
    dfs = []
    for t in TARGETS:
        p = BASE / t / filename
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["target"] = t
        if col_filter and value and col_filter in df.columns:
            df = df[df[col_filter] == value]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def load_concat_parquet(col_filter: str, value: str) -> pd.DataFrame:
    dfs = []
    for t in TARGETS:
        p = BASE / t / "preds.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        df["target"] = t
        df = df[df[col_filter] == value]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def export_model(label: str, model_name: str) -> None:
    out = BASE / label
    out.mkdir(parents=True, exist_ok=True)

    # metrics.csv
    df = load_concat("metrics.csv", "model_name", model_name)
    if not df.empty:
        df.to_csv(out / "metrics.csv", index=False)

    # preds.parquet
    df = load_concat_parquet("model_name", model_name)
    if not df.empty:
        df.to_parquet(out / "preds.parquet", index=False)

    # country_metrics.csv (no model_name column — skip filtering)
    df = load_concat("country_metrics.csv")
    if not df.empty:
        df.to_csv(out / "country_metrics.csv", index=False)

    # coefs_lr.csv  (LR models only)
    if model_name in LR_MODELS:
        df = load_concat("coefs_lr.csv", "model_name", model_name)
        if not df.empty:
            df.to_csv(out / "coefs_lr.csv", index=False)

    # coefs_xgb.csv + shap_importance.csv  (XGB models only)
    if model_name in XGB_MODELS:
        df = load_concat("coefs_xgb.csv", "model_name", model_name)
        if not df.empty:
            df.to_csv(out / "coefs_xgb.csv", index=False)

        df = load_concat("shap_importance.csv", "model_name", model_name)
        if not df.empty:
            df.to_csv(out / "shap_importance.csv", index=False)

        df = load_concat("feature_importance_summary.csv")
        if not df.empty:
            df.to_csv(out / "feature_importance_summary.csv", index=False)

        df = load_concat("calibration_comparison.csv", "model_name", model_name)
        if not df.empty:
            df.to_csv(out / "calibration_comparison.csv", index=False)

    print(f"  {label:3s}  ({model_name})  ->  {out}")


if __name__ == "__main__":
    print(f"Exporting per-model folders to {BASE}\n")
    for label, model_name in MODEL_MAP.items():
        export_model(label, model_name)
    print("\nDone.")
