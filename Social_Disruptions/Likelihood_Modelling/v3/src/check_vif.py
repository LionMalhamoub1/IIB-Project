# Quick diagnostic: reports which features VIF filtering drops for each model.
import sys
from pathlib import Path
import pandas as pd
import numpy as np

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))

import train_backtest as tb
from utils import vif_filter

# Load panel and prepare fold 1 training data
panel = tb.load_panel()
panel["y"] = tb.make_target_gdelt(panel, "protest", 7, tb.EXCLUDE_LOW_COVERAGE)

all_feat_cols = list(dict.fromkeys(
    tb.FEATURES_M0 + tb.FEATURES_M1_ADD + tb.FEATURES_M2_ADD +
    tb.FEATURES_M3_ADD + tb.FEATURES_M4_ADD
))
panel = tb.shift_features_by_horizon(panel, all_feat_cols, 7)

train_df = panel[panel["date"].dt.year < 2020].copy()
baselines = tb.compute_country_baselines(train_df)
train_df = train_df.merge(baselines, on="country_iso3", how="left")
train_labeled = train_df.dropna(subset=["y"])

print(f"Training rows: {len(train_labeled):,}  ({train_labeled['date'].dt.year.min()}–{train_labeled['date'].dt.year.max()})\n")
print(f"{'Model':<22}  {'In':>4}  {'Out':>4}  {'Dropped features'}")
print("-" * 90)

models = {
    **tb.MODEL_SPECS,
    "model5_xgb":       tb.FEATURES_XGB,
    "model6_xgb_nolag": tb.FEATURES_XGB_NOLAG,
}

for model_name, feat_list in models.items():
    feats = tb.available_features(feat_list, list(train_labeled.columns))
    kept, dropped = vif_filter(train_labeled, feats, tb.VIF_THRESHOLD)
    drop_str = ", ".join(f"{f}(VIF={v:.0f})" for f, v in dropped.items()) if dropped else "—"
    print(f"{model_name:<22}  {len(feats):>4}  {len(kept):>4}  {drop_str}")
