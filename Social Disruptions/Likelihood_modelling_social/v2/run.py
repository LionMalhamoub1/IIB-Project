"""
run.py
======
Entry point for the GDELT-label likelihood modelling pipeline (v2).

Steps
-----
1. build_panel  -- merge GDELT labels with the economic feature panel
2. train        -- walk-forward backtest across 4 targets

Outputs land in:
  v2/data/interim/modelling_panel_gdelt.parquet   -- merged feature + label panel
  v2/data/processed/<target>/                     -- predictions, metrics, feature importances
    preds.parquet
    metrics.csv
    coefs_lr.csv      (logistic regression coefficients)
    coefs_xgb.csv     (XGBoost feature importances)
"""

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from build_panel    import run as build_panel
from train_backtest import main as train


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("STEP 1 - Building GDELT modelling panel")
    print("=" * 60 + "\n")
    build_panel()

    print("\n" + "=" * 60)
    print("STEP 2 - Walk-forward backtest")
    print("=" * 60 + "\n")
    train()

    print("\nDone.")
    print(f"  Panel   : {_HERE / 'data' / 'interim' / 'modelling_panel_gdelt.parquet'}")
    print(f"  Results : {_HERE / 'data' / 'processed'}/")
