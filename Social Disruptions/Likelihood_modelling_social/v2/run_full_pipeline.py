"""
run_full_pipeline.py
====================
End-to-end overnight pipeline.  Run this once and it will:

  Stage 1 — Backfill clustering & labels (verification/v2)
      Clusters the three missing date ranges (2017, Dec 2020 gap, 2021),
      rebuilds the country-day label panel, and rebuilds the modelling
      labels (protest_7d / strike_7d etc.) across the full 2017-2021 span.

  Stage 2 — Build modelling panel (Likelihood_modelling_social/v2)
      Merges the new GDELT labels into the 39-country feature panel,
      extended to cover 2017-01-01 to 2021-12-31.

  Stage 3 — Train & backtest (Likelihood_modelling_social/v2)
      Walk-forward backtest with two test folds (2020, 2021).
      Optuna hyperparameter search is enabled (XGB_OPTUNA_TRIALS = 40).
      All models: persistence → markets → full LR → XGBoost (tuned).
"""

import subprocess
import sys
import time
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent          # v2/
_SD      = _HERE.parent.parent                      # Social Disruptions/
_ROOT    = _SD.parent                               # repo root

BACKFILL_SCRIPT  = _SD / "verification" / "v2" / "run_backfill.py"
BUILD_PANEL_DIR  = _HERE / "src"
TRAIN_BT_DIR     = _HERE / "src"


def _banner(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}\n")


def _elapsed(t0: float) -> str:
    s = int(time.time() - t0)
    return f"{s // 3600}h {(s % 3600) // 60}m {s % 60}s"


# ── Stage 1: Backfill clustering + labels ─────────────────────────────────────
def stage1_backfill() -> None:
    _banner("STAGE 1 — Backfill clustering & label panels (2017 / Dec-2020 / 2021)")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(BACKFILL_SCRIPT)],
        cwd=str(BACKFILL_SCRIPT.parent),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Stage 1 failed (exit {result.returncode})")
    print(f"\nStage 1 complete in {_elapsed(t0)}")


# ── Stage 2: Build modelling panel ────────────────────────────────────────────
def stage2_build_panel() -> None:
    _banner("STAGE 2 — Build modelling panel (2017-01-01 to 2021-12-31)")
    t0 = time.time()

    sys.path.insert(0, str(BUILD_PANEL_DIR))
    import build_panel

    # Extend GDELT label range to cover the full 5-year dataset
    build_panel.GDELT_START = "2017-01-01"
    build_panel.GDELT_END   = "2021-12-31"

    build_panel.run()
    print(f"\nStage 2 complete in {_elapsed(t0)}")


# ── Stage 3: Train & backtest ──────────────────────────────────────────────────
def stage3_train() -> None:
    _banner("STAGE 3 — Walk-forward backtest  (test folds: 2020, 2021)")
    t0 = time.time()

    sys.path.insert(0, str(TRAIN_BT_DIR))
    import train_backtest

    # Two test folds now that 2021 labels exist
    train_backtest.FIRST_TEST_YEAR    = 2020
    train_backtest.LAST_TEST_YEAR     = 2021

    # Optuna tuning — 40 trials per fold (fine for overnight)
    train_backtest.XGB_OPTUNA_TRIALS  = 40

    train_backtest.main()
    print(f"\nStage 3 complete in {_elapsed(t0)}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    total_t0 = time.time()
    print("\nStarting full pipeline — estimated runtime: 3-6 hours")

    stage1_backfill()
    stage2_build_panel()
    stage3_train()

    _banner(f"PIPELINE COMPLETE — total runtime {_elapsed(total_t0)}")
    print(f"  Labels      : {_SD / 'verification/v2/output/labels_20170101_20211231.parquet'}")
    print(f"  Panel       : {_HERE / 'data/interim/modelling_panel_gdelt.parquet'}")
    print(f"  Results     : {_HERE / 'data/processed/'}")
