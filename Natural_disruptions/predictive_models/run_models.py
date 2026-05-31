"""
Entry point for the full predictive modelling pipeline.

Runs all three modelling steps in order:
  1. train            — cross-validated Logistic Regression + Random Forest;
                        reports AUC, precision, recall, F1 and saves ROC / CM plots
  2. feature_importance — per-indicator Gini importance (RF) and standardised
                          coefficient magnitude (LR); saves bar charts
  3. ablation         — leave-one-out AUC drop + single-feature AUC for each
                        indicator; saves ablation bar charts

All outputs are saved to:
  Natural_disruptions/results/predictive_models/

Usage:
  python -m Natural_disruptions.predictive_models.run_models

  # Run only a subset of steps:
  python -m Natural_disruptions.predictive_models.run_models --steps train ablation
"""

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "Natural_disruptions" / "results" / "predictive_models"

ALL_STEPS = ["train", "feature_importance", "ablation"]


def main():
    parser = argparse.ArgumentParser(
        description="Predictive modelling pipeline: train, feature importance, ablation"
    )
    parser.add_argument(
        "--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS,
        help="Which steps to run (default: all)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if "train" in args.steps:
        log.info("=== Step 1/3: Training classifiers ===")
        from .helper_scripts.train import run_training
        run_training(RESULTS_DIR)

    if "feature_importance" in args.steps:
        log.info("=== Step 2/3: Feature importance ===")
        from .helper_scripts.feature_importance import run_feature_importance
        run_feature_importance(RESULTS_DIR)

    if "ablation" in args.steps:
        log.info("=== Step 3/3: Ablation study ===")
        from .helper_scripts.ablation import run_ablation
        run_ablation(RESULTS_DIR)

    log.info("=== Modelling complete ===")
    log.info(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
