"""
Entry point for the full indicator validation pipeline.

Runs all steps in order:
  1. generate_baseline  — sample non-flood baseline points
  2. enrich_baseline    — add hydro-climate indicators via GEE (optional if
                          baseline is already enriched)
  3. distribution_analysis — coverage, summary stats, AUC-ROC, KS tests,
                             and plots saved to Natural_disruptions/results/

Usage:
  # Full run (generates + enriches + analyses):
  python -m Natural_disruptions.validating_indicators.run_validation \
         --project <gee-project-id>

  # Analysis only (skip generation/enrichment if already done):
  python -m Natural_disruptions.validating_indicators.run_validation --analyse-only
"""

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
BASELINE_RAW      = ROOT / "cache" / "floods" / "baseline_samples.jsonl"
BASELINE_ENRICHED = ROOT / "cache" / "floods" / "baseline_enriched.jsonl"


def main():
    parser = argparse.ArgumentParser(
        description="Indicator validation pipeline: generate baseline, enrich, and analyse"
    )
    parser.add_argument(
        "--project", default=None,
        help="Google Earth Engine project ID (required unless --analyse-only)"
    )
    parser.add_argument(
        "--analyse-only", action="store_true",
        help="Skip baseline generation and enrichment; jump straight to analysis"
    )
    parser.add_argument(
        "--skip-enrich", action="store_true",
        help="Generate baseline but do not run GEE enrichment (useful for dry runs)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.analyse_only:
        # --- Step 1: Generate baseline ---
        if BASELINE_RAW.exists():
            log.info(f"Baseline already exists ({BASELINE_RAW}), skipping generation")
        else:
            log.info("=== Step 1/3: Generating baseline samples ===")
            from .helper_scripts.generate_baseline import generate_baseline
            generate_baseline()

        # --- Step 2: Enrich baseline ---
        if args.skip_enrich:
            log.info("--skip-enrich set, skipping GEE enrichment")
        elif BASELINE_ENRICHED.exists():
            log.info(f"Enriched baseline already exists ({BASELINE_ENRICHED}), skipping enrichment")
        else:
            if not args.project:
                raise ValueError(
                    "--project <gee-project-id> is required for enrichment. "
                    "Use --skip-enrich to skip, or --analyse-only if baseline is already enriched."
                )
            log.info("=== Step 2/3: Enriching baseline with GEE indicators ===")
            from .helper_scripts.enrich_baseline import enrich_baseline
            enrich_baseline(args.project)
    else:
        log.info("--analyse-only: skipping generation and enrichment steps")

    # --- Step 3: Distribution analysis ---
    log.info("=== Step 3/3: Running distribution analysis ===")
    from .helper_scripts.distribution_analysis import run_distribution_analysis
    metrics = run_distribution_analysis()

    log.info("=== Validation complete ===")
    log.info(
        "Results saved to: "
        + str(ROOT / "Natural_disruptions" / "results" / "validating_indicators")
    )
    return metrics


if __name__ == "__main__":
    main()
