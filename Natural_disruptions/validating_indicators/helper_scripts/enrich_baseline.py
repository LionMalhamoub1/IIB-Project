"""
Enrich baseline samples with the same hydro-climate indicators used for
flood reference events.

Calls the existing enrichment modules from Builder_Reference
sequentially, writing results to cache/floods/baseline_enriched.jsonl.
Each enrichment step is resumable — already-processed samples are skipped.

Requires:
  - Google Earth Engine credentials (for SPI, CHIRPS, ERA5, GPM, static features)

Run:
  python -m Natural_disruptions.validating_indicators.enrich_baseline \
         --project <gee-project-id>

Input:  cache/floods/baseline_samples.jsonl  (from generate_baseline.py)
Output: cache/floods/baseline_enriched.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
BASELINE_PATH  = ROOT / "cache" / "floods" / "baseline_samples.jsonl"
ENRICHED_PATH  = ROOT / "cache" / "floods" / "baseline_enriched.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def _save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")


def enrich_baseline(gee_project: str, input_path: Path = BASELINE_PATH,
                    output_path: Path = ENRICHED_PATH) -> list[dict]:
    """
    Run all enrichment passes on the baseline samples.

    Passes (in order):
      1. SPI-30 (CHIRPS via GEE)
      2. CHIRPS rainfall totals and anomaly (GEE)
      3. ERA5-Land soil moisture and runoff (GEE)
      4. GPM peak intensity (GEE)
      5. Static features: JRC surface water + WorldPop (GEE)

    Each pass writes incrementally; re-running this script is safe.
    """
    sys.path.insert(0, str(ROOT))

    from Builder_Reference.helper_scripts.enrichment.spi           import enrich_events_with_spi
    from Builder_Reference.helper_scripts.enrichment.chirps        import enrich_events_with_chirps
    from Builder_Reference.helper_scripts.enrichment.era5          import enrich_events_with_era5
    from Builder_Reference.helper_scripts.enrichment.gpm           import enrich_events_with_gpm
    from Builder_Reference.helper_scripts.enrichment.static_features import enrich_events_with_static_features

    samples = _load_jsonl(input_path)
    log.info(f"Loaded {len(samples)} baseline samples from {input_path}")

    # Intermediate paths (one per enrichment step)
    step_paths = {
        "spi":    ROOT / "cache" / "floods" / "baseline_spi.jsonl",
        "chirps": ROOT / "cache" / "floods" / "baseline_chirps.jsonl",
        "era5":   ROOT / "cache" / "floods" / "baseline_era5.jsonl",
        "gpm":    ROOT / "cache" / "floods" / "baseline_gpm.jsonl",
        "static": ROOT / "cache" / "floods" / "baseline_static.jsonl",
    }

    def _run_step(fn, current, path, **kwargs):
        """Run enrichment step and reload from file (functions only return new events)."""
        fn(current, output_path=path, resume=True, **kwargs)
        return _load_jsonl(path) if path.exists() else current

    # --- Step 1: SPI ---
    log.info("Step 1/6: SPI-30")
    samples = _run_step(enrich_events_with_spi, samples, step_paths["spi"],
                        project=gee_project, max_workers=30)

    # --- Step 2: CHIRPS ---
    log.info("Step 2/6: CHIRPS")
    samples = _run_step(enrich_events_with_chirps, samples, step_paths["chirps"],
                        project=gee_project, max_workers=30)

    # --- Step 3: ERA5 ---
    log.info("Step 3/6: ERA5")
    samples = _run_step(enrich_events_with_era5, samples, step_paths["era5"],
                        project=gee_project, max_workers=30)

    # --- Step 4: GPM ---
    log.info("Step 4/6: GPM")
    samples = _run_step(enrich_events_with_gpm, samples, step_paths["gpm"],
                        project=gee_project, max_workers=30)

    # --- Step 5: Static features ---
    log.info("Step 5/6: Static (JRC + WorldPop)")
    samples = _run_step(enrich_events_with_static_features, samples, step_paths["static"],
                        project=gee_project, max_workers=30)

    # Write final merged output
    _save_jsonl(samples, output_path)
    log.info(f"Enriched baseline saved to {output_path}")
    return samples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Enrich baseline samples with hydro-climate indicators")
    parser.add_argument("--project", required=True, help="Google Earth Engine project ID")
    parser.add_argument("--input",   default=str(BASELINE_PATH), help="Input baseline JSONL")
    parser.add_argument("--output",  default=str(ENRICHED_PATH), help="Output enriched JSONL")
    args = parser.parse_args()

    enrich_baseline(args.project, Path(args.input), Path(args.output))
