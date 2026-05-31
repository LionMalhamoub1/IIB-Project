"""
Merge enriched reference flood events and baseline samples into a single
analysis-ready pandas DataFrame.

The reference dataset is spread across multiple JSONL files (one per
enrichment step). This module merges them by the event key
(source + source_id + date_start) and attaches label=1. Baseline
samples (label=0) are loaded from the enriched baseline JSONL.

Exported constants:
  INDICATOR_COLS  — ordered list of the six core indicators used throughout
                    the validation and modelling pipeline

Exported functions:
  load_flood_events()   -> pd.DataFrame  (label=1)
  load_baseline()       -> pd.DataFrame  (label=0)
  load_combined()       -> pd.DataFrame  (label = 0 or 1)
"""

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "cache" / "floods"

# Paths for each enrichment layer of the reference dataset
REFERENCE_LAYERS = {
    "spi":    CACHE / "reference_floods_spi.jsonl",
    "chirps": CACHE / "reference_floods_chirps.jsonl",
    "era5":   CACHE / "reference_floods_era5.jsonl",
    "gpm":    CACHE / "reference_floods_gpm.jsonl",
    "static": CACHE / "reference_floods_static.jsonl",
}

BASELINE_PATH = CACHE / "baseline_enriched.jsonl"

INDICATOR_COLS = [
    "spi_30d",
    "chirps_7d_anom_pct",
    "era5_soil_moisture_day0",
    "era5_soil_moisture_deep_day0",
    "era5_soil_moisture_deep_7d_mean",
    "gpm_peak_3h_mm",
    "jrc_recurrence_pct",
    "pop_density_km2",
    "terrain_slope_mean",
]

# Metadata columns carried through for diagnostics
META_COLS = [
    "source", "source_id", "country_iso", "country",
    "lat", "lon", "date_start",
]


def _event_key(row: dict) -> str:
    return f"{row.get('source','?')}|{row.get('source_id','?')}|{row.get('date_start','?')[:10]}"


def _load_layer(path: Path) -> dict[str, dict]:
    """Load one enrichment JSONL file into a key -> record dict."""
    records = {}
    if not path.exists():
        log.warning(f"Layer not found, skipping: {path.name}")
        return records
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records[_event_key(row)] = row
    return records


def load_flood_events() -> pd.DataFrame:
    """
    Load and merge all enrichment layers for the reference flood events.

    Each layer adds its own indicator columns. Events are merged by key
    (source + source_id + date_start). The static layer is used as the
    base because it has the widest coverage (18,567 events).

    Returns a DataFrame with label=1.
    """
    # Use static layer as anchor (widest coverage)
    base_layer = _load_layer(REFERENCE_LAYERS["static"])
    if not base_layer:
        raise FileNotFoundError(
            f"Base layer missing: {REFERENCE_LAYERS['static']}\n"
            "Run the reference enrichment pipeline first."
        )

    # Merge additional indicator columns from other layers
    for layer_name, layer_path in REFERENCE_LAYERS.items():
        if layer_name == "static":
            continue
        layer = _load_layer(layer_path)
        for key, record in layer.items():
            if key in base_layer:
                base_layer[key].update(record)

    rows = list(base_layer.values())
    df = pd.DataFrame(rows)
    df["label"] = 1

    # Coerce indicator columns to numeric
    for col in INDICATOR_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"Loaded {len(df)} flood events (label=1)")
    return df


def load_baseline() -> pd.DataFrame:
    """
    Load enriched baseline samples.

    Falls back gracefully if the baseline has not been enriched yet —
    returns the raw (unenriched) baseline if enriched file is missing.

    Returns a DataFrame with label=0.
    """
    path = BASELINE_PATH
    if not path.exists():
        # Try raw (unenriched) baseline
        raw_path = CACHE / "baseline_samples.jsonl"
        if raw_path.exists():
            log.warning(
                "Enriched baseline not found; loading raw baseline (no indicators). "
                "Run enrich_baseline.py first for full analysis."
            )
            path = raw_path
        else:
            raise FileNotFoundError(
                "Baseline not found. Run generate_baseline.py then enrich_baseline.py."
            )

    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    df["label"] = 0

    for col in INDICATOR_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"Loaded {len(df)} baseline samples (label=0)")
    return df


def load_combined() -> pd.DataFrame:
    """
    Load and concatenate flood events (label=1) and baseline (label=0).

    Keeps only INDICATOR_COLS + META_COLS + label so downstream scripts
    get a clean, consistent DataFrame regardless of which enrichment
    columns happen to be present.

    Returns:
        df  — combined DataFrame, balanced report printed to log
    """
    floods   = load_flood_events()
    baseline = load_baseline()

    # Align columns: keep only what both have
    keep = ["label"] + META_COLS + [c for c in INDICATOR_COLS if c in floods.columns or c in baseline.columns]
    floods_keep   = floods.reindex(  columns=keep)
    baseline_keep = baseline.reindex(columns=keep)

    df = pd.concat([floods_keep, baseline_keep], ignore_index=True)

    n_flood = (df["label"] == 1).sum()
    n_base  = (df["label"] == 0).sum()
    log.info(f"Combined dataset: {len(df)} rows  (floods={n_flood}, baseline={n_base})")

    coverage = {
        col: f"{df[col].notna().sum()}/{len(df)} ({100*df[col].notna().mean():.0f}%)"
        for col in INDICATOR_COLS if col in df.columns
    }
    for col, cov in coverage.items():
        log.info(f"  {col}: {cov}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = load_combined()
    print(df[["label"] + INDICATOR_COLS].describe())
