"""
build_official_reference_dataset.py
====================================
Merges all enrichment layers into a single, analysis-ready JSONL file that
constitutes the official reference flood dataset.

Design rationale
----------------
The enrichment pipeline runs each data source (CHIRPS, GPM, ERA5, static
features, SPI) as an independent job that writes its own JSONL.  This
separation keeps jobs resumable and avoids re-running everything when one
source fails.  This script is the final assembly step: it left-joins every
enrichment layer onto the geocoded base file, producing one flat record per
event with all ~40 fields present.

Join key: ``source|source_id|date_start``
  - ``source``     — originating database (DFO, GDACS, EM-DAT, …)
  - ``source_id``  — original record ID within that database
  - ``date_start`` — truncated to YYYY-MM-DD to handle ISO timestamps

Events that are in the base file but missing from an enrichment layer (e.g.
outside CHIRPS latitude coverage) receive null
values for that layer's fields rather than being dropped.  This ensures the
output always has the same number of rows as the geocoded input.

Private ``_<layer>_processed`` flags from each enrichment JSONL are stripped
from the final output to keep the schema clean.

Usage
-----
    python -m Builder_Reference.helper_scripts.enrichment.build_official_reference_dataset

Output
------
    cache/floods/reference_floods_enriched.jsonl
"""

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
CACHE = ROOT / "cache" / "floods"
OUTPUTS = ROOT / "Builder_Reference" / "outputs"

BASE_PATH = CACHE / "reference_floods_geocoded.jsonl"
OUTPUT_PATH = OUTPUTS / "reference_floods_enriched.jsonl"

# Enrichment layers in join order. Each tuple is (jsonl_path, flag_field).
ENRICHMENT_LAYERS = [
    (CACHE / "reference_floods_chirps.jsonl",  "_chirps_processed"),
    (CACHE / "reference_floods_gpm.jsonl",     "_gpm_processed"),
    (CACHE / "reference_floods_era5.jsonl",    "_era5_processed"),
    (CACHE / "reference_floods_static.jsonl",  "_static_processed"),
    (CACHE / "reference_floods_spi.jsonl",     "_spi_processed"),
]


def _event_key(event: dict) -> str:
    """Stable join key: source|source_id|date_start (date part only)."""
    return (
        f"{event.get('source', '?')}|"
        f"{event.get('source_id', '?')}|"
        f"{str(event.get('date_start', '?'))[:10]}"
    )


def _load_enrichment_index(path: Path, flag_field: str) -> dict:
    """
    Load an enrichment JSONL into a dict keyed by event key.
    Only enrichment fields (not base fields or private flags) are kept,
    so that the join adds new columns without overwriting base data.
    """
    BASE_FIELDS = {
        "source", "source_id", "date_start", "date_end", "country",
        "country_iso", "all_country_iso", "lat", "lon", "location_name",
        "region", "geocode_source", "dead", "injured", "displaced",
        "affected", "indirectly_affected", "houses_destroyed", "houses_damaged",
        "roads_km", "damage_usd_thousands", "severity", "main_cause",
        "event_name", "area_km2", "glide_number",
    }
    index = {}
    if not path.exists():
        log.warning(f"Enrichment file not found, skipping: {path.name}")
        return index

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            key = _event_key(row)
            enrichment_fields = {
                k: v for k, v in row.items()
                if k not in BASE_FIELDS and not k.startswith("_")
            }
            index[key] = enrichment_fields

    log.info(f"  Loaded {len(index):,} records from {path.name}")
    return index


def build_official_reference_dataset():
    log.info(f"Loading base: {BASE_PATH}")
    base_events = []
    with BASE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            base_events.append(json.loads(line))
    log.info(f"  {len(base_events):,} base events")

    # Load all enrichment layers
    log.info("Loading enrichment layers...")
    layers = []
    for path, flag in ENRICHMENT_LAYERS:
        idx = _load_enrichment_index(path, flag)
        layers.append(idx)

    # Merge
    log.info("Merging...")
    merged = []
    n_partial = 0
    for event in base_events:
        key = _event_key(event)
        record = dict(event)
        has_all = True
        for idx in layers:
            if key in idx:
                record.update(idx[key])
            else:
                has_all = False
        if not has_all:
            n_partial += 1
        merged.append(record)

    n_complete = len(merged) - n_partial
    log.info(
        f"Merge complete: {len(merged):,} total events | "
        f"{n_complete:,} fully enriched | {n_partial:,} partially enriched"
    )

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for record in merged:
            f.write(json.dumps(record, default=str) + "\n")

    log.info(f"Written: {OUTPUT_PATH}")


if __name__ == "__main__":
    build_official_reference_dataset()
