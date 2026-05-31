"""
build_combined.py
=================
Assembles the combined flood event dataset from the reference and GDELT pipelines.

Run at any time to rebuild from the current state of all upstream outputs:

    python Builder_Combined/build_combined.py

Inputs (must exist before running):
    Builder_Reference/outputs/reference_floods_enriched.jsonl
    Builder_Matching/outputs/gdelt_floods_matched.jsonl
    Builder_Matching/outputs/match_index.json          <- written by run_matching.py

Outputs:
    Builder_Combined/outputs/combined_floods.jsonl     (full fidelity, streaming)
    Builder_Combined/outputs/combined_floods.csv       (flat, pandas/Excel friendly)
    Builder_Combined/outputs/build_summary.json        (counts and coverage stats)

See APPROACH.txt for design rationale.
"""

import csv
import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parent.parent
REF_PATH   = ROOT / "Builder_Reference" / "outputs" / "reference_floods_enriched.jsonl"
GDELT_PATH = ROOT / "Builder_Matching"  / "outputs" / "gdelt_floods_matched.jsonl"
INDEX_PATH = ROOT / "Builder_Matching"  / "outputs" / "match_index.json"
OUT_DIR    = ROOT / "Builder_Combined"  / "outputs"

# ── Shared hydro-climate fields (identical names in both datasets) ──────────
HYDRO_FIELDS = [
    "chirps_3d_total_mm", "chirps_7d_total_mm", "chirps_14d_total_mm",
    "chirps_30d_total_mm", "chirps_peak_daily_mm", "chirps_7d_baseline_mm",
    "chirps_7d_anom_mm",  "chirps_7d_anom_pct",
    "gpm_1d_total_mm",    "gpm_3d_total_mm",   "gpm_7d_total_mm",
    "gpm_peak_daily_mm",  "gpm_peak_3h_mm",
    "era5_soil_moisture_day0",      "era5_soil_moisture_7d_mean",
    "era5_soil_moisture_30d_mean",  "era5_soil_moisture_deep_day0",
    "era5_soil_moisture_deep_7d_mean", "era5_precip_7d_mm", "era5_runoff_7d_mm",
    "pop_count_25km", "pop_density_km2",
    "jrc_occurrence_pct", "jrc_recurrence_pct", "terrain_slope_mean",
    "spi_30d", "spi_30d_pct",
]

# ── Reference-specific scalar fields ───────────────────────────────────────
REF_SCALAR_FIELDS = [
    "source", "glide_number",
    "date_start", "date_end", "country", "country_iso", "region",
    "lat", "lon", "location_name", "area_km2",
    "dead", "injured", "displaced", "affected", "indirectly_affected",
    "houses_destroyed", "houses_damaged", "roads_km",
    "damage_usd_thousands", "damage_eur2020_thousands",
    "severity", "main_cause", "event_name", "geocode_source",
]

# ── GDELT-specific scalar fields ────────────────────────────────────────────
GDELT_SCALAR_FIELDS = [
    "url", "source_title",
    "event_date", "publish_date", "location_name", "lat", "lon",
    "duration_hours", "severity", "confidence",
    "disruption_type", "classification_source",
    "llm_disruption_type", "expert_disruption_type", "expert_probability",
]

# extras sub-fields (nested dict in GDELT records)
GDELT_EXTRAS_FIELDS = [
    "rainfall_intensity", "rainfall_levels",
    "death_toll", "main_cause", "glide_number", "event_start_day",
]


# ── ID helpers ──────────────────────────────────────────────────────────────

def _ref_cluster_id(ref: dict) -> str:
    """
    Content-based cluster ID for a reference event, stable across rebuilds
    as long as the geocoding and date are unchanged.
    Groups all GDELT rows that matched this reference event.
    """
    key = "|".join([
        (ref.get("date_start") or "")[:10],
        ref.get("country_iso") or ref.get("country") or "",
        str(round(ref.get("lat") or 0.0, 2)),
        str(round(ref.get("lon") or 0.0, 2)),
    ])
    return "ref_" + hashlib.sha1(key.encode()).hexdigest()[:12]


def _gdelt_event_id(gdelt: dict) -> str:
    """Stable row ID for a GDELT event, derived from the article URL."""
    return "gdelt_" + hashlib.sha1((gdelt.get("url") or "").encode()).hexdigest()[:12]


# ── Row builder ─────────────────────────────────────────────────────────────

def _build_row(
    row_type: str,
    cluster_id: str,
    event_id: str,
    gdelt: dict | None,
    ref: dict | None,
    match_score: float | None,
    n_gdelt_in_cluster: int,
) -> dict:
    row: dict = {
        # ── Identity / match metadata ──────────────────────────────────────
        "row_type":           row_type,
        "cluster_id":         cluster_id,
        "event_id":           event_id,
        "matched":            row_type == "matched_gdelt",
        "match_score":        match_score,
        "n_gdelt_in_cluster": n_gdelt_in_cluster,
        "has_ref":            ref   is not None,
        "has_gdelt":          gdelt is not None,
    }

    # ── Canonical fields (reference preferred; fallback to GDELT) ──────────
    if ref is not None:
        row["canonical_date_start"]    = (ref.get("date_start") or "")[:10] or None
        row["canonical_date_end"]      = (ref.get("date_end")   or "")[:10] or None
        row["canonical_lat"]           = ref.get("lat")
        row["canonical_lon"]           = ref.get("lon")
        row["canonical_country"]       = ref.get("country")
        row["canonical_country_iso"]   = ref.get("country_iso")
        row["canonical_location_name"] = ref.get("location_name")
    else:
        row["canonical_date_start"]    = (gdelt.get("event_date") or "")[:10] or None
        row["canonical_date_end"]      = None
        row["canonical_lat"]           = gdelt.get("lat")
        row["canonical_lon"]           = gdelt.get("lon")
        row["canonical_country"]       = None
        row["canonical_country_iso"]   = None
        row["canonical_location_name"] = gdelt.get("location_name")

    # ── Reference fields ───────────────────────────────────────────────────
    for field in REF_SCALAR_FIELDS + HYDRO_FIELDS:
        row[f"ref_{field}"] = ref.get(field) if ref is not None else None

    # List/dict fields from reference (stored as-is for JSONL; serialised for CSV)
    row["ref_matched_sources"] = ref.get("matched_sources") if ref is not None else None
    row["ref_source_ids"]      = ref.get("source_ids")      if ref is not None else None

    # ── GDELT fields ───────────────────────────────────────────────────────
    for field in GDELT_SCALAR_FIELDS + HYDRO_FIELDS:
        row[f"gdelt_{field}"] = gdelt.get(field) if gdelt is not None else None

    extras = (gdelt.get("extras") or {}) if gdelt is not None else {}
    for field in GDELT_EXTRAS_FIELDS:
        row[f"gdelt_extras_{field}"] = extras.get(field)

    return row


# ── CSV serialiser (flatten nested types to JSON strings) ───────────────────

def _csv_value(v) -> str | None:
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v


# ── Main ────────────────────────────────────────────────────────────────────

def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validate inputs
    for path in (REF_PATH, GDELT_PATH, INDEX_PATH):
        if not path.exists():
            log.error(f"Missing input: {path}")
            if path == INDEX_PATH:
                log.error("Run Builder_Matching/matching/run_matching.py first to generate match_index.json")
            sys.exit(1)

    # ── Load ────────────────────────────────────────────────────────────────
    log.info(f"Loading reference: {REF_PATH}")
    ref_events: list[dict] = []
    with REF_PATH.open(encoding="utf-8") as f:
        for line in f:
            ref_events.append(json.loads(line))
    log.info(f"  {len(ref_events):,} reference events")

    log.info(f"Loading GDELT matched: {GDELT_PATH}")
    gdelt_events: list[dict] = []
    with GDELT_PATH.open(encoding="utf-8") as f:
        for line in f:
            gdelt_events.append(json.loads(line))
    log.info(f"  {len(gdelt_events):,} GDELT events")

    log.info(f"Loading match index: {INDEX_PATH}")
    with INDEX_PATH.open(encoding="utf-8") as f:
        raw_index: dict = json.load(f)
    # Positional mapping: gdelt array index → ref array index
    match_index: dict[int, int] = {int(k): int(v) for k, v in raw_index.items()}
    log.info(f"  {len(match_index):,} matched pairs")

    # ── Pre-compute cluster IDs and reverse mapping ──────────────────────────
    ref_cluster_ids = [_ref_cluster_id(r) for r in ref_events]

    # ref_idx → sorted list of gdelt_idx (so cluster members are deterministic)
    ref_to_gdelt: dict[int, list[int]] = {}
    for g_idx, r_idx in match_index.items():
        ref_to_gdelt.setdefault(r_idx, []).append(g_idx)

    matched_ref_idxs   = set(match_index.values())
    matched_gdelt_idxs = set(match_index.keys())

    # ── Build rows ──────────────────────────────────────────────────────────
    rows: list[dict] = []

    # 1. Matched GDELT rows (one per GDELT event, even if multiple share a ref)
    for g_idx, r_idx in match_index.items():
        gdelt   = gdelt_events[g_idx]
        ref     = ref_events[r_idx]
        cid     = ref_cluster_ids[r_idx]
        n_gdelt = len(ref_to_gdelt[r_idx])
        rows.append(_build_row(
            row_type           = "matched_gdelt",
            cluster_id         = cid,
            event_id           = _gdelt_event_id(gdelt),
            gdelt              = gdelt,
            ref                = ref,
            match_score        = gdelt.get("match_score"),
            n_gdelt_in_cluster = n_gdelt,
        ))

    # 2. Unmatched GDELT rows
    for g_idx, gdelt in enumerate(gdelt_events):
        if g_idx in matched_gdelt_idxs:
            continue
        eid = _gdelt_event_id(gdelt)
        rows.append(_build_row(
            row_type           = "unmatched_gdelt",
            cluster_id         = eid,
            event_id           = eid,
            gdelt              = gdelt,
            ref                = None,
            match_score        = None,
            n_gdelt_in_cluster = 1,
        ))

    # 3. Unmatched reference rows
    for r_idx, ref in enumerate(ref_events):
        if r_idx in matched_ref_idxs:
            continue
        cid = ref_cluster_ids[r_idx]
        rows.append(_build_row(
            row_type           = "unmatched_ref",
            cluster_id         = cid,
            event_id           = cid,
            gdelt              = None,
            ref                = ref,
            match_score        = None,
            n_gdelt_in_cluster = 0,
        ))

    n_matched_gdelt   = len(match_index)
    n_unmatched_gdelt = len(gdelt_events) - len(matched_gdelt_idxs)
    n_unmatched_ref   = len(ref_events)   - len(matched_ref_idxs)

    log.info(
        f"Combined rows: {len(rows):,} total  "
        f"({n_matched_gdelt:,} matched_gdelt | "
        f"{n_unmatched_gdelt:,} unmatched_gdelt | "
        f"{n_unmatched_ref:,} unmatched_ref)"
    )

    # Sort by canonical date (ascending) for readability
    rows.sort(key=lambda r: r.get("canonical_date_start") or "")

    # ── Write JSONL ─────────────────────────────────────────────────────────
    jsonl_path = OUT_DIR / "combined_floods.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    log.info(f"Written: {jsonl_path}")

    # ── Write CSV (flat; nested types as JSON strings) ──────────────────────
    csv_path = OUT_DIR / "combined_floods.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: _csv_value(v) for k, v in row.items()})
    log.info(f"Written: {csv_path}")

    # ── Coverage summary ─────────────────────────────────────────────────────
    hydro_ref_filled  = sum(
        1 for r in rows
        if r["row_type"] in ("matched_gdelt", "unmatched_ref")
        and r.get("ref_chirps_7d_total_mm") is not None
    )
    hydro_gdelt_filled = sum(
        1 for r in rows
        if r["row_type"] in ("matched_gdelt", "unmatched_gdelt")
        and r.get("gdelt_chirps_7d_total_mm") is not None
    )
    has_ref_rows   = n_matched_gdelt + n_unmatched_ref
    has_gdelt_rows = n_matched_gdelt + n_unmatched_gdelt

    # GDELT date window: the range actually covered by the GDELT dataset.
    # Reference spans much longer; only events inside this window were ever
    # candidates for matching, so the within-window recall is the meaningful metric.
    gdelt_dates = sorted(
        e.get("event_date", "")[:10] for e in gdelt_events if e.get("event_date")
    )
    gdelt_window_start = gdelt_dates[0]  if gdelt_dates else None
    gdelt_window_end   = gdelt_dates[-1] if gdelt_dates else None

    ref_in_window = sum(
        1 for r in ref_events
        if gdelt_window_start
        and gdelt_window_start <= (r.get("date_start") or "")[:10] <= gdelt_window_end
    )

    summary = {
        # Overall counts
        "ref_events_total":               len(ref_events),
        "ref_events_matched":             len(matched_ref_idxs),
        "ref_events_unmatched":           n_unmatched_ref,
        "gdelt_events_total":             len(gdelt_events),
        "gdelt_events_matched":           n_matched_gdelt,
        "gdelt_events_unmatched":         n_unmatched_gdelt,
        "combined_rows_total":            len(rows),
        # Within-window rates (meaningful: ref outside GDELT window can never match)
        "gdelt_window":                   f"{gdelt_window_start} to {gdelt_window_end}",
        "ref_events_in_gdelt_window":     ref_in_window,
        "ref_recall_in_window_pct":
            round(100 * len(matched_ref_idxs) / ref_in_window, 1) if ref_in_window else None,
        # Whole-dataset rates (denominator includes ref events outside GDELT window)
        "ref_match_rate_overall_pct":     round(100 * len(matched_ref_idxs) / len(ref_events), 1),
        "gdelt_match_rate_pct":           round(100 * n_matched_gdelt / len(gdelt_events), 1),
        # Hydro completeness
        "ref_hydro_completeness_pct":
            round(100 * hydro_ref_filled  / has_ref_rows,   1) if has_ref_rows   else None,
        "gdelt_hydro_completeness_pct":
            round(100 * hydro_gdelt_filled / has_gdelt_rows, 1) if has_gdelt_rows else None,
    }

    summary_path = OUT_DIR / "build_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Written: {summary_path}")

    log.info(
        f"Within-window recall: {summary['ref_recall_in_window_pct']}% "
        f"({len(matched_ref_idxs)} / {ref_in_window} ref events in {summary['gdelt_window']}) | "
        f"GDELT matched: {summary['gdelt_match_rate_pct']}% | "
        f"ref hydro complete: {summary['ref_hydro_completeness_pct']}% | "
        f"GDELT hydro complete: {summary['gdelt_hydro_completeness_pct']}%"
    )
    log.info("Done.")
    return summary


if __name__ == "__main__":
    build()
