"""
analyse_details_coverage.py
===========================
Reads full-extraction JSONL files saved by validate_truncation.py
(in helper_scripts/full_extractions/full_<label>.jsonl) and reports
the fill-rate of every detail field we ask for in the LLM prompt.

This tells you whether each field is worth keeping in the prompt
or can be removed to save tokens.

Usage
-----
    python -m Builder_GDELT.helper_scripts.analyse_details_coverage
    python -m Builder_GDELT.helper_scripts.analyse_details_coverage --labels flood strike
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
FULL_EXTRACTIONS_DIR = HERE / "full_extractions"

# Fields we ask for in the LLM prompt, keyed by disruption type
EXPECTED_FIELDS: dict[str, list[str]] = {
    "flood": [
        "rainfall_intensity",
        "rainfall_levels",
        "death_toll",
        "main_cause",
    ],
    "drought": [
        "rainfall_deviation",
        "reservoir_level",
        "temperature_anomaly",
        "water_restrictions",
    ],
    "cyclone_hurricane": [
        "sea_surface_temp_anomaly",
        "storm_category",
        "wind_speed",
    ],
    "extreme_heat": [
        "temperature_anomaly",
        "power_grid_stress",
    ],
    "landslide": [
        "rainfall_intensity",
        "soil_moisture",
        "deforestation_activity",
    ],
    "earthquake": [
        "seismic_event_count",
        "max_magnitude",
        "foreshock_activity",
    ],
    "mine_accident": [
        "fatalities",
        "injuries",
        "equipment_failure",
    ],
    "labour_strike": [
        "protest_type",
        "protesting_groups",
        "organizations_or_companies",
        "target_of_protest",
        "issue",
        "sector",
        "estimated_participants",
        "event_start_day",
        "reported_day_number",
    ],
    "protests": [
        "protest_type",
        "protesting_groups",
        "organizations_or_companies",
        "target_of_protest",
        "issue",
        "sector",
        "estimated_participants",
        "event_start_day",
        "reported_day_number",
    ],
    "country_relations": [
        "casualties",
        "countries_involved",
    ],
    "trade_embargo": [
        "sanction_count",
        "trade_restrictiveness_index",
    ],
    "tariffs": [
        "tariff_rate",
        "affected_products_count",
        "affected_trade_value",
    ],
}


def _is_filled(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    if isinstance(value, (list, dict)) and not value:
        return False
    return True


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def analyse_label(label: str) -> None:
    path = FULL_EXTRACTIONS_DIR / f"full_{label}.jsonl"
    if not path.exists():
        print(f"\n[{label}] File not found: {path}")
        print(f"  Re-run validate_truncation.py with --label {label} to generate it.")
        return

    records = _load_jsonl(path)
    n_total = len(records)

    # Group by disruption_type as classified by the LLM
    by_type: dict[str, list[dict]] = {}
    for r in records:
        dtype = r.get("disruption_type", "unknown")
        by_type.setdefault(dtype, []).append(r)

    print(f"\n{'='*65}")
    print(f"  Label: {label}  —  {n_total} articles")
    type_counts = {k: len(v) for k, v in by_type.items()}
    for dtype, cnt in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {dtype}: {cnt}")
    print(f"{'='*65}")

    # For each disruption type present in this label's data
    for dtype in sorted(by_type.keys()):
        if dtype not in EXPECTED_FIELDS:
            continue
        records_of_type = by_type[dtype]
        n = len(records_of_type)
        expected = EXPECTED_FIELDS[dtype]

        print(f"\n  [{dtype}]  n={n}")
        print(f"  {'Field':<35} {'Filled':>8}  {'%':>6}  Verdict")
        print(f"  {'-'*35}  {'-'*8}  {'-'*6}  -------")

        unexpected_keys: Counter = Counter()
        field_counts: dict[str, int] = {f: 0 for f in expected}

        for r in records_of_type:
            details = r.get("details") or {}
            for field in expected:
                if _is_filled(details.get(field)):
                    field_counts[field] += 1
            for key in details:
                if key not in expected:
                    unexpected_keys[key] += 1

        for field in expected:
            count = field_counts[field]
            pct = count / n * 100
            verdict = "KEEP" if pct >= 10 else "DROP?"
            print(f"  {field:<35} {count:>5}/{n:<3}  {pct:>5.1f}%  {verdict}")

        if unexpected_keys:
            print(f"\n  Unexpected keys returned by model (not in prompt schema):")
            for key, cnt in unexpected_keys.most_common():
                print(f"    {key}: {cnt}/{n} ({100*cnt/n:.0f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Labels to analyse (e.g. flood strike earthquake). "
             "Default: all files found in full_extractions/",
    )
    args = parser.parse_args()

    if not FULL_EXTRACTIONS_DIR.exists():
        print(f"No full_extractions/ directory found at:\n  {FULL_EXTRACTIONS_DIR}")
        print("Re-run validate_truncation.py with --label <name> to generate files.")
        return

    if args.labels:
        labels = args.labels
    else:
        labels = sorted(p.stem.removeprefix("full_")
                        for p in FULL_EXTRACTIONS_DIR.glob("full_*.jsonl"))

    if not labels:
        print("No full_*.jsonl files found. Re-run validate_truncation.py first.")
        return

    print("\nDetails field coverage analysis")
    print(f"Source: {FULL_EXTRACTIONS_DIR}")
    print("Verdict: KEEP = filled >=10% of articles  |  DROP? = <10%")

    for label in labels:
        analyse_label(label)

    print("\nDone.")


if __name__ == "__main__":
    main()
