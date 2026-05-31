"""
split_by_type.py
================
Reads all_consolidated.jsonl and writes one JSONL file per disruption_type
into this directory, e.g.:
    flood.jsonl
    earthquake.jsonl
    protests.jsonl
    ...

Each output file is sorted by confidence descending so the highest-quality
records (best candidates for manual seeding / extras investigation) appear first.

Usage
-----
    python Builder_GDELT/results/combined/results_by_type/split_by_type.py
    python Builder_GDELT/results/combined/results_by_type/split_by_type.py --min-confidence 0.6
    python Builder_GDELT/results/combined/results_by_type/split_by_type.py --type flood earthquake
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE   = Path(__file__).resolve().parent
INPUT  = HERE.parent / "all_consolidated.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="Split all_consolidated.jsonl by disruption_type")
    parser.add_argument(
        "--min-confidence", type=float, default=0.0,
        help="Only include records with confidence >= this value (default: 0.0 = all)"
    )
    parser.add_argument(
        "--type", nargs="+", default=None,
        help="Only write files for these types (default: all types)"
    )
    args = parser.parse_args()

    if not INPUT.exists():
        raise FileNotFoundError(f"Input not found: {INPUT}")

    # Load and group
    by_type: dict[str, list[dict]] = defaultdict(list)
    total = 0
    skipped = 0
    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            conf = float(record.get("confidence") or 0)
            if conf < args.min_confidence:
                skipped += 1
                continue
            dtype = record.get("disruption_type") or "unknown"
            by_type[dtype].append(record)

    print(f"Loaded {total:,} records ({skipped:,} skipped by confidence filter)")
    print(f"Types found: {sorted(by_type)}\n")

    # Filter to requested types
    types_to_write = sorted(args.type) if args.type else sorted(by_type)

    for dtype in types_to_write:
        records = by_type.get(dtype, [])
        if not records:
            print(f"  {dtype}: 0 records — skipping")
            continue

        # Sort by confidence descending, then by event_date descending
        records.sort(key=lambda r: (-(r.get("confidence") or 0), -(r.get("event_date") or "").__len__()))

        out_path = HERE / f"{dtype}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Quick extras summary
        with_extras = sum(
            1 for r in records
            if isinstance(r.get("extras"), dict) and any(
                v is not None and v != "" and v != []
                for v in r["extras"].values()
            )
        )
        print(f"  {dtype:<22}  {len(records):>6,} records  |  {with_extras:>5,} have non-empty extras  ->  {out_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
