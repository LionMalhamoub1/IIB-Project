"""
Quick inspection of the combined or consolidated flood reference dataset.

Usage:
    python -m Builder_Reference.helper_scripts.reference.inspect                      # auto-detects file
    python -m Builder_Reference.helper_scripts.reference.inspect --file path/to.jsonl
    python -m Builder_Reference.helper_scripts.reference.inspect --n 50              # show more rows
"""

import argparse
import json
from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 35)


DEFAULT_PATHS = [
    Path("cache/floods/reference_floods_enriched.jsonl"),
    Path("cache/floods/reference_floods_consolidated.jsonl"),
    Path("cache/floods/reference_floods_combined.jsonl"),
]

DISPLAY_COLS = [
    "source",
    "matched_sources",
    "country",
    "country_iso",
    "date_start",
    "date_end",
    "dead",
    "displaced",
    "affected",
    "damage_usd_thousands",
    "severity",
    "main_cause",
    "event_name",
    "glide_number",
]


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--n", type=int, default=30, help="Rows to display")
    args = parser.parse_args()

    path = args.file
    if path is None:
        for p in DEFAULT_PATHS:
            if p.exists():
                path = p
                break

    if path is None or not path.exists():
        print("No reference file found. Run combine/consolidate first.")
        return

    print(f"\nLoading: {path}")
    df = load_jsonl(path)
    print(f"Total records: {len(df):,}\n")

    # --- Source breakdown ---
    if "source" in df.columns:
        print("=== Source breakdown ===")
        print(df["source"].value_counts().to_string())
        print()

    # --- Matched sources breakdown (consolidated only) ---
    if "matched_sources" in df.columns:
        df["_src_key"] = df["matched_sources"].apply(
            lambda v: "+".join(sorted(v)) if isinstance(v, list) else str(v)
        )
        print("=== Matched-source combinations ===")
        print(df["_src_key"].value_counts().to_string())
        print()
        df.drop(columns=["_src_key"], inplace=True)

    # --- Field fill rates ---
    enrichment_cols = [
        "country_iso", "date_start", "date_end",
        "lat", "lon", "dead", "displaced", "affected",
        "damage_usd_thousands", "severity", "glide_number",
        "spi_30d", "chirps_7d_total_mm", "gpm_1d_total_mm",
        "era5_soil_moisture_day0", "pop_density_km2",
        "jrc_recurrence_pct", "glofas_discharge_m3s",
    ]
    present = [c for c in enrichment_cols if c in df.columns]
    fill = (df[present].notna().sum() / len(df) * 100).round(1)
    print("=== Field fill rates (%) ===")
    print(fill.to_string())
    print()

    # --- Sample rows ---
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    print(f"=== Sample rows (first {args.n}) ===")
    print(df[cols].head(args.n).to_string(index=False))


if __name__ == "__main__":
    main()
