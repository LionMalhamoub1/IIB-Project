#!/usr/bin/env python3
"""
Vector population metrics (1): Feature fill-rate diagnostics + heatmap.

Reads:
  - Builder_GDELT/results/combined/all_consolidated.csv
    (falls back to .jsonl if csv missing)

Writes:
  - Builder_GDELT/Vector Metrics/feature_fill_rates_overall.csv
  - Builder_GDELT/Vector Metrics/feature_fill_rates_by_type_wide.csv
  - Builder_GDELT/Vector Metrics/feature_fill_rates_by_type_heatmap.png

Heatmap:
  - rows: disruption_type
  - cols (fixed order):
        urls
        num_articles
        confidence
        location_name
        publish_date
        event_date
        source_title
        extras
        duration_hours
  - values: filled % (0-100)
"""

from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt


# PATHS

COMBINED_RESULTS_DIR = Path("Builder_GDELT/results/combined")
INPUT_STEM = "all_consolidated"

OUT_DIR = Path("Builder_GDELT") / "Vector Metrics" / "feature_fill_rate_diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# LOAD

def _load_all_consolidated() -> pd.DataFrame:
    csv_path = COMBINED_RESULTS_DIR / f"{INPUT_STEM}.csv"
    jsonl_path = COMBINED_RESULTS_DIR / f"{INPUT_STEM}.jsonl"

    if csv_path.exists():
        return pd.read_csv(csv_path)
    if jsonl_path.exists():
        return pd.read_json(jsonl_path, lines=True)

    raise FileNotFoundError(
        f"Could not find {csv_path.name} or {jsonl_path.name}"
    )


# HELPERS

def _is_nan(x: Any) -> bool:
    try:
        return isinstance(x, float) and math.isnan(x)
    except Exception:
        return False


def _parse_json_like(x: Any) -> Any:
    if x is None or _is_nan(x):
        return None
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return ""
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return x
        return x
    return x


def _filled_mask(series: pd.Series, feature: str) -> pd.Series:
    s = series.map(_parse_json_like)

    if feature in {"event_date", "publish_date"}:
        return (
            s.notna()
            & s.astype(str).str.strip().ne("")
            & s.astype(str).str.lower().ne("null")
        )

    if feature in {"location_name", "source_title"}:
        return s.notna() & s.astype(str).str.strip().ne("")

    if feature in {"duration_hours", "confidence", "num_articles"}:
        return s.notna()

    if feature == "urls":
        def _urls_filled(v):
            v = _parse_json_like(v)
            if v is None or _is_nan(v):
                return False
            if isinstance(v, list):
                return len(v) > 0
            if isinstance(v, str):
                return v.strip() != ""
            return False
        return s.map(_urls_filled)

    if feature == "extras":
        def _extras_filled(v):
            v = _parse_json_like(v)
            if v is None or _is_nan(v):
                return False
            if isinstance(v, dict):
                return len(v) > 0
            if isinstance(v, str):
                t = v.strip()
                if not t or t == "{}":
                    return False
                return True
            return False
        return s.map(_extras_filled)

    return s.notna() & s.astype(str).str.strip().ne("")


def _compute_fill_rates(df: pd.DataFrame, features: list[str]) -> dict:
    n = len(df)
    result = {}
    for f in features:
        if f not in df.columns:
            result[f] = 0.0
            continue
        mask = _filled_mask(df[f], f)
        filled_pct = (mask.sum() / n * 100.0) if n else 0.0
        result[f] = round(float(filled_pct), 2)
    return result


# HEATMAP

def _plot_heatmap(matrix: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(12, max(6, 0.4 * len(matrix))))
    im = plt.imshow(matrix.values, aspect="auto", interpolation="nearest")

    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(matrix.index)), matrix.index)

    cbar = plt.colorbar(im)
    cbar.set_label("Filled (%)")

    plt.title("Feature fill rate by disruption type")
    plt.xlabel("Feature")
    plt.ylabel("Disruption type")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# MAIN

def main():
    df = _load_all_consolidated()

    if "disruption_type" not in df.columns:
        raise ValueError("Column 'disruption_type' missing.")

    # FIXED FEATURE ORDER
    features = [
        "urls",
        "num_articles",
        "confidence",
        "source_title",
        "location_name",
        "publish_date",
        "event_date",
        "extras",
        "duration_hours",
    ]

    rows = []

    for dtype, g in df.groupby(df["disruption_type"].fillna("unknown").astype(str)):
        row = {"disruption_type": dtype, "n": len(g)}
        fill_rates = _compute_fill_rates(g, features)
        row.update(fill_rates)
        rows.append(row)

    wide_df = pd.DataFrame(rows)
    wide_df = wide_df.sort_values("n", ascending=False)

    # Save wide CSV
    wide_out = OUT_DIR / "feature_fill_rates_by_type_wide.csv"
    wide_df.to_csv(wide_out, index=False)

    # Prepare heatmap matrix (remove n column)
    heatmap_df = wide_df.set_index("disruption_type")[features]

    heatmap_out = OUT_DIR / "feature_fill_rates_by_type_heatmap.png"
    _plot_heatmap(heatmap_df, heatmap_out)

    print(f"Wrote: {wide_out}")
    print(f"Wrote: {heatmap_out}")


if __name__ == "__main__":
    main()