#!/usr/bin/env python3
"""
Vector population metrics (2): Per-record vector density + distributions.

Idea:
  Each record is a "vector" with multiple features. We score how populated it is.

Density score (unweighted):
  density = (# filled features) / (# considered features)

Also writes:
  - density histogram (overall)
  - density boxplot by disruption_type
  - summary table by disruption_type

Reads:
  - Builder_GDELT/results/combined/all_consolidated.csv
    (falls back to .jsonl if csv missing)

Writes:
  - Builder_GDELT/Vector Metrics/vector_density_records.csv
  - Builder_GDELT/Vector Metrics/vector_density_by_type.csv
  - Builder_GDELT/Vector Metrics/vector_density_hist.png
  - Builder_GDELT/Vector Metrics/vector_density_boxplot_by_type.png
"""

from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any, Dict, List

import pandas as pd
import matplotlib.pyplot as plt


# PATHS

COMBINED_RESULTS_DIR = Path("Builder_GDELT/results/combined")
INPUT_STEM = "all_consolidated"

OUT_DIR = Path("Builder_GDELT") / "Vector Metrics" / "vector_densities_distributions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# LOAD

def _load_all_consolidated() -> pd.DataFrame:
    csv_path = COMBINED_RESULTS_DIR / f"{INPUT_STEM}.csv"
    jsonl_path = COMBINED_RESULTS_DIR / f"{INPUT_STEM}.jsonl"

    if csv_path.exists():
        return pd.read_csv(csv_path)
    if jsonl_path.exists():
        return pd.read_json(jsonl_path, lines=True)

    raise FileNotFoundError(f"Could not find {csv_path.name} or {jsonl_path.name}")


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


def _filled_value(v: Any, feature: str) -> bool:
    v = _parse_json_like(v)

    if v is None or _is_nan(v):
        return False

    if feature in {"event_date", "publish_date"}:
        s = str(v).strip()
        return s != "" and s.lower() != "null"

    if feature in {"location_name", "source_title"}:
        return str(v).strip() != ""

    if feature in {"duration_hours", "confidence", "num_articles"}:
        return True  # non-null already checked

    if feature == "urls":
        if isinstance(v, list):
            return len(v) > 0
        if isinstance(v, str):
            return v.strip() != ""
        return False

    if feature == "extras":
        if isinstance(v, dict):
            return len(v) > 0
        if isinstance(v, str):
            t = v.strip()
            return t != "" and t != "{}"
        return False

    return str(v).strip() != ""


# MAIN

def main() -> None:
    df = _load_all_consolidated()

    if "disruption_type" not in df.columns:
        raise ValueError("Column 'disruption_type' missing from all_consolidated.")

    features = [
        "urls",
        "num_articles",
        "confidence",
        "location_name",
        "publish_date",
        "event_date",
        "source_title",
        "extras",
        "duration_hours",
    ]

    # ensure all columns exist (missing columns treated as always empty)
    for f in features:
        if f not in df.columns:
            df[f] = None

    filled_counts: List[int] = []
    for _, row in df.iterrows():
        c = 0
        for f in features:
            if _filled_value(row[f], f):
                c += 1
        filled_counts.append(c)

    df_out = pd.DataFrame({
        "disruption_type": df["disruption_type"].fillna("unknown").astype(str),
        "filled_features": filled_counts,
        "total_features": len(features),
        "density": [c / len(features) for c in filled_counts],
        "confidence": df["confidence"] if "confidence" in df.columns else None,
        "num_articles": df["num_articles"] if "num_articles" in df.columns else None,
    })

    # save per-record densities
    records_out = OUT_DIR / "vector_density_records.csv"
    df_out.to_csv(records_out, index=False)

    # summary by type
    by_type = (
        df_out.groupby("disruption_type")
        .agg(
            n=("density", "size"),
            density_mean=("density", "mean"),
            density_median=("density", "median"),
            density_p10=("density", lambda s: s.quantile(0.10)),
            density_p90=("density", lambda s: s.quantile(0.90)),
            filled_mean=("filled_features", "mean"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )

    by_type_out = OUT_DIR / "vector_density_by_type.csv"
    by_type.to_csv(by_type_out, index=False)

    # histogram overall
    hist_out = OUT_DIR / "vector_density_hist.png"
    plt.figure()
    plt.hist(df_out["density"], bins=20)
    plt.xlabel("Vector density (filled / total)")
    plt.ylabel("Count")
    plt.title("Vector density distribution (overall)")
    plt.tight_layout()
    plt.savefig(hist_out, dpi=200)
    plt.close()

    # boxplot by type (top 20 types by n for readability)
    top_types = by_type["disruption_type"].head(20).tolist()
    plot_df = df_out[df_out["disruption_type"].isin(top_types)].copy()

    box_out = OUT_DIR / "vector_density_boxplot_by_type.png"
    plt.figure(figsize=(max(10, 0.6 * len(top_types) + 4), 6))
    data = [plot_df.loc[plot_df["disruption_type"] == t, "density"].values for t in top_types]
    plt.boxplot(data, labels=top_types, vert=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Vector density (filled / total)")
    plt.title("Vector density by disruption type (top 20 by count)")
    plt.tight_layout()
    plt.savefig(box_out, dpi=200)
    plt.close()

    print(f"Wrote: {records_out}")
    print(f"Wrote: {by_type_out}")
    print(f"Wrote: {hist_out}")
    print(f"Wrote: {box_out}")


if __name__ == "__main__":
    main()