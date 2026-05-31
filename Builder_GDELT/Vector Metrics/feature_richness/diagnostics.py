#!/usr/bin/env python3
"""
Vector population metrics (3): URL aggregation strength (clean, minimal).

Goal:
  Show how often records are single-source vs multi-source, and the shape of the
  url_count distribution (heavy mass at 1 + long tail).

Reads:
  - Builder_GDELT/results/combined/all_consolidated.csv
    (falls back to .jsonl if csv missing)

Writes:
  - Builder_GDELT/Vector Metrics/url_count_per_record.csv
  - Builder_GDELT/Vector Metrics/url_count_distribution_logy.png
  - Builder_GDELT/Vector Metrics/url_count_buckets_overall.png
  - Builder_GDELT/Vector Metrics/url_count_buckets_by_type_stacked.png

Plots:
  1) url_count_distribution_logy.png
     Frequency by exact url_count with log y-scale.

  2) url_count_buckets_overall.png
     Overall bucketed %: 1, 2-3, 4-5, 6+

  3) url_count_buckets_by_type_stacked.png
     Stacked bar chart by disruption_type (top 20 by count):
       segments = buckets (1, 2-3, 4-5, 6+)
       bar height = 100% (composition within each type)
"""

from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Any, List

import pandas as pd
import matplotlib.pyplot as plt


# ------------------ PATHS ------------------ #

COMBINED_RESULTS_DIR = Path("Builder_GDELT/results/combined")
INPUT_STEM = "all_consolidated"

OUT_DIR = Path("Builder_GDELT") / "Vector Metrics" /"feature_richness"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------ LOAD ------------------ #

def _load_all_consolidated() -> pd.DataFrame:
    csv_path = COMBINED_RESULTS_DIR / f"{INPUT_STEM}.csv"
    jsonl_path = COMBINED_RESULTS_DIR / f"{INPUT_STEM}.jsonl"

    if csv_path.exists():
        return pd.read_csv(csv_path)
    if jsonl_path.exists():
        return pd.read_json(jsonl_path, lines=True)

    raise FileNotFoundError(f"Could not find {csv_path.name} or {jsonl_path.name}")


# ------------------ HELPERS ------------------ #

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
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                return json.loads(s)
            except Exception:
                return x
        return x
    return x


def _count_urls(v: Any) -> int:
    v = _parse_json_like(v)
    if v is None or _is_nan(v):
        return 0
    if isinstance(v, list):
        return len(v)
    if isinstance(v, str):
        return 1 if v.strip() else 0
    return 0


def _bucket_label(url_count: int) -> str:
    if url_count <= 1:
        return "1"
    if 2 <= url_count <= 3:
        return "2-3"
    if 4 <= url_count <= 5:
        return "4-5"
    return "6+"


# ------------------ PLOTS ------------------ #

def _plot_logfreq(url_counts: pd.Series, out_path: Path) -> None:
    freq = url_counts.value_counts().sort_index()
    x = freq.index.astype(int).tolist()
    y = freq.values.tolist()

    plt.figure(figsize=(10, 5))
    plt.bar([str(v) for v in x], y)
    plt.yscale("log")
    plt.title("URL count per record (frequency; log y-scale)")
    plt.xlabel("url_count")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_bucket_overall(buckets: pd.Series, out_path: Path) -> None:
    bucket_order = ["1", "2-3", "4-5", "6+"]
    pct = (buckets.value_counts(normalize=True) * 100.0).reindex(bucket_order).fillna(0.0)

    plt.figure(figsize=(8, 5))
    plt.bar(bucket_order, pct.tolist())
    plt.ylim(0, 100)
    plt.title("URL aggregation buckets (overall)")
    plt.xlabel("url_count bucket")
    plt.ylabel("Percent of records (%)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _plot_bucket_stacked_by_type(df: pd.DataFrame, out_path: Path, top_n_types: int = 20) -> None:
    """
    100% stacked bars for top-N disruption types by count.
    Each bar shows the composition across url_count buckets.
    """
    bucket_order = ["1", "2-3", "4-5", "6+"]

    type_counts = df["disruption_type"].value_counts()
    top_types = type_counts.head(top_n_types).index.tolist()

    d = df[df["disruption_type"].isin(top_types)].copy()

    # counts per type x bucket
    ct = pd.crosstab(d["disruption_type"], d["url_bucket"]).reindex(index=top_types, columns=bucket_order, fill_value=0)

    # convert to % within type
    pct = ct.div(ct.sum(axis=1).replace(0, 1), axis=0) * 100.0

    plt.figure(figsize=(max(10, 0.6 * len(top_types) + 4), 6))

    bottom = [0.0] * len(pct)
    x = list(range(len(pct.index)))

    for b in bucket_order:
        vals = pct[b].tolist()
        plt.bar(x, vals, bottom=bottom, label=b)
        bottom = [bottom[i] + vals[i] for i in range(len(bottom))]

    plt.xticks(x, pct.index.tolist(), rotation=45, ha="right")
    plt.ylim(0, 100)
    plt.title(f"URL aggregation buckets by disruption type (top {top_n_types}; 100% stacked)")
    plt.xlabel("disruption_type")
    plt.ylabel("Percent of records (%)")
    plt.legend(title="url_count bucket")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ------------------ MAIN ------------------ #

def main() -> None:
    df = _load_all_consolidated()

    if "disruption_type" not in df.columns:
        raise ValueError("Column 'disruption_type' missing from all_consolidated.")
    if "urls" not in df.columns:
        raise ValueError("Column 'urls' missing from all_consolidated.")

    out = pd.DataFrame({
        "disruption_type": df["disruption_type"].fillna("unknown").astype(str),
        "url_count": df["urls"].map(_count_urls),
    })
    out["url_bucket"] = out["url_count"].map(_bucket_label)

    # ---- save simple per-record table ----
    table_out = OUT_DIR / "url_count_per_record.csv"
    out.to_csv(table_out, index=False)

    # ---- plots ----
    logfreq_out = OUT_DIR / "url_count_distribution_logy.png"
    _plot_logfreq(out["url_count"], logfreq_out)

    buckets_overall_out = OUT_DIR / "url_count_buckets_overall.png"
    _plot_bucket_overall(out["url_bucket"], buckets_overall_out)

    stacked_out = OUT_DIR / "url_count_buckets_by_type_stacked.png"
    _plot_bucket_stacked_by_type(out, stacked_out, top_n_types=20)

    print(f"Wrote: {table_out}")
    print(f"Wrote: {logfreq_out}")
    print(f"Wrote: {buckets_overall_out}")
    print(f"Wrote: {stacked_out}")


if __name__ == "__main__":
    main()