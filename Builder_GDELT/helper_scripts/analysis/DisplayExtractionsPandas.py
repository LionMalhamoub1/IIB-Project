#!/usr/bin/env python3
"""
Display consolidated extractions using pandas.

Refactored to:
- Accept DataFrame directly
- Be callable from master pipeline
- Still runnable standalone

Update:
- Disruption Type Counts now prints BEFORE vs AFTER consolidation if df_before is provided.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


# LOAD (standalone only)

def load_extractions(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    if input_path.suffix.lower() == ".csv":
        return pd.read_csv(input_path)
    if input_path.suffix.lower() == ".jsonl":
        return pd.read_json(input_path, lines=True)

    raise ValueError("Input must be .csv or .jsonl")


# HELPERS

def _truncate(s: str, max_chars: int) -> str:
    s = str(s)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3].rstrip() + "..."


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _print_coverage_stats(df: pd.DataFrame):
    """
    Print coverage statistics for dates and locations
    (known disruption types only).
    """
    known = df[df["disruption_type"] != "unknown"].copy()
    total = len(known)

    if total == 0:
        print("\nNo known disruption types found  -  skipping coverage stats.\n")
        return

    known["event_date"] = pd.to_datetime(known.get("event_date"), errors="coerce")
    known["publish_date"] = pd.to_datetime(known.get("publish_date"), errors="coerce")

    has_event_date = known["event_date"].notna()
    has_publish_only = known["event_date"].isna() & known["publish_date"].notna()
    has_no_date = known["event_date"].isna() & known["publish_date"].isna()

    has_location = known["location_name"].astype(str).str.strip().ne("")

    print("\n=== Coverage diagnostics (known disruption types only) ===\n")

    print("Date coverage:")
    print(
        f"- Event date extracted       : {has_event_date.sum():5d} / {total} "
        f"({100 * has_event_date.mean():5.1f}%)"
    )
    print(
        f"- Publish date only (proxy)  : {has_publish_only.sum():5d} / {total} "
        f"({100 * has_publish_only.mean():5.1f}%)"
    )
    print(
        f"- No date available          : {has_no_date.sum():5d} / {total} "
        f"({100 * has_no_date.mean():5.1f}%)"
    )

    print("\nLocation coverage:")
    print(
        f"- Location extracted         : {has_location.sum():5d} / {total} "
        f"({100 * has_location.mean():5.1f}%)"
    )

    print("\n=========================================================\n")


# PUBLIC ENTRY POINT

def run_display_extractions(
    df_after: pd.DataFrame,
    df_before: Optional[pd.DataFrame] = None,
    max_rows: int = 30,
    title_max_chars: int = 40,
    location_max_chars: int = 40,
):
    """
    Display consolidated extractions.

    If df_before is provided, prints disruption type counts
    before vs after consolidation.
    """

    df_after = df_after.copy().fillna("")

    print(
        "\nDate notation used below:\n"
        "- YYYY-MM-DD : event date extracted from article text (preferred)\n"
        "- YYYY/MM/DD : article publish date (metadata proxy, used when no event date)\n"
    )

    # Coverage diagnostics
    _print_coverage_stats(df_after)

    # Disruption type counts (BEFORE vs AFTER)
    print("\n=== Disruption Type Counts ===\n")

    after_counts = df_after["disruption_type"].value_counts().sort_index()

    if df_before is not None:
        df_before = df_before.copy().fillna("")
        before_counts = df_before["disruption_type"].value_counts().sort_index()

        all_types = sorted(set(before_counts.index).union(set(after_counts.index)))

        counts_df = pd.DataFrame(
            {
                "before": [int(before_counts.get(t, 0)) for t in all_types],
                "after": [int(after_counts.get(t, 0)) for t in all_types],
            },
            index=all_types,
        )
        counts_df.index.name = "type"
        print(counts_df.to_string())

    else:
        print(after_counts.to_frame(name="after").to_string())

    print("\n=== Classification Source Breakdown (Pre-Consolidation) ===\n")
    print(df_before["classification_source"].value_counts().to_string())

    # Date parsing for display
    df_after["event_date"] = (
        pd.to_datetime(df_after.get("event_date"), errors="coerce", utc=True)
        .dt.tz_convert(None)
    )
    df_after["publish_date"] = (
        pd.to_datetime(df_after.get("publish_date"), errors="coerce", utc=True)
        .dt.tz_convert(None)
    )

    def display_date(row) -> str:
        if pd.notna(row["event_date"]):
            return row["event_date"].strftime("%Y-%m-%d")
        if pd.notna(row["publish_date"]):
            return row["publish_date"].strftime("%Y/%m/%d")
        return ""

    df_after["display_date"] = df_after.apply(display_date, axis=1)

    df_after["title_short"] = df_after["source_title"].apply(lambda s: _truncate(s, title_max_chars))
    df_after["location_short"] = df_after["location_name"].apply(lambda s: _truncate(s, location_max_chars))

    view = df_after[[
        "title_short",
        "disruption_type",
        "display_date",
        "location_short",
        "duration_hours",
        "confidence",
    ]].rename(columns={
        "title_short": "title",
        "disruption_type": "type",
        "display_date": "date",
        "location_short": "location",
        "duration_hours": "duration_h",
    })

    view["confidence"] = _to_numeric(view["confidence"])
    view["duration_h"] = _to_numeric(view["duration_h"])

    view = view.sort_values("confidence", ascending=False, na_position="last")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 140)
    pd.set_option("display.max_colwidth", None)

    # Print floods first, then the rest
    ordered_types = []
    if "flood" in view["type"].unique():
        ordered_types.append("flood")
    ordered_types += sorted(t for t in view["type"].unique() if t not in ("flood", "unknown"))

    for dtype in ordered_types:
        group = view[view["type"] == dtype]
        if len(group) == 0:
            continue

        print(f"\n=== {dtype.upper()} ({len(group)} events) ===\n")

        if len(group) <= max_rows:
            print(group.to_string(index=False))
        else:
            print(group.head(max_rows).to_string(index=False))
            print(f"\n... ({len(group) - max_rows} more rows not shown) ...\n")


# STANDALONE SUPPORT

if __name__ == "__main__":
    # Example: compare raw vs consolidated using the default naming convention
    base_dir = Path(__file__).resolve().parents[1]  # helper_scripts/ -> project root
    results_dir = base_dir / "results"

    before_path = Path("Builder_GDELT/results/daily/20250101/extractions.csv")
    after_path = Path("Builder_GDELT/results/daily/20250101/extractionsConsolidated.csv")

    df_before = load_extractions(before_path) if before_path.exists() else None
    df_after = load_extractions(after_path)

    run_display_extractions(df_after, df_before=df_before)