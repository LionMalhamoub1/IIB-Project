#!/usr/bin/env python3
"""
Unified date-debugging and coverage diagnostics script.

Refactored to:
- Accept DataFrames directly
- Be callable from master pipeline
- Still runnable standalone
"""

from pathlib import Path
import json
import pandas as pd


# ------------------ LOAD HELPERS (standalone only) ------------------ #

def load_df(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)

    elif input_path.suffix.lower() == ".jsonl":
        records = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)

    else:
        raise ValueError("Input must be .csv or .jsonl")

    df["event_date"] = (
        pd.to_datetime(df.get("event_date"), errors="coerce", utc=True)
        .dt.tz_convert(None)
    )
    df["publish_date"] = (
        pd.to_datetime(df.get("publish_date"), errors="coerce", utc=True)
        .dt.tz_convert(None)
    )

    return df


def has_any_date(df: pd.DataFrame) -> pd.Series:
    return df["event_date"].notna() | df["publish_date"].notna()


# ------------------ COVERAGE DIAGNOSTICS ------------------ #

def coverage_stats_known_only(df: pd.DataFrame, label: str):
    print(f"\n================ {label.upper()} =================\n")

    df = df[df["disruption_type"].fillna("unknown") != "unknown"].copy()
    total = len(df)

    if total == 0:
        print("No known disruption types.\n")
        return

    has_location = df["location_name"].notna() & (
        df["location_name"].astype(str).str.strip() != ""
    )
    has_event_date = df["event_date"].notna()
    has_publish_date = df["publish_date"].notna()
    has_any = has_event_date | has_publish_date

    print("Overall coverage (known only):")
    print(f"- Location      : {has_location.sum():5d} / {total} ({100*has_location.mean():5.1f}%)")
    print(f"- Event date    : {has_event_date.sum():5d} / {total} ({100*has_event_date.mean():5.1f}%)")
    print(f"- Publish date  : {has_publish_date.sum():5d} / {total} ({100*has_publish_date.mean():5.1f}%)")
    print(f"- Any date      : {has_any.sum():5d} / {total} ({100*has_any.mean():5.1f}%)")


# ------------------ PUBLIC ENTRY POINT ------------------ #

def run_debugger_and_metrics(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame
):
    """
    Main callable function for master pipeline.

    Accepts:
        df_before  = raw extractions
        df_after   = consolidated extractions
    """

    # ---- 1) Coverage diagnostics ----
    coverage_stats_known_only(df_before, "Before Consolidating")
    coverage_stats_known_only(df_after, "After Consolidating")

    # ---- 2) Parsed date coverage ----
    any_before = has_any_date(df_before)
    any_after = has_any_date(df_after)

    print("\n=== DATE COVERAGE (PARSED DATETIMES ONLY) ===\n")
    print(f"Extractions total        : {len(df_before)}")
    print(f"Consolidated total       : {len(df_after)}")
    print()
    print(f"With any date (before)   : {any_before.sum()} ({100*any_before.mean():.1f}%)")
    print(f"With any date (after)    : {any_after.sum()} ({100*any_after.mean():.1f}%)")

    # ---- 3) Event_date audit (before only) ----
    event_present = df_before["event_date"].notna()
    print("\n=== EVENT_DATE AUDIT (BEFORE) ===\n")
    print(f"With event_date          : {event_present.sum()} ({100*event_present.mean():.1f}%)")

    # ---- 4) Publish_date audit (before only) ----
    publish_present = df_before["publish_date"].notna()
    print("\n=== PUBLISH_DATE AUDIT (BEFORE) ===\n")
    print(f"With publish_date        : {publish_present.sum()} ({100*publish_present.mean():.1f}%)")

    # ---- 5) Strict publish_date validation ----
    publish_non_null = df_before["publish_date"].dropna()
    failures = 0

    for v in publish_non_null:
        try:
            pd.to_datetime(v, utc=True, errors="raise")
        except Exception:
            failures += 1

    print("\n=== STRICT PUBLISH_DATE VALIDATION ===\n")
    print(f"Non-null publish_date values : {len(publish_non_null)}")
    print(f"Valid datetimes              : {len(publish_non_null) - failures}")
    print(f"Invalid datetimes            : {failures}")


# ------------------ STANDALONE SUPPORT ------------------ #

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"

    before = load_df(results_dir / "weekly_extractions_202601.jsonl")
    after = load_df(results_dir / "weekly_extractions_202601Consolidated.jsonl")

    run_debugger_and_metrics(before, after)