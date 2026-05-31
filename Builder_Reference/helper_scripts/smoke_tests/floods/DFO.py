#!/usr/bin/env python3
"""
DFO FloodArchive.xlsx smoke test (local file).

Assumes you have downloaded FloodArchive.xlsx (DFO-style flood event archive)
and placed it next to this script.

Headers (per screenshot):
ID, GlideNumber, Country, OtherCountry, long, lat, Area, Began, Ended,
Validation, Dead, Displaced, MainCause, Severity

This script:
  - loads the Excel file,
  - checks required columns exist,
  - parses Began/Ended dates (handles Excel date cells or dd/mm/yyyy strings),
  - prints row count, date coverage, and a small sample.
"""

from pathlib import Path
import pandas as pd


# ------------------CONFIG------------------ #

FILENAME = "DFO.xlsx"   # expected next to this script
SHEET_NAME = None               # set if needed; otherwise uses first sheet
SAMPLE_N = 10

REQUIRED_COLS = [
    "ID",
    "Country",
    "long",
    "lat",
    "Area",
    "Began",
    "Ended",
    "MainCause",
    "Severity",
]


# ------------------HELPERS------------------ #

def _load_excel(path: Path, sheet_name=None):
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path.resolve()}")

    xls = pd.ExcelFile(path)
    use_sheet = sheet_name or xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=use_sheet)
    return df, xls.sheet_names, use_sheet


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise AssertionError(
            "Missing required columns:\n"
            f"  {missing}\n\n"
            "Columns found:\n"
            f"  {list(df.columns)}"
        )


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Handle both cases:
    #  - true Excel date cells (already datetime-like)
    #  - strings like '01/01/1985' (dd/mm/yyyy) or '1985-01-01'
    df["Began_dt"] = pd.to_datetime(df["Began"], errors="coerce", dayfirst=True)
    df["Ended_dt"] = pd.to_datetime(df["Ended"], errors="coerce", dayfirst=True)

    return df


# ------------------MAIN------------------ #

def main():
    base_dir = Path(__file__).resolve().parent
    path = base_dir / FILENAME

    print(f"Loading DFO FloodArchive file: {path}")
    df, sheets, used_sheet = _load_excel(path, sheet_name=SHEET_NAME)

    print(f"\nSheets found: {sheets}")
    print(f"Using sheet: {used_sheet!r}")

    print(f"\nLoaded: {len(df):,} rows × {len(df.columns)} columns")

    _check_required_columns(df)
    print("Required columns present OK.")

    df = _parse_dates(df)

    began_ok = df["Began_dt"].notna().sum()
    ended_ok = df["Ended_dt"].notna().sum()

    print(f"\nParsed Began dates: {began_ok:,}/{len(df):,}")
    print(f"Parsed Ended dates: {ended_ok:,}/{len(df):,}")

    if began_ok:
        print(f"Date range (Began): {df['Began_dt'].min().date()} → {df['Began_dt'].max().date()}")
    if ended_ok:
        print(f"Date range (Ended): {df['Ended_dt'].min().date()} → {df['Ended_dt'].max().date()}")

    print("\nBasic stats:")
    print(f"  Unique countries: {df['Country'].nunique(dropna=True):,}")

    if "Severity" in df.columns:
        try:
            sev_counts = df["Severity"].value_counts(dropna=False).sort_index()
            print("  Severity counts:")
            print(sev_counts.to_string())
        except Exception:
            pass

    # Print a small sample
    sample_cols = [c for c in [
        "ID", "GlideNumber", "Country", "OtherCountry", "Began", "Ended",
        "lat", "long", "Area", "Validation", "Dead", "Displaced", "MainCause", "Severity"
    ] if c in df.columns]

    print(f"\nSample rows (first {SAMPLE_N}):")
    print(df[sample_cols].head(SAMPLE_N).to_string(index=False))

    # Optional: quick “sorted by date?” check
    if began_ok:
        began_series = df["Began_dt"].dropna()
        if len(began_series) >= 2:
            print("\nOrdering check (Began_dt):")
            print(f"  monotonic increasing: {began_series.is_monotonic_increasing}")
            print(f"  monotonic decreasing: {began_series.is_monotonic_decreasing}")

    print("\nDFO FloodArchive smoke test completed successfully.")


if __name__ == "__main__":
    main()
