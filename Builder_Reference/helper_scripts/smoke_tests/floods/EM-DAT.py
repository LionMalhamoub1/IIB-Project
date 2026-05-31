#!/usr/bin/env python3
"""
EM-DAT (downloaded local file) smoke test.

Assumes you have downloaded an EM-DAT export as an Excel file and placed it
next to this script.

This script:
  - loads the Excel file,
  - checks required columns exist,
  - builds parsed start/end dates from Start Year/Month/Day + End Year/Month/Day,
  - prints row count, date coverage, disaster-type counts,
  - prints a small sample of rows.

Edit FILENAME if your file name differs.
"""

from pathlib import Path
import pandas as pd


# ------------------CONFIG------------------ #

FILENAME = "EM-DAT.xlsx"   # expected next to this script
SHEET_NAME = None         # set if needed; otherwise uses first sheet
SAMPLE_N = 10

REQUIRED_COLS = [
    "DisNo.",
    "Disaster Type",
    "ISO",
    "Country",
    "Start Year",
    "End Year",
    "Total Deaths",
    "Total Affected",
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


def _to_int_series(s: pd.Series) -> pd.Series:
    """Coerce to pandas nullable integer (handles blanks/NaNs)."""
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _build_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build StartDate / EndDate from Start Year/Month/Day, End Year/Month/Day.

    If month/day are missing, we default:
      - month -> 1
      - day   -> 1
    so we can still compute a comparable timeline for smoke testing.
    """
    df = df.copy()

    sy = _to_int_series(df["Start Year"])
    sm = _to_int_series(df.get("Start Month", pd.Series([pd.NA] * len(df))))
    sd = _to_int_series(df.get("Start Day", pd.Series([pd.NA] * len(df))))

    ey = _to_int_series(df["End Year"])
    em = _to_int_series(df.get("End Month", pd.Series([pd.NA] * len(df))))
    ed = _to_int_series(df.get("End Day", pd.Series([pd.NA] * len(df))))

    # Defaults where missing (only for building a date)
    sm = sm.fillna(1)
    sd = sd.fillna(1)
    em = em.fillna(1)
    ed = ed.fillna(1)

    df["StartDate"] = pd.to_datetime(
        dict(year=sy.astype("float"), month=sm.astype("float"), day=sd.astype("float")),
        errors="coerce",
        utc=True,
    )
    df["EndDate"] = pd.to_datetime(
        dict(year=ey.astype("float"), month=em.astype("float"), day=ed.astype("float")),
        errors="coerce",
        utc=True,
    )

    return df


# ------------------MAIN------------------ #

def main():
    base_dir = Path(__file__).resolve().parent
    path = base_dir / FILENAME

    print(f"Loading EM-DAT file: {path}")
    df, sheets, used_sheet = _load_excel(path, sheet_name=SHEET_NAME)

    print(f"\nSheets found: {sheets}")
    print(f"Using sheet: {used_sheet!r}")

    print(f"\nLoaded: {len(df):,} rows × {len(df.columns)} columns")

    _check_required_columns(df)
    print("Required columns present OK.")

    df = _build_dates(df)

    start_ok = df["StartDate"].notna().sum()
    end_ok = df["EndDate"].notna().sum()

    print(f"\nParsed StartDate: {start_ok:,}/{len(df):,}")
    print(f"Parsed EndDate:   {end_ok:,}/{len(df):,}")

    if start_ok:
        print(f"StartDate range: {df['StartDate'].min().date()} → {df['StartDate'].max().date()}")
    if end_ok:
        print(f"EndDate range:   {df['EndDate'].min().date()} → {df['EndDate'].max().date()}")

    # Disaster type counts (useful sanity check)
    print("\nDisaster Type counts (top 15):")
    type_counts = df["Disaster Type"].fillna("UNKNOWN").value_counts().head(15)
    print(type_counts.to_string())

    # Optional: Flood-only count (since you're in Floods/Smoke Tests)
    flood_count = df["Disaster Type"].astype(str).str.contains("flood", case=False, na=False).sum()
    print(f"\nFlood rows detected (Disaster Type contains 'flood'): {flood_count:,}")

    # Sample rows
    sample_cols = [c for c in [
        "DisNo.", "Event Name", "Disaster Group", "Disaster Subgroup",
        "Disaster Type", "Disaster Subtype",
        "ISO", "Country", "Region", "Subregion", "Location",
        "Start Year", "Start Month", "Start Day", "End Year", "End Month", "End Day",
        "StartDate", "EndDate",
        "Total Deaths", "Total Affected", "Total Damage ('000 US$)"
    ] if c in df.columns]

    print(f"\nSample rows (first {SAMPLE_N}):")
    print(df[sample_cols].head(SAMPLE_N).to_string(index=False))

    # Informative ordering check (not enforced)
    if start_ok:
        s = df["StartDate"].dropna()
        if len(s) >= 2:
            print("\nOrdering check (StartDate):")
            print(f"  monotonic increasing: {s.is_monotonic_increasing}")
            print(f"  monotonic decreasing: {s.is_monotonic_decreasing}")

    print("\nEM-DAT local smoke test completed successfully.")


if __name__ == "__main__":
    main()
