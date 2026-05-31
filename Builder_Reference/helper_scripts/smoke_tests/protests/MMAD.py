import sys
from pathlib import Path

import pandas as pd


# ------------------ CONFIG ------------------ #
REQUIRED_COLUMNS = [
    "id", "cowcode", "location", "latitude", "longitude", "asciiname", "event_date",
    "side", "actors", "issue", "scope", "part_violence", "sec_engagement",
    "numparticipants", "avg_numparticipants", "source", "version"
]


def load_reports_csv() -> pd.DataFrame:
    """Load MMAD reports.csv from the same directory as this script."""
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "reports.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find MMAD file at: {csv_path}")

    df = pd.read_csv(csv_path)
    return df


def assert_schema(df: pd.DataFrame, enforce_order: bool = False) -> None:
    cols = list(df.columns)

    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    extra = [c for c in cols if c not in REQUIRED_COLUMNS]

    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    if extra:
        # Not fatal, but you probably want to know
        print(f"Warning: extra columns present (ignored by smoke test): {extra}")

    if enforce_order:
        if cols[: len(REQUIRED_COLUMNS)] != REQUIRED_COLUMNS:
            raise RuntimeError(
                "Column order does not match REQUIRED_COLUMNS. "
                "Set enforce_order=False if you don't care about order."
            )


def basic_sanity_checks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("reports.csv loaded successfully but contains 0 rows")

    # Coerce lat/lon
    df["latitude_num"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude_num"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Coerce event_date
    df["event_date_parsed"] = pd.to_datetime(df["event_date"], errors="coerce", utc=True)

    # Report null rates
    lat_na = df["latitude_num"].isna().mean()
    lon_na = df["longitude_num"].isna().mean()
    date_na = df["event_date_parsed"].isna().mean()

    print("\nNull-rate sanity checks:")
    print(f"  latitude not parseable:  {lat_na:.1%}")
    print(f"  longitude not parseable: {lon_na:.1%}")
    print(f"  event_date not parseable:{date_na:.1%}")

    # Hard fail only if it's *basically all broken*
    if date_na > 0.95:
        raise RuntimeError("More than 95% of event_date values failed to parse — likely wrong format/column")

    # Lat/lon might legitimately be missing sometimes; only fail if almost all missing
    if lat_na > 0.99 or lon_na > 0.99:
        print("Warning: almost all lat/lon values are missing or non-numeric.")

    return df


def print_summary(df: pd.DataFrame) -> None:
    print("\nBasic summary:")
    print(f"  Rows: {len(df):,}")
    print(f"  Versions: {df['version'].nunique() if 'version' in df.columns else 'N/A'}")
    print(f"  Sources: {df['source'].nunique() if 'source' in df.columns else 'N/A'}")

    # Year coverage if dates parse
    if "event_date_parsed" in df.columns and df["event_date_parsed"].notna().any():
        years = df["event_date_parsed"].dt.year.dropna().astype(int)
        if not years.empty:
            print(f"  Year range: {years.min()}–{years.max()}")

    # A couple of samples (keep it readable)
    show_cols = [
        "id", "cowcode", "asciiname", "event_date", "location",
        "latitude", "longitude", "actors", "issue", "scope", "source", "version"
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    print("\nSample rows:")
    print(df[show_cols].head(2).to_string(index=False))


def run_mmad_smoke_test(enforce_order: bool = False) -> None:
    print("Loading MMAD reports.csv...")
    df = load_reports_csv()

    print("Checking schema...")
    assert_schema(df, enforce_order=enforce_order)

    print("Running sanity checks...")
    df = basic_sanity_checks(df)

    print_summary(df)

    print("\nMMAD smoke test PASSED")


if __name__ == "__main__":
    try:
        # set True if you want to enforce the exact column ordering too
        run_mmad_smoke_test(enforce_order=False)
    except Exception as e:
        print(f"\nMMAD smoke test FAILED: {e}")
        sys.exit(1)
