import re
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import download
import filter
import enrichv3 as enrich
import fix_title_description
import relevant_urls
import cleanup_intermediates


DATE_FMT = "%Y%m%d"


def _parse_dates(user_input: str) -> list[str]:
    """Parse a date string into a sorted list of YYYYMMDD strings — single date, range, or comma list."""
    s = (user_input or "").strip()

    # Comma-separated list — split and validate each token individually
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        dates = []
        for p in parts:
            _validate_date(p)
            dates.append(p)
        return sorted(set(dates))

    # Hyphen-separated range — expand every day between start and end inclusive
    if "-" in s:
        a, b = [p.strip() for p in s.split("-", 1)]
        _validate_date(a)
        _validate_date(b)
        start = datetime.strptime(a, DATE_FMT)
        end = datetime.strptime(b, DATE_FMT)
        if end < start:
            start, end = end, start  # swap so we always iterate forward

        out = []
        cur = start
        while cur <= end:
            out.append(cur.strftime(DATE_FMT))
            cur += timedelta(days=1)
        return out

    # Single date
    _validate_date(s)
    return [s]


def _validate_date(d: str) -> None:
    if not re.fullmatch(r"\d{8}", d or ""):
        raise ValueError(f"Invalid date '{d}'. Expected YYYYMMDD.")
    # Parse through strptime to catch impossible dates like 20230231
    datetime.strptime(d, DATE_FMT)


def run_one_date(date: str) -> None:
    print(f"\n==============================")
    print(f"Processing date: {date}")
    print(f"==============================")

    print(f"\n>> Step 1: Downloading...")
    download.main(date)

    print(f"\n>> Step 2: Filtering & Deduping...")
    filter.main(date)

    print(f"\n>> Step 3: Fetching Titles & Meta Tags...")
    enrich.main(date)

    print("\n>> Step 4: Cleaning Titles & Meta Tags...")
    fix_title_description.main(date)

    print("\n>> Step 5: Relevant URLs...")
    relevant_urls.main(date)

    print("\n>> Cleaning up intermediates...")
    cleanup_intermediates.cleanup_day(date)

    print("\nDONE:", date)


def start_pipeline(start_date=None, end_date=None):
    """Run the retrieval pipeline over a date range. If called without arguments, prompts interactively."""
    if start_date is not None:
        if end_date is None:
            dates = _parse_dates(start_date)
        else:
            dates = _parse_dates(f"{start_date}-{end_date}")
    else:
        user_in = input(
            "Enter date(s) to process:\n"
            "  - Single: YYYYMMDD\n"
            "  - Range:  YYYYMMDD-YYYYMMDD\n"
            "  - List:   YYYYMMDD,YYYYMMDD,...\n"
            "> "
        ).strip()

        try:
            dates = _parse_dates(user_in)
        except Exception as e:
            print(f"\nError: {e}")
            return

    print(f"\nWill process {len(dates)} date(s): {', '.join(dates)}")

    for i, d in enumerate(dates, start=1):
        print(f"\n[{i}/{len(dates)}]")
        try:
            run_one_date(d)
        except Exception as e:
            print(f"\n!!! Failed for {d}: {repr(e)}")
            continue

    print("\nALL STEPS COMPLETE")


if __name__ == "__main__":
    start_pipeline()
