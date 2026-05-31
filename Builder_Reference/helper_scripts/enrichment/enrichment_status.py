"""
enrichment_status.py
=====================
Quick status check for all enrichment jobs running against the reference dataset.

Shows progress, completion percentage, error counts, and last log activity
for each enrichment layer.

Usage
-----
    python -m Builder_Reference.helper_scripts.enrichment.enrichment_status
"""

import json
import re
from datetime import datetime
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[3]
CACHE = ROOT / "cache" / "floods"
LOGS  = ROOT / "logs"

DATE_FROM = "2017-01-01"
DATE_TO   = "2021-12-31"

def _count_in_range() -> int:
    """Count geocoded events within the enrichment date range."""
    p = CACHE / "reference_floods_geocoded.jsonl"
    if not p.exists():
        return 0
    count = 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                ds = (r.get("date_start") or "")[:10]
                if DATE_FROM <= ds <= DATE_TO:
                    count += 1
            except Exception:
                pass
    return count

TOTAL_EVENTS = _count_in_range()

LAYERS = [
    ("CHIRPS",  CACHE / "reference_floods_chirps.jsonl",  "_chirps_processed",  LOGS / "chirps.log"),
    ("GPM",     CACHE / "reference_floods_gpm.jsonl",     "_gpm_processed",     LOGS / "gpm.log"),
    ("ERA5",    CACHE / "reference_floods_era5.jsonl",    "_era5_processed",    LOGS / "era5.log"),
    ("Static",  CACHE / "reference_floods_static.jsonl",  "_static_processed",  LOGS / "static.log"),
    ("SPI",     CACHE / "reference_floods_spi.jsonl",     "_spi_processed",     LOGS / "spi.log"),
]


def _count_jsonl(path: Path, flag_field: str) -> tuple[int, int, int]:
    """Return (total_lines, enriched, errors) from a JSONL output file."""
    if not path.exists():
        return 0, 0, 0
    total = enriched = errors = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                total += 1
                flag = row.get(flag_field)
                if flag is True:
                    enriched += 1
                elif flag is False:
                    errors += 1
            except json.JSONDecodeError:
                pass
    return total, enriched, errors



def _last_log_line(log_path: Path) -> str:
    """Return the last non-empty line of a log file."""
    if not log_path.exists():
        return "no log file"
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            if line.strip():
                return line.strip()
    except Exception:
        pass
    return "unreadable"


def _bar(pct: float, width: int = 25) -> str:
    filled = int(pct / 100 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def main():
    print(f"\nEnrichment status — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total reference events: {TOTAL_EVENTS:,}")
    print("=" * 75)

    all_done = True
    for name, jsonl_path, flag_field, log_path in LAYERS:
        total, enriched, errors = _count_jsonl(jsonl_path, flag_field)
        pct = total / TOTAL_EVENTS * 100 if TOTAL_EVENTS else 0
        skipped = total - enriched - errors
        done = (total >= TOTAL_EVENTS) and TOTAL_EVENTS > 0
        status = "DONE" if done else "running" if total > 0 else "not started"
        if status != "DONE":
            all_done = False

        bar = _bar(pct)
        print(f"\n{name:<8} {bar} {pct:5.1f}%  ({total:,}/{TOTAL_EVENTS:,})")
        print(f"         enriched={enriched:,}  errors={errors:,}  skipped={skipped:,}  [{status}]")

        last = _last_log_line(log_path)
        # Trim long lines
        if len(last) > 72:
            last = last[:69] + "..."
        print(f"         last: {last}")

    print("\n" + "=" * 75)
    if all_done:
        print("All layers complete. Ready to run build_official_reference_dataset.py")
    else:
        remaining = [n for n, p, f, _ in LAYERS
                     if not p.exists() or _count_jsonl(p, f)[0] < TOTAL_EVENTS]
        print(f"Still running: {', '.join(remaining)}")
    print()


if __name__ == "__main__":
    main()
