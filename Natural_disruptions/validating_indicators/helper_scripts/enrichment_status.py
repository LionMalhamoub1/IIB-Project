"""
enrichment_status.py
=====================
Quick status check for all enrichment steps running against the baseline samples.

Shows progress, completion percentage, enriched/error/skipped counts, and the
last log line for each enrichment layer. Run at any point during enrichment to
check how far along each step is.

Usage
-----
    python -m Natural_disruptions.validating_indicators.enrichment_status
"""

import json
from datetime import datetime
from pathlib import Path

ROOT  = Path(__file__).resolve().parents[3]
CACHE = ROOT / "cache" / "floods"
LOGS  = ROOT / "logs"

BASELINE_PATH = CACHE / "baseline_samples.jsonl"

LAYERS = [
    ("SPI",    CACHE / "baseline_spi.jsonl",    "_spi_processed",    LOGS / "baseline_spi.log"),
    ("CHIRPS", CACHE / "baseline_chirps.jsonl", "_chirps_processed", LOGS / "baseline_chirps.log"),
    ("ERA5",   CACHE / "baseline_era5.jsonl",   "_era5_processed",   LOGS / "baseline_era5.log"),
    ("GPM",    CACHE / "baseline_gpm.jsonl",    "_gpm_processed",    LOGS / "baseline_gpm.log"),
    ("Static", CACHE / "baseline_static.jsonl", "_static_processed", LOGS / "baseline_static.log"),
]

FINAL_OUTPUT = CACHE / "baseline_enriched.jsonl"


def _total_baseline() -> int:
    """Count lines in the baseline samples file."""
    if not BASELINE_PATH.exists():
        return 0
    with BASELINE_PATH.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


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


def _log_complete(log_path: Path) -> bool:
    """True if the log file contains a completion message."""
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return "Complete." in text or "Output written" in text
    except Exception:
        return False


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
    total_baseline = _total_baseline()

    print(f"\nBaseline enrichment status — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total baseline samples: {total_baseline:,}  ({BASELINE_PATH.name})")
    print("=" * 75)

    if total_baseline == 0:
        print("  baseline_samples.jsonl not found. Run generate_baseline.py first.")
        print()
        return

    all_done = True
    for name, jsonl_path, flag_field, log_path in LAYERS:
        total, enriched, errors = _count_jsonl(jsonl_path, flag_field)
        pct = total / total_baseline * 100 if total_baseline else 0
        skipped = total - enriched - errors
        done = (total >= total_baseline) or _log_complete(log_path)
        if not jsonl_path.exists():
            status = "not started"
        elif done:
            status = "DONE"
        else:
            status = "running"
        if status != "DONE":
            all_done = False

        bar = _bar(pct)
        print(f"\n{name:<8} {bar} {pct:5.1f}%  ({total:,}/{total_baseline:,})")
        print(f"         enriched={enriched:,}  errors={errors:,}  skipped={skipped:,}  [{status}]")

        last = _last_log_line(log_path)
        if len(last) > 72:
            last = last[:69] + "..."
        print(f"         last: {last}")

    # Final merged output
    print("\n" + "-" * 75)
    if FINAL_OUTPUT.exists():
        with FINAL_OUTPUT.open() as f:
            n_final = sum(1 for _ in f)
        pct_final = n_final / total_baseline * 100 if total_baseline else 0
        print(f"Final output  {_bar(pct_final)} {pct_final:5.1f}%  ({n_final:,}/{total_baseline:,})")
        print(f"              {FINAL_OUTPUT.name}")
    else:
        print(f"Final output  {'not yet written':<40} {FINAL_OUTPUT.name}")

    print("\n" + "=" * 75)
    if all_done:
        print("All layers complete. Ready to run distribution_analysis.py")
    else:
        remaining = [
            name for name, path, flag, log in LAYERS
            if not path.exists() or (_count_jsonl(path, flag)[0] < total_baseline and not _log_complete(log))
        ]
        print(f"Still pending: {', '.join(remaining)}")
    print()


if __name__ == "__main__":
    main()
