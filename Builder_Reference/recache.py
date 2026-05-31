"""
Re-cache all automatable flood reference sources from 2016-01-01 to today.

Sources handled (no manual files required):
  HANZE, GDACS, IFRC, Copernicus, ReliefWeb, Desinventar

DFO and EM-DAT are skipped — they require a manually downloaded CSV/XLSX.

Usage:
    python -m Builder_Reference.recache [--skip-desinventar] [--skip-copernicus]
    python -m Builder_Reference.recache --only gdacs
"""

import argparse
from datetime import date
from pathlib import Path

START = date(2016, 1, 1)
END   = date(2026, 12, 31)

BASE = Path(__file__).parent.parent / "cache" / "floods"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-desinventar", action="store_true",
                        help="Skip Desinventar (slow — scrapes 87 countries)")
    parser.add_argument("--skip-copernicus", action="store_true",
                        help="Skip Copernicus (slow — one HTTP request per activation)")
    parser.add_argument("--only", default=None,
                        help="Run only this source (hanze/gdacs/ifrc/reliefweb/copernicus/desinventar)")
    args = parser.parse_args()

    only = args.only.lower() if args.only else None

    # ---- HANZE ----
    if not only or only == "hanze":
        print(f"\n{'='*60}")
        print(f"[recache] HANZE  {START} -> {END}")
        print(f"{'='*60}")
        from Builder_Reference.helper_scripts.reference.cache.hanze import cache_hanze_floods
        cache_hanze_floods(
            start_date=START,
            end_date=END,
            output_path=BASE / "hanze.json",
            request_delay=0.5,
        )

    # ---- GDACS ----
    if not only or only == "gdacs":
        print(f"\n{'='*60}")
        print(f"[recache] GDACS  {START} -> {END}")
        print(f"{'='*60}")
        from Builder_Reference.helper_scripts.reference.cache.gdacs import cache_gdacs_floods
        cache_gdacs_floods(
            start_date=START,
            end_date=END,
            output_path=BASE / "gdacs.json",
        )

    # ---- IFRC ----
    if not only or only == "ifrc":
        print(f"\n{'='*60}")
        print(f"[recache] IFRC  {START} -> {END}")
        print(f"{'='*60}")
        from Builder_Reference.helper_scripts.reference.cache.ifrc import cache_ifrc_floods
        cache_ifrc_floods(
            start_date=START,
            end_date=END,
            output_path=BASE / "ifrc.json",
        )

    # ---- ReliefWeb ----
    if not only or only == "reliefweb":
        print(f"\n{'='*60}")
        print(f"[recache] ReliefWeb  {START} -> {END}")
        print(f"{'='*60}")
        from Builder_Reference.helper_scripts.reference.cache.reliefweb import cache_reliefweb_floods
        cache_reliefweb_floods(
            start_date=START,
            end_date=END,
            output_path=BASE / "reliefweb.json",
            appname="iib-flood-project",
        )

    # ---- Copernicus ----
    if (not only or only == "copernicus") and not args.skip_copernicus:
        print(f"\n{'='*60}")
        print(f"[recache] Copernicus  {START} -> {END}")
        print(f"{'='*60}")
        from Builder_Reference.helper_scripts.reference.cache.copernicus import cache_copernicus_floods
        cache_copernicus_floods(
            start_date=START,
            end_date=END,
            output_path=BASE / "copernicus.json",
        )

    # ---- Desinventar ----
    if (not only or only == "desinventar") and not args.skip_desinventar:
        print(f"\n{'='*60}")
        print(f"[recache] Desinventar  {START} -> {END}")
        print(f"{'='*60}")
        from Builder_Reference.helper_scripts.reference.cache.desinventar_web import cache_desinventar_floods_web
        cache_desinventar_floods_web(
            start_date=START,
            end_date=END,
            output_path=BASE / "desinventar.json",
        )

    print("\n[recache] Done.")


if __name__ == "__main__":
    main()
