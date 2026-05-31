"""
Build the consolidated official flood reference dataset.

Runs the full pipeline:
  1. Cache DFO            (local xlsx)
  2. Cache EM-DAT         (local xlsx)
  3. Cache GDACS          (live API)
  4. Cache ReliefWeb      (live API)
  5. Cache IFRC GO        (live API)
  6. Cache Copernicus EMS (local CSV — see instructions below)
  7. Cache Desinventar    (local xlsx/csv — see instructions below)
  8. Combine all sources into a single JSONL
  9. Consolidate (cross-source deduplication)

OUTPUT
------
  cache/floods/reference_floods_consolidated.jsonl

CONFIG
------
Edit the variables in the CONFIG block below before running.

FILES YOU NEED TO DOWNLOAD MANUALLY
-------------------------------------
Copernicus EMS:
  1. Go to https://emergency.copernicus.eu/mapping/list-of-activations-rapid
  2. Click "Export" / "Download CSV" (top-right of the table)
  3. Save to: Builder_Reference/helper_scripts/smoke_tests/floods/Copernicus.csv
  (or update COPERNICUS_SOURCE below)

Desinventar:
  1. Go to https://www.desinventar.net/DesInventar/profiletab_main.jsp
  2. Select the region/countries of interest and export as Excel or CSV
     (the global Sendai Monitor export from UNDRR also works:
      https://sendaimonitor.undrr.org/analytics/country-global/1)
  3. Save to: Builder_Reference/helper_scripts/smoke_tests/floods/Desinventar.xlsx
  (or update DESINVENTAR_SOURCE below)

Usage:
    python -m Builder_Reference.helper_scripts.reference.build
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ============================================================
# CONFIG — edit these before running
# ============================================================

DATE_START = date(2016, 1, 1)
DATE_END   = date.today()

# Paths to local source files
DFO_SOURCE          = Path("Builder_Reference/helper_scripts/smoke_tests/floods/DFO.xlsx")
EMDAT_SOURCE        = Path("Builder_Reference/helper_scripts/smoke_tests/floods/EM-DAT.xlsx")
COPERNICUS_SOURCE   = Path("Builder_Reference/helper_scripts/smoke_tests/floods/Copernicus.csv")
DESINVENTAR_SOURCE  = Path("Builder_Reference/helper_scripts/smoke_tests/floods/Desinventar.xlsx")

# ReliefWeb requires a short app-name string (any identifier is fine)
RELIEFWEB_APPNAME = "ben-iib-project"

# Where to write cache and output files
CACHE_DIR  = Path("cache/floods")
OUTPUT_DIR = Path("cache/floods")

# ============================================================
# PIPELINE
# ============================================================

from Builder_Reference.helper_scripts.reference.cache.dfo          import cache_dfo_floods
from Builder_Reference.helper_scripts.reference.cache.emdat        import cache_emdat_floods
from Builder_Reference.helper_scripts.reference.cache.gdacs        import cache_gdacs_floods
from Builder_Reference.helper_scripts.reference.cache.reliefweb    import cache_reliefweb_floods
from Builder_Reference.helper_scripts.reference.cache.ifrc         import cache_ifrc_floods
from Builder_Reference.helper_scripts.reference.cache.copernicus   import cache_copernicus_floods
from Builder_Reference.helper_scripts.reference.cache.desinventar  import cache_desinventar_floods
from Builder_Reference.helper_scripts.reference.combine            import combine_flood_references
from Builder_Reference.helper_scripts.reference.consolidate        import consolidate_flood_references


def run():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Building flood reference dataset")
    print(f"  Date range : {DATE_START} to {DATE_END}")
    print(f"  Cache dir  : {CACHE_DIR}")
    print(f"{'='*60}\n")

    # --------------------------------------------------
    # 1) DFO
    # --------------------------------------------------
    dfo_path = CACHE_DIR / "dfo.json"
    print("[ 1/9 ] Caching DFO ...")
    if not DFO_SOURCE.exists():
        print(f"        WARNING: {DFO_SOURCE} not found — skipping DFO")
    else:
        cache_dfo_floods(DATE_START, DATE_END, DFO_SOURCE, dfo_path)
        print(f"        Done -> {dfo_path}")

    # --------------------------------------------------
    # 2) EM-DAT
    # --------------------------------------------------
    emdat_path = CACHE_DIR / "emdat.json"
    print("[ 2/9 ] Caching EM-DAT ...")
    if not EMDAT_SOURCE.exists():
        print(f"        WARNING: {EMDAT_SOURCE} not found — skipping EM-DAT")
    else:
        cache_emdat_floods(DATE_START, DATE_END, EMDAT_SOURCE, emdat_path)
        print(f"        Done -> {emdat_path}")

    # --------------------------------------------------
    # 3) GDACS (live API)
    # --------------------------------------------------
    gdacs_path = CACHE_DIR / "gdacs.json"
    print("[ 3/9 ] Caching GDACS (API) ...")
    try:
        cache_gdacs_floods(DATE_START, DATE_END, gdacs_path)
        print(f"        Done -> {gdacs_path}")
    except Exception as e:
        print(f"        WARNING: GDACS failed ({e}) — skipping")

    # --------------------------------------------------
    # 4) ReliefWeb (live API)
    # --------------------------------------------------
    reliefweb_path = CACHE_DIR / "reliefweb.json"
    print("[ 4/9 ] Caching ReliefWeb (API) ...")
    try:
        cache_reliefweb_floods(DATE_START, DATE_END, reliefweb_path, RELIEFWEB_APPNAME)
        print(f"        Done -> {reliefweb_path}")
    except Exception as e:
        print(f"        WARNING: ReliefWeb failed ({e}) — skipping")

    # --------------------------------------------------
    # 5) IFRC GO (live API)
    # --------------------------------------------------
    ifrc_path = CACHE_DIR / "ifrc.json"
    print("[ 5/9 ] Caching IFRC GO (API) ...")
    try:
        cache_ifrc_floods(DATE_START, DATE_END, ifrc_path)
        print(f"        Done -> {ifrc_path}")
    except Exception as e:
        print(f"        WARNING: IFRC failed ({e}) — skipping")

    # --------------------------------------------------
    # 6) Copernicus EMS (local CSV)
    # --------------------------------------------------
    copernicus_path = CACHE_DIR / "copernicus.json"
    print("[ 6/9 ] Caching Copernicus EMS ...")
    if not COPERNICUS_SOURCE.exists():
        print(f"        SKIPPED: {COPERNICUS_SOURCE} not found")
        print(f"        Download from: https://emergency.copernicus.eu/mapping/list-of-activations-rapid")
        copernicus_path = None
    else:
        try:
            cache_copernicus_floods(DATE_START, DATE_END, COPERNICUS_SOURCE, copernicus_path)
            print(f"        Done -> {copernicus_path}")
        except Exception as e:
            print(f"        WARNING: Copernicus failed ({e}) — skipping")
            copernicus_path = None

    # --------------------------------------------------
    # 7) Desinventar (local file)
    # --------------------------------------------------
    desinventar_path = CACHE_DIR / "desinventar.json"
    print("[ 7/9 ] Caching Desinventar ...")
    if not DESINVENTAR_SOURCE.exists():
        print(f"        SKIPPED: {DESINVENTAR_SOURCE} not found")
        print(f"        Download from: https://www.desinventar.net/DesInventar/profiletab_main.jsp")
        desinventar_path = None
    else:
        try:
            cache_desinventar_floods(DATE_START, DATE_END, DESINVENTAR_SOURCE, desinventar_path)
            print(f"        Done -> {desinventar_path}")
        except Exception as e:
            print(f"        WARNING: Desinventar failed ({e}) — skipping")
            desinventar_path = None

    # --------------------------------------------------
    # 8) Combine
    # --------------------------------------------------
    combined_path = OUTPUT_DIR / "reference_floods_combined.jsonl"
    print("[ 8/9 ] Combining ...")
    combine_flood_references(
        dfo_path          = dfo_path,
        emdat_path        = emdat_path,
        gdacs_path        = gdacs_path,
        reliefweb_path    = reliefweb_path,
        ifrc_path         = ifrc_path,
        copernicus_path   = copernicus_path,
        desinventar_path  = desinventar_path,
        output_path       = combined_path,
    )

    # --------------------------------------------------
    # 9) Consolidate
    # --------------------------------------------------
    consolidated_path = OUTPUT_DIR / "reference_floods_consolidated.jsonl"
    print("[ 9/9 ] Consolidating ...")
    consolidate_flood_references(
        combined_path = combined_path,
        output_path   = consolidated_path,
    )

    print(f"\n{'='*60}")
    print(f"Complete. Final dataset: {consolidated_path}")
    print(f"{'='*60}\n")
    print("Next step: python -m Builder_Reference.explore")


if __name__ == "__main__":
    run()
