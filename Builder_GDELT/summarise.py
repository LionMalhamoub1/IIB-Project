"""
summarise.py - GDELT flood enrichment dataset diagnostic report.

Prints a structured report covering:
  - Overview         : total records, enrichment rate, skip-reason breakdown
  - Field completeness : coverage for each key field (mirrors reference builder)
  - Impact summary   : deaths, affected, displaced, flooded area stats
  - Temporal         : year-by-year counts, monthly seasonality
  - Coordinate quality : geo_source split, _geo_verified rate, rejections
  - GEE field fill rates : every satellite field with fill % and value ranges
  - Classification   : expert vs LLM split, confidence distribution
  - Data quality flags : missing dates, missing coords, coord rejections
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

# -- Paths --------------------------------------------------------------------

ROOT         = Path(__file__).resolve().parents[1]
ENRICHED_ALL = ROOT / "Builder_GDELT" / "results" / "enriched_floods" / "all_floods_enriched.jsonl"
ENRICHED_DIR = ROOT / "Builder_GDELT" / "results" / "enriched_floods"

# -- Satellite fields to report on --------------------------------------------
#
# Grouped by instrument/source. Each tuple: (field_name, short_label, unit).

GEE_FIELDS = [
    # GPM - high-resolution rainfall intensity
    ("gpm_1d_total_mm",      "GPM 1-day total",       "mm"),
    ("gpm_3d_total_mm",      "GPM 3-day total",       "mm"),
    ("gpm_7d_total_mm",      "GPM 7-day total",       "mm"),
    ("gpm_peak_daily_mm",    "GPM peak daily",         "mm"),
    ("gpm_peak_3h_mm",       "GPM peak 3-hour",        "mm"),
    # CHIRPS - daily rainfall with long climatological baseline
    ("chirps_3d_total_mm",   "CHIRPS 3-day total",    "mm"),
    ("chirps_7d_total_mm",   "CHIRPS 7-day total",    "mm"),
    ("chirps_14d_total_mm",  "CHIRPS 14-day total",   "mm"),
    ("chirps_30d_total_mm",  "CHIRPS 30-day total",   "mm"),
    ("chirps_peak_daily_mm", "CHIRPS peak daily",      "mm"),
    ("chirps_7d_baseline_mm","CHIRPS 7d baseline",    "mm"),
    ("chirps_7d_anom_mm",    "CHIRPS 7d anomaly",     "mm"),
    ("chirps_7d_anom_pct",   "CHIRPS 7d anomaly",      "%"),
    # ERA5-Land - soil moisture and runoff
    ("era5_soil_moisture_day0",       "ERA5 soil moist. day-0",   "m3/m3"),
    ("era5_soil_moisture_7d_mean",    "ERA5 soil moist. 7d mean", "m3/m3"),
    ("era5_soil_moisture_30d_mean",   "ERA5 soil moist. 30d mean","m3/m3"),
    ("era5_soil_moisture_deep_day0",  "ERA5 deep moist. day-0",   "m3/m3"),
    ("era5_soil_moisture_deep_7d_mean","ERA5 deep moist. 7d mean","m3/m3"),
    ("era5_precip_7d_mm",             "ERA5 precip 7d",           "mm"),
    ("era5_runoff_7d_mm",             "ERA5 runoff 7d",           "mm"),
    # SPI - statistical rainfall anomaly
    ("spi_30d",              "SPI-30",                 "index"),
    ("spi_30d_pct",          "SPI-30 percentile",      "%"),
    # Static / exposure
    ("pop_count_25km",       "Population (25 km)",    "persons"),
    ("pop_density_km2",      "Pop. density",          "persons/km2"),
    ("jrc_occurrence_pct",   "JRC water occurrence",   "%"),
    ("jrc_recurrence_pct",   "JRC water recurrence",   "%"),
    ("terrain_slope_mean",   "Terrain slope",          "deg"),
]

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]


# -- Helpers ------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _load_date_range(start: date, end: date) -> list[dict]:
    records = []
    d = start
    while d <= end:
        p = ENRICHED_DIR / d.strftime("%Y%m%d") / "floods_enriched.jsonl"
        if p.exists():
            records.extend(_load_jsonl(p))
        d += timedelta(days=1)
    return records


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "  n/a"
    return f"{100 * num / den:5.1f}%"


def _bar(value: float, width: int = 30, char: str = "#") -> str:
    filled = round(max(0.0, min(1.0, value)) * width)
    return char * filled + "." * (width - filled)


def _median(vals: list) -> float:
    s = sorted(vals)
    n = len(s)
    return (s[n // 2] + s[(n - 1) // 2]) / 2 if n else 0.0


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# -- Section printers ---------------------------------------------------------

def print_overview(records: list[dict]) -> None:
    section("OVERVIEW")

    total    = len(records)
    enriched = sum(1 for r in records if r.get("_enriched"))
    not_enr  = total - enriched

    print(f"\n  Total flood records      : {total:>8,}")
    print(f"  Enriched (_enriched=True): {enriched:>8,}  {_pct(enriched, total)}")
    print(f"  Not enriched             : {not_enr:>8,}  {_pct(not_enr, total)}")

    skip = Counter(
        r.get("_enrich_skip_reason") or "unknown"
        for r in records if not r.get("_enriched")
    )
    if skip:
        print(f"\n  Skip reasons:")
        for reason, cnt in skip.most_common():
            print(f"    {reason:<25} {cnt:>7,}  {_pct(cnt, not_enr)}")

    date_src = Counter(r.get("_date_source") or "none" for r in records)
    print(f"\n  Date source breakdown:")
    for src, cnt in date_src.most_common():
        print(f"    {src:<25} {cnt:>7,}  {_pct(cnt, total)}")


def _field_val(record: dict, field: str):
    """Return the value of a field, checking top-level then extras dict."""
    v = record.get(field)
    if v is None:
        v = (record.get("extras") or {}).get(field)
    return v


def print_field_completeness(records: list[dict]) -> None:
    section("FIELD COMPLETENESS")

    n = len(records)
    if n == 0:
        return

    # Mirror the reference builder field list  -  using GDELT equivalents.
    # LLM detail fields (deaths, affected, etc.) live inside the 'extras' dict;
    # _field_val() checks both top-level and extras so counts are correct.
    fields = [
        ("event_date",        "Start date"),
        # End date: GDELT articles are point-in-time reports, no end date extracted
        # Country ISO: used during geocoding validation but not written to the record
        ("lat",               "Latitude"),
        ("lon",               "Longitude"),
        ("death_toll",        "Deaths"),
        ("affected_count",    "Affected persons"),
        ("displaced_count",   "Displaced"),
        # Damage (USD k): not in the LLM extraction prompt
        ("area_affected_km2", "Flooded area km2"),
        ("main_cause",        "Main cause"),
        ("severity",          "Severity"),
        # GLIDE number: not in GDELT source data
        # Event name: not extracted
        ("location_name",     "Location name"),
    ]

    print(f"\n  {'Field':<28} {'Present':>7}  {'%':>6}  Coverage bar")
    print(f"  {'-'*70}")
    for field, label in fields:
        cnt = sum(
            1 for r in records
            if _field_val(r, field) is not None
            and str(_field_val(r, field)).strip() not in ("", "None")
        )
        bar = _bar(cnt / n, width=30)
        print(f"  {label:<28} {cnt:>7,}  {_pct(cnt, n)}  {bar}")

    print(f"\n  Not available from GDELT pipeline:")
    print(f"    End date, Country ISO, Damage (USD k), GLIDE number, Event name")


def print_impact_summary(records: list[dict]) -> None:
    section("IMPACT DATA SUMMARY  (LLM-extracted fields)")

    n = len(records)

    def _stats(field: str):
        vals = []
        for r in records:
            v = _field_val(r, field)
            if v is None:
                continue
            try:
                fv = float(v)
                if fv > 0:
                    vals.append(fv)
            except (TypeError, ValueError):
                pass
        if not vals:
            return None
        vals.sort()
        return {
            "count":  len(vals),
            "sum":    sum(vals),
            "min":    vals[0],
            "p25":    vals[len(vals) // 4],
            "median": vals[len(vals) // 2],
            "p75":    vals[3 * len(vals) // 4],
            "max":    vals[-1],
        }

    for field, label, unit in [
        ("death_toll",        "Deaths",       "persons"),
        ("affected_count",    "Affected",     "persons"),
        ("displaced_count",   "Displaced",    "persons"),
        ("area_affected_km2", "Flooded area", "km2"),
    ]:
        s = _stats(field)
        if s is None:
            print(f"\n  {label}: no data")
            continue
        print(f"\n  {label} ({unit})  [{s['count']:,} events with data / {n:,} total]")
        print(f"    Total  : {s['sum']:>15,.0f}")
        print(f"    Min    : {s['min']:>15,.0f}")
        print(f"    P25    : {s['p25']:>15,.0f}")
        print(f"    Median : {s['median']:>15,.0f}")
        print(f"    P75    : {s['p75']:>15,.0f}")
        print(f"    Max    : {s['max']:>15,.0f}")


def print_temporal(records: list[dict]) -> None:
    section("TEMPORAL COVERAGE")

    year_counts  = Counter()
    month_counts = Counter()

    for r in records:
        d = r.get("event_date") or r.get("publish_date") or ""
        if len(d) >= 4:
            year_counts[d[:4]] += 1
        if len(d) >= 7:
            try:
                month_counts[int(d[5:7])] += 1
            except ValueError:
                pass

    if year_counts:
        max_yr = max(year_counts.values())
        print(f"\n  {'Year':<6} {'Records':>8}  Sparkline")
        print(f"  {'-' * 55}")
        for yr in sorted(year_counts):
            cnt = year_counts[yr]
            print(f"  {yr:<6} {cnt:>8,}  {_bar(cnt / max_yr, width=40)}")

    if month_counts:
        max_mo = max(month_counts.values())
        print(f"\n  Monthly distribution (all years combined):")
        print(f"  {'Month':<6} {'Records':>8}  Seasonality")
        print(f"  {'-' * 55}")
        for m in range(1, 13):
            cnt = month_counts.get(m, 0)
            print(f"  {MONTHS[m-1]:<6} {cnt:>8,}  {_bar(cnt / max_mo, width=40)}")


def print_coordinate_quality(records: list[dict]) -> None:
    section("COORDINATE & GEOCODING QUALITY")

    total = len(records)

    geo_src = Counter(r.get("geo_source") or "None" for r in records)
    print(f"\n  Geo source breakdown:")
    for src, cnt in geo_src.most_common():
        print(f"    {str(src):<30} {cnt:>7,}  {_pct(cnt, total)}  {_bar(cnt / total, width=25)}")

    verified   = sum(1 for r in records if r.get("_geo_verified"))
    unverified = total - verified
    print(f"\n  _geo_verified = True  : {verified:>8,}  {_pct(verified, total)}")
    print(f"  _geo_verified = False : {unverified:>8,}  {_pct(unverified, total)}")

    rejected = [r for r in records if r.get("_coord_rejected_km")]
    if rejected:
        dists = sorted(r["_coord_rejected_km"] for r in rejected)
        print(f"\n  Coords rejected (GDELT centroid >500 km from Nominatim):")
        print(f"    Count   : {len(rejected):,}")
        print(f"    Min km  : {dists[0]:,.0f}")
        print(f"    Median  : {_median(dists):,.0f}")
        print(f"    Max km  : {dists[-1]:,.0f}")

    no_coords = sum(1 for r in records if r.get("lat") is None or r.get("lon") is None)
    print(f"\n  No coordinates (lat/lon null) : {no_coords:>6,}  {_pct(no_coords, total)}")


def print_gee_fields(records: list[dict]) -> None:
    section("GEE ENRICHMENT - FIELD FILL RATES & VALUE RANGES")

    enriched = [r for r in records if r.get("_enriched")]
    n = len(enriched)
    if n == 0:
        print("  No enriched records found.")
        return

    print(f"\n  Computed over {n:,} enriched records.\n")
    print(f"  {'Field':<30} {'Fill':>6}   {'Min':>10}  {'Median':>10}  {'Max':>10}  {'Unit'}")
    print(f"  {'-' * 82}")

    prev_prefix = ""
    for field, label, unit in GEE_FIELDS:
        prefix = field.split("_")[0]
        if prev_prefix and prefix != prev_prefix:
            print()
        prev_prefix = prefix

        vals = [r[field] for r in enriched if r.get(field) is not None]
        fill_pct = _pct(len(vals), n)

        if vals:
            mn  = min(vals)
            med = _median(vals)
            mx  = max(vals)
            print(f"  {label:<30} {fill_pct}   {mn:>10.2f}  {med:>10.2f}  {mx:>10.2f}  {unit}")
        else:
            print(f"  {label:<30} {fill_pct}   {'n/a':>10}  {'n/a':>10}  {'n/a':>10}  {unit}")


def print_classification(records: list[dict]) -> None:
    section("CLASSIFICATION SOURCE & LLM CONFIDENCE")

    total = len(records)

    cls_src = Counter(r.get("classification_source") or "unknown" for r in records)
    print(f"\n  Classification source:")
    for src, cnt in cls_src.most_common():
        print(f"    {src:<20} {cnt:>7,}  {_pct(cnt, total)}  {_bar(cnt / total, width=30)}")

    llm = [r for r in records if r.get("classification_source") == "llm"]
    if llm:
        conf_vals = [r.get("confidence") or 0.0 for r in llm]
        buckets = Counter()
        for c in conf_vals:
            if c == 0.0:
                buckets["0.0 (unknown/failed)"] += 1
            elif c < 0.5:
                buckets["0.0-0.5"] += 1
            elif c < 0.7:
                buckets["0.5-0.7"] += 1
            elif c < 0.9:
                buckets["0.7-0.9"] += 1
            else:
                buckets["0.9-1.0"] += 1

        n_llm = len(llm)
        print(f"\n  LLM confidence distribution ({n_llm:,} LLM records):")
        for bucket in ["0.0 (unknown/failed)", "0.0-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]:
            cnt = buckets.get(bucket, 0)
            print(f"    {bucket:<22} {cnt:>7,}  {_pct(cnt, n_llm)}  {_bar(cnt / n_llm, width=28)}")

        with_loc = sum(1 for r in llm if r.get("location_name", "").strip())
        print(f"\n  LLM records with location_name : {with_loc:,} / {n_llm:,}  {_pct(with_loc, n_llm)}")


def print_quality_flags(records: list[dict]) -> None:
    section("DATA QUALITY FLAGS")

    total = len(records)

    no_event_date   = sum(1 for r in records if not r.get("event_date"))
    no_publish_date = sum(1 for r in records if not r.get("publish_date"))
    no_lat          = sum(1 for r in records if r.get("lat") is None)
    no_location     = sum(1 for r in records if not (r.get("location_name") or "").strip())
    no_gee          = sum(1 for r in records
                          if r.get("_enriched") and r.get("gpm_7d_total_mm") is None
                          and r.get("chirps_7d_total_mm") is None)
    date_fallback   = sum(1 for r in records if r.get("_date_source") == "yyyymmdd_fallback")

    print(f"\n  Total records            : {total:,}")
    print(f"  Missing event_date       : {no_event_date:>8,}  {_pct(no_event_date, total)}")
    print(f"  Missing publish_date     : {no_publish_date:>8,}  {_pct(no_publish_date, total)}")
    print(f"  Missing latitude         : {no_lat:>8,}  {_pct(no_lat, total)}")
    print(f"  Missing location_name    : {no_location:>8,}  {_pct(no_location, total)}")
    print(f"  Enriched but no GEE data : {no_gee:>8,}  {_pct(no_gee, total)}")
    print(f"  Date from folder fallback: {date_fallback:>8,}  {_pct(date_fallback, total)}"
          f"  (no_date events recovered by Stage 3)")


# -- Entry point --------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise the GDELT flood enrichment dataset."
    )
    parser.add_argument("--file", type=Path, default=None,
                        help="Path to a specific JSONL file (overrides default)")
    parser.add_argument("--from", dest="date_from", metavar="YYYYMMDD", default=None,
                        help="Start date for day-folder scan (inclusive)")
    parser.add_argument("--to",   dest="date_to",   metavar="YYYYMMDD", default=None,
                        help="End date for day-folder scan (inclusive)")
    args = parser.parse_args()

    if args.file:
        path = args.file
        if not path.exists():
            print(f"ERROR: {path} not found.")
            return
        print(f"\nLoading {path} ...")
        records = _load_jsonl(path)
        label = path.name

    elif args.date_from or args.date_to:
        try:
            start = date(int(args.date_from[:4]), int(args.date_from[4:6]), int(args.date_from[6:]))
            end   = date(int(args.date_to[:4]),   int(args.date_to[4:6]),   int(args.date_to[6:]))
        except (TypeError, ValueError):
            print("ERROR: --from and --to must both be YYYYMMDD.")
            return
        print(f"\nScanning day folders {args.date_from} - {args.date_to} ...")
        records = _load_date_range(start, end)
        label = f"{args.date_from} - {args.date_to}"

    else:
        if not ENRICHED_ALL.exists():
            print(f"ERROR: {ENRICHED_ALL} not found. Run the pipeline first.")
            return
        print(f"\nLoading {ENRICHED_ALL} ...")
        records = _load_jsonl(ENRICHED_ALL)
        label = "all_floods_enriched.jsonl"

    if not records:
        print("No records found.")
        return

    print(f"Loaded {len(records):,} records  [{label}]\n")

    print_overview(records)
    print_field_completeness(records)
    print_impact_summary(records)
    print_temporal(records)
    print_coordinate_quality(records)
    print_gee_fields(records)
    print_classification(records)
    print_quality_flags(records)

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
