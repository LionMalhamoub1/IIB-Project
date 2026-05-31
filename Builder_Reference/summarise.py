"""
Summarise the consolidated official flood reference database.

Prints a structured report covering:
  - Record counts (total, by source, consolidated vs raw)
  - Temporal coverage (year-by-year breakdown, coverage heatmap)
  - Cross-source merging stats (matched-source combinations)
  - Geographic coverage (top countries, continent distribution)
  - Field completeness (what fraction of events have each key field)
  - Impact data availability (deaths / affected / damage)
  - Data quality flags (missing dates, missing ISO, short vs long events)

Usage:
    python -m Builder_Reference.helper_scripts.reference.summarise
    python -m Builder_Reference.helper_scripts.reference.summarise --consolidated   # default
    python -m Builder_Reference.helper_scripts.reference.summarise --combined       # raw pre-consolidation
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pycountry

BASE = Path("cache/floods")
ENRICHED     = BASE / "reference_floods_enriched.jsonl"
CONSOLIDATED = BASE / "reference_floods_consolidated.jsonl"
COMBINED     = BASE / "reference_floods_combined.jsonl"

SOURCES = ["DFO", "EM-DAT", "GDACS", "HANZE", "IFRC", "ReliefWeb", "Copernicus",
           "Desinventar", "Desinventar-AGG", "CONSOLIDATED"]

# Comprehensive ISO-3 -> continent mapping
_ISO3_CONTINENT: dict[str, str] = {
    # Africa
    "AGO":"Africa","BDI":"Africa","BEN":"Africa","BFA":"Africa","BWA":"Africa",
    "CAF":"Africa","CIV":"Africa","CMR":"Africa","COD":"Africa","COG":"Africa",
    "COM":"Africa","CPV":"Africa","DJI":"Africa","DZA":"Africa","EGY":"Africa",
    "ERI":"Africa","ETH":"Africa","GAB":"Africa","GHA":"Africa","GIN":"Africa",
    "GMB":"Africa","GNB":"Africa","GNQ":"Africa","KEN":"Africa","LBR":"Africa",
    "LBY":"Africa","LSO":"Africa","MAR":"Africa","MDG":"Africa","MLI":"Africa",
    "MOZ":"Africa","MRT":"Africa","MUS":"Africa","MWI":"Africa","MYT":"Africa",
    "NAM":"Africa","NER":"Africa","NGA":"Africa","REU":"Africa","RWA":"Africa",
    "SDN":"Africa","SEN":"Africa","SHN":"Africa","SLE":"Africa","SOM":"Africa",
    "SSD":"Africa","STP":"Africa","SWZ":"Africa","SYC":"Africa","TCD":"Africa",
    "TGO":"Africa","TUN":"Africa","TZA":"Africa","UGA":"Africa","ZAF":"Africa",
    "ZMB":"Africa","ZWE":"Africa",
    # Asia
    "AFG":"Asia","ARE":"Asia","ARM":"Asia","AZE":"Asia","BGD":"Asia","BHR":"Asia",
    "BRN":"Asia","BTN":"Asia","CHN":"Asia","CYP":"Asia","GEO":"Asia","HKG":"Asia",
    "IDN":"Asia","IND":"Asia","IRN":"Asia","IRQ":"Asia","ISR":"Asia","JOR":"Asia",
    "JPN":"Asia","KAZ":"Asia","KGZ":"Asia","KHM":"Asia","KOR":"Asia","KWT":"Asia",
    "LAO":"Asia","LBN":"Asia","LKA":"Asia","MAC":"Asia","MDV":"Asia","MMR":"Asia",
    "MNG":"Asia","MYS":"Asia","NPL":"Asia","OMN":"Asia","PAK":"Asia","PHL":"Asia",
    "PRK":"Asia","PSE":"Asia","QAT":"Asia","SAU":"Asia","SGP":"Asia","SYR":"Asia",
    "THA":"Asia","TJK":"Asia","TKM":"Asia","TLS":"Asia","TWN":"Asia","UZB":"Asia",
    "VNM":"Asia","YEM":"Asia",
    # Europe
    "ALB":"Europe","AND":"Europe","AUT":"Europe","BEL":"Europe","BGR":"Europe",
    "BIH":"Europe","BLR":"Europe","CHE":"Europe","CYP":"Europe","CZE":"Europe",
    "DEU":"Europe","DNK":"Europe","ESP":"Europe","EST":"Europe","FIN":"Europe",
    "FRA":"Europe","FRO":"Europe","GBR":"Europe","GIB":"Europe","GRC":"Europe",
    "HRV":"Europe","HUN":"Europe","IMN":"Europe","IRL":"Europe","ISL":"Europe",
    "ITA":"Europe","KOS":"Europe","LIE":"Europe","LTU":"Europe","LUX":"Europe",
    "LVA":"Europe","MCO":"Europe","MDA":"Europe","MKD":"Europe","MLT":"Europe",
    "MNE":"Europe","NLD":"Europe","NOR":"Europe","POL":"Europe","PRT":"Europe",
    "ROU":"Europe","RUS":"Europe","SMR":"Europe","SRB":"Europe","SVK":"Europe",
    "SVN":"Europe","SWE":"Europe","TUR":"Europe","UKR":"Europe","VAT":"Europe",
    "XKX":"Europe",
    # N. America (includes Central America & Caribbean)
    "ABW":"N. America","AIA":"N. America","ATG":"N. America","BHS":"N. America",
    "BLZ":"N. America","BMU":"N. America","BRB":"N. America","CAN":"N. America",
    "CRI":"N. America","CUB":"N. America","CYM":"N. America","DMA":"N. America",
    "DOM":"N. America","GLP":"N. America","GRD":"N. America","GTM":"N. America",
    "HND":"N. America","HTI":"N. America","JAM":"N. America","KNA":"N. America",
    "LCA":"N. America","MEX":"N. America","MSR":"N. America","MTQ":"N. America",
    "NIC":"N. America","PAN":"N. America","PRI":"N. America","SLV":"N. America",
    "TCA":"N. America","TTO":"N. America","USA":"N. America","VCT":"N. America",
    "VIR":"N. America",
    # S. America
    "ARG":"S. America","BOL":"S. America","BRA":"S. America","CHL":"S. America",
    "COL":"S. America","ECU":"S. America","FLK":"S. America","GUF":"S. America",
    "GUY":"S. America","PER":"S. America","PRY":"S. America","SUR":"S. America",
    "URY":"S. America","VEN":"S. America",
    # Oceania
    "AUS":"Oceania","COK":"Oceania","FJI":"Oceania","FSM":"Oceania","GUM":"Oceania",
    "KIR":"Oceania","MHL":"Oceania","MNP":"Oceania","NCL":"Oceania","NFK":"Oceania",
    "NIU":"Oceania","NRU":"Oceania","NZL":"Oceania","PYF":"Oceania","PLW":"Oceania",
    "PNG":"Oceania","SLB":"Oceania","TKL":"Oceania","TON":"Oceania","TUV":"Oceania",
    "VUT":"Oceania","WSM":"Oceania",
}


def _continent(iso3: str | None) -> str:
    if not iso3:
        return "Unknown"
    return _ISO3_CONTINENT.get(iso3, "Unknown")


def _load(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _year(rec: dict) -> str | None:
    d = rec.get("date_start") or ""
    return d[:4] if len(d) >= 4 else None


def _bar(value: float, width: int = 30, char: str = "#") -> str:
    filled = round(value * width)
    return char * filled + "." * (width - filled)


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "  n/a"
    return f"{100*num/den:5.1f}%"


# ---------------------------------------------------------
# SECTION PRINTERS
# ---------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_overview(records: list[dict], label: str) -> None:
    section(f"OVERVIEW  [{label}]")

    total = len(records)
    print(f"  Total records         : {total:>8,}")

    # Exclude Desinventar children (they are sub-records, not events)
    events = [r for r in records if r.get("source") != "Desinventar"]
    print(f"  Event-level records   : {len(events):>8,}  (excl. Desinventar district children)")

    # Source breakdown
    source_counts = Counter(r.get("source", "?") for r in records)
    print(f"\n  {'Source':<22} {'Records':>8}  {'of total':>8}")
    print(f"  {'-'*42}")
    for src in sorted(source_counts, key=lambda s: -source_counts[s]):
        cnt = source_counts[src]
        print(f"  {src:<22} {cnt:>8,}  {_pct(cnt, total)}")

    # Multi-source merges
    multi = [r for r in records if len(r.get("matched_sources", [])) > 1]
    if multi:
        print(f"\n  Multi-source merged   : {len(multi):>8,}  events confirmed by >=2 sources")
        src_pair_counts = Counter()
        for r in multi:
            srcs = sorted(r.get("matched_sources", []))
            for i in range(len(srcs)):
                for j in range(i+1, len(srcs)):
                    src_pair_counts[(srcs[i], srcs[j])] += 1
        print(f"\n  Top source pairings (cross-confirmed events):")
        for (s1, s2), cnt in src_pair_counts.most_common(10):
            print(f"    {s1:<18} x {s2:<18} : {cnt:>5,}")


def print_temporal(records: list[dict]) -> None:
    section("TEMPORAL COVERAGE")

    events = [r for r in records if r.get("source") != "Desinventar"]
    year_counts: dict[str, Counter] = defaultdict(Counter)

    for r in events:
        y = _year(r)
        if y:
            src = r.get("source", "?")
            # For consolidated, show by matched sources
            if src == "CONSOLIDATED":
                for s in (r.get("matched_sources") or [src]):
                    year_counts[y][s] += 1
            else:
                year_counts[y][src] += 1

    all_years = sorted(year_counts.keys())
    if not all_years:
        print("  No dated records.")
        return

    # Total per year
    year_totals = {y: sum(year_counts[y].values()) for y in all_years}
    max_total   = max(year_totals.values()) if year_totals else 1

    print(f"\n  {'Year':<6} {'Events':>7}  Sparkline")
    print(f"  {'-'*55}")
    for y in all_years:
        t = year_totals[y]
        bar = _bar(t / max_total, width=40)
        print(f"  {y:<6} {t:>7,}  {bar}")

    no_date = sum(1 for r in events if not _year(r))
    if no_date:
        print(f"\n  Records with no date  : {no_date:,}")

    print(f"\n  Date range            : {all_years[0]} - {all_years[-1]}")
    print(f"  Years with data       : {len(all_years)}")


def print_source_year_matrix(records: list[dict]) -> None:
    section("SOURCE x YEAR MATRIX  (event counts)")

    events = [r for r in records if r.get("source") not in ("Desinventar",)]

    # For consolidated records expand matched_sources
    src_year: dict[str, Counter] = defaultdict(Counter)
    for r in events:
        y = _year(r)
        if not y:
            continue
        src = r.get("source", "?")
        if src == "CONSOLIDATED":
            for s in (r.get("matched_sources") or [src]):
                src_year[s][y] += 1
        else:
            src_year[src][y] += 1

    all_sources = sorted(src_year.keys())
    all_years   = sorted({y for c in src_year.values() for y in c})

    if not all_years or not all_sources:
        return

    col_w = 6
    hdr = f"  {'Source':<22}" + "".join(f"{y:>{col_w}}" for y in all_years)
    print(hdr)
    print("  " + "-" * (22 + col_w * len(all_years)))
    for src in all_sources:
        row = f"  {src:<22}"
        for y in all_years:
            c = src_year[src].get(y, 0)
            row += f"{'':>{col_w-len(str(c))}}{c if c else '.'}" if c else f"{'':>{col_w-1}}."
        print(row)


def print_geographic(records: list[dict]) -> None:
    section("GEOGRAPHIC COVERAGE")

    events = [r for r in records if r.get("source") not in ("Desinventar",)]

    iso_counts   = Counter(r.get("country_iso") for r in events if r.get("country_iso"))
    no_iso       = sum(1 for r in events if not r.get("country_iso"))
    continent_counts = Counter(_continent(iso) for iso in iso_counts.elements())

    print(f"\n  Unique countries      : {len(iso_counts):>6}")
    print(f"  Records missing ISO   : {no_iso:>6,}")

    print(f"\n  Continent distribution:")
    total_iso = sum(iso_counts.values())
    for cont, cnt in continent_counts.most_common():
        bar = _bar(cnt / total_iso, width=35)
        print(f"    {cont:<14} {cnt:>6,}  {_pct(cnt, total_iso)}  {bar}")

    print(f"\n  Top 30 countries:")
    print(f"  {'ISO':<6} {'Country':<30} {'Events':>7}")
    print(f"  {'-'*46}")
    for iso, cnt in iso_counts.most_common(30):
        try:
            name = pycountry.countries.get(alpha_3=iso).name[:29]
        except Exception:
            name = iso
        print(f"  {iso:<6} {name:<30} {cnt:>7,}")


def print_field_completeness(records: list[dict]) -> None:
    section("FIELD COMPLETENESS")

    events = [r for r in records if r.get("source") not in ("Desinventar",)]
    n = len(events)
    if n == 0:
        return

    fields = [
        ("date_start",           "Start date"),
        ("date_end",             "End date"),
        ("country_iso",          "ISO-3 country"),
        ("lat",                  "Latitude"),
        ("lon",                  "Longitude"),
        ("dead",                 "Deaths"),
        ("affected",             "Affected persons"),
        ("displaced",            "Displaced"),
        ("damage_usd_thousands", "Damage (USD k)"),
        ("area_km2",             "Flooded area km2"),
        ("main_cause",           "Main cause"),
        ("severity",             "Severity"),
        ("glide_number",         "GLIDE number"),
        ("event_name",           "Event name"),
        ("location_name",        "Location name"),
    ]

    print(f"\n  {'Field':<28} {'Present':>7}  {'%':>6}  Coverage bar")
    print(f"  {'-'*70}")
    for field, label in fields:
        cnt = sum(1 for r in events if r.get(field) is not None)
        frac = cnt / n
        bar = _bar(frac, width=30)
        print(f"  {label:<28} {cnt:>7,}  {_pct(cnt, n)}  {bar}")


def print_impact_summary(records: list[dict]) -> None:
    section("IMPACT DATA SUMMARY")

    events = [r for r in records if r.get("source") not in ("Desinventar",)]
    n = len(events)

    def _stats(field: str):
        vals = [r[field] for r in events if r.get(field) is not None and r[field] > 0]
        if not vals:
            return None
        vals.sort()
        return {
            "count": len(vals),
            "sum":   sum(vals),
            "min":   vals[0],
            "p25":   vals[len(vals)//4],
            "median":vals[len(vals)//2],
            "p75":   vals[3*len(vals)//4],
            "max":   vals[-1],
        }

    for field, label, unit in [
        ("dead",                 "Deaths",           "persons"),
        ("affected",             "Affected",         "persons"),
        ("displaced",            "Displaced",        "persons"),
        ("damage_usd_thousands", "Damage",           "USD thousands"),
        ("area_km2",             "Flooded area",     "km2"),
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


def print_quality_flags(records: list[dict]) -> None:
    section("DATA QUALITY FLAGS")

    events = [r for r in records if r.get("source") not in ("Desinventar",)]
    n = len(events)

    from datetime import date as _date

    def _parse(s):
        try:
            p = s[:10].split("-")
            return _date(int(p[0]), int(p[1]), int(p[2]))
        except Exception:
            return None

    no_start  = sum(1 for r in events if not r.get("date_start"))
    no_end    = sum(1 for r in events if not r.get("date_end"))
    no_iso    = sum(1 for r in events if not r.get("country_iso"))
    no_coord  = sum(1 for r in events if r.get("lat") is None or r.get("lon") is None)
    no_impact = sum(1 for r in events
                    if all(r.get(f) is None for f in
                           ("dead","affected","displaced","damage_usd_thousands")))

    # Long events (>180 days) — likely umbrella/aggregate entries
    long_events = []
    for r in events:
        ds = _parse(r.get("date_start", ""))
        de = _parse(r.get("date_end", ""))
        if ds and de and (de - ds).days > 180:
            long_events.append(((de - ds).days, r.get("source"), r.get("country_iso"), str(ds)))

    long_events.sort(reverse=True)

    print(f"\n  Total event records   : {n:,}")
    print(f"  Missing start date    : {no_start:>7,}  {_pct(no_start, n)}")
    print(f"  Missing end date      : {no_end:>7,}  {_pct(no_end, n)}")
    print(f"  Missing ISO country   : {no_iso:>7,}  {_pct(no_iso, n)}")
    print(f"  Missing coordinates   : {no_coord:>7,}  {_pct(no_coord, n)}")
    print(f"  No impact data at all : {no_impact:>7,}  {_pct(no_impact, n)}")
    print(f"  Events >180 days long : {len(long_events):>7,}  {_pct(len(long_events), n)}")

    if long_events:
        print(f"\n  Longest events (likely umbrella entries):")
        print(f"  {'Days':>6}  {'Source':<16}  {'ISO':<5}  Start")
        print(f"  {'-'*44}")
        for days, src, iso, start in long_events[:15]:
            print(f"  {days:>6}  {(src or '?'):<16}  {(iso or '?'):<5}  {start}")


def print_consolidation_stats(records: list[dict]) -> None:
    """Only meaningful for the consolidated file."""
    consolidated = [r for r in records if r.get("source") == "CONSOLIDATED"]
    if not consolidated:
        return

    section("CONSOLIDATION STATS")

    n = len(consolidated)
    source_counts_per_event = Counter(
        len(r.get("matched_sources", [])) for r in consolidated
    )

    print(f"\n  Consolidated events   : {n:,}")
    print(f"\n  Sources per event:")
    for k in sorted(source_counts_per_event):
        cnt = source_counts_per_event[k]
        bar = _bar(cnt / n, width=35)
        print(f"    {k} source(s)  {cnt:>7,}  {_pct(cnt, n)}  {bar}")

    # Which source combinations appear most?
    combo_counts = Counter(
        tuple(sorted(r.get("matched_sources", []))) for r in consolidated
        if len(r.get("matched_sources", [])) > 1
    )
    print(f"\n  Most common multi-source combinations:")
    print(f"  {'Combination':<55} {'Count':>6}")
    print(f"  {'-'*64}")
    for combo, cnt in combo_counts.most_common(15):
        label = " + ".join(combo)
        print(f"  {label:<55} {cnt:>6,}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def _is_gdacs_green(record: dict) -> bool:
    """True if this record is a GDACS green alert (or came from one after merging)."""
    src = record.get("source", "")
    sev = (record.get("severity") or "").lower()
    if src == "GDACS":
        return sev == "green"
    if src == "CONSOLIDATED":
        # Merged record: flag as green only if GDACS is the SOLE source and severity=green
        srcs = record.get("matched_sources", [])
        if srcs == ["GDACS"]:
            return sev == "green"
    return False


def print_desinventar_section(records: list[dict]) -> None:
    section("DESINVENTAR — DISTRICT CHILDREN & AGGREGATION")

    all_desagg    = [r for r in records if r.get("source") == "Desinventar-AGG"]
    all_children  = [r for r in records if r.get("source") == "Desinventar"]

    print(f"""
  Desinventar records events at sub-national (district/municipality) level —
  one row per administrative unit affected. A single real-world flood can
  appear as dozens of separate rows across different departments/provinces.

  To make these comparable to country-level sources (EM-DAT, DFO, etc.),
  the consolidation pipeline runs a pre-aggregation step (Pass 0):

    1. District records within {7} days of each other in the same country
       are grouped into one synthetic "Desinventar-AGG" cluster.
    2. The AGG cluster sums impact fields (deaths, affected, etc.) across
       all constituent districts — giving a country-level estimate.
    3. The AGG record participates in cross-source matching (Passes 1 & 2).
    4. The original district records are kept as children in the output,
       each tagged with a cluster_id pointing to their parent AGG record.

  So yes — all districts belonging to the same flood are represented as
  ONE Desinventar-AGG record for matching/consolidation purposes, while
  the underlying district detail is preserved and queryable via cluster_id.
""")

    n_clusters  = len(all_desagg)
    n_children  = len(all_children)
    if n_clusters == 0:
        print("  No Desinventar records found.")
        return

    cluster_sizes = Counter(r.get("cluster_size", 1) for r in all_desagg)
    avg_size = n_children / n_clusters if n_clusters else 0

    print(f"  AGG clusters (synthetic country-level records) : {n_clusters:>7,}")
    print(f"  District children (original sub-national rows) : {n_children:>7,}")
    print(f"  Average districts per cluster                  : {avg_size:>10.1f}")

    print(f"\n  Cluster size distribution:")
    for size in sorted(cluster_sizes)[:12]:
        cnt = cluster_sizes[size]
        bar = _bar(cnt / n_clusters, width=30)
        print(f"    {size:>4} district(s)  {cnt:>6,}  {_pct(cnt, n_clusters)}  {bar}")
    if len(cluster_sizes) > 12:
        large = sum(v for k, v in cluster_sizes.items() if k > 12)
        print(f"    >12 district(s)  {large:>6,}  {_pct(large, n_clusters)}")

    # How many AGG records merged with another source?
    merged_agg = [r for r in records
                  if r.get("source") == "CONSOLIDATED"
                  and "Desinventar-AGG" in r.get("matched_sources", [])]
    print(f"\n  AGG clusters matched to another source         : {len(merged_agg):>7,}  "
          f"({_pct(len(merged_agg), n_clusters)} of clusters)")

    # Countries with most clusters
    country_clusters = Counter(r.get("country_iso") for r in all_desagg)
    print(f"\n  Countries with most Desinventar clusters (top 15):")
    print(f"  {'ISO':<6} {'Clusters':>8}  {'Districts':>10}")
    print(f"  {'-'*28}")
    child_by_country = Counter(r.get("country_iso") for r in all_children)
    for iso, cnt in country_clusters.most_common(15):
        try:
            name = pycountry.countries.get(alpha_3=iso).name[:20] if iso else "?"
        except Exception:
            name = iso or "?"
        print(f"  {(iso or '?'):<6} {cnt:>8,}  {child_by_country.get(iso, 0):>10,}  {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, default=None,
                        help="Path to a JSONL file (default: reference_floods_enriched.jsonl)")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--enriched", action="store_true", default=False,
                     help="Summarise the enriched file (default)")
    grp.add_argument("--consolidated", action="store_true", default=False,
                     help="Summarise the consolidated (pre-enrichment) file")
    grp.add_argument("--combined", action="store_true",
                     help="Summarise the pre-consolidation combined file")
    parser.add_argument("--include-green", action="store_true", default=False,
                        help="Include GDACS green alerts (excluded by default)")
    args = parser.parse_args()

    if args.file:
        path  = args.file
        label = path.name
    elif args.combined:
        path  = COMBINED
        label = "combined (pre-consolidation)"
    elif args.consolidated:
        path  = CONSOLIDATED
        label = "consolidated"
    else:
        path  = ENRICHED if ENRICHED.exists() else CONSOLIDATED
        label = "enriched" if path == ENRICHED else "consolidated"

    if not path.exists():
        print(f"ERROR: {path} not found. Run python -m Builder_Reference.helper_scripts.reference.build first.")
        return

    print(f"\nLoading {path} ...")
    records = _load(path)

    # Filter GDACS green alerts unless --include-green is set
    if not args.include_green:
        before = len(records)
        records = [r for r in records if not _is_gdacs_green(r)]
        removed = before - len(records)
        label += f", excl. GDACS green alerts ({removed:,} removed)"

    print_overview(records, label)
    print_temporal(records)
    print_source_year_matrix(records)
    print_geographic(records)
    print_field_completeness(records)
    print_impact_summary(records)
    print_consolidation_stats(records)
    print_desinventar_section(records)
    print_quality_flags(records)

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
