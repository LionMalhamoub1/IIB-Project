"""
Compute disruption event database statistics and save to clustering_stats.txt.
"""
import json
import glob
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
import pycountry

# ── paths ──────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parents[1]
CONSOLIDATED   = ROOT / "Builder_GDELT/results/combined/all_consolidated.jsonl"
DAILY_DIR      = ROOT / "Builder_GDELT/results/daily"
SVM_DAILY_DIR  = ROOT / "data/processed/model_scored_daily"
PANEL_PARQUET  = ROOT / "Social Disruptions/Likelihood_modelling_social/v2/data/interim/modelling_panel_gdelt.parquet"
OUT_FILE       = ROOT / "notes" / "clustering_stats.txt"

YEARS = [2017, 2018, 2019, 2020, 2021]

# ── 1. Pre-clustering SVM positives by year ────────────────────────────────
# Count rows in *_interesting_urls_experts_only.csv across all days
print("Counting SVM positives...")
svm_by_year = defaultdict(int)
for year in YEARS:
    files = glob.glob(str(SVM_DAILY_DIR / str(year) / "**" / "*_interesting_urls_experts_only.csv"), recursive=True)
    for f in files:
        try:
            svm_by_year[year] += len(pd.read_csv(f))
        except Exception:
            pass

# ── 2. Pre-clustering LLM extractions by year (Stage 5-6, before global cluster) ─
print("Counting pre-clustering LLM extractions...")
pre_cluster_by_year = defaultdict(int)
for day_dir in sorted(DAILY_DIR.iterdir()):
    ej = day_dir / "extractions.jsonl"
    if not ej.exists():
        continue
    try:
        year = int(day_dir.name[:4])
    except ValueError:
        continue
    if year not in YEARS:
        continue
    with open(ej, encoding="utf-8") as f:
        pre_cluster_by_year[year] += sum(1 for _ in f)

# ── 3. Load consolidated events ────────────────────────────────────────────
print("Loading consolidated events...")
events = []
with open(CONSOLIDATED, encoding="utf-8") as f:
    for line in f:
        events.append(json.loads(line))

df = pd.DataFrame(events)
df["event_date"]   = pd.to_datetime(df["event_date"],   errors="coerce")
df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
df["best_date"]    = df["event_date"].fillna(df["publish_date"])
df["year"]         = df["best_date"].dt.year

# keep only 2017-2021
df_period = df[df["year"].isin(YEARS)].copy()

# ── 4. Post-clustering counts by year and type ─────────────────────────────
post_total    = df_period.groupby("year").size()
post_protest  = df_period[df_period["disruption_type"] == "protests"].groupby("year").size()
post_strike   = df_period[df_period["disruption_type"] == "labour_strike"].groupby("year").size()
post_combined = (post_protest.reindex(YEARS, fill_value=0) +
                 post_strike.reindex(YEARS, fill_value=0))

# ── 5. Clustering reduction ratio ─────────────────────────────────────────
reduction = {}
for y in YEARS:
    pre  = pre_cluster_by_year.get(y, 0)
    post = int(post_total.get(y, 0))
    reduction[y] = round(pre / post, 2) if post > 0 else None

# ── 6. Panel countries and zero-event analysis ────────────────────────────
print("Computing country stats...")
panel = pd.read_parquet(PANEL_PARQUET)
panel_countries = sorted(panel["country_iso3"].unique())
n_panel = len(panel_countries)

# map location_name to ISO3
def location_to_iso3(loc: str) -> str | None:
    if not loc:
        return None
    country_str = loc.split(",")[0].strip()
    # direct pycountry lookup
    try:
        c = pycountry.countries.search_fuzzy(country_str)
        return c[0].alpha_3
    except Exception:
        return None

df_period = df_period.copy()
df_period["iso3"] = df_period["location_name"].apply(location_to_iso3)

# focus on protests + strikes
ps = df_period[df_period["disruption_type"].isin(["protests", "labour_strike"])]
country_counts = ps["iso3"].value_counts().dropna()

# panel countries with zero events
zero_event = [c for c in panel_countries if c not in country_counts.index]

# top 10 / bottom 5 (among panel countries only, excluding zeros)
panel_counts = country_counts[country_counts.index.isin(panel_countries)]
top10  = panel_counts.nlargest(10)
bottom5 = panel_counts[panel_counts > 0].nsmallest(5)

def iso3_to_name(code):
    try:
        return pycountry.countries.get(alpha_3=code).name
    except Exception:
        return code

# ── 7. Total unique events ─────────────────────────────────────────────────
total_unique = len(df_period)
total_protest = int((df_period["disruption_type"] == "protests").sum())
total_strike  = int((df_period["disruption_type"] == "labour_strike").sum())

# ── Write output ───────────────────────────────────────────────────────────
lines = []
SEP = "=" * 70

lines.append(SEP)
lines.append("DISRUPTION EVENT DATABASE — CLUSTERING STATISTICS")
lines.append(f"Period: 2017–2021  |  Panel: {n_panel} countries")
lines.append(SEP)

lines.append("")
lines.append("1. PRE-CLUSTERING ARTICLE COUNTS (SVM positives entering pipeline)")
lines.append("-" * 50)
lines.append(f"  {'Year':<8} {'SVM positives':>15} {'LLM extractions':>17}")
for y in YEARS:
    lines.append(f"  {y:<8} {svm_by_year.get(y,0):>15,} {pre_cluster_by_year.get(y,0):>17,}")
lines.append(f"  {'TOTAL':<8} {sum(svm_by_year.values()):>15,} {sum(pre_cluster_by_year.values()):>17,}")

lines.append("")
lines.append("2. POST-CLUSTERING EVENT COUNTS BY YEAR AND TYPE")
lines.append("-" * 50)
lines.append(f"  {'Year':<8} {'All types':>10} {'Protests':>10} {'Strikes':>10} {'Combined':>10}")
for y in YEARS:
    a = int(post_total.get(y, 0))
    p = int(post_protest.get(y, 0))
    s = int(post_strike.get(y, 0))
    c = p + s
    lines.append(f"  {y:<8} {a:>10,} {p:>10,} {s:>10,} {c:>10,}")
lines.append(f"  {'TOTAL':<8} {int(post_total.sum()):>10,} {int(post_protest.sum()):>10,} {int(post_strike.sum()):>10,} {int(post_combined.sum()):>10,}")

lines.append("")
lines.append("3. TOTAL UNIQUE EVENTS IN FINAL DATABASE (2017–2021)")
lines.append("-" * 50)
lines.append(f"  All types (2017–2021):          {total_unique:>8,}")
lines.append(f"  Protests:                       {total_protest:>8,}")
lines.append(f"  Labour strikes:                 {total_strike:>8,}")
lines.append(f"  Protests + strikes combined:    {total_protest+total_strike:>8,}")
lines.append(f"  Overall total (all years):      {len(df):>8,}")

lines.append("")
lines.append("4. CLUSTERING REDUCTION RATIO BY YEAR")
lines.append("   (LLM extractions ÷ post-clustering events)")
lines.append("-" * 50)
lines.append(f"  {'Year':<8} {'Pre':>8} {'Post':>8} {'Ratio':>8}")
for y in YEARS:
    pre  = pre_cluster_by_year.get(y, 0)
    post = int(post_total.get(y, 0))
    r    = f"{reduction[y]:.2f}x" if reduction[y] else "N/A"
    lines.append(f"  {y:<8} {pre:>8,} {post:>8,} {r:>8}")

lines.append("")
lines.append("5. CLUSTERING PARAMETERS")
lines.append("-" * 50)
lines.append("  Method:                      Greedy sequential clustering")
lines.append("  Similarity criterion:        Location token overlap (>=1 common")
lines.append("                               token, >2 chars) + date proximity")
lines.append("  Time window — event/event:   ±1 day")
lines.append("  Time window — event/publish: ±2 days")
lines.append("  Time window — pub/pub:       ±3 days")
lines.append("  Cluster representative:      First article; merge takes")
lines.append("                               earliest date, longest location,")
lines.append("                               max confidence, Nominatim coords")
lines.append("                               preferred over GDELT actiongeo")
lines.append("  Scope:                       Per disruption type (types not")
lines.append("                               merged across categories)")

lines.append("")
lines.append("6. PANEL COUNTRIES WITH ZERO PROTEST/STRIKE EVENTS (2017–2021)")
lines.append("-" * 50)
lines.append(f"  Total panel countries:  {n_panel}")
lines.append(f"  Zero-event countries:   {len(zero_event)}")
if zero_event:
    for iso in sorted(zero_event):
        lines.append(f"    {iso}  {iso3_to_name(iso)}")

lines.append("")
lines.append("7. TOP 10 PANEL COUNTRIES BY TOTAL PROTEST/STRIKE EVENT COUNT")
lines.append("-" * 50)
for rank, (iso, cnt) in enumerate(top10.items(), 1):
    lines.append(f"  {rank:>2}. {iso}  {iso3_to_name(iso):<35}  {int(cnt):>5,}")

lines.append("")
lines.append("8. BOTTOM 5 PANEL COUNTRIES BY TOTAL PROTEST/STRIKE EVENT COUNT")
lines.append("   (excluding zero-event countries)")
lines.append("-" * 50)
for rank, (iso, cnt) in enumerate(bottom5.items(), 1):
    lines.append(f"  {rank:>2}. {iso}  {iso3_to_name(iso):<35}  {int(cnt):>5,}")

lines.append("")
lines.append(SEP)

output = "\n".join(lines)
print("\n" + output)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    f.write(output + "\n")

print(f"\nSaved to {OUT_FILE}")
