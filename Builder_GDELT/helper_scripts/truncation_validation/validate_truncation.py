"""
validate_truncation.py
=======================
Validates whether truncating article body text to a character cap causes
meaningful loss of extraction quality, across a range of caps.

Design
------
The title is intentionally EXCLUDED from the truncated call (--no-title mode
is the default) so that the test isolates what information is carried by the
body text alone.  Without this, low caps (e.g. 50 chars) pass because the
title — which is never truncated — already contains type/location/date.

Four metrics are compared between full-body and truncated-body extraction:
  1. disruption_type  — high-level event class (flood, earthquake, etc.)
  2. event_date       — date of the event
  3. country          — first element of the location array
  4. details_fields   — count of populated indicator fields (rainfall_anomaly,
                        death_toll, river_discharge, etc.) — these only appear
                        deep in the article body, not in the headline.

Efficiency
----------
Full extraction runs once per article.  For each cap, truncated extraction
only runs on articles longer than the cap; shorter articles trivially agree.

Usage
-----
    python -m Builder_GDELT.helper_scripts.truncation_validation.validate_truncation
    python -m Builder_GDELT.helper_scripts.truncation_validation.validate_truncation --n 200 --caps 50 100 200 400 800 1600 3200 6400 --highlight 800
    python -m Builder_GDELT.helper_scripts.truncation_validation.validate_truncation --with-title   # use title in both calls (less strict)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]

for _p in [ROOT / ".env", Path(__file__).resolve().parent / ".env"]:
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=True)
        break

from openai import OpenAI
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CACHE_FILE   = ROOT / "API Costs" / "scrape_cache.jsonl"
RESULTS_FILE = Path(__file__).resolve().parent / "truncation_validation_results.jsonl"
PLOT_FILE    = Path(__file__).resolve().parent / "truncation_validation_plot.png"
FULL_EXTRACTIONS_DIR = Path(__file__).resolve().parent / "full_extractions"

DEFAULT_N    = 50
DEFAULT_CAPS = [50, 100, 200, 400, 800, 1600, 3200]


# ---------------------------------------------------------------------------
# Data loading / scraping
# ---------------------------------------------------------------------------

def _load_successful_urls(urls_file: str | None = None) -> list[str]:
    """Load URLs from a plain-text file (one per line) or from the general scrape cache."""
    if urls_file:
        p = Path(urls_file)
        if not p.exists():
            raise FileNotFoundError(f"URLs file not found: {p}")
        return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Scrape cache not found: {CACHE_FILE}\nRun estimate_llm_costs.py first.")
    urls = []
    with CACHE_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if row.get("success"):
                    urls.append(row["url"])
            except Exception:
                pass
    return urls


def _scrape(url: str) -> dict:
    from Builder_GDELT.helper_scripts.pipeline.webscraper import extract_article_text
    try:
        result = extract_article_text(url, timeout=20)
        return {"url": url, "text": result.get("text") or "", "title": result.get("title") or ""}
    except Exception as e:
        return {"url": url, "text": "", "title": "", "error": str(e)}


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

def _extract(url: str, title: str, text: str) -> dict:
    system_prompt = """\
You are an information extraction engine for supply chain disruptions.

Your job is to read a news article and extract a REAL physical or policy disruption event. Stay faithful to the article text.

You MUST output a single JSON object and NOTHING else.
"""

    user_prompt = f"""Extract a single main supply chain disruption event.

If none, return:
{{
  "disruption_type": "unknown",
  "event_date": null,
  "location": ["", "", "", ""],
  "duration_hours": null,
  "details": {{}},
  "confidence": 0.0
}}

Allowed disruption_type:
flood, drought, cyclone_hurricane, extreme_heat, landslide, earthquake, mine_accident, labour_strike, protests, country_relations, trade_embargo, tariffs, unknown

Return JSON only.

Schema:
{{
  "disruption_type": "...",
  "event_date": "YYYY-MM-DD" or null,
  "location": ["country", "region_or_state", "city", "specific_location"],
  "duration_hours": number or null,
  "details": {{ ... }},
  "confidence": 0.0
}}

IMPORTANT:
- Our threshold for classing a disruption is a confidence of 0.6
- Ignore metaphorical disruptions e.g. "a flood of criticism".

LOCATION FIELD:
- "location" must always be an array of four strings in descending geographic scale:
  ["country", "region_or_state", "city", "specific_location"]
- If a level is not mentioned in the article, leave it as "".

DETAILS FIELD:
- "details" is the ONLY place where disruption-specific indicators or structured information may appear.
- "details" must be an object.
- Only include fields listed below for the relevant disruption type.
- Only include information if explicitly mentioned in the article.
- Do not infer, estimate, or fabricate values.
- If no relevant information is mentioned, details must be {{}}.

Details by disruption type:

flood:
- rainfall_intensity
- rainfall_levels
- death_toll
- main_cause

drought:
- rainfall_deviation
- reservoir_level
- temperature_anomaly
- water_restrictions

cyclone_hurricane:
- sea_surface_temp_anomaly
- storm_category
- wind_speed

extreme_heat:
- temperature_anomaly
- power_grid_stress

landslide:
- rainfall_intensity
- soil_moisture
- deforestation_activity

earthquake:
- seismic_event_count
- max_magnitude
- foreshock_activity

mine_accident:
- fatalities
- injuries
- equipment_failure

labour_strike or protests:
- protest_type
- protesting_groups
- organizations_or_companies
- target_of_protest
- issue
- sector
- estimated_participants
- event_start_day
- reported_day_number


country_relations:
- casualties
- countries_involved

trade_embargo:
- sanction_count
- trade_restrictiveness_index

tariffs:
- tariff_rate
- affected_products_count
- affected_trade_value


Now process this article:

URL: {url}
TITLE: {title}
TEXT: {text}
"""
    try:
        completion = _client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            timeout=60,
        )
        data = json.loads(completion.choices[0].message.content or "{}")
    except Exception as e:
        return {"disruption_type": "error", "event_date": None, "location": [], "details": {}, "confidence": 0.0, "_error": str(e)}
    data.setdefault("disruption_type", "unknown")
    data.setdefault("event_date", None)
    data.setdefault("location", ["", "", "", ""])
    data.setdefault("details", {})
    data.setdefault("confidence", 0.0)
    return data


# ---------------------------------------------------------------------------
# Completeness helpers
# ---------------------------------------------------------------------------

def _details_count(details) -> int:
    """Count non-empty fields in the details dict."""
    if not isinstance(details, dict):
        return 0
    return sum(1 for v in details.values() if v is not None and v != "" and v != [])


def _completeness(result: dict) -> dict:
    """Measure absolute completeness of a single extraction result."""
    has_date    = bool(result.get("event_date"))
    loc         = result.get("location") or []
    has_country = bool(loc[0].strip()) if isinstance(loc, list) and loc else False
    det_count   = _details_count(result.get("details"))
    has_details = det_count > 0
    confidence  = float(result.get("confidence") or 0.0)
    return {
        "has_date":    has_date,
        "has_country": has_country,
        "det_count":   det_count,
        "has_details": has_details,
        "confidence":  confidence,
    }


def _agg_completeness(results: list[dict]) -> dict:
    """Aggregate completeness stats across a list of extraction results."""
    n = len(results)
    if n == 0:
        return {}
    cms = [_completeness(r) for r in results]
    return {
        "n":              n,
        "pct_date":       sum(c["has_date"]    for c in cms) / n * 100,
        "pct_country":    sum(c["has_country"] for c in cms) / n * 100,
        "pct_details":    sum(c["has_details"] for c in cms) / n * 100,
        "avg_det_fields": sum(c["det_count"]   for c in cms) / n,
        "avg_confidence": sum(c["confidence"]  for c in cms) / n,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(cap_stats: list[dict], highlight: int | None, label: str = "", plot_file: Path | None = None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
    except ImportError:
        print("matplotlib not available — skipping plot. Install with: pip install matplotlib")
        return

    # "full" entry has cap=None — plot it at a synthetic x position to the right
    full_stat  = next((s for s in cap_stats if s["cap"] is None), None)
    trunc_stats = [s for s in cap_stats if s["cap"] is not None]

    caps         = [s["cap"]             for s in trunc_stats]
    date_pct     = [s["pct_date"]        for s in trunc_stats]
    country_pct  = [s["pct_country"]     for s in trunc_stats]
    details_pct  = [s["pct_details"]     for s in trunc_stats]
    avg_det      = [s["avg_det_fields"]  for s in trunc_stats]
    trunc_pct    = [s["pct_truncated"]   for s in trunc_stats]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(caps, date_pct,    "s--", color="#ff7f0e", label="has event_date (%)",      linewidth=2)
    ax1.plot(caps, country_pct, "^:",  color="#2ca02c", label="has country (%)",          linewidth=2)
    ax1.plot(caps, details_pct, "D-.", color="#d62728", label="has ≥1 detail field (%)", linewidth=2)

    # Full-text baselines as horizontal dashed lines
    if full_stat:
        ax1.axhline(full_stat["pct_date"],    color="#ff7f0e", linestyle=":", linewidth=1, alpha=0.5)
        ax1.axhline(full_stat["pct_country"], color="#2ca02c", linestyle=":", linewidth=1, alpha=0.5)
        ax1.axhline(full_stat["pct_details"], color="#d62728", linestyle=":", linewidth=1, alpha=0.5)

    ax1.set_xlabel("Character cap (body text only, log scale)", fontsize=12)
    ax1.set_ylabel("Articles with field populated (%)", fontsize=11)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.set_ylim(0, 105)
    ax1.set_xscale("log")

    # Secondary axis: avg detail fields count
    ax2 = ax1.twinx()
    ax2.plot(caps, avg_det, "o-", color="#9467bd", linewidth=2, label="avg detail fields filled")
    if full_stat:
        ax2.axhline(full_stat["avg_det_fields"], color="#9467bd", linestyle=":", linewidth=1, alpha=0.5)
    ax2.set_ylabel("Avg detail fields populated", fontsize=11, color="#9467bd")
    ax2.tick_params(axis="y", labelcolor="#9467bd")
    ax2.set_ylim(bottom=0)

    # Grey fill for % articles truncated (tertiary info via ax1 scale)
    ax1.fill_between(caps, trunc_pct, alpha=0.08, color="grey", label="% articles truncated")

    if highlight is not None:
        ax1.axvline(x=highlight, color="red", linestyle="--", linewidth=1.5,
                    label=f"Selected cap ({highlight:,} chars)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    full_note = f"  (dotted lines = full-text baseline)" if full_stat else ""
    type_label = f" — {label}" if label else ""
    ax1.set_title(f"LLM extraction completeness vs body-text character cap{type_label}\n"
                  f"(title excluded from truncated call){full_note}", fontsize=12)
    ax1.grid(True, alpha=0.3, which="both")

    out = plot_file or PLOT_FILE
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Plot saved to: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int, default=DEFAULT_N)
    parser.add_argument("--caps",       type=int, nargs="+", default=DEFAULT_CAPS)
    parser.add_argument("--highlight",  type=int, default=3300)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--with-title", action="store_true",
                        help="Include title in truncated call (less strict — default is to exclude it)")
    parser.add_argument("--urls-file",  type=str, default=None,
                        help="Path to plain-text file with one URL per line (overrides scrape cache)")
    parser.add_argument("--label",      type=str, default="",
                        help="Label for this run, shown in plot title (e.g. 'floods')")
    parser.add_argument("--plot-file",  type=str, default=None,
                        help="Output path for the plot PNG (default: truncation_validation_plot.png)")
    args = parser.parse_args()
    caps     = sorted(args.caps)
    no_title = not args.with_title
    plot_file = Path(args.plot_file) if args.plot_file else None

    random.seed(args.seed)

    print(f"Mode: {'title EXCLUDED from truncated call (strict)' if no_title else 'title included in all calls'}")

    # --- Load & scrape ---
    print("\nLoading URLs...")
    all_urls = _load_successful_urls(args.urls_file)
    print(f"  {len(all_urls):,} available")
    sample_urls = random.sample(all_urls, min(args.n, len(all_urls)))
    print(f"  Sampled {len(sample_urls)} (seed={args.seed})")

    print(f"\nScraping {len(sample_urls)} articles (20 workers)...")
    articles = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_scrape, url): url for url in sample_urls}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            articles[r["url"]] = r
            if i % 10 == 0 or i == len(sample_urls):
                ok = sum(1 for a in articles.values() if len(a.get("text", "")) > 100)
                print(f"  {i}/{len(sample_urls)} fetched | usable: {ok}")
                sys.stdout.flush()

    usable = {url: a for url, a in articles.items() if len(a.get("text", "")) > 100}
    print(f"\n{len(usable)} articles usable")

    # --- Full extraction (with real title) ---
    print(f"\nRunning full extraction on {len(usable)} articles...")
    full_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_extract, url, a["title"], a["text"]): url
            for url, a in usable.items()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            url = futures[fut]
            full_results[url] = fut.result()
            if i % 10 == 0 or i == len(usable):
                print(f"  {i}/{len(usable)} done")
                sys.stdout.flush()

    n_with_details = sum(1 for r in full_results.values() if _details_count(r.get("details")) > 0)
    print(f"  {n_with_details}/{len(usable)} full articles have non-empty details fields")

    # Full-text completeness baseline
    full_agg = _agg_completeness(list(full_results.values()))
    print(f"\n  [Full baseline] date={full_agg['pct_date']:.1f}%  country={full_agg['pct_country']:.1f}%  "
          f"details={full_agg['pct_details']:.1f}%  avg_det_fields={full_agg['avg_det_fields']:.2f}")

    cap_stats = [{
        "cap":           None,
        "label":         "full",
        "n":             full_agg["n"],
        "pct_truncated": 0.0,
        **{k: full_agg[k] for k in ("pct_date","pct_country","pct_details","avg_det_fields","avg_confidence")},
    }]
    all_raw_results: list[dict] = []

    # --- Per-cap truncated extraction ---
    for cap in caps:
        needs_trunc   = [url for url, a in usable.items() if len(a["text"]) > cap]
        pct_truncated = len(needs_trunc) / len(usable) * 100

        print(f"\nCap {cap:,} chars — {len(needs_trunc)}/{len(usable)} truncated ({pct_truncated:.0f}%)")
        sys.stdout.flush()

        cap_results: dict[str, dict] = {}
        # Articles shorter than cap: use full extraction result (no info lost)
        for url in usable:
            if url not in [u for u in needs_trunc]:
                cap_results[url] = full_results[url]

        if needs_trunc:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {
                    executor.submit(
                        _extract,
                        url,
                        "" if no_title else usable[url]["title"],
                        usable[url]["text"][:cap],
                    ): url
                    for url in needs_trunc
                }
                for fut in as_completed(futures):
                    url = futures[fut]
                    cap_results[url] = fut.result()
                    r = cap_results[url]
                    all_raw_results.append({
                        "url": url, "cap": cap,
                        "chars_full": len(usable[url]["text"]),
                        **_completeness(r),
                    })

        agg = _agg_completeness(list(cap_results.values()))
        print(f"  date={agg['pct_date']:.1f}%  country={agg['pct_country']:.1f}%  "
              f"details={agg['pct_details']:.1f}%  avg_det_fields={agg['avg_det_fields']:.2f}  "
              f"conf={agg['avg_confidence']:.2f}")

        cap_stats.append({
            "cap":           cap,
            "label":         str(cap),
            "n":             agg["n"],
            "pct_truncated": pct_truncated,
            **{k: agg[k] for k in ("pct_date","pct_country","pct_details","avg_det_fields","avg_confidence")},
        })
        sys.stdout.flush()

    # --- Summary table ---
    print(f"\n{'='*90}")
    print(f"{'Cap':>8}  {'% trunc':>8}  {'date%':>7}  {'ctry%':>7}  {'det%':>7}  {'avg_det':>8}  {'conf':>6}")
    print(f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*8}  {'-'*6}")
    for s in cap_stats:
        cap_label = "full" if s["cap"] is None else f"{s['cap']:,}"
        marker = " <--" if args.highlight and s["cap"] == args.highlight else ""
        print(f"{cap_label:>8}  {s['pct_truncated']:>7.0f}%  {s['pct_date']:>6.1f}%  "
              f"{s['pct_country']:>6.1f}%  {s['pct_details']:>6.1f}%  "
              f"{s['avg_det_fields']:>8.2f}  {s['avg_confidence']:>6.2f}{marker}")
    print(f"{'='*90}")

    # --- Save & plot ---
    with RESULTS_FILE.open("w", encoding="utf-8") as f:
        for row in all_raw_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nResults saved to: {RESULTS_FILE}")

    # Save full extraction results (with raw details dict) for field-level analysis
    FULL_EXTRACTIONS_DIR.mkdir(exist_ok=True)
    label_slug = args.label.replace(" ", "_") or "unlabelled"
    full_out = FULL_EXTRACTIONS_DIR / f"full_{label_slug}.jsonl"
    with full_out.open("w", encoding="utf-8") as f:
        for url, result in full_results.items():
            f.write(json.dumps({"url": url, **result}, ensure_ascii=False) + "\n")
    print(f"Full extractions saved to: {full_out}")

    _plot(cap_stats, highlight=args.highlight, label=args.label, plot_file=plot_file)


if __name__ == "__main__":
    main()
