"""
extract_full_for_coverage.py
=============================
Scrapes and runs full (uncapped) LLM extraction on a seeded URL list,
saving raw details dicts to full_extractions/full_<label>.jsonl.
Much faster than running validate_truncation.py with all caps.

Usage
-----
    python -m Builder_GDELT.helper_scripts.truncation_validation.extract_full_for_coverage \
        --urls-file Builder_GDELT/helper_scripts/truncation_validation/urls_flood_seeded.txt \
        --n 200 --label flood

    python -m Builder_GDELT.helper_scripts.truncation_validation.extract_full_for_coverage \
        --urls-file Builder_GDELT/helper_scripts/truncation_validation/urls_strike_seeded.txt \
        --n 200 --label strike
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FULL_EXTRACTIONS_DIR = HERE / "full_extractions"

for _p in [ROOT / ".env", HERE / ".env"]:
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=True)
        break

from openai import OpenAI
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from Builder_GDELT.helper_scripts.pipeline.webscraper import extract_article_text


def _scrape(url: str) -> dict:
    try:
        result = extract_article_text(url, timeout=20)
        return {"url": url, "text": result.get("text") or "", "title": result.get("title") or ""}
    except Exception as e:
        return {"url": url, "text": "", "title": "", "error": str(e)}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urls-file", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--label", required=True)
    parser.add_argument("--cap", type=int, default=None, help="Truncate article body to this many chars before LLM call")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    all_urls = [l.strip() for l in Path(args.urls_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    sample = random.sample(all_urls, min(args.n, len(all_urls)))
    print(f"Sampled {len(sample)} URLs from {len(all_urls)} (seed={args.seed})")

    print(f"Scraping {len(sample)} articles...")
    articles = {}
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(_scrape, url): url for url in sample}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            articles[r["url"]] = r
            if i % 20 == 0 or i == len(sample):
                ok = sum(1 for a in articles.values() if len(a.get("text", "")) > 100)
                print(f"  {i}/{len(sample)} fetched | usable: {ok}")
                sys.stdout.flush()

    usable = {url: a for url, a in articles.items() if len(a.get("text", "")) > 100}
    print(f"{len(usable)} usable articles")

    cap_note = f" (capped at {args.cap} chars)" if args.cap else ""
    print(f"Running extraction on {len(usable)} articles{cap_note}...")
    results = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_extract, url, a["title"], a["text"][:args.cap] if args.cap else a["text"]): url
            for url, a in usable.items()
        }
        for i, fut in enumerate(as_completed(futures), 1):
            url = futures[fut]
            results[url] = fut.result()
            if i % 20 == 0 or i == len(usable):
                print(f"  {i}/{len(usable)} extracted")
                sys.stdout.flush()

    FULL_EXTRACTIONS_DIR.mkdir(exist_ok=True)
    out = FULL_EXTRACTIONS_DIR / f"full_{args.label}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for url, result in results.items():
            f.write(json.dumps({"url": url, **result}, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(results)} records to: {out}")


if __name__ == "__main__":
    main()
