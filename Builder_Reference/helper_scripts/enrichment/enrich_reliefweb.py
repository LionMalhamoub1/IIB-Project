"""
enrich_reliefweb.py
===================
Uses cached ReliefWeb situation reports to extract two things per disaster:
  1. Sub-national location name (for geocoding — more specific than country)
  2. Impact figures (dead, injured, displaced, affected, houses_destroyed,
     damage_usd_thousands) to fill in gaps in the reference dataset

Reads:   Builder_Reference/outputs/reference_floods_enriched.jsonl
         Builder_Reference/cache/reliefweb_reports/{disaster_id}.json  (one per disaster)
Writes:  reference_floods_enriched.jsonl  (patched in-place)

LLM approach
------------
Follows the same async + semaphore pattern as DisruptionExtractorAsync.py:
  - AsyncOpenAI with MAX_API_CONCURRENT simultaneous in-flight calls
  - asyncio.Semaphore gates API calls
  - JSON-mode response format (response_format={"type": "json_object"})
  - One prompt per disaster (not per event — many events share the same source_id)

Only null/missing fields in the reference record are updated.  Extracted values
are tagged with reliefweb_llm_extracted=True so downstream scripts can
distinguish LLM-derived figures from original source figures.

Run reliefweb_reports.py first to populate the cache.

Usage
-----
  python -m Builder_Reference.helper_scripts.enrichment.enrich_reliefweb
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Env + client setup  (mirrors DisruptionExtractorAsync.py)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

_ENV_CANDIDATES = [
    ROOT / ".env",
    Path(__file__).resolve().parents[1] / ".env",
    Path(__file__).resolve().parent / ".env",
    Path.cwd() / ".env",
]
for _p in _ENV_CANDIDATES:
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=True)
        break

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found.")

ENRICHED      = ROOT / "Builder_Reference" / "outputs" / "reference_floods_enriched.jsonl"
REPORTS_CACHE = ROOT / "Builder_Reference" / "cache" / "reliefweb_reports"

DEFAULT_MODEL    = "gpt-4o-mini"
MAX_API_CONCURRENT = 40

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an information extraction engine for flood disaster reports.

Your job is to read situation report titles and body excerpts for a known flood disaster and extract structured data.
Stay faithful to the source text — do not infer or fabricate values not explicitly stated.

You MUST output a single JSON object and NOTHING else."""


def _build_user_prompt(disaster_name: str, country: str, reports: list[dict], disaster_description: str = "") -> str:
    titles   = "\n".join(f"- {r['title']}" for r in reports if r.get("title"))
    snippets = "\n\n".join(
        f"[{r['title']}]\n{r['body']}"
        for r in reports
        if r.get("body")
    )

    return f"""\
Extract flood impact data for a known flood disaster.

DISASTER: {disaster_name}
COUNTRY:  {country}

Official disaster description (most reliable source):
{disaster_description if disaster_description else "(not available)"}

Report titles ({len(reports)} reports):
{titles if titles else "(none)"}

Body excerpts from key reports:
{snippets if snippets else "(none available)"}

IMPORTANT:
- Only include information explicitly stated in the reports — do not infer or fabricate.
- If multiple figures appear for the same field, use the highest or most recent value.
- For location: use the most specific sub-national location where flooding occurred (region/district/city). If only the country is mentioned, return ["", "", "", ""].

Schema:
{{
  "location": ["country", "region_or_state", "city", "specific_location"],
  "details": {{
    "death_toll":        <int or null>,
    "injured_count":     <int or null>,
    "affected_count":    <int or null>,
    "displaced_count":   <int or null>,
    "area_affected_km2": <number or null>,
    "flood_type":        <"flash" | "riverine" | "coastal" | "dam_failure" | "storm_surge" | "snowmelt" | "unknown" | null>,
    "severity":          <"minor" | "moderate" | "major" | "catastrophic" | null>,
    "main_cause":        <string or null>,
    "rainfall_intensity": <string or null>,
    "rainfall_levels":   <string or null>,
    "houses_destroyed":  <int or null>,
    "damage_usd_thousands": <number or null>,
    "event_start_day":   <"YYYY-MM-DD" or null>,
    "event_end_day":     <"YYYY-MM-DD" or null>,
    "glide_number":      <string or null>
  }},
  "confidence": <0.0–1.0 reflecting certainty in the extracted values>
}}"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

async def _extract(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    disaster_id: str,
    disaster_name: str,
    country: str,
    reports: list[dict],
    model: str,
    disaster_description: str = "",
) -> dict:
    """Call the LLM for one disaster. Returns extracted dict or empty dict on failure."""
    prompt = _build_user_prompt(disaster_name, country, reports, disaster_description)
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                timeout=60,
            )
            raw = resp.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as exc:
            log.warning(f"LLM failed for disaster {disaster_id}: {exc}")
            return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_cache(disaster_id: str) -> Optional[dict]:
    path = REPORTS_CACHE / f"{disaster_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _patch_event(event: dict, extracted: dict) -> bool:
    """
    Apply extracted fields to event, only filling null/missing values.
    Maps from the GDELT-aligned details schema onto the reference field names.
    Returns True if any field was updated.
    """
    details = extracted.get("details") or {}
    updated = False

    # Mapping from LLM details field → reference dataset field name
    FIELD_MAP = {
        "death_toll":           "dead",
        "injured_count":        "injured",
        "displaced_count":      "displaced",
        "affected_count":       "affected",
        "houses_destroyed":     "houses_destroyed",
        "damage_usd_thousands": "damage_usd_thousands",
        "area_affected_km2":    "area_km2",
        "severity":             "severity",
        "main_cause":           "main_cause",
        "flood_type":           "flood_type",
        "glide_number":         "glide_number",
    }
    for llm_field, ref_field in FIELD_MAP.items():
        val = details.get(llm_field)
        if val is not None and event.get(ref_field) is None:
            event[ref_field] = val
            updated = True

    # Location — use the 4-element array ["country", "region", "city", "specific"]
    # Only update if elements 1-3 have sub-national content
    location = extracted.get("location") or []
    if isinstance(location, list) and len(location) == 4:
        sub_national = [p.strip() for p in location[1:] if isinstance(p, str) and p.strip()]
        if sub_national and not event.get("lat"):
            country_part = location[0].strip() if isinstance(location[0], str) else (event.get("country") or "")
            loc_name = ", ".join([country_part] + sub_national) if country_part else ", ".join(sub_national)
            event["location_name"] = loc_name
            updated = True

    if updated:
        event["reliefweb_llm_extracted"] = True

    return updated


# ---------------------------------------------------------------------------
# Async runner
# ---------------------------------------------------------------------------

async def _run(events: list[dict], model: str) -> list[dict]:
    # Group events by source_id — many events share the same disaster
    by_id: dict[str, list[dict]] = {}
    for e in events:
        did = str(e.get("source_id", ""))
        if did:
            by_id.setdefault(did, []).append(e)

    # Only process disasters where cache exists
    to_process = {did: evs for did, evs in by_id.items() if _load_cache(did)}
    skipped_no_cache = len(by_id) - len(to_process)
    log.info(
        f"  {len(by_id)} unique disasters | "
        f"{len(to_process)} have cached reports | "
        f"{skipped_no_cache} skipped (run reliefweb_reports.py first)"
    )

    if not to_process:
        return events

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sem    = asyncio.Semaphore(MAX_API_CONCURRENT)

    # Build tasks
    async def _task(did: str, evs: list[dict]):
        cache = _load_cache(did)
        extracted = await _extract(
            client, sem, did,
            cache.get("disaster_name", ""),
            cache.get("country", ""),
            cache.get("reports", []),
            model,
            cache.get("disaster_description", ""),
        )
        return did, extracted

    tasks = [asyncio.create_task(_task(did, evs)) for did, evs in to_process.items()]

    # Collect results and apply patches
    extraction_map: dict[str, dict] = {}
    done = 0
    for coro in asyncio.as_completed(tasks):
        did, extracted = await coro
        extraction_map[did] = extracted
        done += 1
        if done % 50 == 0 or done == len(tasks):
            log.info(f"  [{done}/{len(tasks)}] LLM extractions complete")

    # Patch events in-place
    patched = 0
    for e in events:
        did = str(e.get("source_id", ""))
        extracted = extraction_map.get(did, {})
        if extracted and _patch_event(e, extracted):
            patched += 1

    log.info(f"  {patched} events updated with extracted fields")
    return events


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model: str = DEFAULT_MODEL) -> None:
    if not ENRICHED.exists():
        log.error(f"Input not found: {ENRICHED}")
        sys.exit(1)

    log.info(f"Loading events from {ENRICHED}")
    events = []
    with ENRICHED.open("r", encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line))
    log.info(f"  {len(events):,} total events")

    reliefweb_events = [e for e in events if e.get("source") == "ReliefWeb"]
    log.info(f"  {len(reliefweb_events):,} ReliefWeb events to enrich")

    enriched = asyncio.run(_run(reliefweb_events, model))

    # Merge enriched ReliefWeb events back into full list
    rw_by_id = {id(e): e for e in enriched}
    final = [rw_by_id.get(id(e), e) for e in events]

    with ENRICHED.open("w", encoding="utf-8") as f:
        for e in final:
            f.write(json.dumps(e, default=str) + "\n")
    log.info(f"Written back to {ENRICHED}")

    # Summary
    llm_events  = [e for e in enriched if e.get("reliefweb_llm_extracted")]
    loc_filled  = sum(1 for e in llm_events if e.get("location_name") and e["location_name"].lower() != (e.get("country") or "").lower())
    dead_filled = sum(1 for e in llm_events if e.get("dead") is not None)
    disp_filled = sum(1 for e in llm_events if e.get("displaced") is not None)
    log.info(f"  Events updated by LLM:           {len(llm_events)}")
    log.info(f"  Sub-national locations:          {loc_filled}")
    log.info(f"  Death toll extracted:            {dead_filled}")
    log.info(f"  Displaced count extracted:       {disp_filled}")


if __name__ == "__main__":
    main()
