"""
Async version of DisruptionExtractor.

Replaces ThreadPoolExecutor with asyncio + aiohttp for web fetching,
and AsyncOpenAI for LLM calls. Drop-in replacement for run_batch().

Usage:
    from .DisruptionExtractorAsync import run_batch
    run_batch(input_csv="...", input_yyyymmdd="20260101")
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import trafilatura
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pathlib import Path
from tqdm import tqdm


# ENV + CLIENT SETUP

ENV_CANDIDATES = [
    Path(__file__).resolve().parents[3] / ".env",
    Path(__file__).resolve().parents[1] / ".env",
    Path(__file__).resolve().parent / ".env",
    Path.cwd() / ".env",
]

env_loaded = False
for _p in ENV_CANDIDATES:
    if _p.exists():
        load_dotenv(dotenv_path=_p, override=True)
        print(f"[env] Loaded .env from: {_p}")
        env_loaded = True
        break

if not env_loaded:
    print(f"[env] No .env found. Tried: {', '.join(str(p) for p in ENV_CANDIDATES)}")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found after loading .env.")

DEFAULT_MODEL = "gpt-4o-mini"

# Max simultaneous in-flight requests for each stage.
# Tune these if you hit rate limits or connection errors.
MAX_SCRAPE_CONCURRENT = 40   # aiohttp fetches
MAX_API_CONCURRENT    = 40   # OpenAI calls

SCRAPE_TIMEOUT_S = 20
SCRAPE_MAX_BYTES = 1024 * 1024   # 1 MB cap on article HTML

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

META_DATE_TAGS = [
    ("property", "article:published_time"),
    ("property", "article:modified_time"),
    ("name",     "pubdate"),
    ("name",     "publish-date"),
    ("name",     "publication_date"),
    ("itemprop", "datePublished"),
    ("itemprop", "dateModified"),
]


# DATA MODEL

@dataclass
class ExtractRecord:
    url: str
    source_title: str

    llm_disruption_type: str

    expert_disruption_type: Optional[str]
    expert_probability: Optional[float]

    disruption_type: str
    classification_source: str

    event_date: Optional[str]
    publish_date: Optional[str]
    location_name: str
    duration_hours: Optional[float]
    extras: Dict[str, Any]
    confidence: float
    lat: Optional[float]
    lon: Optional[float]
    event_description: Optional[str]


# DATE NORMALISATION

def _normalise_date(value: Any, *, date_only: bool) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        ts = pd.to_datetime(s, utc=True, errors="raise")
    except Exception:
        return None
    return ts.date().isoformat() if date_only else ts.isoformat()


def _parse_date(val: str) -> Optional[str]:
    try:
        return dateparser.parse(val).isoformat()
    except Exception:
        return None


def _prep(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u00a0", " ").replace("\r", " ")
    return " ".join(s.split()).strip()


# ASYNC WEB SCRAPER

async def _fetch_html(url: str, session: aiohttp.ClientSession) -> Optional[str]:
    """Fetch raw HTML for a URL, capped at SCRAPE_MAX_BYTES."""
    timeout = aiohttp.ClientTimeout(total=SCRAPE_TIMEOUT_S)
    try:
        async with session.get(url, timeout=timeout, allow_redirects=True) as resp:
            if resp.status != 200:
                return None
            buf = b""
            async for chunk in resp.content.iter_chunked(8192):
                buf += chunk
                if len(buf) >= SCRAPE_MAX_BYTES:
                    break
            encoding = resp.charset or "utf-8"
            try:
                return buf.decode(encoding, errors="replace")
            except (LookupError, UnicodeDecodeError):
                return buf.decode("utf-8", errors="replace")
    except Exception:
        return None


def _extract_from_html(url: str, html: str) -> Dict[str, Optional[str]]:
    """
    Synchronous extraction from already-fetched HTML.
    Mirrors webscraper.py logic (trafilatura + BeautifulSoup).
    Called via run_in_executor to avoid blocking the event loop.
    """
    soup = BeautifulSoup(html, "html.parser")

    publish_date: Optional[str] = None

    # 1) Meta tags
    for attr, key in META_DATE_TAGS:
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            parsed = _parse_date(tag["content"])
            if parsed:
                publish_date = parsed
                break

    # 2) <time datetime="...">
    if publish_date is None:
        time_tag = soup.find("time", datetime=True)
        if time_tag:
            publish_date = _parse_date(time_tag["datetime"])

    # 3) JSON-LD
    if publish_date is None:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    for key in ("datePublished", "dateModified"):
                        if key in data:
                            publish_date = _parse_date(data[key])
                            if publish_date:
                                break
                if publish_date:
                    break
            except Exception:
                pass

    # 4) trafilatura
    text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
    title = ""
    meta = trafilatura.bare_extraction(html)
    if isinstance(meta, dict):
        title = meta.get("title") or ""
        if publish_date is None:
            raw = meta.get("date") or meta.get("published")
            if raw:
                publish_date = _parse_date(raw)

    return {
        "url": url,
        "title": _prep(title),
        "text": _prep(text),
        "publish_date": publish_date,
    }


async def fetch_article_async(
    url: str,
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
) -> Dict[str, Optional[str]]:
    """Fetch + parse article. HTML fetch is async; parsing runs in executor."""
    empty = {"url": url, "title": "", "text": "", "publish_date": None}

    async with sem:
        html = await _fetch_html(url, session)

    if not html:
        return empty

    # Run CPU-bound parsing off the event loop thread
    art = await loop.run_in_executor(None, _extract_from_html, url, html)
    return art


# ASYNC LLM CALL

async def _call_openai_async(
    url: str,
    title: str,
    text: str,
    publish_date: Optional[str],
    expert_prob: Optional[float],
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
) -> Dict[str, Any]:

    system_prompt = """\
You are an information extraction engine for supply chain disruptions.

Your job is to read a news article and extract a REAL physical disruption event. Stay faithful to the article text.

You MUST output a single JSON object and NOTHING else.
"""

    user_prompt = f"""Extract a single main disruption event from the article below.

Only extract events of these types:
- flood  (including flash floods, river flooding, inundation)
- labour_strike  (worker strikes, walkouts)
- protests  (demonstrations, civil unrest affecting operations)

CLASSIFIER_CONFIDENCE: {f"{expert_prob:.3f}" if expert_prob is not None else "unknown"}

Extraction threshold rules:
- If CLASSIFIER_CONFIDENCE >= 0.65: always extract all available fields and return the best matching disruption_type. Do not return unknown solely because you are uncertain — set confidence to reflect your certainty but always attempt extraction.
- If CLASSIFIER_CONFIDENCE < 0.65 or unknown: only extract if your own confidence is >= 0.6. Otherwise return:
{{
  "disruption_type": "unknown",
  "event_date": null,
  "location": ["", "", "", ""],
  "duration_hours": null,
  "details": {{}},
  "confidence": 0.0,
  "event_description": ""
}}

Allowed disruption_type values: flood, labour_strike, protests, unknown

Return JSON only.

Schema:
{{
  "disruption_type": "...",
  "event_date": "YYYY-MM-DD" or null,
  "location": ["country", "region_or_state", "city", "specific_location"],
  "duration_hours": number or null,
  "details": {{ ... }},
  "confidence": 0.0-1.0,
  "event_description": "string"
}}


IMPORTANT:
- Only extract events of the three allowed types above. Classify everything else as "unknown".
- Ignore metaphorical language e.g. "a flood of criticism", "a wave of protests" when no real event is described.
- Articles about past events, legal cases, anniversary reports, or policy discussions referencing a historical flood are NOT active flood events — classify as "unknown".

EVENT_DESCRIPTION:
- For floods: always return "".
- For labour_strike or protests: one sentence — [group] [action] in [most specific location mentioned] over [specific cause]. Name groups, legislation, and organisations explicitly. Never use vague terms like "economic problems".

EVENT_DATE FIELD:
- Return the date the event occurred (not the article's publish date).
- Use YYYY-MM-DD. For multi-day floods, use the earliest date mentioned.
- If the article uses relative references ("last Tuesday", "on Monday", "this week") and is clearly reporting a current or recent event, resolve them using the ARTICLE_PUBLISH_DATE above as anchor.
- If ARTICLE_PUBLISH_DATE is unknown, or the article is not reporting a live/recent event, do not guess — return null.
- If genuinely unknowable, return null.

LOCATION FIELD:
- Always four strings in descending geographic scale: ["country", "region_or_state", "city", "specific_location"]
- Use the most specific location where the event physically occurred, not where the article was published.
- Leave unused levels as "".

DETAILS FIELD:
- Only include fields listed below for the matched disruption type.
- Only include information explicitly stated in the article — do not infer or fabricate.
- If nothing relevant is mentioned, details must be {{}}.

flood:
- rainfall_intensity
- rainfall_levels
- death_toll           (int)
- injured_count        (int)
- affected_count       (int)
- displaced_count      (int)
- area_affected_km2    (number)
- flood_type           (flash | riverine | coastal | dam_failure | storm_surge | unknown | snowmelt...)
- severity             (minor | moderate | major | catastrophic)
- main_cause
- event_start_day      (YYYY-MM-DD)
- event_end_day        (YYYY-MM-DD)
- glide_number

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

Now process this article:

ARTICLE_PUBLISH_DATE: {publish_date or "unknown"}
URL: {url}
TITLE: {title}
TEXT: {text}
"""

    async with sem:
        try:
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                timeout=60,
            )
            raw = completion.choices[0].message.content or ""
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {}
        except Exception:
            data = {}

    data.setdefault("disruption_type", "unknown")
    data.setdefault("event_date", None)
    data.setdefault("location", ["", "", "", ""])
    data.setdefault("duration_hours", None)
    data.setdefault("details", {})
    data.setdefault("confidence", 0.0)
    data.setdefault("event_description", None)
    return data


# SINGLE URL (ASYNC)

async def extract_from_url_async(
    url: str,
    expert_type: Optional[str],
    expert_prob: Optional[float],
    lat: Optional[float],
    lon: Optional[float],
    session: aiohttp.ClientSession,
    openai_client: AsyncOpenAI,
    sem_scrape: asyncio.Semaphore,
    sem_api: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
    threshold: float = 0.6,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:

    art = await fetch_article_async(url, session, sem_scrape, loop)
    title = art.get("title", "") or ""
    body  = (art.get("text", "") or "")[:3200]
    publish_date = art.get("publish_date")

    if len(body.strip()) < 50:
        # Dead link, paywall, or empty page  -  skip LLM to avoid wasting tokens
        llm_out = {
            "disruption_type": "unknown",
            "event_date": None,
            "location": ["", "", "", ""],
            "duration_hours": None,
            "details": {},
            "confidence": 0.0,
            "event_description": None,
        }
    else:
        llm_out  = await _call_openai_async(url, title, body, publish_date, expert_prob, openai_client, sem_api, model)
    llm_type = llm_out.get("disruption_type") or "unknown"

    publish_date_norm = _normalise_date(publish_date, date_only=False)
    event_date_norm   = _normalise_date(llm_out.get("event_date"), date_only=True)

    expert_type_clean = expert_type.strip() if expert_type else None

    if expert_type_clean and expert_prob is not None and expert_prob >= threshold:
        final_type = expert_type_clean
        classification_source = "expert"
    else:
        final_type = llm_type
        classification_source = "llm"

    rec = ExtractRecord(
        url=url,
        source_title=title,
        llm_disruption_type=llm_type,
        expert_disruption_type=expert_type_clean,
        expert_probability=expert_prob,
        disruption_type=final_type,
        classification_source=classification_source,
        event_date=event_date_norm,
        publish_date=publish_date_norm,
        location_name=", ".join(p for p in (llm_out.get("location") or []) if p),
        duration_hours=llm_out.get("duration_hours"),
        extras=llm_out.get("details") or {},
        confidence=round(float(llm_out.get("confidence") or 0.0), 3),
        lat=lat,
        lon=lon,
        event_description=llm_out.get("event_description") or None,
    )
    return rec.__dict__


# ASYNC BATCH RUNNER

async def _run_async(
    records_input: List[Tuple],
    out_jsonl: str,
    model: str,
) -> Tuple[List[Dict], List[Dict]]:

    results: List[Dict] = []
    errors:  List[Dict] = []
    write_lock = asyncio.Lock()

    sem_scrape = asyncio.Semaphore(MAX_SCRAPE_CONCURRENT)
    sem_api    = asyncio.Semaphore(MAX_API_CONCURRENT)
    loop       = asyncio.get_event_loop()

    openai_client = AsyncOpenAI(api_key=api_key)

    connector = aiohttp.TCPConnector(limit=MAX_SCRAPE_CONCURRENT + 10, ssl=False)
    async with aiohttp.ClientSession(headers=HEADERS, connector=connector) as session:

        async def _worker(record_tuple):
            url, expert_type, expert_prob, lat, lon = record_tuple
            try:
                data = await extract_from_url_async(
                    url=url,
                    expert_type=expert_type,
                    expert_prob=expert_prob,
                    lat=lat,
                    lon=lon,
                    session=session,
                    openai_client=openai_client,
                    sem_scrape=sem_scrape,
                    sem_api=sem_api,
                    loop=loop,
                    model=model,
                )
                async with write_lock:
                    with open(out_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
                return ("ok", data)
            except Exception as e:
                return ("err", {"url": url, "error": str(e)})

        tasks = [asyncio.create_task(_worker(r)) for r in records_input]

        pbar = tqdm(total=len(tasks), desc="Extracting (async)")
        for coro in asyncio.as_completed(tasks):
            status, payload = await coro
            if status == "ok":
                results.append(payload)
            else:
                errors.append(payload)
            pbar.update(1)
        pbar.close()

    return results, errors


# PUBLIC ENTRY POINT

def run_batch(
    input_csv: str,
    input_yyyymmdd: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_scrape_concurrent: int = MAX_SCRAPE_CONCURRENT,
    max_api_concurrent: int = MAX_API_CONCURRENT,
):
    """
    Drop-in replacement for DisruptionExtractor.run_batch().
    Reads the same CSV format and writes to the same output paths.
    """
    global MAX_SCRAPE_CONCURRENT, MAX_API_CONCURRENT
    MAX_SCRAPE_CONCURRENT = max_scrape_concurrent
    MAX_API_CONCURRENT    = max_api_concurrent

    if not input_csv or str(input_csv).strip() == "":
        raise ValueError("input_csv must be provided")

    input_path = str(input_csv).strip()
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find input CSV: {input_path}")

    if not input_yyyymmdd or str(input_yyyymmdd).strip() == "":
        input_yyyymmdd = datetime.now().strftime("%Y%m%d")
    input_yyyymmdd = str(input_yyyymmdd).strip()

    base_dir    = Path(__file__).resolve().parents[2]  # Builder_GDELT/
    results_dir = str(base_dir / "results" / "daily" / input_yyyymmdd)
    os.makedirs(results_dir, exist_ok=True)

    out_jsonl  = os.path.join(results_dir, "extractions.jsonl")
    out_errors = os.path.join(results_dir, "errors.jsonl")

    df = pd.read_csv(input_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"url_normalized", "top_expert", "top_expert_p"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    has_coords = "actiongeo_lat" in df.columns and "actiongeo_lon" in df.columns
    coord_cols = (
        ["url_normalized", "top_expert", "top_expert_p", "actiongeo_lat", "actiongeo_lon"]
        if has_coords
        else ["url_normalized", "top_expert", "top_expert_p"]
    )
    rows = df[coord_cols].dropna(subset=["url_normalized"])

    records_input = [
        (
            str(row["url_normalized"]).strip(),
            str(row["top_expert"]).strip() if pd.notna(row["top_expert"]) else None,
            float(row["top_expert_p"]) if pd.notna(row["top_expert_p"]) else None,
            float(row["actiongeo_lat"]) if has_coords and pd.notna(row["actiongeo_lat"]) else None,
            float(row["actiongeo_lon"]) if has_coords and pd.notna(row["actiongeo_lon"]) else None,
        )
        for _, row in rows.iterrows()
    ]

    total = len(records_input)
    print(f"Processing {total} URLs (async) | model={model}")
    print(f"  scrape concurrency: {MAX_SCRAPE_CONCURRENT}  |  api concurrency: {MAX_API_CONCURRENT}")
    print(f"Input:  {input_path}")
    print(f"Output: {results_dir}\n")

    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    results, errors = asyncio.run(_run_async(records_input, out_jsonl, model))

    if results:
        print(f"\nSaved outputs to:\n  {out_jsonl}")

    if errors:
        with open(out_errors, "w", encoding="utf-8") as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Some URLs failed — see:\n  {out_errors}")
    else:
        print("No errors recorded.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to input CSV")
    parser.add_argument("--date",   default=None,  help="YYYYMMDD for output folder")
    parser.add_argument("--model",  default=DEFAULT_MODEL)
    parser.add_argument("--scrape-concurrency", type=int, default=MAX_SCRAPE_CONCURRENT)
    parser.add_argument("--api-concurrency",    type=int, default=MAX_API_CONCURRENT)
    args = parser.parse_args()

    run_batch(
        input_csv=args.input,
        input_yyyymmdd=args.date,
        model=args.model,
        max_scrape_concurrent=args.scrape_concurrency,
        max_api_concurrent=args.api_concurrency,
    )
