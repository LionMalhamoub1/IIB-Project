from __future__ import annotations

import os
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from .webscraper import extract_article_text

# ------------------ ENV + OPENAI CLIENT SETUP ------------------ #
from pathlib import Path

# Try repo-root .env (common) + fallback to script-folder .env
ENV_CANDIDATES = [
    Path(__file__).resolve().parents[1] / ".env",  # repo root if script is /src/...
    Path(__file__).resolve().parent / ".env",      # same folder as this file
    Path.cwd() / ".env",                           # current working dir (last resort)
]

env_loaded = False
for p in ENV_CANDIDATES:
    if p.exists():
        load_dotenv(dotenv_path=p, override=True)
        print(f"[env] Loaded .env from: {p}")
        env_loaded = True
        break

if not env_loaded:
    print(f"[env] No .env found. Tried: {', '.join(str(p) for p in ENV_CANDIDATES)}")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found after loading .env. Check location + file contents.")

client = OpenAI(api_key=api_key)

DEFAULT_MODEL = "gpt-5-mini"
MAX_WORKERS = 20


# ------------------ DATA MODEL ------------------ #

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


# ------------------ DATE NORMALISATION ------------------ #

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


# ------------------ LLM EXTRACTION ------------------ #

def _call_chatgpt_extractor(
    url: str,
    title: str,
    text: str,
    model: str = DEFAULT_MODEL,
    timeout: int = 60,
) -> Dict[str, Any]:

    system_prompt = """\
You are an information extraction engine for supply chain disruptions.

Your job is to read a news article and extract a REAL physical disruption event. Stay faithful to the article text.

You MUST output a single JSON object and NOTHING else.
"""

    user_prompt = f"""Extract a single main disruption event from the article below.

Only extract events of these types:
- flood  (including flash floods, river flooding, inundation...)
- labour_strike  (worker strikes, walkouts...)
- protests  (demonstrations, civil unrest affecting operations...)

If the article describes none of these, or confidence is below 0.6, return:
{{
  "disruption_type": "unknown",
  "event_date": null,
  "location": ["", "", "", ""],
  "duration_hours": null,
  "details": {{}},
  "confidence": 0.0
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
  "confidence": 0.0–1.0
}}

IMPORTANT:
- Only extract events of the three allowed types above. Classify everything else as "unknown".
- Confidence threshold for a valid disruption is 0.6.
- Ignore metaphorical language e.g. "a flood of criticism" when no real event is described.

LOCATION FIELD:
- Always four strings in descending geographic scale: ["country", "region_or_state", "city", "specific_location"]
- Leave unused levels as "".

DETAILS FIELD:
- Only include fields listed below for the matched disruption type.
- Only include information explicitly mentioned in the article — do not infer or fabricate.
- If nothing relevant is mentioned, details must be {{}}.

flood:
- rainfall_intensity
- rainfall_levels
- death_toll
- main_cause
- glide_number

protests:
- protest_type
- protesting_groups
- target_of_protest
- issue
- sector
- estimated_participants
- event_start_day
- reported_day_number
 
labour_strike:
- strike_type
- workers_or_unions_involved
- organizations_or_companies
- target_of_strike
- issue
- sector
- estimated_participants
- event_start_day
- reported_day_number

Now process this article:

URL: {url}
TITLE: {title}
TEXT: {text}
"""


    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        timeout=timeout,
    )

    raw = completion.choices[0].message.content or ""

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    data.setdefault("disruption_type", "unknown")
    data.setdefault("event_date", None)
    data.setdefault("location", ["", "", "", ""])
    data.setdefault("duration_hours", None)
    data.setdefault("details", {})
    data.setdefault("confidence", 0.0)

    return data


# ------------------ SINGLE URL EXTRACTION ------------------ #

def extract_from_url_llm_single_pass(
    url: str,
    expert_type: Optional[str],
    expert_prob: Optional[float],
    threshold: float = 0.6,
    model: str = DEFAULT_MODEL,
) -> ExtractRecord:

    art = extract_article_text(url)
    title = art.get("title", "") or ""
    body = (art.get("text", "") or "")[:3200]
    publish_date = art.get("publish_date")

    if len(body.strip()) < 50:
        # Dead link, paywall, or empty page — skip LLM to avoid wasting tokens
        llm_out = {
            "disruption_type": "unknown",
            "event_date": None,
            "location": ["", "", "", ""],
            "duration_hours": None,
            "details": {},
            "confidence": 0.0,
        }
    else:
        llm_out = _call_chatgpt_extractor(url, title, body, model=model)
    llm_type = llm_out.get("disruption_type") or "unknown"

    publish_date_norm = _normalise_date(publish_date, date_only=False)
    event_date_norm = _normalise_date(llm_out.get("event_date"), date_only=True)

    expert_type_clean = expert_type.strip() if expert_type else None

    # Hybrid override logic
    if expert_type_clean and expert_prob is not None and expert_prob >= threshold:
        final_type = expert_type_clean
        classification_source = "expert"
    else:
        final_type = llm_type
        classification_source = "llm"

    return ExtractRecord(
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
    )


# ------------------ BATCH RUNNER ------------------ #

def run_batch(
    input_csv: str,
    input_yyyymmdd: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_workers: int = MAX_WORKERS,
):

    if input_csv is None or str(input_csv).strip() == "":
        raise ValueError("input_csv must be provided")

    input_path = str(input_csv).strip()

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find input CSV: {input_path}")

    if input_yyyymmdd is None or str(input_yyyymmdd).strip() == "":
        input_yyyymmdd = datetime.now().strftime("%Y%m%d")
    input_yyyymmdd = str(input_yyyymmdd).strip()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results", "daily", input_yyyymmdd)
    os.makedirs(results_dir, exist_ok=True)

    out_jsonl = os.path.join(results_dir, "extractions.jsonl")
    out_csv = os.path.join(results_dir, "extractions.csv")
    out_errors = os.path.join(results_dir, "errors.csv")

    df = pd.read_csv(input_path)
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"url_normalized", "top_expert", "top_expert_p"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    has_coords = "actiongeo_lat" in df.columns and "actiongeo_lon" in df.columns
    coord_cols = ["url_normalized", "top_expert", "top_expert_p", "actiongeo_lat", "actiongeo_lon"] if has_coords else ["url_normalized", "top_expert", "top_expert_p"]
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

    if os.path.exists(out_jsonl):
        os.remove(out_jsonl)

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    write_lock = threading.Lock()

    def _worker(record_tuple) -> Dict[str, Any]:
        url, expert_type, expert_prob, lat, lon = record_tuple
        rec = extract_from_url_llm_single_pass(
            url=url,
            expert_type=expert_type,
            expert_prob=expert_prob,
            threshold=0.6,
            model=model,
        )
        rec.lat = lat
        rec.lon = lon
        return rec.__dict__

    print(f"Processing {total} URLs with model={model} using up to {max_workers} workers...\n")
    print(f"Input:  {input_path}")
    print(f"Output: {results_dir}\n")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_input = {
            executor.submit(_worker, record): record[0]
            for record in records_input
        }

        for fut in tqdm(as_completed(future_to_input), total=total, desc="Extracting"):
            url = future_to_input[fut]
            try:
                data = fut.result()
                results.append(data)

                with write_lock:
                    with open(out_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")

            except Exception as e:
                errors.append({"url": url, "error": str(e)})

    if results:
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\nSaved outputs to:\n  {out_jsonl}\n  {out_csv}")

    if errors:
        pd.DataFrame(errors).to_csv(out_errors, index=False, encoding="utf-8")
        print(f"Some URLs failed — see:\n  {out_errors}")
    else:
        print("No errors recorded.")

if __name__ == "__main__":
    run_batch()