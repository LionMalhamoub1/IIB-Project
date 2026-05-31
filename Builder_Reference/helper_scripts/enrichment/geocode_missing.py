"""
geocode_missing.py
==================
Backfills lat/lon for reference sources that were never geocoded:
  HANZE           — geocode from location_name via Nominatim REST API
  Desinventar-AGG — geocode from location_name/region via Nominatim REST API
  ReliefWeb       — geocode from location_name/country via Nominatim REST API

Uses the same approach as Builder_GDELT/helper_scripts/pipeline/geocode_locations.py:
  - Nominatim queried via requests (not geopy) with 1.1s delay in finally block
  - Bounding-box span check rejects coarse results (country/region centroids)
  - On 429, waits 10s and retries once before giving up
  - If nothing specific found, lat/lon stays None

Multiple workers split the unique query list. Each worker sleeps NOMINATIM_DELAY
between requests. Some 429s may occur when workers overlap; these are retried.

Usage
-----
  python -m Builder_Reference.helper_scripts.enrichment.geocode_missing
"""

import json
import logging
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parents[3]
ENRICHED    = ROOT / "Builder_Reference" / "outputs" / "reference_floods_enriched.jsonl"
CACHE_PATH  = ROOT / "Builder_Reference" / "cache" / "geocoding" / "missing_sources_cache.json"

SOURCES         = {"HANZE", "Desinventar-AGG", "ReliefWeb"}
NOMINATIM_URL   = "https://nominatim.openstreetmap.org/search"
NOMINATIM_UA    = "IIB-Project-GeocodeLocations/1.0 (cambridge.ac.uk)"
NOMINATIM_DELAY = 1.5    # seconds between requests (always fires in finally)
RETRY_WAIT      = 60.0   # seconds to wait on a 429 before retrying — longer block needs longer wait
BBOX_MAX_KM     = 400    # reject results coarser than this
WORKERS         = 1      # single worker — Nominatim rate-limits by IP, not user agent


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Cache helpers (cache_lock guards all writes)
# ---------------------------------------------------------------------------

_cache_lock = threading.Lock()


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _cache_lock:
        with CACHE_PATH.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)


def _geocode_key(source: str, event: dict) -> str:
    location_name = (event.get("location_name") or "").strip()
    region        = (event.get("region") or "").strip()
    country       = (event.get("country") or "").strip()
    return f"{source}|{location_name}|{region}|{country}".lower()


# ---------------------------------------------------------------------------
# Nominatim query — single call with 429 retry
# ---------------------------------------------------------------------------

def _nominatim_query(query: str) -> Optional[tuple[float, float, float]]:
    """
    Query Nominatim. Returns (lat, lon, bbox_span_km) or None.
    On 429 waits RETRY_WAIT seconds and retries once.
    Always sleeps NOMINATIM_DELAY in finally (even on failure).
    """
    def _do_request():
        return requests.get(
            NOMINATIM_URL,
            params={"q": query, "format": "json", "limit": 1, "addressdetails": 1},
            headers={"User-Agent": NOMINATIM_UA},
            timeout=10,
        )

    try:
        resp = _do_request()
        if resp.status_code == 429:
            log.debug(f"429 for {query!r}, waiting {RETRY_WAIT}s then retrying...")
            time.sleep(RETRY_WAIT)
            resp = _do_request()

        resp.raise_for_status()
        results = resp.json()
        if not results:
            return None

        r = results[0]
        lat = float(r["lat"])
        lon = float(r["lon"])
        bbox_span_km = 0.0
        try:
            bb = r.get("boundingbox", [])
            if len(bb) == 4:
                bbox_span_km = _haversine_km(
                    float(bb[0]), float(bb[2]), float(bb[1]), float(bb[3])
                )
        except Exception:
            pass
        return lat, lon, bbox_span_km

    except Exception as exc:
        log.warning(f"Nominatim error for {query!r}: {exc}")
        return None
    finally:
        time.sleep(NOMINATIM_DELAY)


def _geocode_with_bbox_check(query: str) -> Optional[tuple[float, float]]:
    result = _nominatim_query(query)
    if result is None:
        return None
    lat, lon, bbox_span_km = result
    if bbox_span_km > BBOX_MAX_KM:
        log.debug(f"Rejected {query!r}: bbox {bbox_span_km:.0f} km > {BBOX_MAX_KM} km")
        return None
    return lat, lon


# ---------------------------------------------------------------------------
# Per-source query strategy
# ---------------------------------------------------------------------------

def _build_queries(source: str, event: dict) -> list[str]:
    """Return ordered list of query strings to try for this event."""
    location_name = (event.get("location_name") or "").strip()
    region        = (event.get("region") or "").strip()
    country       = (event.get("country") or "").strip()
    queries = []

    if source == "HANZE":
        # "Korana; Kupa, Croatia" → try "Korana, Croatia", "Kupa, Croatia", full string
        if location_name and location_name.lower() != country.lower():
            components = [c.strip() for c in location_name.split(";")]
            for comp in components:
                if country.lower() not in comp.lower() and country:
                    queries.append(f"{comp}, {country}")
                else:
                    queries.append(comp)
            if len(components) > 1:
                queries.append(location_name)

    elif source == "Desinventar-AGG":
        if region and country:
            queries.append(f"{region}, {country}")
        if location_name and location_name.lower() != country.lower():
            queries.append(location_name)

    elif source == "ReliefWeb":
        if location_name and location_name.lower() != country.lower():
            queries.append(location_name)

    return queries


def _geocode_event(source: str, event: dict) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Try each query in order, return first specific result or (None, None, None)."""
    for query in _build_queries(source, event):
        result = _geocode_with_bbox_check(query)
        if result:
            return result[0], result[1], query
    return None, None, None


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker(
    worker_id: int,
    items: list[tuple[str, str, dict]],
    shared_cache: dict,
    counter: list,
    total: int,
    pbar: tqdm,
) -> None:
    """Process a chunk of (key, source, event) items, writing results into shared_cache."""
    for key, source, event in items:
        lat, lon, query_used = _geocode_event(source, event)
        result = {
            "lat": lat,
            "lon": lon,
            "geocode_source": "nominatim" if lat else None,
            "query": query_used,
        }

        with _cache_lock:
            shared_cache[key] = result
            counter[0] += 1
            done = counter[0]

        label = (event.get("location_name") or event.get("country") or "")[:45]
        status = f"({lat:.2f},{lon:.2f})" if lat else "—"
        pbar.set_postfix_str(f"{label} {status}", refresh=False)
        pbar.update(1)

        if done % 20 == 0:
            _save_cache(shared_cache)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not ENRICHED.exists():
        log.error(f"Input not found: {ENRICHED}")
        sys.exit(1)

    log.info(f"Loading events from {ENRICHED}")
    events = []
    with ENRICHED.open("r", encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line))
    log.info(f"  {len(events):,} total events")

    cache = _load_cache()

    # Clear stale country-centroid fallbacks from previous runs
    stale = [k for k, v in cache.items() if isinstance(v, dict) and v.get("geocode_source") == "country_centroid"]
    if stale:
        for k in stale:
            del cache[k]
        log.info(f"  Cleared {len(stale)} stale country-centroid cache entries")
        _save_cache(cache)

    # Reset country-centroid coordinates written by previous runs so we retry properly
    reset_count = 0
    for e in events:
        if e.get("source") in SOURCES and e.get("geocode_source") == "country_centroid":
            e["lat"] = e["lon"] = e["latitude"] = e["longitude"] = None
            e["geocode_source"] = None
            reset_count += 1
    if reset_count:
        log.info(f"  Reset {reset_count} country-centroid coordinates for re-geocoding")

    to_geocode = [e for e in events if e.get("source") in SOURCES and not e.get("lat")]
    log.info(f"  {len(to_geocode):,} events need geocoding")

    # Deduplicate to unique keys
    unique: dict[str, tuple[str, dict]] = {}
    for e in to_geocode:
        key = _geocode_key(e["source"], e)
        if key not in unique:
            unique[key] = (e["source"], e)

    cache_misses = [(key, src, ev) for key, (src, ev) in unique.items() if key not in cache]
    log.info(
        f"  {len(unique):,} unique locations | "
        f"{len(unique) - len(cache_misses)} cached | "
        f"{len(cache_misses)} to query"
    )

    if cache_misses:
        n_workers = min(WORKERS, len(cache_misses))
        chunks = [cache_misses[i::n_workers] for i in range(n_workers)]
        counter = [0]
        est_min = len(cache_misses) * NOMINATIM_DELAY / n_workers / 60
        log.info(f"  Querying Nominatim: {len(cache_misses)} locations, est. {est_min:.1f} min")

        with tqdm(total=len(cache_misses), unit="loc", ncols=90,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(_worker, i, chunk, cache, counter, len(cache_misses), pbar)
                    for i, chunk in enumerate(chunks)
                ]
                for f in as_completed(futures):
                    f.result()

        _save_cache(cache)
        log.info(f"Cache saved: {CACHE_PATH}")

    # Apply cache to events
    filled = failed = 0
    for e in events:
        if e.get("source") not in SOURCES or e.get("lat"):
            continue
        entry = cache.get(_geocode_key(e["source"], e), {})
        if entry and entry.get("lat"):
            e["lat"]            = entry["lat"]
            e["lon"]            = entry["lon"]
            e["latitude"]       = entry["lat"]
            e["longitude"]      = entry["lon"]
            e["geocode_source"] = entry["geocode_source"]
            filled += 1
        else:
            failed += 1

    log.info(f"Applied geocodes: {filled} filled, {failed} no specific result (left as None)")
    for src in sorted(SOURCES):
        src_events = [e for e in events if e.get("source") == src]
        src_filled = sum(1 for e in src_events if e.get("lat"))
        log.info(f"  {src}: {src_filled}/{len(src_events)} have coordinates")

    with ENRICHED.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, default=str) + "\n")
    log.info(f"Written back to {ENRICHED}")


if __name__ == "__main__":
    main()
