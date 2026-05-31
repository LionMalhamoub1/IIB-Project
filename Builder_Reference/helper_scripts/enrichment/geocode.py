"""
Geocoding cache and lookup for Desinventar events.

Desinventar records have region + country but no lat/lon. This module:
  1. Collects all unique (region, country) pairs from the dataset
  2. Geocodes them via Nominatim (OpenStreetMap) at 1 req/sec
  3. Persists results to a JSON cache file
  4. Provides a lookup function used by the backfill script

geocode_source values written to events:
  "precise"          original lat/lon from the source dataset
  "admin1_centroid"  geocoded from region + country (Desinventar)
  "country_centroid" geocoded from country only (region was empty)
  None               no coordinates available (geocoding failed)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

log = logging.getLogger(__name__)

NOMINATIM_USER_AGENT = "iib-project-flood-geocoder"
CACHE_PATH = Path(__file__).resolve().parents[2] / "cache" / "geocoding" / "desinventar_cache.json"
REQUEST_DELAY = 1.1   # Nominatim rate limit: 1 req/sec


def load_cache() -> dict:
    if CACHE_PATH.exists():
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _nominatim_query(geolocator, query: str) -> Optional[tuple[float, float]]:
    """Return (lat, lon) for a query string, or None on failure."""
    try:
        location = geolocator.geocode(query, timeout=10)
        if location:
            return (location.latitude, location.longitude)
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        log.warning(f"Geocoding failed for {repr(query)}: {e}")
    return None


def build_geocode_cache(events: list[dict]) -> dict:
    """
    Geocode all unique (region, country) pairs from Desinventar events
    that are missing lat/lon. Results are persisted to CACHE_PATH.

    Returns the full cache dict: {cache_key: {"lat": ..., "lon": ..., "geocode_source": ...}}
    """
    cache = load_cache()

    # Collect unique pairs that need geocoding
    pairs: set[tuple[str, str]] = set()
    for e in events:
        if e.get("source") != "Desinventar":
            continue
        if e.get("lat") and e.get("lon"):
            continue
        region = (e.get("region") or "").strip()
        country = (e.get("country") or "").strip()
        if country:
            pairs.add((region, country))

    new_pairs = [(r, c) for (r, c) in pairs if _cache_key(r, c) not in cache]
    log.info(f"Pairs to geocode: {len(new_pairs)} new / {len(pairs)} total")

    if not new_pairs:
        return cache

    geolocator = Nominatim(user_agent=NOMINATIM_USER_AGENT)

    for i, (region, country) in enumerate(new_pairs):
        key = _cache_key(region, country)

        if region:
            # Try "Region, Country" first, fall back to country alone
            query = f"{region}, {country}"
            result = _nominatim_query(geolocator, query)
            time.sleep(REQUEST_DELAY)

            if result is None:
                log.debug(f"Region query failed, trying country only: {repr(country)}")
                result = _nominatim_query(geolocator, country)
                time.sleep(REQUEST_DELAY)
                source = "country_centroid" if result else None
            else:
                source = "admin1_centroid"
        else:
            # No region — geocode country only
            result = _nominatim_query(geolocator, country)
            time.sleep(REQUEST_DELAY)
            source = "country_centroid" if result else None

        if result:
            cache[key] = {"lat": result[0], "lon": result[1], "geocode_source": source}
            log.info(f"  [{i+1}/{len(new_pairs)}] {repr(query if region else country)} -> {result} ({source})")
        else:
            cache[key] = {"lat": None, "lon": None, "geocode_source": None}
            log.warning(f"  [{i+1}/{len(new_pairs)}] Failed: {repr(region)}, {repr(country)}")

        # Save incrementally every 10 entries
        if (i + 1) % 10 == 0:
            save_cache(cache)

    save_cache(cache)
    log.info(f"Cache saved to {CACHE_PATH}")
    return cache


def apply_geocode_cache(events: list[dict], cache: dict) -> list[dict]:
    """
    Return a copy of events with lat/lon filled in for Desinventar records
    that were missing coordinates. Adds 'geocode_source' to every event.
    """
    enriched = []
    filled = 0
    failed = 0

    for e in events:
        row = dict(e)

        if row.get("lat") and row.get("lon"):
            row["geocode_source"] = "precise"
        elif row.get("source") == "Desinventar":
            region  = (row.get("region") or "").strip()
            country = (row.get("country") or "").strip()
            key = _cache_key(region, country)
            entry = cache.get(key, {})
            if entry.get("lat") and entry.get("lon"):
                row["lat"] = entry["lat"]
                row["lon"] = entry["lon"]
                row["geocode_source"] = entry["geocode_source"]
                filled += 1
            else:
                row["geocode_source"] = None
                failed += 1
        else:
            row["geocode_source"] = None

        enriched.append(row)

    log.info(f"Applied geocodes: {filled} filled, {failed} failed/missing")
    return enriched


def _cache_key(region: str, country: str) -> str:
    return f"{region.lower().strip()}|{country.lower().strip()}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

import json
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

ROOT        = Path(__file__).resolve().parents[3]
INPUT_PATH  = ROOT / "cache" / "floods" / "reference_floods_consolidated.jsonl"
OUTPUT_PATH = ROOT / "cache" / "floods" / "reference_floods_geocoded.jsonl"


def main():
    if not INPUT_PATH.exists():
        log.error(f"Input not found: {INPUT_PATH}")
        sys.exit(1)

    log.info(f"Loading events from {INPUT_PATH}")
    events = []
    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line))
    log.info(f"Loaded {len(events)} events")

    before = len(events)
    events = [e for e in events if e.get("source") != "Desinventar"]
    log.info(f"Filtered to event-level records: {len(events)} (excluded {before - len(events)} Desinventar district children)")

    log.info("Building geocode cache (Nominatim, ~1 req/sec)...")
    cache = build_geocode_cache(events)

    log.info("Applying geocodes to events...")
    geocoded = apply_geocode_cache(events, cache)

    sources = {}
    for e in geocoded:
        s = e.get("geocode_source") or "none"
        sources[s] = sources.get(s, 0) + 1
    log.info("geocode_source breakdown:")
    for s, count in sorted(sources.items(), key=lambda x: -x[1]):
        log.info(f"  {s}: {count}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in geocoded:
            f.write(json.dumps(row, default=str) + "\n")

    log.info(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
