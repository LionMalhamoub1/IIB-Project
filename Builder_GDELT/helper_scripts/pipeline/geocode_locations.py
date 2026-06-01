"""
geocode_locations.py
====================
Improves lat/lon coordinates in extraction records by replacing or filling
GDELT actiongeo coordinates with Nominatim-geocoded versions of the
LLM-extracted location_name, where confidence justifies it.

Rules applied per event
-----------------------
  High confidence (>= CONFIDENCE_THRESHOLD) AND non-empty location_name:
    1. Geocode location_name via Nominatim (result cached to avoid repeat calls).
    2. Pre-screen: if the resolved place's bounding box spans > BBOX_MAX_SPAN_KM,
       the location is too coarse (e.g. a country or continent centroid) to be
       useful for localised flood enrichment  -  discard the Nominatim result and
       fall back to GDELT coords.
    3. Country validation: if location_name starts with a recognisable country
       name, check it matches Nominatim's returned country_code.
    4. If Nominatim result is fine-grained AND country check passes:
         lat/lon       <- Nominatim result
         geo_source    <- "nominatim_location_name"
         _geo_verified <- True   (coordinates cross-validated)
    5. Otherwise (coarse bbox, country mismatch, geocoding failure):
         keep original GDELT lat/lon (may be None)
         geo_source    <- "actiongeo" or None
         _geo_verified <- False  (coordinates unverified)

  Low confidence OR empty location_name:
    Keep original GDELT lat/lon unchanged.
    geo_source    <- "actiongeo" (if coords present) or None
    _geo_verified <- False  (no cross-validation attempted)

Post-processing: actiongeo coord rejection
------------------------------------------
  For events that ended up using GDELT actiongeo coords, if a Nominatim result
  IS available for the same location_name (e.g. confidence was too low but
  Nominatim succeeded in a prior run), compute the haversine distance between
  the GDELT coord and the Nominatim point.

  - Distance <= COORD_SUSPICIOUS_KM: GDELT coord is plausible.
      _geo_verified <- True   (GDELT coord confirmed by Nominatim proximity)

  - Distance > COORD_SUSPICIOUS_KM: GDELT coord is implausible (likely a
      country centroid assigned by GDELT when it couldn't resolve the location).
      lat/lon            <- None  (nulled  -  enrichment will skip as no_coords)
      geo_source         <- None
      _geo_verified      <- False
      _coord_rejected_km <- rounded distance (diagnostic, for write-up)

  If no Nominatim reference is available for comparison:
      _geo_verified <- False  (uncertainty acknowledged, coord kept)

_geo_verified flag
------------------
  Every event receives a boolean _geo_verified field:
    True   -  coordinates have been cross-validated against an independent
            geocode of the LLM-extracted location name; the two sources agree
            within COORD_SUSPICIOUS_KM km of each other.
    False  -  coordinates are unverified: either only GDELT actiongeo is
            available with no independent reference to compare against, or the
            Nominatim result was too coarse / mismatched / rejected.

  Intended use in analysis: compute results on all records, then repeat on
  _geo_verified=True only as a sensitivity check  -  if findings are stable,
  coordinate uncertainty is not materially affecting conclusions.

Country sanity check
--------------------
  Parses the first comma-separated component of location_name as a country
  name, resolves it to an ISO alpha-2 code via pycountry fuzzy search, and
  compares against Nominatim's returned country_code.  If no country can be
  parsed from the name (e.g. "Winnipeg") the check is skipped (pass-through).

Cache
-----
  Geocode results are cached to GEOCODE_CACHE_PATH as JSON so reruns never
  re-call Nominatim for the same string.  Format per entry:
      "<location_name>": [lat, lon, country_code, bbox_span_km]  # success
      "<location_name>": null                                      # geocoding failed
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

# -- Config --------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.7
NOMINATIM_URL        = "https://nominatim.openstreetmap.org/search"
NOMINATIM_UA         = "IIB-Project-GeocodeLocations/1.0 (cambridge.ac.uk)"
NOMINATIM_DELAY      = 1.1   # seconds between requests (Nominatim ToS: max 1/s)

# If the place Nominatim resolves has a bounding box whose diagonal span
# exceeds this threshold, the result is considered too coarse to be useful
# as a flood event location.  For example, "South Asia" might resolve to a
# point in India but with a bounding box spanning thousands of km  -  assigning
# rainfall/soil moisture data to that centroid would be meaningless.
# 500 km accepts city and district-level resolutions while rejecting country-
# and large-region-level ones.  The GDELT actiongeo coord is used instead.
BBOX_MAX_SPAN_KM = 500

# If a GDELT actiongeo coord is farther than this from the Nominatim geocode
# of the same location_name, the GDELT coord is almost certainly a country
# centroid assigned when GDELT could not resolve the specific location.
# Enriching at a centroid would associate rainfall/soil moisture data from
# an entirely different area to the event, so the coord is nulled out and
# the event is skipped by enrichment (treated as no_coords).
COORD_SUSPICIOUS_KM = 500

# Shared with verify_gdelt_coords.py so both tools reuse the same cache
GEOCODE_CACHE_PATH = (
    Path(__file__).parent.parent.parent
    / "results" / "coord_verification" / "geocode_cache.json"
)


# -- Haversine distance --------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# -- Cache I/O -----------------------------------------------------------------

def _load_cache() -> dict:
    if GEOCODE_CACHE_PATH.exists():
        with open(GEOCODE_CACHE_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        # Normalise legacy cache entries to the current 4-element format:
        # [lat, lon, country_code, bbox_span_km].
        # Older pipeline runs stored only [lat, lon] or [lat, lon, country_code].
        normalised = {}
        for k, v in raw.items():
            if v is None:
                normalised[k] = None
            elif isinstance(v, list):
                if len(v) == 2:
                    normalised[k] = [v[0], v[1], None, None]
                elif len(v) == 3:
                    normalised[k] = [v[0], v[1], v[2], None]
                else:
                    normalised[k] = v   # already 4+ elements, keep as-is
        return normalised
    return {}


def _save_cache(cache: dict) -> None:
    GEOCODE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GEOCODE_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


# -- Nominatim call ------------------------------------------------------------

def _nominatim_geocode(location_name: str) -> tuple[float, float, str, float] | None:
    """
    Query Nominatim for a location string.

    Returns (lat, lon, country_code, bbox_span_km) or None if not found.
      - country_code   ISO 3166-1 alpha-2 lowercase, e.g. "my", "gb"
      - bbox_span_km   diagonal extent of Nominatim's bounding box for the
                       matched place  -  used downstream to screen out coarse
                       resolutions such as countries or continents
    """
    try:
        resp = requests.get(
            NOMINATIM_URL,
            params={"q": location_name, "format": "json", "limit": 1,
                    "addressdetails": 1},
            headers={"User-Agent": NOMINATIM_UA},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            r = results[0]
            lat = float(r["lat"])
            lon = float(r["lon"])
            cc  = (r.get("address") or {}).get("country_code", "").lower()

            # Compute the diagonal span of Nominatim's bounding box.
            # Nominatim returns boundingbox as [min_lat, max_lat, min_lon, max_lon].
            # A large span (e.g. > 500 km) indicates the result resolved to a
            # whole country or region rather than a specific place.
            bbox_span_km = 0.0
            try:
                bb = r.get("boundingbox", [])
                if len(bb) == 4:
                    min_lat, max_lat = float(bb[0]), float(bb[1])
                    min_lon, max_lon = float(bb[2]), float(bb[3])
                    bbox_span_km = _haversine_km(min_lat, min_lon, max_lat, max_lon)
            except Exception:
                pass   # if bbox parsing fails, keep span=0 (treated as fine-grained)

            return lat, lon, cc, bbox_span_km
        return None
    except Exception as exc:
        log.warning(f"[geocode] Nominatim error for {location_name!r}: {exc}")
        return None
    finally:
        time.sleep(NOMINATIM_DELAY)


# -- Country validation --------------------------------------------------------

def _expected_country_code(location_name: str) -> str | None:
    """
    Tries to resolve the first comma-separated component of location_name to an
    ISO alpha-2 country code using pycountry fuzzy search.
    Returns lowercase code or None if no match.
    """
    try:
        import pycountry
        candidate = location_name.split(",")[0].strip()
        matches = pycountry.countries.search_fuzzy(candidate)
        if matches:
            return matches[0].alpha_2.lower()
    except Exception:
        pass
    return None


def _country_check_passes(location_name: str, geocoded_cc: str) -> bool:
    """
    Returns True if:
      - no country can be parsed from location_name (check is skipped), OR
      - the parsed country matches the geocoded country code.
    """
    expected = _expected_country_code(location_name)
    if expected is None:
        return True   # can't determine expected country -> don't penalise
    return expected == geocoded_cc.lower()


# -- Core geocoding function ---------------------------------------------------

def geocode_events(
    events: list[dict],
    cache: dict | None = None,
    save_cache: bool = True,
) -> list[dict]:
    """
    Process a list of event dicts and return an updated list with improved
    coordinates and a _geo_verified flag on every record.

    For each event, may update:
      - lat, lon             (filled or replaced based on geocoding rules)
      - geo_source           ("nominatim_location_name", "actiongeo", or None)
      - _geo_verified        (True if coordinates cross-validated, else False)
      - _location_too_coarse (True if Nominatim resolved to a region/country)
      - _coord_rejected_km   (set if a suspicious GDELT coord was nulled out)

    Parameters
    ----------
    events      : list of event dicts (as loaded from extractions.jsonl)
    cache       : pre-loaded geocode cache dict; if None the file cache is loaded
    save_cache  : whether to persist the updated cache after processing
    """
    if cache is None:
        cache = _load_cache()

    # Collect unique location names not yet in cache that meet the confidence
    # bar  -  these are the only ones worth calling Nominatim for.
    to_geocode = {
        e["location_name"]
        for e in events
        if (
            e.get("location_name", "").strip()
            and (e.get("confidence") or 0) >= CONFIDENCE_THRESHOLD
            and e["location_name"] not in cache
        )
    }

    if to_geocode:
        log.info(f"[geocode] Calling Nominatim for {len(to_geocode)} new locations...")
        for loc in to_geocode:
            result = _nominatim_geocode(loc)
            cache[loc] = list(result) if result else None

        if save_cache:
            _save_cache(cache)
        log.info(f"[geocode] Done. Cache now has {len(cache)} entries.")

    # -- Assign coordinates to each event --------------------------------------

    updated       = []
    n_nominatim   = 0
    n_actiongeo   = 0
    n_no_coords   = 0
    n_mismatch    = 0
    n_bbox_coarse = 0

    for event in events:
        row  = dict(event)
        loc  = (row.get("location_name") or "").strip()
        conf = row.get("confidence") or 0.0

        if loc and conf >= CONFIDENCE_THRESHOLD:
            cached = cache.get(loc)
            if cached is not None:
                geo_lat      = cached[0]
                geo_lon      = cached[1]
                geo_cc       = cached[2] or ""
                bbox_span_km = cached[3] if len(cached) >= 4 and cached[3] is not None else 0.0

                # Pre-screen: reject Nominatim results that resolved to a place
                # that is geographically too large to be a meaningful flood
                # location.  "South Asia" or "West Africa" might resolve to a
                # point, but the bounding box spanning thousands of km reveals
                # it is a regional centroid.  Assigning GEE enrichment data
                # (rainfall, soil moisture, SPI) to such a centroid would
                # produce values unrelated to the actual event location.
                if bbox_span_km > BBOX_MAX_SPAN_KM:
                    log.debug(
                        f"[geocode] Bbox too large for {loc!r}: "
                        f"{bbox_span_km:.0f} km > {BBOX_MAX_SPAN_KM} km "
                        f"— falling back to actiongeo"
                    )
                    n_bbox_coarse += 1
                    row["_location_too_coarse"] = True
                    _set_actiongeo_source(row)
                    # _geo_verified is assigned in the validation pass below
                    n_actiongeo += 1 if row.get("lat") else 0
                    n_no_coords += 1 if not row.get("lat") else 0

                elif _country_check_passes(loc, geo_cc):
                    # Nominatim result is fine-grained and in the expected country.
                    # Use it as the authoritative coordinate for this event.
                    row["lat"]           = round(geo_lat, 5)
                    row["lon"]           = round(geo_lon, 5)
                    row["geo_source"]    = "nominatim_location_name"
                    # Both the bounding box and country checks passed  -  this is
                    # the highest-confidence coordinate assignment in the pipeline.
                    row["_geo_verified"] = True
                    n_nominatim += 1

                else:
                    # Country mismatch  -  Nominatim resolved the string to a
                    # different country than expected (e.g. a city name that
                    # exists in multiple countries).  Fall back to GDELT's coord.
                    log.debug(
                        f"[geocode] Country mismatch for {loc!r}: "
                        f"expected {_expected_country_code(loc)!r}, got {geo_cc!r} "
                        f"— keeping GDELT coord"
                    )
                    n_mismatch += 1
                    _set_actiongeo_source(row)
                    n_actiongeo += 1 if row.get("lat") else 0
                    n_no_coords += 1 if not row.get("lat") else 0

            else:
                # Nominatim returned no result for this location string.
                # Fall back to GDELT's actiongeo coord (may still be usable).
                _set_actiongeo_source(row)
                n_actiongeo += 1 if row.get("lat") else 0
                n_no_coords += 1 if not row.get("lat") else 0

        else:
            # LLM confidence too low or no location extracted  -  GDELT's
            # actiongeo coord is the only option available.
            _set_actiongeo_source(row)
            n_actiongeo += 1 if row.get("lat") else 0
            n_no_coords += 1 if not row.get("lat") else 0

        updated.append(row)

    # -- Actiongeo coord validation and _geo_verified assignment ---------------
    #
    # Events still using GDELT's actiongeo coord are validated by comparing
    # the GDELT coord against the Nominatim result for the same location_name
    # (if one exists in cache from any previous geocoding call, regardless of
    # whether this event's confidence was above the threshold).
    #
    # This catches the common GDELT failure mode of assigning a country centroid
    # when it cannot resolve the specific place mentioned in the article.
    #
    # Three outcomes:
    #   1. Nominatim reference available, distance ok (<= COORD_SUSPICIOUS_KM):
    #        _geo_verified = True    -  GDELT coord confirmed by independent geocode
    #   2. Nominatim reference available, distance too large:
    #        coord nulled, _geo_verified = False, _coord_rejected_km recorded
    #   3. No Nominatim reference available:
    #        _geo_verified = False   -  coord kept but cannot be verified

    n_coord_verified    = 0
    n_coord_rejected    = 0
    n_coord_unverifiable = 0

    for row in updated:
        # Nominatim-sourced coords are already marked _geo_verified=True above.
        if row.get("geo_source") == "nominatim_location_name":
            continue

        # Events with no coordinates cannot be verified or enriched.
        if row.get("lat") is None or row.get("lon") is None:
            row.setdefault("_geo_verified", False)
            continue

        loc    = (row.get("location_name") or "").strip()
        cached = cache.get(loc) if loc else None

        if not cached:
            # No Nominatim reference available  -  the actiongeo coord is kept
            # but we cannot confirm whether it is accurate.  This is the
            # residual uncertainty that cannot be resolved without additional
            # data sources.  Flagged for downstream sensitivity analysis.
            row["_geo_verified"] = False
            n_coord_unverifiable += 1
            continue

        # Compare GDELT coord against the Nominatim point.
        dist = _haversine_km(row["lat"], row["lon"], cached[0], cached[1])

        if dist <= COORD_SUSPICIOUS_KM:
            # The two independent coordinate sources agree  -  the GDELT coord
            # is in the right general area even if not precisely accurate.
            row["_geo_verified"] = True
            n_coord_verified += 1
        else:
            # The GDELT coord is far from where both the LLM and Nominatim
            # place the event.  Almost certainly a country centroid.  Null
            # it out so GEE enrichment is skipped for this event rather than
            # producing data for the wrong location.
            row["lat"]               = None
            row["lon"]               = None
            row["geo_source"]        = None
            row["_geo_verified"]     = False
            row["_coord_rejected_km"] = round(dist)
            n_coord_rejected += 1

    log.info(
        f"[geocode] Results: nominatim={n_nominatim} actiongeo={n_actiongeo} "
        f"no_coords={n_no_coords} country_mismatch={n_mismatch} "
        f"bbox_too_coarse={n_bbox_coarse} | "
        f"coord_verified={n_coord_verified} coord_rejected={n_coord_rejected} "
        f"coord_unverifiable={n_coord_unverifiable}"
    )
    return updated


def _set_actiongeo_source(row: dict) -> None:
    """Set geo_source based on whether GDELT coords are present."""
    row["geo_source"] = "actiongeo" if (row.get("lat") and row.get("lon")) else None


# -- File-level entry point (used by pipelineRunner) ---------------------------

def geocode_jsonl_inplace(jsonl_path: Path) -> None:
    """
    Read a .jsonl extractions file, apply geocoding to all records, and
    overwrite it with the updated records.
    """
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not records:
        return

    cache   = _load_cache()
    updated = geocode_events(records, cache=cache, save_cache=True)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in updated:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info(f"[geocode] Updated coords written to {jsonl_path}")
