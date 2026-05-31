"""
Generate a matched non-flood baseline dataset for indicator validation.

For each country that appears in the reference flood events, sample an equal
number of random (lat, lon, date) points drawn from the same geographic
bounding box and year/month distribution. Any candidate sample that falls
within 200 km and 60 days of a known flood event is discarded and resampled,
preventing contamination of the negative class.

The output JSONL has the same schema as the enriched reference events (with
indicator fields left blank) so that enrich_baseline.py can process it
using the same enrichment scripts.

Output: cache/floods/baseline_samples.jsonl
"""

import json
import logging
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

FLOOD_EVENTS_PATH = Path("cache/floods/reference_floods_static.jsonl")
OUTPUT_PATH       = Path("cache/floods/baseline_samples.jsonl")

EXCLUSION_KM  = 200    # reject baseline sample within this radius of any flood
EXCLUSION_DAYS = 60    # ... and within this many days
SAMPLES_PER_FLOOD = 3  # how many baseline samples to generate per flood event
MAX_ATTEMPTS  = 20     # max resampling attempts before giving up on a slot
RANDOM_SEED   = 42

DATE_FROM = "2017-01-01"  # only use flood events with enriched indicator data
DATE_TO   = "2020-12-31"


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _parse_date(date_str: str) -> datetime | None:
    """Parse a date string to datetime, returning None if malformed."""
    try:
        return datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _date_offset_months(date_str: str, offset_months: int) -> str:
    """Shift a date by a whole number of months, clamping to valid calendar dates."""
    dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
    month = dt.month + offset_months
    year  = dt.year + (month - 1) // 12
    month = ((month - 1) % 12) + 1
    # clamp day to valid range for that month
    import calendar
    max_day = calendar.monthrange(year, month)[1]
    day = min(dt.day, max_day)
    return datetime(year, month, day).strftime("%Y-%m-%d")


# ------------------------------------------------------------------
# Main generation logic
# ------------------------------------------------------------------

def generate_baseline(
    flood_path: Path = FLOOD_EVENTS_PATH,
    output_path: Path = OUTPUT_PATH,
    samples_per_flood: int = SAMPLES_PER_FLOOD,
    exclusion_km: float = EXCLUSION_KM,
    exclusion_days: int = EXCLUSION_DAYS,
    seed: int = RANDOM_SEED,
    date_from: str = DATE_FROM,
    date_to: str = DATE_TO,
) -> list[dict]:
    """
    Sample non-flood baseline points and write them to output_path.

    Strategy:
      - Only flood events within date_from/date_to are used, matching the
        window for which indicator enrichment data is available.
      - Per-country bounding boxes are computed directly from the flood event
        coordinates so that baseline samples are drawn from the same regions.
      - For each flood event one baseline sample is generated at the same
        country location but with the date shifted by a random offset of
        ±6–11 months, avoiding the same seasonal window.
      - Samples within exclusion_km / exclusion_days of any flood are rejected
        and resampled (up to MAX_ATTEMPTS times).

    Returns the list of generated baseline dicts.
    """
    rng = random.Random(seed)

    # Load flood events filtered to the validation date window
    flood_events = []
    n_bad_dates = 0
    n_outside_window = 0
    with open(flood_path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e.get("lat") and e.get("lon") and e.get("date_start") and e.get("country_iso"):
                if _parse_date(e["date_start"]) is None:
                    n_bad_dates += 1
                    continue
                if not (date_from <= e["date_start"][:10] <= date_to):
                    n_outside_window += 1
                    continue
                flood_events.append(e)
    if n_bad_dates:
        log.warning(f"Skipped {n_bad_dates} events with malformed date_start")
    log.info(f"Skipped {n_outside_window} events outside {date_from}–{date_to}")
    if n_bad_dates:
        log.warning(f"Skipped {n_bad_dates} events with malformed date_start")

    log.info(f"Loaded {len(flood_events)} flood events with coordinates")

    # Build per-country bounding boxes from actual event coordinates
    country_coords: dict[str, list[tuple]] = defaultdict(list)
    for e in flood_events:
        country_coords[e["country_iso"]].append((float(e["lat"]), float(e["lon"])))

    country_bbox: dict[str, tuple] = {}
    for iso, coords in country_coords.items():
        lats = [c[0] for c in coords]
        lons = [c[1] for c in coords]
        country_bbox[iso] = (min(lats), max(lats), min(lons), max(lons))

    # Build spatial-temporal index for exclusion check (list of (lat, lon, datetime))
    # Dates already validated above so _parse_date is guaranteed to succeed
    flood_index = [
        (float(e["lat"]), float(e["lon"]), _parse_date(e["date_start"]))
        for e in flood_events
    ]

    def _too_close(lat: float, lon: float, date_str: str) -> bool:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        for (flat, flon, fdate) in flood_index:
            if abs((fdate - dt).days) > exclusion_days:
                continue
            if _haversine_km(lat, lon, flat, flon) < exclusion_km:
                return True
        return False

    # Date offset pool: ±6 to ±11 months (avoids ±0–5 months to stay away from event season)
    offset_pool = list(range(6, 12)) + list(range(-11, -5))

    baseline_samples = []
    n_skipped = 0

    for flood in flood_events:
        iso = flood["country_iso"]
        bbox = country_bbox.get(iso)
        if bbox is None:
            continue

        lat_min, lat_max, lon_min, lon_max = bbox
        # Add small jitter to bbox so we don't always sample exact event coords
        pad = 0.5
        lat_min = max(-90, lat_min - pad)
        lat_max = min(90,  lat_max + pad)
        lon_min = max(-180, lon_min - pad)
        lon_max = min(180,  lon_max + pad)

        for _ in range(samples_per_flood):
            accepted = False
            for _ in range(MAX_ATTEMPTS):
                lat  = rng.uniform(lat_min, lat_max)
                lon  = rng.uniform(lon_min, lon_max)
                offset = rng.choice(offset_pool)
                date = _date_offset_months(flood["date_start"][:10], offset)

                if not _too_close(lat, lon, date):
                    baseline_samples.append({
                        "source":      "baseline",
                        "source_id":   f"b_{len(baseline_samples):06d}",
                        "country_iso": iso,
                        "country":     flood.get("country", ""),
                        "lat":         round(lat, 4),
                        "lon":         round(lon, 4),
                        "date_start":  date,
                        "label":       0,
                    })
                    accepted = True
                    break

            if not accepted:
                n_skipped += 1

    log.info(
        f"Generated {len(baseline_samples)} baseline samples "
        f"({n_skipped} slots could not be filled after {MAX_ATTEMPTS} attempts)"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in baseline_samples:
            f.write(json.dumps(row) + "\n")

    log.info(f"Saved baseline to {output_path}")
    return baseline_samples


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate_baseline()
