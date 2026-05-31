"""
candidate_generation.py
========================
Generates plausible (GDELT event, reference event) candidate pairs for scoring,
using fast spatial and temporal filters to avoid an expensive all-pairs comparison.

Design rationale
----------------
With ~18k reference events and potentially similar GDELT coverage, an all-pairs
comparison is O(N²) — roughly 324 million pairs.  Most of these are obviously
wrong (different continent, different year).  Candidate generation prunes this
down to a small, tractable set using two cheap filters applied in order:

  1. Temporal filter  — the GDELT event date must fall within the reference
     event's date range, extended by ±MAX_DAYS on each side.  This is an
     overlap check, not just a proximity check, so long-duration floods
     (e.g. 90-day monsoon events) are correctly matched to GDELT articles
     published at any point during the flood.  The ±MAX_DAYS tolerance
     accounts for reporting lag — an article may appear 1–2 weeks after
     the flood begins, or a reference database may record the start date
     slightly late.

  2. Spatial filter   — haversine distance between event centroids must be
     ≤ MAX_KM.  300 km is chosen as the default: large enough to accommodate
     geocoding imprecision (Desinventar admin-1 centroids can be 100+ km
     from the actual flood location), but tight enough to exclude events in
     neighbouring countries.

These two filters together typically reduce the candidate set by >99% relative
to all-pairs, making subsequent scoring fast.

No scoring or match decisions are made here.
"""

import math
from datetime import timedelta, date
from typing import Optional


MAX_KM  = 300   # spatial filter radius (km)
MAX_DAYS = 14   # temporal tolerance each side of date range (days)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points (degrees) in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _parse_date(val) -> Optional[date]:
    """Parse a date string (YYYY-MM-DD or ISO timestamp) to a date object."""
    if val is None:
        return None
    if isinstance(val, date):
        return val
    s = str(val)[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _dates_overlap(
    gdelt_date: date,
    ref_start: Optional[date],
    ref_end: Optional[date],
    max_days: int,
) -> bool:
    """
    True if gdelt_date falls within [ref_start - max_days, ref_end + max_days].
    Uses ref_start as ref_end when ref_end is missing (point event).
    """
    if ref_start is None:
        return False
    ref_end = ref_end or ref_start
    return (
        ref_start - timedelta(days=max_days)
        <= gdelt_date
        <= ref_end + timedelta(days=max_days)
    )


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def generate_candidates(
    gdelt_events: list[dict],
    reference_events: list[dict],
    max_km: float = MAX_KM,
    max_days: int = MAX_DAYS,
) -> list[tuple[int, int]]:
    """
    Return a list of (gdelt_index, reference_index) candidate pairs that pass
    both the spatial and temporal filters.

    Parameters
    ----------
    gdelt_events      : list of enriched GDELT event dicts
    reference_events  : list of enriched reference event dicts
    max_km            : maximum haversine distance (km)
    max_days          : date tolerance applied each side of reference date range

    Returns
    -------
    List of (i, j) index pairs into the input lists.
    """
    candidates = []
    n_no_loc = 0

    for i, gdelt in enumerate(gdelt_events):
        glat = gdelt.get("lat")
        glon = gdelt.get("lon")
        gdate = _parse_date(gdelt.get("event_date") or gdelt.get("date_start"))

        if glat is None or glon is None or gdate is None:
            n_no_loc += 1
            continue

        for j, ref in enumerate(reference_events):
            rlat = ref.get("lat")
            rlon = ref.get("lon")
            rstart = _parse_date(ref.get("date_start"))
            rend = _parse_date(ref.get("date_end"))

            if rlat is None or rlon is None:
                continue

            # Temporal filter first (cheap)
            if not _dates_overlap(gdate, rstart, rend, max_days):
                continue

            # Spatial filter (slightly more expensive)
            if _haversine_km(glat, glon, rlat, rlon) > max_km:
                continue

            candidates.append((i, j))

    return candidates
