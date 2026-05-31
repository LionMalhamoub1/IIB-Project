"""
Metadata inference + normalisation for extracted events.

This file converts raw ExtractedEvent objects into CanonicalEvent objects
used by later matching stages.

It performs only lightweight inference:
- Date: prefer extracted event_date, otherwise fall back to published_at
- Location: use extracted location if present

It also tags weak metadata so it can be analysed later:
- date_is_weak / date_source
- location_is_weak / location_source

Flowchart role:
- 'Metadata Inference' stage on the extracted-events stream
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Optional

from Builder_Matching.helper_scripts.models import ExtractedEvent, CanonicalEvent


# ------------------ DATE PARSING ------------------ #

def _parse_date(s: Optional[str]) -> Optional[date]:
    """
    Parse common date formats into a datetime.date.
    Returns None if parsing fails.
    """
    if not s:
        return None

    s = str(s).strip()
    if not s:
        return None

    # Try common formats (keep minimal; extend if needed)
    fmts = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ]

    for fmt in fmts:
        try:
            dt = datetime.strptime(s[:19], fmt)
            return dt.date()
        except Exception:
            continue

    # Try ISO parsing as last resort
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt.date()
    except Exception:
        return None


# ------------------ CANONICALISATION ------------------ #

def to_canonical_extracted(e: ExtractedEvent) -> CanonicalEvent:
    """
    Convert a raw extracted event to a canonical representation.
    """
    # Date inference with weakness tagging
    extracted_date = _parse_date(e.event_date_raw)
    published_date = _parse_date(e.published_at_raw)

    if extracted_date:
        date_start = extracted_date
        date_end = extracted_date
        date_source = "extracted_event_date"
        date_is_weak = False
    elif published_date:
        date_start = published_date
        date_end = published_date
        date_source = "publication_date_fallback"
        date_is_weak = True
    else:
        date_start = None
        date_end = None
        date_source = "missing"
        date_is_weak = True

    # Location inference with weakness tagging
    loc = (e.location_raw or "").strip()
    if loc and loc.lower() not in {"unknown", "n/a", "na", "none"}:
        location_name = loc
        location_source = "extracted_location"
        location_is_weak = False
    else:
        location_name = None
        location_source = "missing"
        location_is_weak = True

    # Matchable text (keep simple)
    title = (e.title or "").strip()
    text = (e.text or "").strip()
    match_text = " ".join([t for t in [title, text] if t]).strip()

    return CanonicalEvent(
        id=e.event_id,
        kind=(e.disruption_type or "unknown").lower(),
        date_start=date_start,
        date_end=date_end,
        location_name=location_name,
        country=None,
        lat=None,
        lon=None,
        text=match_text,
        meta={
            "url": e.url,
            "event_date_raw": e.event_date_raw,
            "published_at_raw": e.published_at_raw,
            "location_raw": e.location_raw,
            "date_source": date_source,
            "date_is_weak": date_is_weak,
            "location_source": location_source,
            "location_is_weak": location_is_weak,
        },
    )
