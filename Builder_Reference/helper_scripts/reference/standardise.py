"""
Standardisation of reference events.

This file converts RefEvent objects (loaded from individual reference
datasets) into CanonicalEvent objects used by the matching pipeline.

Flowchart role:
- 'Reference Standardisation' stage on the reference-data stream

This module performs light normalisation only.
It does not query external data or infer missing information.
"""

from typing import List

from database_validation.helper_scripts.models import RefEvent, CanonicalEvent


def standardise_reference_events(ref_events: List[RefEvent]) -> List[CanonicalEvent]:
    """
    Convert reference events into canonical form.
    """
    canonical: List[CanonicalEvent] = []

    for r in ref_events:
        text = (r.text or r.location_name or "").strip()

        canonical.append(
            CanonicalEvent(
                id=r.ref_id,
                kind=(r.ref_type or "unknown").lower(),
                date_start=r.date_start,
                date_end=r.date_end or r.date_start,
                location_name=r.location_name,
                country=r.country,
                lat=r.lat,
                lon=r.lon,
                text=text,
                meta={
                    "dataset": r.dataset,
                    "raw_ref_id": r.ref_id,
                },
            )
        )

    return canonical
