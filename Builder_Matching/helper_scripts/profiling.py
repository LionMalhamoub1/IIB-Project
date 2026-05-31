"""
Profiling utilities for extracted events.

This file computes simple completeness and quality metrics on the
LLM-extracted events prior to validation.

Flowchart role:
- Part of the 'Filtering & Profiling' stage on the extracted-events stream

The output is used for diagnostics and reporting, not for filtering.
"""

from typing import Dict, List
from Builder_Matching.helper_scripts.models import ExtractedEvent


def profile_extracted_events(events: List[ExtractedEvent]) -> Dict[str, float]:
    """
    Compute basic profiling statistics for a set of extracted events.
    """
    n = len(events)
    if n == 0:
        return {
            "n_events": 0,
            "missing_date_rate": 0.0,
            "missing_location_rate": 0.0,
            "unknown_type_rate": 0.0,
        }

    missing_date = sum(1 for e in events if not e.event_date_raw)
    missing_location = sum(1 for e in events if not e.location_raw)
    unknown_type = sum(
        1 for e in events
        if (e.disruption_type or "unknown") == "unknown"
    )

    return {
        "n_events": n,
        "missing_date_rate": missing_date / n,
        "missing_location_rate": missing_location / n,
        "unknown_type_rate": unknown_type / n,
    }
