"""
Filtering utilities for extracted events.

This file applies simple scope-based filters to the extracted events
before validation is performed.

Flowchart role:
- Part of the 'Filtering & Profiling' stage on the extracted-events stream

This module does not modify event contents; it only includes or excludes
events based on predefined criteria.
"""

from typing import List, Optional, Set
from Builder_Matching.helper_scripts.models import ExtractedEvent


def filter_by_type(
    events: List[ExtractedEvent],
    allowed_types: Optional[Set[str]] = None
) -> List[ExtractedEvent]:
    """
    Keep only extracted events whose disruption_type is in allowed_types.
    If allowed_types is None, no filtering is applied.
    """
    if allowed_types is None:
        return events

    allowed = {t.lower() for t in allowed_types}

    return [
        e for e in events
        if (e.disruption_type or "unknown").lower() in allowed
    ]
