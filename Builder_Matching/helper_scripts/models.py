"""
Data models for the validation pipeline.

This file defines the structured objects passed between validation stages.
It contains no logic and no I/O.

Flowchart role:
- Defines the data representations for:
  - Raw extracted events
  - Raw reference events
  - Canonical (normalised) events
  - Candidate matches
  - Final match decisions

All downstream validation code should depend on these models.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from datetime import date


# ------------------ RAW INPUT OBJECTS ------------------ #

@dataclass
class ExtractedEvent:
    """
    Raw disruption event produced by the LLM extraction pipeline.
    Fields may be missing, ambiguous, or inconsistent.
    """
    event_id: str
    disruption_type: str
    event_date_raw: Optional[str]
    location_raw: Optional[str]
    title: Optional[str]
    text: Optional[str]
    published_at_raw: Optional[str]
    url: Optional[str]
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefEvent:
    """
    Raw event record from an external reference dataset.
    Dataset-specific fields are preserved in `raw`.
    """
    ref_id: str
    dataset: str
    ref_type: str
    date_start: Optional[date]
    date_end: Optional[date]
    location_name: Optional[str]
    country: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    text: Optional[str]
    raw: Dict[str, Any] = field(default_factory=dict)


# ------------------ CANONICAL MATCH OBJECT ------------------ #

@dataclass
class CanonicalEvent:
    """
    Normalised event representation used for matching.
    Both extracted and reference events are converted to this form.
    """
    id: str
    kind: str
    date_start: Optional[date]
    date_end: Optional[date]
    location_name: Optional[str]
    country: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


# ------------------ MATCHING OBJECTS ------------------ #

@dataclass
class CandidateMatch:
    """
    One extracted–reference candidate pair with similarity features.
    """
    extracted_id: str
    ref_id: str
    dataset: str
    features: Dict[str, float]
    score: float


@dataclass
class MatchDecision:
    """
    Final decision for a forward or inverse validation pass.
    """
    source_id: str              # extracted_id (forward) or ref_id (inverse)
    matched_id: Optional[str]
    matched_dataset: Optional[str]
    score: Optional[float]
    passed: bool
    reason: str
