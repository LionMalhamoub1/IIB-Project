"""
scoring.py
===========
Computes a weighted similarity score for each candidate (GDELT, reference)
event pair.

Design rationale
----------------
Four components are combined into a single score in [0, 1]:

  Component              Weight   Rationale
  ---------------------  ------   -------------------------------------------------
  Geo distance            0.30    Primary spatial signal.  Two events describing
                                  the same flood must be geographically close.
                                  Uses exponential decay: score = exp(-d / 100km),
                                  giving 1.0 at 0 km, 0.37 at 100 km, 0.05 at
                                  300 km.  Exponential decay is preferred over
                                  linear because precision matters most at short
                                  distances — the difference between 5 km and
                                  50 km is more meaningful than 250 km vs 300 km.

  Temporal proximity      0.20    Secondary signal.  Events within the reference
                                  date range score 1.0; outside they decay by
                                  1 / (1 + days_outside).  Lower weight than geo
                                  because date reporting in both GDELT and reference
                                  sources has known lag and imprecision.

  Hydro fingerprint       0.35    Highest weight.  The hydro-climate feature vector
                                  (SPI, GloFAS percentile, CHIRPS anomaly, etc.) is
                                  an objective, source-independent measure of the
                                  physical event.  Two events in the same region
                                  and week could be different floods — only the
                                  hydro fingerprint can distinguish them.  Cosine
                                  similarity of normalised 7-D vectors (see
                                  feature_vectors.py).

  Location text           0.15    Weakest signal.  Simple token overlap on country
                                  and location_name strings.  Included because it
                                  catches obvious mismatches (wrong country) that
                                  might survive the 300 km spatial filter in densely
                                  populated border regions.

Weight justification: hydro fingerprint carries the highest weight because it
is the hardest signal to fabricate by coincidence — a 0.35 weight for the most
discriminative feature follows standard feature importance practice.  The geo
weight (0.30) is second because location is the most reliable attribute in both
datasets.  Temporal (0.20) is penalised for known reporting lag.  Text (0.15)
is treated as a tiebreaker only.
"""

import math
import numpy as np
from datetime import date
from typing import Optional

from Builder_Matching.matching.helper_scripts.feature_vectors import hydro_similarity
from Builder_Matching.matching.helper_scripts.candidate_generation import _parse_date, _haversine_km


# ---------------------------------------------------------------------------
# Component score functions
# ---------------------------------------------------------------------------

def _geo_score(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Exponential decay on haversine distance, half-life ~100 km."""
    d = _haversine_km(lat1, lon1, lat2, lon2)
    return math.exp(-d / 100.0)


def _temporal_score(gdelt_event: dict, ref_event: dict) -> float:
    """
    Score 1.0 if GDELT date falls within reference date range,
    decaying by 1/(1+days) outside it.
    """
    gdate = _parse_date(gdelt_event.get("event_date") or gdelt_event.get("date_start"))
    rstart = _parse_date(ref_event.get("date_start"))
    rend = _parse_date(ref_event.get("date_end"))

    if gdate is None or rstart is None:
        return 0.0

    rend = rend or rstart

    if rstart <= gdate <= rend:
        return 1.0

    days_outside = min(abs((gdate - rstart).days), abs((gdate - rend).days))
    return 1.0 / (1.0 + days_outside)


def _location_text_score(gdelt_event: dict, ref_event: dict) -> float:
    """
    Jaccard token overlap over the concatenated country + location_name fields.
    Provides a weak sanity check against cross-country false positives.
    """
    def tokens(event):
        parts = [
            str(event.get("country") or ""),
            str(event.get("location_name") or ""),
        ]
        return set(" ".join(parts).lower().split())

    t_g = tokens(gdelt_event)
    t_r = tokens(ref_event)
    if not t_g or not t_r:
        return 0.0
    intersection = t_g & t_r
    union = t_g | t_r
    return len(intersection) / len(union) if union else 0.0


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------

W_GEO   = 0.30
W_TIME  = 0.20
W_HYDRO = 0.35
W_TEXT  = 0.15


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def score_pair(
    gdelt_event: dict,
    ref_event: dict,
    gdelt_vec: np.ndarray,
    ref_vec: np.ndarray,
) -> dict:
    """
    Compute a weighted similarity score for one candidate pair.

    Parameters
    ----------
    gdelt_event : enriched GDELT event dict
    ref_event   : enriched reference event dict
    gdelt_vec   : hydro feature vector for the GDELT event
    ref_vec     : hydro feature vector for the reference event

    Returns
    -------
    dict with keys: score, geo, temporal, hydro, text
    """
    glat = gdelt_event.get("lat")
    glon = gdelt_event.get("lon")
    rlat = ref_event.get("lat")
    rlon = ref_event.get("lon")

    geo = (
        _geo_score(glat, glon, rlat, rlon)
        if all(v is not None for v in [glat, glon, rlat, rlon])
        else 0.0
    )
    temporal = _temporal_score(gdelt_event, ref_event)
    hydro    = hydro_similarity(gdelt_vec, ref_vec)
    text     = _location_text_score(gdelt_event, ref_event)

    score = W_GEO * geo + W_TIME * temporal + W_HYDRO * hydro + W_TEXT * text

    return {
        "score":    round(score, 4),
        "geo":      round(geo, 4),
        "temporal": round(temporal, 4),
        "hydro":    round(hydro, 4),
        "text":     round(text, 4),
    }


def score_all_candidates(
    gdelt_events: list[dict],
    reference_events: list[dict],
    gdelt_matrix: np.ndarray,
    ref_matrix: np.ndarray,
    candidates: list[tuple[int, int]],
) -> list[dict]:
    """
    Score all candidate pairs.

    Parameters
    ----------
    gdelt_events    : list of enriched GDELT event dicts
    reference_events: list of enriched reference event dicts
    gdelt_matrix    : (N_gdelt, F) feature matrix
    ref_matrix      : (N_ref, F) feature matrix
    candidates      : list of (gdelt_index, ref_index) from candidate_generation

    Returns
    -------
    List of scored pair dicts, each containing gdelt_idx, ref_idx, and score components.
    """
    scored = []
    for i, j in candidates:
        components = score_pair(
            gdelt_events[i],
            reference_events[j],
            gdelt_matrix[i],
            ref_matrix[j],
        )
        scored.append({
            "gdelt_idx": i,
            "ref_idx":   j,
            **components,
        })
    return scored
