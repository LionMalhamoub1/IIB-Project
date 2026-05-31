"""
feature_vectors.py
===================
Builds normalised hydro-climate feature vectors from enriched event records
for use in similarity scoring during matching.

Design rationale
----------------
The core insight behind feature-vector matching is that two records describing
the same physical flood will have been exposed to the same objective
hydro-climatic conditions, regardless of which database or article sourced
them.  A flood in Bangladesh in August 2017 will show the same SPI anomaly,
the same GloFAS discharge percentile, and the same CHIRPS rainfall whether it
appears in EM-DAT, GDACS, or a GDELT news article.

Feature selection
-----------------
Seven features are used, chosen to be complementary and capture different
aspects of flood character:

  spi_30d                  — Standardised Precipitation Index (z-score).
                             Measures rainfall anomaly relative to a 40-year
                             climatology.  Distinguishes genuine extreme-rainfall
                             events from routine wet-season flooding.

  chirps_7d_anom_pct       — 7-day rainfall as a % anomaly above the 1991–2020
                             baseline.  Captures synoptic-scale rainfall events
                             that SPI alone might miss at the 30-day window.

  gpm_peak_3h_mm           — Peak 3-hour GPM rainfall intensity.  Distinguishes
                             flash floods (short intense bursts) from slow-onset
                             riverine floods with similar totals.

  era5_soil_moisture_day0  — Volumetric soil water content on the event day.
                             High antecedent soil moisture amplifies flood
                             response; this provides a pre-event saturation signal.

  jrc_recurrence_pct       — JRC surface water recurrence (% of wet years with
                             water present).  Encodes the structural flood
                             propensity of the location — a permanent floodplain
                             vs. an unusual inundation.

  pop_density_km2          — Population density within 25 km.  While not a
                             hydro feature, it strongly constrains whether a
                             GDELT article would even exist — only events near
                             people get reported.  Including it helps match
                             urban vs. rural events at similar coordinates.

Missing value handling
----------------------
Not all events have all features (e.g. CHIRPS has no data above 50°N; GloFAS
may not have processed all events yet).  Missing values are imputed with the
median of all available values for that feature across the full dataset.
Cosine similarity is then used rather than Euclidean distance, making the
result insensitive to the absolute scale of each feature.
"""

import math
from typing import Optional
import numpy as np


HYDRO_FEATURES = [
    "spi_30d",
    # glofas_discharge_pct_rank removed: not present in either GDELT or reference enrichment
    "chirps_7d_anom_pct",
    "gpm_peak_3h_mm",
    "era5_soil_moisture_day0",
    "jrc_recurrence_pct",
    "pop_density_km2",
]


def compute_medians(events: list[dict]) -> dict[str, float]:
    """
    Compute per-feature medians from a list of event dicts.
    Used to impute missing values before building feature vectors.
    """
    medians = {}
    for feat in HYDRO_FEATURES:
        vals = [
            float(e[feat]) for e in events
            if e.get(feat) is not None and not math.isnan(float(e[feat]))
        ]
        medians[feat] = float(np.median(vals)) if vals else 0.0
    return medians


def build_feature_vector(event: dict, medians: dict[str, float]) -> np.ndarray:
    """
    Build a normalised feature vector for a single event.
    Missing values are replaced with dataset-level medians.
    Returns a 1-D numpy array of length len(HYDRO_FEATURES).
    """
    vec = []
    for feat in HYDRO_FEATURES:
        val = event.get(feat)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            val = medians.get(feat, 0.0)
        vec.append(float(val))
    return np.array(vec, dtype=np.float64)


def hydro_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine similarity between two feature vectors, in [0, 1].

    Cosine similarity is used rather than Euclidean distance because:
      1. Features have different scales (SPI in [-3,3], pop_density in [0,50000+])
         and normalising each feature individually would introduce fragility.
      2. Cosine similarity is naturally bounded to [-1, 1] and, after
         median imputation, will be non-negative for realistic event pairs.
      3. It captures the *shape* of the hydro fingerprint — the relative
         pattern of anomalies — which is more informative than absolute values
         when comparing across different climate regimes.

    Returns 0.0 if either vector is the zero vector.
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    raw = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    # Clamp to [0, 1] — negative cosine similarity means anti-correlated
    # hydro fingerprints, which is strong evidence against a match
    return max(0.0, raw)


def build_feature_index(
    events: list[dict],
    medians: Optional[dict[str, float]] = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Build a (N, F) matrix of feature vectors for a list of events, and
    return the medians used for imputation.

    Parameters
    ----------
    events  : list of enriched event dicts
    medians : pre-computed medians (optional — computed from events if None)

    Returns
    -------
    matrix  : np.ndarray of shape (len(events), len(HYDRO_FEATURES))
    medians : dict used for imputation
    """
    if medians is None:
        medians = compute_medians(events)
    matrix = np.stack([build_feature_vector(e, medians) for e in events])
    return matrix, medians
