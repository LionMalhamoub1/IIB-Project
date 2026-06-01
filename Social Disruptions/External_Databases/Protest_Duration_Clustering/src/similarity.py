from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import List, Set

import numpy as np
import pandas as pd

# ── Stop-words for issue token matching ──────────────────────────────────────

_STOPWORDS: Set[str] = {
    "the", "a", "an", "and", "or", "of", "in", "at", "on", "for", "to",
    "is", "are", "was", "were", "be", "been", "have", "has", "had", "by",
    "with", "against", "from", "this", "that", "their", "its", "over",
    "about", "during", "following", "after", "before", "into", "between",
    "against", "across", "due", "amid", "over",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenise(text: str) -> Set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """Weights and thresholds for the composite similarity score.

    Weights are normalised internally so they need not sum to 1.
    """
    w_location: float = 0.40
    w_actor:    float = 0.30
    w_issue:    float = 0.20
    w_time:     float = 0.10
    time_window_days: int = 14


# ── Sub-scores ────────────────────────────────────────────────────────────────

def _location_score(a: pd.Series, b: pd.Series) -> float:
    """Score 0–1 based on how precisely the locations match."""
    city_a = str(a.get("city", "")).strip().lower()
    city_b = str(b.get("city", "")).strip().lower()
    if city_a and city_b and city_a == city_b:
        return 1.0

    spec_a = str(a.get("specific_location", "")).strip().lower()
    spec_b = str(b.get("specific_location", "")).strip().lower()
    if spec_a and spec_b:
        if spec_a == spec_b or spec_a in spec_b or spec_b in spec_a:
            return 0.8

    reg_a = str(a.get("region_or_state", "")).strip().lower()
    reg_b = str(b.get("region_or_state", "")).strip().lower()
    if reg_a and reg_b and reg_a == reg_b:
        return 0.5

    return 0.0


def _actor_set(row: pd.Series) -> Set[str]:
    groups = row.get("protesting_groups") or []
    orgs   = row.get("organizations_or_companies") or []
    target = str(row.get("target_of_protest", "") or "").strip().lower()
    actors: Set[str] = set()
    if isinstance(groups, list):
        actors.update(g for g in groups if g)
    if isinstance(orgs, list):
        actors.update(o for o in orgs if o)
    if target:
        actors.add(target)
    return actors


def _actor_score(a: pd.Series, b: pd.Series) -> float:
    return _jaccard(_actor_set(a), _actor_set(b))


def _issue_score(a: pd.Series, b: pd.Series) -> float:
    ta = _tokenise(str(a.get("issue", "") or ""))
    tb = _tokenise(str(b.get("issue", "") or ""))
    base = _jaccard(ta, tb)

    # Boost if sector also matches
    sec_a = str(a.get("sector", "") or "").strip().lower()
    sec_b = str(b.get("sector", "") or "").strip().lower()
    if sec_a and sec_b and sec_a == sec_b:
        base = min(1.0, base + 0.15)
    return base


def _time_score(da: date, db: date, window_days: int) -> float:
    delta = abs((da - db).days)
    if delta >= window_days:
        return 0.0
    return 1.0 - (delta / window_days)


# ── Composite score ───────────────────────────────────────────────────────────

def compute_similarity(
    a: pd.Series,
    b: pd.Series,
    cfg: SimConfig,
) -> float:
    """Return a composite similarity score in [0, 1].

    The four sub-scores are computed and combined using ``cfg`` weights
    (normalised so they sum to 1).
    """
    da = a.get("published_date")
    db = b.get("published_date")

    total_w = cfg.w_location + cfg.w_actor + cfg.w_issue + cfg.w_time
    if total_w == 0:
        return 0.0

    loc = _location_score(a, b)
    act = _actor_score(a, b)
    iss = _issue_score(a, b)

    if isinstance(da, date) and isinstance(db, date):
        tim = _time_score(da, db, cfg.time_window_days)
    else:
        tim = 0.0

    score = (
        cfg.w_location * loc
        + cfg.w_actor   * act
        + cfg.w_issue   * iss
        + cfg.w_time    * tim
    ) / total_w

    return float(np.clip(score, 0.0, 1.0))


# ── Block distance matrix ─────────────────────────────────────────────────────

def _passes_candidate_filter(
    a: pd.Series,
    b: pd.Series,
    time_window_days: int,
) -> bool:
    """Stage-1 blocking filter for a pair of articles.

    Returns True only if:
    - date gap ≤ time_window_days, AND
    - at least one of: same city, shared org/company, shared protesting group.
    """
    da = a.get("published_date")
    db = b.get("published_date")
    if isinstance(da, date) and isinstance(db, date):
        if abs((da - db).days) > time_window_days:
            return False

    city_a = str(a.get("city", "") or "").strip().lower()
    city_b = str(b.get("city", "") or "").strip().lower()
    if city_a and city_b and city_a == city_b:
        return True

    orgs_a  = set(a.get("organizations_or_companies") or [])
    orgs_b  = set(b.get("organizations_or_companies") or [])
    if orgs_a & orgs_b:
        return True

    grps_a = set(a.get("protesting_groups") or [])
    grps_b = set(b.get("protesting_groups") or [])
    if grps_a & grps_b:
        return True

    return False


def compute_block_distance_matrix(
    block_df: pd.DataFrame,
    cfg: SimConfig,
) -> np.ndarray:
    """Compute an N×N pairwise distance matrix for articles in one block.

    Pairs that fail the Stage-1 candidate filter receive distance = 1.0.
    All other pairs receive distance = 1 - composite_similarity.

    Parameters
    ----------
    block_df:
        Rows belonging to a single blocking group (same country & type bucket).
    cfg:
        Similarity configuration (weights, time window).

    Returns
    -------
    Symmetric float64 ndarray of shape (N, N) with zeros on the diagonal.
    """
    n = len(block_df)
    rows = [block_df.iloc[i] for i in range(n)]
    dist = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(dist, 0.0)

    for i in range(n):
        for j in range(i + 1, n):
            if _passes_candidate_filter(rows[i], rows[j], cfg.time_window_days):
                sim = compute_similarity(rows[i], rows[j], cfg)
                d = 1.0 - sim
            else:
                d = 1.0
            dist[i, j] = d
            dist[j, i] = d

    return dist
