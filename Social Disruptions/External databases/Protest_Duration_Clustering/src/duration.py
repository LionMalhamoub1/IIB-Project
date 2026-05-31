from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import pandas as pd

from parsing import (
    parse_explicit_duration,
    parse_event_start_reference,
    parse_reported_day_number,
)

logger = logging.getLogger(__name__)

# ── Duration result ────────────────────────────────────────────────────────────

@dataclass
class DurationResult:
    cluster_id:                  int
    country:                     str
    city:                        str
    protest_type:                str
    issue:                       str
    top_actors:                  str
    cluster_start_date:          Optional[date]
    cluster_end_date:            Optional[date]
    lower_bound_duration_days:   int
    parsed_start_date:           Optional[date]
    parsed_explicit_duration_days: Optional[int]
    parsed_reported_day_number:  Optional[int]
    estimated_duration_days:     int
    duration_source:             str
    duration_confidence:         str
    n_articles:                  int
    n_unique_sources:            int
    sample_urls:                 str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _modal(values: pd.Series, top_n: int = 1) -> str:
    counts = Counter(v for v in values if v and str(v).strip())
    return ", ".join(k for k, _ in counts.most_common(top_n))


def _top_actors(rows: pd.DataFrame) -> str:
    actors: List[str] = []
    for _, row in rows.iterrows():
        actors.extend(row.get("protesting_groups") or [])
        actors.extend(row.get("organizations_or_companies") or [])
        t = str(row.get("target_of_protest", "") or "").strip()
        if t:
            actors.append(t)
    counts = Counter(actors)
    return ", ".join(k for k, _ in counts.most_common(5))


def _url_domain(url: str) -> str:
    m = re.search(r"https?://(?:www\.)?([^/]+)", str(url))
    return m.group(1) if m else str(url)[:40]


def _sample_urls(rows: pd.DataFrame, n: int = 3) -> str:
    urls = [str(u) for u in rows.get("url", pd.Series(dtype=str)).dropna().unique()[:n]]
    return " | ".join(urls)


# ── Per-cluster duration estimation ──────────────────────────────────────────

def estimate_cluster_duration(
    cluster_rows: pd.DataFrame,
    resolve_relative: bool = False,
) -> DurationResult:
    """Estimate duration and summarise one cluster.

    Parameters
    ----------
    cluster_rows:
        All rows belonging to a single cluster_id.
    resolve_relative:
        Passed through to :func:`~parsing.parse_event_start_reference`.

    Returns
    -------
    A :class:`DurationResult` with all duration metrics and confidence flags.
    """
    cid = int(cluster_rows["cluster_id"].iloc[0])

    dates = pd.to_datetime(
        cluster_rows["published_date"].map(
            lambda d: d.isoformat() if isinstance(d, date) else None
        ),
        errors="coerce",
    ).dt.date.dropna()

    cluster_start = dates.min() if not dates.empty else None
    cluster_end   = dates.max() if not dates.empty else None

    if cluster_start and cluster_end:
        lower_bound = (cluster_end - cluster_start).days + 1
    else:
        lower_bound = 1

    # Aggregate parsed duration signals across all articles in the cluster
    best_explicit:    Optional[int]  = None
    best_reported:    Optional[int]  = None
    best_start_date:  Optional[date] = None

    for _, row in cluster_rows.iterrows():
        exp = parse_explicit_duration(str(row.get("explicit_duration", "") or ""))
        if exp and (best_explicit is None or exp > best_explicit):
            best_explicit = exp

        rep = parse_reported_day_number(str(row.get("reported_day_number", "") or ""))
        if rep and (best_reported is None or rep > best_reported):
            best_reported = rep

        pub = row.get("published_date")
        pub_date = pub if isinstance(pub, date) else None
        start = parse_event_start_reference(
            str(row.get("event_start_reference", "") or ""),
            pub_date=pub_date,
            resolve_relative=resolve_relative,
        )
        if start and (best_start_date is None or start < best_start_date):
            best_start_date = start

    # Duration from resolved start date + cluster end date
    ref_duration: Optional[int] = None
    if best_start_date and cluster_end:
        delta = (cluster_end - best_start_date).days + 1
        if delta > 0:
            ref_duration = delta

    # Build candidate durations and pick the winner.
    # Tiebreak by source quality so explicit_duration beats date_range_only
    # when both return the same number.
    _PRIORITY = {
        "explicit_duration":    4,
        "reported_day_number":  3,
        "event_start_reference":2,
        "date_range_only":      1,
    }
    candidates = {"date_range_only": lower_bound}
    if best_explicit:
        candidates["explicit_duration"] = best_explicit
    if best_reported:
        candidates["reported_day_number"] = best_reported
    if ref_duration:
        candidates["event_start_reference"] = ref_duration

    best_source = max(candidates, key=lambda k: (candidates[k], _PRIORITY[k]))
    estimated   = candidates[best_source]

    # Confidence
    if best_source == "explicit_duration":
        confidence = "high"
    elif best_source in ("reported_day_number", "event_start_reference"):
        confidence = "medium"
    else:
        # date_range_only: medium if well-covered, low otherwise
        n = len(cluster_rows)
        confidence = "medium" if (n >= 3 and lower_bound > 1) else "low"

    # Aggregate metadata
    country     = _modal(cluster_rows.get("country", pd.Series(dtype=str)), top_n=1)
    city        = _modal(cluster_rows.get("city",    pd.Series(dtype=str)), top_n=1)
    ptype       = _modal(cluster_rows.get("protest_type", pd.Series(dtype=str)), top_n=1)
    issue       = _modal(cluster_rows.get("issue",  pd.Series(dtype=str)), top_n=1)
    actors      = _top_actors(cluster_rows)

    url_col = cluster_rows.get("url") if "url" in cluster_rows.columns else pd.Series(dtype=str)
    n_unique_sources = url_col.dropna().map(_url_domain).nunique()
    sample          = _sample_urls(cluster_rows)

    return DurationResult(
        cluster_id                   = cid,
        country                      = country,
        city                         = city,
        protest_type                 = ptype,
        issue                        = issue,
        top_actors                   = actors,
        cluster_start_date           = cluster_start,
        cluster_end_date             = cluster_end,
        lower_bound_duration_days    = lower_bound,
        parsed_start_date            = best_start_date,
        parsed_explicit_duration_days= best_explicit,
        parsed_reported_day_number   = best_reported,
        estimated_duration_days      = estimated,
        duration_source              = best_source,
        duration_confidence          = confidence,
        n_articles                   = len(cluster_rows),
        n_unique_sources             = n_unique_sources,
        sample_urls                  = sample,
    )


# ── Build the full summaries table ────────────────────────────────────────────

def build_cluster_summaries(
    articles_df: pd.DataFrame,
    resolve_relative: bool = False,
) -> pd.DataFrame:
    """Build the cluster-summary table from the clustered articles DataFrame.

    Parameters
    ----------
    articles_df:
        Output of :func:`~clustering.run_clustering`, which must contain a
        ``cluster_id`` column alongside the expanded event fields.
    resolve_relative:
        Passed through to :func:`estimate_cluster_duration`.

    Returns
    -------
    One row per cluster with duration estimates and metadata.
    """
    rows: List[DurationResult] = []
    for cid, group in articles_df.groupby("cluster_id"):
        result = estimate_cluster_duration(group, resolve_relative=resolve_relative)
        rows.append(result)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame([r.__dict__ for r in rows])
    summary = summary.sort_values("cluster_id").reset_index(drop=True)

    logger.info(
        "Cluster summaries built: %d clusters, median estimated duration = %.0f days.",
        len(summary),
        summary["estimated_duration_days"].median(),
    )
    return summary
