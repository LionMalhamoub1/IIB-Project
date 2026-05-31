"""
evaluation.py
==============
Measures the quality of the matching result and produces summary statistics
for use in the dissertation.

Design rationale
----------------
The dual-gate concept — evaluating matching in both directions — comes from
information retrieval, where precision and recall measure complementary aspects
of a retrieval system's performance.  Applied to flood event matching:

  Precision (forward: GDELT → reference)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Of all GDELT flood events, what fraction matched a reference event?
  High precision means GDELT is extracting real, verifiable floods.
  Low precision means GDELT contains noise — articles about non-flood topics
  mis-classified as floods, or geolocation errors placing events in the wrong
  country.

  Recall (inverse: reference → GDELT)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  Of all reference events, what fraction was captured by at least one GDELT
  article?  High recall means GDELT's coverage of real floods is comprehensive.
  Low recall reveals systematic gaps — floods in low-media-coverage regions
  (rural Africa, central Asia) that official databases record but the news
  never reports.  This is one of the dissertation's key findings.

Both directions are computed from the same match_index, so no additional
matching is needed.  The results are broken down by:
  - Year (to show temporal trends in GDELT coverage)
  - Country/region (to show geographic bias)
  - Source dataset (to show which official databases GDELT best represents)
  - Match score distribution (to show confidence levels)

These breakdowns directly support the dissertation analysis sections.
"""

from collections import defaultdict


def compute_precision_recall(
    gdelt_events: list[dict],
    reference_events: list[dict],
    match_index: dict,
) -> dict:
    """
    Compute overall precision and recall.

    Parameters
    ----------
    gdelt_events     : all GDELT events (matched and unmatched)
    reference_events : all reference events
    match_index      : gdelt_idx -> ref_idx mapping from matching.run_matching()

    Returns
    -------
    dict with precision, recall, and counts
    """
    n_gdelt = len(gdelt_events)
    n_ref   = len(reference_events)
    n_matched_gdelt = len(match_index)
    n_matched_ref   = len(set(match_index.values()))

    precision = n_matched_gdelt / n_gdelt if n_gdelt > 0 else 0.0
    recall    = n_matched_ref   / n_ref   if n_ref   > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "n_gdelt_total":    n_gdelt,
        "n_gdelt_matched":  n_matched_gdelt,
        "n_gdelt_unmatched": n_gdelt - n_matched_gdelt,
        "n_ref_total":      n_ref,
        "n_ref_matched":    n_matched_ref,
        "n_ref_unmatched":  n_ref - n_matched_ref,
        "precision":        round(precision, 4),
        "recall":           round(recall, 4),
        "f1":               round(f1, 4),
    }


def breakdown_by_year(
    gdelt_events: list[dict],
    reference_events: list[dict],
    match_index: dict,
) -> dict:
    """Precision and recall broken down by event year."""
    matched_ref_indices = set(match_index.values())

    # GDELT precision by year
    gdelt_by_year = defaultdict(lambda: {"total": 0, "matched": 0})
    for i, e in enumerate(gdelt_events):
        year = str(e.get("event_date") or e.get("date_start") or "")[:4]
        gdelt_by_year[year]["total"] += 1
        if i in match_index:
            gdelt_by_year[year]["matched"] += 1

    # Reference recall by year
    ref_by_year = defaultdict(lambda: {"total": 0, "matched": 0})
    for j, e in enumerate(reference_events):
        year = str(e.get("date_start") or "")[:4]
        ref_by_year[year]["total"] += 1
        if j in matched_ref_indices:
            ref_by_year[year]["matched"] += 1

    result = {}
    for year in sorted(set(list(gdelt_by_year) + list(ref_by_year))):
        g = gdelt_by_year.get(year, {"total": 0, "matched": 0})
        r = ref_by_year.get(year, {"total": 0, "matched": 0})
        result[year] = {
            "gdelt_precision": round(g["matched"] / g["total"], 4) if g["total"] else 0,
            "ref_recall":      round(r["matched"] / r["total"], 4) if r["total"] else 0,
            "gdelt_total":     g["total"],
            "ref_total":       r["total"],
        }
    return result


def breakdown_by_source(
    reference_events: list[dict],
    match_index: dict,
) -> dict:
    """Recall broken down by reference source dataset (DFO, GDACS, EM-DAT, etc.)."""
    matched_ref_indices = set(match_index.values())
    by_source = defaultdict(lambda: {"total": 0, "matched": 0})

    for j, e in enumerate(reference_events):
        src = e.get("source", "unknown")
        by_source[src]["total"] += 1
        if j in matched_ref_indices:
            by_source[src]["matched"] += 1

    return {
        src: {
            "recall": round(v["matched"] / v["total"], 4) if v["total"] else 0,
            "total":   v["total"],
            "matched": v["matched"],
        }
        for src, v in sorted(by_source.items())
    }


def score_distribution(
    scored_pairs: list[dict],
    match_index: dict,
    bins: int = 10,
) -> dict:
    """
    Histogram of match scores split into matched vs. all candidate pairs.
    Useful for visualising the score gap between true and false matches.
    """
    all_scores     = [p["score"] for p in scored_pairs]
    matched_scores = [
        scored_pairs[k]["score"]
        for k, pair in enumerate(scored_pairs)
        if pair.get("gdelt_idx") in match_index
        and match_index[pair["gdelt_idx"]] == pair.get("ref_idx")
    ]

    def histogram(scores):
        width = 1.0 / bins
        counts = [0] * bins
        for s in scores:
            idx = min(int(s / width), bins - 1)
            counts[idx] += 1
        return {
            f"{i*width:.1f}-{(i+1)*width:.1f}": counts[i]
            for i in range(bins)
        }

    return {
        "all_candidates": histogram(all_scores),
        "matched_pairs":  histogram(matched_scores),
        "n_candidates":   len(all_scores),
        "n_matched":      len(matched_scores),
    }
