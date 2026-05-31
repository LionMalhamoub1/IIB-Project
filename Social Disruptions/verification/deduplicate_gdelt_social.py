"""
deduplicate_gdelt_social.py
===========================
Collapses multiple GDELT records describing the same real-world protest or
labour strike into a single canonical event record.

The problem
-----------
Each GDELT record corresponds to one news article.  A single real-world event
will generate many articles over its duration, each extracted as a separate
record.  Matching those raw records against ACLED inflates apparent coverage
and makes precision/recall metrics meaningless.

Approach
--------
1. Build a weighted similarity graph over GDELT events.
   Nodes  = individual GDELT records.
   Edges  = pairs of records likely describing the same event.

2. Threshold edges at EDGE_THRESHOLD and find connected components.
   Each component is treated as one real-world event.

3. Aggregate each component into a canonical event record.

Note on connected components and chaining
-----------------------------------------
Connected components can over-merge: if A~B and B~C both score above
threshold, A and C are merged even if score(A,C) < threshold.  To guard
against this, any cluster whose internal mean pairwise score falls below
CHAIN_WARN_THRESHOLD triggers a warning in the output.  If this is common,
lower EDGE_THRESHOLD or switch to a stricter linkage method.

Similarity scoring with sparse data
-------------------------------------
The score is the weighted average of ONLY the features available in BOTH
records, renormalised so that missing features don't penalise the score.
A pair with only temporal + location data is on the same 0–1 scale as a
pair with all five features.

Hard gates (score = 0, no edge):
  • Different iso3, or iso3 unresolved in either record
  • Different disruption_type
  • Date gap > TYPE_WINDOW[disruption_type]

Soft features (included only when present in both):
  temporal     0.35  linear decay over window — always present
  location     0.30  fuzzy city/region match
  sector       0.15  exact match
  issue        0.10  fuzzy token match
  protest_type 0.10  exact match

reported_day_number bonus (+0.10, capped at 1.0)
  When both records carry reported_day_number and those numbers are
  consistent with the date gap (|Δdays - Δday_num| ≤ 1), a bonus is added.

Aggregation
-----------
  event_date           earliest date in cluster
  event_end_date       max(latest article date,
                           start + max(reported_day_number) - 1)
  duration_days        event_end_date - event_date + 1
  confidence           max
  n_source_articles    count
  num_articles         sum
  estimated_participants  max
  reported_day_number_max max
  sector / issue / protest_type / protesting_groups /
  organizations / target_of_protest
                       coalesce: unanimous → use value;
                                 conflict  → most common + flag
  subloc               most common non-empty value
  conflicted_fields    list of fields where articles disagreed
  mean_internal_score  mean pairwise similarity within cluster
                       (low values flag potential over-merging)
  source_articles      full provenance list

Outputs
-------
  verification/gdelt_social_deduped.jsonl
  verification/gdelt_social_deduped.parquet   ← flat table for modelling
  verification/dedup_report.json
  verification/figures/
      fig_dedup_cluster_sizes.png
      fig_dedup_events_per_month.png
      fig_dedup_edge_weights.png

Usage
-----
  python "Social Disruptions/verification/deduplicate_gdelt_social.py"
  python "Social Disruptions/verification/deduplicate_gdelt_social.py" \\
      --gdelt path/to/all_consolidated.jsonl \\
      --out   path/to/output_dir \\
      --threshold 0.55
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from _utils import extract_iso3, extract_subloc, fuzzy_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SD   = _HERE.parent
_ROOT = _SD.parent

DEFAULT_GDELT_PATH = _ROOT / "Builder_GDELT" / "results" / "combined" / "all_consolidated.jsonl"
DEFAULT_OUT_DIR    = _HERE

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SOCIAL_TYPES:         frozenset[str] = frozenset({"protests", "labour_strike"})
CONFIDENCE_THRESHOLD: float          = 0.6
EDGE_THRESHOLD:       float          = 0.50
DAY_NUMBER_BONUS:     float          = 0.10
CHAIN_WARN_THRESHOLD: float          = 0.40   # warn if mean internal score < this

TYPE_WINDOW: dict[str, int] = {
    "protests":      3,
    "labour_strike": 21,
}

BASE_WEIGHTS: dict[str, float] = {
    "temporal":     0.35,
    "location":     0.30,
    "sector":       0.15,
    "issue":        0.10,
    "protest_type": 0.10,
}

# ---------------------------------------------------------------------------
# Event loading
# ---------------------------------------------------------------------------

def load_events(path: Path) -> list[dict]:
    """Load all qualifying GDELT social events from a JSONL file."""
    log.info("Loading GDELT events: %s", path)
    records = []
    skipped: dict[str, int] = defaultdict(int)

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                skipped["parse"] += 1
                continue
            if e.get("disruption_type") not in SOCIAL_TYPES:
                skipped["type"] += 1
                continue
            if (e.get("confidence") or 0.0) < CONFIDENCE_THRESHOLD:
                skipped["conf"] += 1
                continue

            raw_date = e.get("event_date") or e.get("publish_date")
            try:
                date = pd.Timestamp(raw_date).normalize()
            except Exception:
                skipped["date"] += 1
                continue

            details = e.get("details") or e.get("extras") or {}
            records.append({
                "idx":                    len(records),
                "event_date":             date,
                "disruption_type":        e["disruption_type"],
                "iso3":                   extract_iso3(e),
                "subloc":                 extract_subloc(e),
                "confidence":             float(e.get("confidence") or 0.0),
                "num_articles":           int(e.get("num_articles") or 1),
                "location_raw":           json.dumps(
                                              e.get("location")
                                              or e.get("location_name") or ""
                                          ),
                "sector":                 details.get("sector") or None,
                "issue":                  details.get("issue") or None,
                "protest_type":           details.get("protest_type") or None,
                "protesting_groups":      details.get("protesting_groups") or None,
                "organizations":          details.get("organizations_or_companies") or None,
                "target_of_protest":      details.get("target_of_protest") or None,
                "estimated_participants": details.get("estimated_participants") or None,
                "event_start_day":        details.get("event_start_day") or None,
                "reported_day_number":    details.get("reported_day_number") or None,
            })

    log.info(
        "Loaded %d events | skipped: type=%d conf=%d date=%d parse=%d",
        len(records),
        skipped["type"], skipped["conf"], skipped["date"], skipped["parse"],
    )
    return records


# ---------------------------------------------------------------------------
# Pairwise similarity
# ---------------------------------------------------------------------------

def _similarity(a: dict, b: dict) -> float:
    """
    Compute similarity between two GDELT event records.
    Returns 0.0 if hard gates fail; otherwise a score in (0, 1].
    Score is normalised over the features available in BOTH records.
    """
    # Hard gates
    if a["iso3"] is None or b["iso3"] is None or a["iso3"] != b["iso3"]:
        return 0.0
    if a["disruption_type"] != b["disruption_type"]:
        return 0.0

    max_days = TYPE_WINDOW[a["disruption_type"]]
    days_gap = abs((a["event_date"] - b["event_date"]).days)
    if days_gap > max_days:
        return 0.0

    scores:  dict[str, float] = {}
    weights: dict[str, float] = {}

    # Temporal — always present
    scores["temporal"]  = 1.0 - days_gap / max_days
    weights["temporal"] = BASE_WEIGHTS["temporal"]

    # Location — include if at least one record has subloc
    sa, sb = a["subloc"], b["subloc"]
    if sa or sb:
        scores["location"]  = fuzzy_similarity(sa, sb) if (sa and sb) else 0.35
        weights["location"] = BASE_WEIGHTS["location"]

    # Sector — only if both present
    if a["sector"] and b["sector"]:
        scores["sector"]  = 1.0 if a["sector"].lower() == b["sector"].lower() else 0.0
        weights["sector"] = BASE_WEIGHTS["sector"]

    # Issue — fuzzy, only if both present
    if a["issue"] and b["issue"]:
        scores["issue"]  = fuzzy_similarity(a["issue"].lower(), b["issue"].lower())
        weights["issue"] = BASE_WEIGHTS["issue"]

    # Protest type — only if both present
    if a["protest_type"] and b["protest_type"]:
        scores["protest_type"] = (
            1.0 if a["protest_type"].lower() == b["protest_type"].lower() else 0.2
        )
        weights["protest_type"] = BASE_WEIGHTS["protest_type"]

    total_weight = sum(weights.values())
    score = sum(scores[k] * weights[k] for k in scores) / total_weight

    # reported_day_number consistency bonus
    da = a.get("reported_day_number")
    db = b.get("reported_day_number")
    if da is not None and db is not None:
        try:
            if abs(abs(float(da) - float(db)) - days_gap) <= 1:
                score = min(1.0, score + DAY_NUMBER_BONUS)
        except (TypeError, ValueError):
            pass

    return round(score, 4)


# ---------------------------------------------------------------------------
# Graph construction and clustering
# ---------------------------------------------------------------------------

def build_similarity_graph(events: list[dict], threshold: float) -> "nx.Graph":
    """
    Build the GDELT self-similarity graph.

    Pairs are only considered within the same (iso3, disruption_type) bucket
    and within the TYPE_WINDOW date range.  Events are sorted by date so the
    inner loop can break early once the window is exceeded, giving roughly
    O(N × W) comparisons rather than O(N²) where W is the window size.
    """
    import networkx as nx

    log.info("Building similarity graph (%d events, threshold=%.2f)...",
             len(events), threshold)

    G = nx.Graph()
    for e in events:
        G.add_node(e["idx"], **{k: v for k, v in e.items() if k != "idx"})

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for e in events:
        if e["iso3"] is not None:
            buckets[(e["iso3"], e["disruption_type"])].append(e)

    n_edges = 0
    for (_, dtype), bucket in buckets.items():
        bucket_sorted = sorted(bucket, key=lambda x: x["event_date"])
        window = TYPE_WINDOW[dtype]
        for i, ea in enumerate(bucket_sorted):
            for eb in bucket_sorted[i + 1:]:
                if (eb["event_date"] - ea["event_date"]).days > window:
                    break
                sim = _similarity(ea, eb)
                if sim >= threshold:
                    G.add_edge(ea["idx"], eb["idx"], weight=sim)
                    n_edges += 1

    log.info("Graph: %d nodes | %d edges above threshold %.2f",
             G.number_of_nodes(), n_edges, threshold)
    return G


def get_clusters(G: "nx.Graph") -> list[list[int]]:
    """
    Return connected components as sorted lists of node IDs, largest first.
    """
    import networkx as nx
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    n_multi = sum(1 for c in components if len(c) > 1)
    log.info("Clusters: %d total | %d multi-article | %d singletons",
             len(components), n_multi, len(components) - n_multi)
    return [sorted(c) for c in components]


def _mean_internal_score(G: "nx.Graph", node_ids: list[int]) -> float:
    """Mean edge weight among all edges within a cluster."""
    if len(node_ids) < 2:
        return 1.0
    edges = [
        G[u][v]["weight"]
        for u in node_ids for v in node_ids
        if u < v and G.has_edge(u, v)
    ]
    return float(np.mean(edges)) if edges else 0.0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _coalesce_string(
    values: list[Optional[str]],
) -> tuple[Optional[str], bool]:
    """
    Merge string values from cluster members.
    Returns (value, conflicted).
    If all non-null values agree → value, no conflict.
    If they disagree → most common value, mark conflicted.
    If all null → None, no conflict.
    """
    non_null = [v.strip().lower() for v in values if v and str(v).strip()]
    if not non_null:
        return None, False
    if len(set(non_null)) == 1:
        return non_null[0], False
    return Counter(non_null).most_common(1)[0][0], True


def _most_common_nonempty(values: list[str]) -> str:
    non_empty = [v for v in values if v and v.strip()]
    if not non_empty:
        return ""
    return Counter(non_empty).most_common(1)[0][0]


def aggregate_cluster(
    cluster_events: list[dict],
    cluster_id: str,
    mean_internal_score: float,
) -> dict:
    """Merge a cluster of GDELT event dicts into one canonical record."""
    dates      = [e["event_date"] for e in cluster_events]
    start_date = min(dates)
    latest_date = max(dates)

    day_numbers = [
        float(e["reported_day_number"])
        for e in cluster_events
        if e.get("reported_day_number") is not None
    ]
    max_day_number: Optional[float] = max(day_numbers) if day_numbers else None

    if max_day_number is not None:
        estimated_end = max(
            latest_date,
            start_date + pd.Timedelta(days=int(max_day_number) - 1),
        )
    else:
        estimated_end = latest_date

    conflicted: list[str] = []

    def _coalesce(field: str) -> Optional[str]:
        val, conflict = _coalesce_string([e.get(field) for e in cluster_events])
        if conflict:
            conflicted.append(field)
        return val

    participants = [
        float(e["estimated_participants"])
        for e in cluster_events
        if e.get("estimated_participants") is not None
    ]

    return {
        "cluster_id":              cluster_id,
        "n_source_articles":       len(cluster_events),
        "mean_internal_score":     round(mean_internal_score, 4),
        "event_date":              start_date.strftime("%Y-%m-%d"),
        "event_end_date":          estimated_end.strftime("%Y-%m-%d"),
        "duration_days":           int((estimated_end - start_date).days + 1),
        "disruption_type":         cluster_events[0]["disruption_type"],
        "iso3":                    cluster_events[0]["iso3"],
        "subloc":                  _most_common_nonempty(
                                       [e["subloc"] for e in cluster_events]
                                   ),
        "confidence":              max(e["confidence"] for e in cluster_events),
        "num_articles":            sum(e["num_articles"] for e in cluster_events),
        "sector":                  _coalesce("sector"),
        "issue":                   _coalesce("issue"),
        "protest_type":            _coalesce("protest_type"),
        "protesting_groups":       _coalesce("protesting_groups"),
        "organizations":           _coalesce("organizations"),
        "target_of_protest":       _coalesce("target_of_protest"),
        "estimated_participants":  int(max(participants)) if participants else None,
        "reported_day_number_max": int(max_day_number) if max_day_number is not None else None,
        "conflicted_fields":       conflicted,
        "source_articles": [
            {
                "event_date":            e["event_date"].strftime("%Y-%m-%d"),
                "confidence":            e["confidence"],
                "num_articles":          e["num_articles"],
                "subloc":                e["subloc"],
                "sector":                e.get("sector"),
                "issue":                 e.get("issue"),
                "protest_type":          e.get("protest_type"),
                "estimated_participants": e.get("estimated_participants"),
                "reported_day_number":   e.get("reported_day_number"),
                "location_raw":          e.get("location_raw"),
            }
            for e in sorted(cluster_events, key=lambda x: x["event_date"])
        ],
    }


# ---------------------------------------------------------------------------
# Flat parquet output for likelihood modelling
# ---------------------------------------------------------------------------

def to_modelling_dataframe(deduped: list[dict]) -> pd.DataFrame:
    """
    Convert the deduplicated event list into a flat DataFrame suitable for
    likelihood modelling.

    source_articles and conflicted_fields are dropped (too nested for tabular
    use); everything else is kept as columns.  Dates are proper datetime64.
    """
    flat = []
    for e in deduped:
        row = {k: v for k, v in e.items()
               if k not in ("source_articles", "conflicted_fields")}
        row["has_conflict"] = bool(e.get("conflicted_fields"))
        flat.append(row)

    df = pd.DataFrame(flat)
    df["event_date"]     = pd.to_datetime(df["event_date"])
    df["event_end_date"] = pd.to_datetime(df["event_end_date"])
    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Saved: %s", path)


def fig_cluster_sizes(clusters: list[list[int]], out: Path) -> None:
    import matplotlib.pyplot as plt
    sizes     = [len(c) for c in clusters]
    max_size  = max(sizes)
    bins      = range(1, min(max_size + 2, 20))
    singleton_pct = 100 * sizes.count(1) / len(sizes)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sizes, bins=bins, align="left", color="#457B9D",
            edgecolor="white", rwidth=0.8)
    ax.set_xlabel("Articles per cluster")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Distribution of cluster sizes  (each cluster = one real-world event)")
    ax.set_xticks(list(bins))
    ax.text(0.97, 0.95, f"Singletons: {singleton_pct:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color="grey")
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_events_per_month(
    raw_events: list[dict],
    deduped: list[dict],
    out: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    raw_dates   = pd.to_datetime([e["event_date"] for e in raw_events]).to_period("M")
    dedup_dates = pd.to_datetime([e["event_date"] for e in deduped]).to_period("M")

    raw_counts   = raw_dates.value_counts().sort_index()
    dedup_counts = dedup_dates.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(raw_counts.index.to_timestamp(),   raw_counts.values,
            label="Raw GDELT records",     color="#E63946", lw=1.5, alpha=0.8)
    ax.plot(dedup_counts.index.to_timestamp(), dedup_counts.values,
            label="Deduplicated events",   color="#457B9D", lw=1.8)
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.set_title("Raw GDELT records vs deduplicated events per month")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend()
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_edge_weight_distribution(
    G: "nx.Graph",
    threshold: float,   # passed explicitly — do not read module constant
    out: Path,
) -> None:
    import matplotlib.pyplot as plt

    weights = [d["weight"] for _, _, d in G.edges(data=True)]
    if not weights:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(weights, bins=20, color="#2A9D8F", edgecolor="white", alpha=0.85)
    ax.axvline(threshold, color="#E63946", lw=1.5, linestyle="--",
               label=f"Edge threshold ({threshold})")
    ax.set_xlabel("Pairwise similarity score")
    ax.set_ylabel("Number of edges")
    ax.set_title("Distribution of inter-article similarity scores")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    gdelt_path: Path  = DEFAULT_GDELT_PATH,
    out_dir:    Path  = DEFAULT_OUT_DIR,
    threshold:  float = EDGE_THRESHOLD,
) -> list[dict]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not gdelt_path.exists():
        log.error("GDELT file not found: %s", gdelt_path)
        sys.exit(1)

    events = load_events(gdelt_path)
    if not events:
        log.error("No qualifying GDELT social events found.")
        sys.exit(1)

    G        = build_similarity_graph(events, threshold)
    clusters = get_clusters(G)

    event_by_idx = {e["idx"]: e for e in events}
    deduped: list[dict] = []
    n_chain_warnings = 0

    for ci, node_ids in enumerate(clusters):
        cluster_events = [event_by_idx[n] for n in node_ids]
        mean_score     = _mean_internal_score(G, node_ids)

        if len(node_ids) > 2 and mean_score < CHAIN_WARN_THRESHOLD:
            n_chain_warnings += 1

        iso3  = cluster_events[0]["iso3"] or "UNK"
        dtype = cluster_events[0]["disruption_type"]
        date  = min(e["event_date"] for e in cluster_events).strftime("%Y%m%d")
        cluster_id = f"{iso3}_{dtype}_{date}_c{ci:04d}"

        deduped.append(aggregate_cluster(cluster_events, cluster_id, mean_score))

    if n_chain_warnings:
        log.warning(
            "%d clusters have mean_internal_score < %.2f — possible over-merging "
            "due to transitivity.  Consider lowering --threshold or reviewing "
            "clusters with low mean_internal_score in the output.",
            n_chain_warnings, CHAIN_WARN_THRESHOLD,
        )

    log.info(
        "Deduplication: %d raw → %d canonical events  (%.1f%% reduction)",
        len(events), len(deduped),
        100 * (1 - len(deduped) / len(events)),
    )

    # Conflict summary
    n_conflicted = sum(1 for e in deduped if e["conflicted_fields"])
    if n_conflicted:
        field_counts = Counter(f for e in deduped for f in e["conflicted_fields"])
        log.info("%d events have conflicting field values: %s",
                 n_conflicted, dict(field_counts.most_common()))

    # Write JSONL
    out_jsonl = out_dir / "gdelt_social_deduped.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for record in deduped:
            f.write(json.dumps(record, default=str) + "\n")
    log.info("JSONL written: %s", out_jsonl)

    # Write flat parquet for likelihood modelling
    df_model = to_modelling_dataframe(deduped)
    out_parquet = out_dir / "gdelt_social_deduped.parquet"
    df_model.to_parquet(out_parquet, index=False)
    log.info("Parquet written: %s", out_parquet)

    # Dedup report
    report = {
        "parameters": {
            "edge_threshold":       threshold,
            "chain_warn_threshold": CHAIN_WARN_THRESHOLD,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "type_windows":         TYPE_WINDOW,
            "day_number_bonus":     DAY_NUMBER_BONUS,
        },
        "raw_records":       len(events),
        "canonical_events":  len(deduped),
        "reduction_pct":     round(100 * (1 - len(deduped) / len(events)), 1),
        "chain_warnings":    n_chain_warnings,
        "conflicted_events": n_conflicted,
        "cluster_size_distribution": {
            str(s): sum(1 for c in clusters if len(c) == s)
            for s in sorted({len(c) for c in clusters})
        },
        "type_breakdown": {
            dtype: {
                "raw":    sum(1 for e in events  if e["disruption_type"] == dtype),
                "deduped": sum(1 for e in deduped if e["disruption_type"] == dtype),
            }
            for dtype in SOCIAL_TYPES
        },
    }
    with (out_dir / "dedup_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    fig_cluster_sizes(clusters, out=fig_dir / "fig_dedup_cluster_sizes.png")
    fig_events_per_month(events, deduped, out=fig_dir / "fig_dedup_events_per_month.png")
    fig_edge_weight_distribution(G, threshold=threshold,
                                 out=fig_dir / "fig_dedup_edge_weights.png")
    log.info("Done.")
    return deduped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deduplicate GDELT social events using graph clustering"
    )
    parser.add_argument("--gdelt",     type=Path,  default=DEFAULT_GDELT_PATH)
    parser.add_argument("--out",       type=Path,  default=DEFAULT_OUT_DIR)
    parser.add_argument("--threshold", type=float, default=EDGE_THRESHOLD,
                        help=f"Min similarity for an edge (default {EDGE_THRESHOLD})")
    args = parser.parse_args()
    main(gdelt_path=args.gdelt, out_dir=args.out, threshold=args.threshold)
