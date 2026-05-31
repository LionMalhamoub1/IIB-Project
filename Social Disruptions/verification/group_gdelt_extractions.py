"""
group_gdelt_extractions.py
==========================
Groups raw GDELT extractions (pre-consolidation) into canonical events using
a two-level weighted similarity graph.

LEVEL 1 — Article clustering (specific events)
-----------------------------------------------
Each node is one extracted article.  An edge is added when two articles are
likely reporting on the same real-world event.  Connected components become
canonical event clusters.

LEVEL 2 — Movement linking (related events)
--------------------------------------------
Each Level-1 cluster is represented by the centroid of its article embeddings.
A second, looser graph pass links clusters whose centroids are semantically
similar, share a country and disruption type, and fall within a broader date
window.  Connected components become movements — groups of city-level events
that are part of the same broader campaign or disaster.

Hard gates for Level 1 (score = 0 immediately):
  1. Different disruption_type
  2. Different iso3, or iso3 unresolved in either record
  3. Date gap > dynamic window (base TYPE_WINDOW, extended by duration_hours,
     capped at MAX_WINDOW)
  4. Sub-location conflict: both records have a resolved city/region AND their
     fuzzy similarity < SUBLOC_HARD_GATE.
     Rationale: a protest in Los Angeles and one in New York are distinct
     events even if they share a date, country, and topic.  Country-wide
     movements are unaffected because those articles typically lack city-level
     sub-location data and pass through.

Soft features — social disruptions (protests / labour_strike):

  Primary path — semantic embeddings (when both records have content):
    temporal        0.15  linear decay over dynamic window
    location        0.30  fuzzy sub-location match
    embedding       0.55  cosine similarity of content embeddings
                          (event_description, issue, target, groups, type,
                           sector — location deliberately excluded)

  Fallback path — sparse records without extractable content:
    temporal        0.30
    location        0.30
    sector          0.10  exact match
    issue           0.10  fuzzy token match
    protest_type    0.10  exact match
    participants    0.10  ratio of counts (min/max)

Soft features — flood:
    temporal        0.40
    location        0.40
    main_cause      0.20  fuzzy match

Edge weighting:
    Final similarity is multiplied by min(conf_a, conf_b) before threshold
    comparison.  Low-confidence articles therefore require a stronger content
    match to form an edge, preventing sparse records from chaining unrelated
    events together.

Bonus:
    reported_day_number consistency (+0.10, capped at 1.0)

Dynamic temporal window:
    Base window: TYPE_WINDOW[dtype].  If either article has duration_hours,
    the window is extended to cover the event duration (capped at
    MAX_WINDOW[dtype]).  This handles multi-day strikes and occupations
    without setting a blanket wide window for all events of that type.

Usage
-----
  python group_gdelt_extractions.py --date 20180102
  python group_gdelt_extractions.py --range 20180101 20180115
  python group_gdelt_extractions.py --input path/to/extractions.jsonl
  python group_gdelt_extractions.py --date 20180102 --out path/to/dir --threshold 0.45

Output
------
  verification/grouped/YYYYMMDD_grouped.jsonl    (Level-1 clusters)
  verification/grouped/YYYYMMDD_movements.jsonl  (Level-2 movements)
  verification/grouped/YYYYMMDD_grouped_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
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

DEFAULT_OUT_DIR = _HERE / "grouped"

# ---- Set your date range here ---- #
START_DATE = "20180328"
END_DATE   = "20180415"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
KNOWN_TYPES           = frozenset({"protests", "labour_strike"})
CONFIDENCE_THRESHOLD  = 0.6
EDGE_THRESHOLD        = 0.45
CHAIN_WARN_THRESHOLD  = 0.40
DAY_NUMBER_BONUS      = 0.10
SUBLOC_HARD_GATE      = 0.20   # Block pairs where both sublocs are known but dissimilar
EMBEDDING_MODEL       = "all-MiniLM-L6-v2"

# Level-1 base windows (days). Extended dynamically by duration_hours.
TYPE_WINDOW: dict[str, int] = {
    "protests":      3,
    "labour_strike": 7,   # reduced from 21 — duration_hours extends as needed
    "flood":         7,
}

# Hard cap on dynamic window extension
MAX_WINDOW: dict[str, int] = {
    "protests":      7,
    "labour_strike": 21,
    "flood":         14,
}

# Level-2 movement linking
MOVEMENT_THRESHOLD = 0.72   # centroid cosine similarity
MOVEMENT_WINDOW    = 14     # days between cluster start dates

# Weights when semantic embeddings are available (primary path)
BASE_WEIGHTS_SOCIAL_EMB: dict[str, float] = {
    "temporal":  0.15,
    "location":  0.30,
    "embedding": 0.55,
}

# Weights when no embedding content exists (fallback path)
BASE_WEIGHTS_SOCIAL_FALLBACK: dict[str, float] = {
    "temporal":     0.30,
    "location":     0.30,
    "sector":       0.10,
    "issue":        0.10,
    "protest_type": 0.10,
    "participants": 0.10,
}

BASE_WEIGHTS_FLOOD: dict[str, float] = {
    "temporal":   0.40,
    "location":   0.40,
    "main_cause": 0.20,
}

# ---------------------------------------------------------------------------
# Location normalisation
# ---------------------------------------------------------------------------

# Common abbreviations / variants → canonical form used in location_name
_LOC_NORM_PATTERNS: list[tuple[str, str]] = [
    (r"\bU\.S\.A\.?\b",                       "United States"),
    (r"\bUSA\b",                               "United States"),
    (r"\bU\.S\.\b",                            "United States"),
    (r"\bUS\b(?!\s+[A-Z]{2}\b)",               "United States"),  # "US" but not "US CA"
    (r"\bU\.K\.?\b",                           "United Kingdom"),
    (r"\bUK\b",                                "United Kingdom"),
    (r"\bDR\.?\s*Congo\b",                     "Democratic Republic of the Congo"),
    (r"\bDRC\b",                               "Democratic Republic of the Congo"),
]
_LOC_NORM_RE: list[tuple[re.Pattern, str]] = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in _LOC_NORM_PATTERNS
]


def _normalise_location(loc_name: str) -> str:
    """Apply canonical substitutions to a location_name string."""
    for pattern, repl in _LOC_NORM_RE:
        loc_name = pattern.sub(repl, loc_name)
    return loc_name


# ---------------------------------------------------------------------------
# Dynamic temporal window
# ---------------------------------------------------------------------------

def _get_window(dtype: str, dur_a: Optional[float], dur_b: Optional[float]) -> int:
    """Return per-pair temporal window in days.

    Starts from TYPE_WINDOW[dtype].  If either article records a duration,
    the window is extended to cover the full event, capped at MAX_WINDOW.
    This lets a 10-day strike match articles from day 1 and day 9 without
    forcing a blanket wide window on all strikes.
    """
    base = TYPE_WINDOW[dtype]
    cap  = MAX_WINDOW[dtype]
    if dur_a or dur_b:
        max_dur_days = max((dur_a or 0.0), (dur_b or 0.0)) / 24.0
        return min(cap, max(base, int(np.ceil(max_dur_days)) + 1))
    return base


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model (%s)...", EMBEDDING_MODEL)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("Embedding model ready.")
    return _embedding_model


def _make_embedding_text(record: dict) -> str:
    """Build the text to embed for a record.

    Only includes LLM-extracted fields that describe *what* the event is about.
    Location is deliberately excluded — it is handled separately by the hard
    gate and the subloc similarity score — so the embedding captures event
    identity independent of geography.
    """
    extras = record.get("extras") or {}

    def _val(v) -> str:
        if isinstance(v, list):
            return " ".join(str(x) for x in v if x)
        return str(v).strip() if v else ""

    parts = []

    desc = _val(record.get("event_description") or "")
    if desc:
        parts.append(desc)

    issue = _val(record.get("issue") or extras.get("issue") or "")
    if issue:
        parts.append(issue)

    target = _val(extras.get("target_of_protest") or "")
    if target:
        parts.append(target)

    groups = _val(extras.get("protesting_groups") or "")
    if groups:
        parts.append(groups)

    ptype = _val(record.get("protest_type") or extras.get("protest_type") or "")
    if ptype:
        parts.append(ptype)

    sector = _val(record.get("sector") or extras.get("sector") or "")
    if sector:
        parts.append(sector)

    return " | ".join(parts)


def compute_embeddings(records: list[dict]) -> None:
    """Compute and store unit-norm embeddings in-place for all records.

    Records without extractable content receive embedding=None and fall back
    to the fuzzy-string similarity path.
    """
    texts     = [_make_embedding_text(r) for r in records]
    non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]

    if not non_empty:
        for r in records:
            r["embedding"] = None
        return

    model  = _get_embedding_model()
    idxs   = [i for i, _ in non_empty]
    chunks = [t for _, t in non_empty]

    vecs = model.encode(
        chunks,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=False,
    )

    embed_map = dict(zip(idxs, vecs))
    for i, r in enumerate(records):
        r["embedding"] = embed_map.get(i, None)

    log.info(
        "Embeddings: %d / %d records had extractable content",
        len(non_empty), len(records),
    )


def _cluster_centroid(article_records: list[dict]) -> Optional[np.ndarray]:
    """Return unit-norm centroid of article embeddings, or None if unavailable."""
    vecs = [r["embedding"] for r in article_records if r.get("embedding") is not None]
    if not vecs:
        return None
    centroid = np.mean(vecs, axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-9:
        return None
    return centroid / norm


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_extractions(path: Path) -> list[dict]:
    log.info("Loading: %s", path)
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

            # Trust LLM classification over expert — if LLM said "unknown"
            # the article lacked usable content regardless of expert confidence
            raw_llm = e.get("llm_disruption_type") or ""
            if isinstance(raw_llm, list):
                raw_llm = raw_llm[0] if raw_llm else ""
            dtype = str(raw_llm).strip().lower()
            if dtype not in KNOWN_TYPES:
                skipped["type"] += 1
                continue

            if (e.get("confidence") or 0.0) < CONFIDENCE_THRESHOLD:
                skipped["conf"] += 1
                continue

            raw_date = e.get("event_date") or e.get("publish_date")
            try:
                ts = pd.Timestamp(raw_date)
                if ts.tzinfo is not None:
                    ts = ts.tz_convert("UTC").tz_localize(None)
                date = ts.normalize()
            except Exception:
                skipped["date"] += 1
                continue

            extras    = e.get("extras") or {}
            loc_name  = _normalise_location((e.get("location_name") or "").strip())

            records.append({
                "idx":                    len(records),
                "url":                    e.get("url", ""),
                "source_title":           e.get("source_title", ""),
                "disruption_type":        dtype,
                "event_date":             date,
                "publish_date":           e.get("publish_date"),
                "iso3":                   extract_iso3(e),
                "subloc":                 extract_subloc({**e, "location_name": loc_name}),
                "location_name":          loc_name,
                "confidence":             float(e.get("confidence") or 0.0),
                "lat":                    e.get("lat"),
                "lon":                    e.get("lon"),
                "duration_hours":         e.get("duration_hours"),
                "event_description":      e.get("event_description") or None,
                "llm_disruption_type":    e.get("llm_disruption_type"),
                "expert_disruption_type": e.get("expert_disruption_type"),
                "expert_probability":     e.get("expert_probability"),
                "classification_source":  e.get("classification_source"),
                # extras flattened for similarity scoring
                "sector":                 extras.get("sector") or None,
                "issue":                  extras.get("issue") or None,
                "protest_type":           extras.get("protest_type") or None,
                "estimated_participants": extras.get("estimated_participants") or None,
                "reported_day_number":    extras.get("reported_day_number") or None,
                "main_cause":             extras.get("main_cause") or None,
                # full extras kept for output
                "extras":                 extras,
            })

    log.info(
        "Loaded %d records | skipped: type=%d conf=%d date=%d parse=%d",
        len(records), skipped["type"], skipped["conf"], skipped["date"], skipped["parse"],
    )
    return records


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _participants_sim(a: Any, b: Any) -> Optional[float]:
    try:
        pa, pb = float(a), float(b)
        if pa <= 0 or pb <= 0:
            return None
        return min(pa, pb) / max(pa, pb)
    except (TypeError, ValueError):
        return None


def _similarity(a: dict, b: dict) -> float:
    # Hard gate 1: disruption type and country must match
    if a["iso3"] is None or b["iso3"] is None or a["iso3"] != b["iso3"]:
        return 0.0
    if a["disruption_type"] != b["disruption_type"]:
        return 0.0

    # Hard gate 2: dynamic date window
    dtype    = a["disruption_type"]
    max_days = _get_window(dtype, a.get("duration_hours"), b.get("duration_hours"))
    days_gap = abs((a["event_date"] - b["event_date"]).days)
    if days_gap > max_days:
        return 0.0

    # Hard gate 3: sub-location conflict
    # Block pairs where both records resolve to a city/region that is
    # geographically dissimilar.  Prevents cross-city merging in large
    # countries while leaving country-wide movements unaffected (those
    # articles typically have no city-level sub-location).
    sa, sb = a["subloc"], b["subloc"]
    if sa and sb and fuzzy_similarity(sa, sb) < SUBLOC_HARD_GATE:
        return 0.0

    def _str(v) -> str:
        if isinstance(v, list):
            v = v[0] if v else ""
        return str(v).strip().lower()

    scores:  dict[str, float] = {}
    weights: dict[str, float] = {}

    if dtype == "flood":
        scores["temporal"]  = 1.0 - days_gap / max_days
        weights["temporal"] = BASE_WEIGHTS_FLOOD["temporal"]

        if sa or sb:
            scores["location"]  = fuzzy_similarity(sa, sb) if (sa and sb) else 0.35
            weights["location"] = BASE_WEIGHTS_FLOOD["location"]

        if a["main_cause"] and b["main_cause"]:
            scores["main_cause"]  = fuzzy_similarity(
                _str(a["main_cause"]), _str(b["main_cause"])
            )
            weights["main_cause"] = BASE_WEIGHTS_FLOOD["main_cause"]

    else:
        ea, eb = a.get("embedding"), b.get("embedding")
        use_embeddings = ea is not None and eb is not None

        if use_embeddings:
            scores["temporal"]  = 1.0 - days_gap / max_days
            weights["temporal"] = BASE_WEIGHTS_SOCIAL_EMB["temporal"]

            if sa or sb:
                scores["location"]  = fuzzy_similarity(sa, sb) if (sa and sb) else 0.35
                weights["location"] = BASE_WEIGHTS_SOCIAL_EMB["location"]

            scores["embedding"]  = float(np.dot(ea, eb))
            weights["embedding"] = BASE_WEIGHTS_SOCIAL_EMB["embedding"]

        else:
            scores["temporal"]  = 1.0 - days_gap / max_days
            weights["temporal"] = BASE_WEIGHTS_SOCIAL_FALLBACK["temporal"]

            if sa or sb:
                scores["location"]  = fuzzy_similarity(sa, sb) if (sa and sb) else 0.35
                weights["location"] = BASE_WEIGHTS_SOCIAL_FALLBACK["location"]

            if a["sector"] and b["sector"]:
                scores["sector"]  = 1.0 if _str(a["sector"]) == _str(b["sector"]) else 0.0
                weights["sector"] = BASE_WEIGHTS_SOCIAL_FALLBACK["sector"]

            if a["issue"] and b["issue"]:
                scores["issue"]  = fuzzy_similarity(_str(a["issue"]), _str(b["issue"]))
                weights["issue"] = BASE_WEIGHTS_SOCIAL_FALLBACK["issue"]

            if a["protest_type"] and b["protest_type"]:
                scores["protest_type"] = (
                    1.0 if _str(a["protest_type"]) == _str(b["protest_type"]) else 0.2
                )
                weights["protest_type"] = BASE_WEIGHTS_SOCIAL_FALLBACK["protest_type"]

            psim = _participants_sim(
                a.get("estimated_participants"), b.get("estimated_participants")
            )
            if psim is not None:
                scores["participants"]  = psim
                weights["participants"] = BASE_WEIGHTS_SOCIAL_FALLBACK.get("participants", 0.10)

    if not weights:
        return 0.0

    score = sum(scores[k] * weights[k] for k in scores) / sum(weights.values())

    # reported_day_number consistency bonus
    da, db = a.get("reported_day_number"), b.get("reported_day_number")
    if da is not None and db is not None:
        try:
            if abs(abs(float(da) - float(db)) - days_gap) <= 1:
                score = min(1.0, score + DAY_NUMBER_BONUS)
        except (TypeError, ValueError):
            pass

    return round(score, 4)


# ---------------------------------------------------------------------------
# Level-1 graph construction + clustering
# ---------------------------------------------------------------------------

def build_graph(records: list[dict], threshold: float):
    """Build article-level similarity graph with confidence-weighted edges.

    Edge weight = similarity × min(conf_a, conf_b).  This means a low-quality
    article (confidence 0.6) needs a stronger content match to form an edge,
    preventing sparse records from acting as bridges between unrelated events.
    """
    import networkx as nx

    G = nx.Graph()
    for r in records:
        G.add_node(r["idx"])

    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        if r["iso3"] is not None:
            buckets[(r["iso3"], r["disruption_type"])].append(r)

    n_edges = 0
    for (_, dtype), bucket in buckets.items():
        bucket_sorted = sorted(bucket, key=lambda x: x["event_date"])
        for i, ra in enumerate(bucket_sorted):
            for rb in bucket_sorted[i + 1:]:
                # Quick early-exit on the strictest possible window
                if (rb["event_date"] - ra["event_date"]).days > MAX_WINDOW[dtype]:
                    break
                sim = _similarity(ra, rb)
                if sim <= 0.0:
                    continue
                # Confidence weighting: penalise low-quality articles without being
                # too aggressive — sqrt softens the penalty so a 0.6-confidence
                # article needs ~0.65 raw similarity rather than 0.83
                conf_weight   = np.sqrt(min(ra["confidence"], rb["confidence"]))
                weighted_sim  = round(sim * conf_weight, 4)
                if weighted_sim >= threshold:
                    G.add_edge(ra["idx"], rb["idx"], weight=weighted_sim)
                    n_edges += 1

    log.info("Graph: %d nodes | %d edges (threshold=%.2f)",
             G.number_of_nodes(), n_edges, threshold)
    return G


def get_clusters(G) -> list[list[int]]:
    import networkx as nx
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    n_multi = sum(1 for c in components if len(c) > 1)
    log.info("Clusters: %d total | %d multi-article | %d singletons",
             len(components), n_multi, len(components) - n_multi)
    return [sorted(c) for c in components]


def _mean_internal_score(G, node_ids: list[int]) -> float:
    if len(node_ids) < 2:
        return 1.0
    edges = [G[u][v]["weight"] for u in node_ids for v in node_ids
             if u < v and G.has_edge(u, v)]
    return float(np.mean(edges)) if edges else 0.0


# ---------------------------------------------------------------------------
# Level-1 aggregation
# ---------------------------------------------------------------------------

def _aggregate_cluster(
    records: list[dict],
    cluster_id: str,
    mean_score: float,
) -> dict:
    dates = [r["event_date"] for r in records]
    lats  = [float(r["lat"]) for r in records if r.get("lat") is not None]
    lons  = [float(r["lon"]) for r in records if r.get("lon") is not None]

    articles = [
        {
            "url":                    r["url"],
            "source_title":           r["source_title"],
            "event_date":             r["event_date"].strftime("%Y-%m-%d"),
            "publish_date":           r["publish_date"],
            "confidence":             r["confidence"],
            "event_description":      r["event_description"],
            "llm_disruption_type":    r["llm_disruption_type"],
            "expert_disruption_type": r["expert_disruption_type"],
            "expert_probability":     r["expert_probability"],
            "classification_source":  r["classification_source"],
            "location_name":          r["location_name"],
            "lat":                    r["lat"],
            "lon":                    r["lon"],
            "duration_hours":         r["duration_hours"],
            "indicators":             r["extras"],
        }
        for r in sorted(records, key=lambda x: x["event_date"])
    ]

    cluster = {
        "cluster_id":          cluster_id,
        "parent_event_id":     None,   # filled in by link_movements
        "disruption_type":     records[0]["disruption_type"],
        "event_date":          min(dates).strftime("%Y-%m-%d"),
        "event_end_date":      max(dates).strftime("%Y-%m-%d"),
        "iso3":                records[0]["iso3"],
        "location_name":       max(
                                   (r["location_name"] for r in records if r["location_name"]),
                                   key=len, default=""
                               ),
        "n_articles":          len(records),
        "confidence_max":      max(r["confidence"] for r in records),
        "mean_internal_score": round(mean_score, 4),
        "lat":                 round(float(np.median(lats)), 5) if lats else None,
        "lon":                 round(float(np.median(lons)), 5) if lons else None,
        "articles":            articles,
    }

    # Store centroid internally for Level-2 movement linking (stripped before output)
    cluster["_centroid"] = _cluster_centroid(records)

    return cluster


# ---------------------------------------------------------------------------
# Level-2 movement linking
# ---------------------------------------------------------------------------

def link_movements(grouped: list[dict]) -> tuple[list[dict], list[dict]]:
    """Link Level-1 clusters into movements via centroid similarity.

    Two clusters are candidates for the same movement if they share iso3 and
    disruption_type, their start dates are within MOVEMENT_WINDOW days, and
    their centroid cosine similarity >= MOVEMENT_THRESHOLD.

    Returns (updated grouped list, movements list).  Each cluster in grouped
    has its parent_event_id field populated; standalone clusters keep None.
    """
    import networkx as nx

    linkable = [g for g in grouped if g.get("_centroid") is not None]

    G = nx.Graph()
    for i in range(len(linkable)):
        G.add_node(i)

    for i, ca in enumerate(linkable):
        da = pd.Timestamp(ca["event_date"])
        for j in range(i + 1, len(linkable)):
            cb = linkable[j]
            if ca["iso3"] != cb["iso3"]:
                continue
            if ca["disruption_type"] != cb["disruption_type"]:
                continue
            db = pd.Timestamp(cb["event_date"])
            if abs((da - db).days) > MOVEMENT_WINDOW:
                continue
            sim = float(np.dot(ca["_centroid"], cb["_centroid"]))
            if sim >= MOVEMENT_THRESHOLD:
                G.add_edge(i, j, weight=round(sim, 4))

    movements = []
    n_linked  = 0

    for mi, component in enumerate(nx.connected_components(G)):
        component = sorted(component)
        if len(component) == 1:
            continue   # singleton — not part of a movement

        # Require at least two multi-article clusters as evidence — a single
        # substantial cluster linking to singletons is not a reliable movement
        if sum(1 for i in component if linkable[i]["n_articles"] > 1) < 2:
            continue

        cluster_subset = [linkable[i] for i in component]
        iso3  = cluster_subset[0]["iso3"]
        dtype = cluster_subset[0]["disruption_type"]
        dates     = [pd.Timestamp(c["event_date"]) for c in cluster_subset]
        end_dates = [pd.Timestamp(c["event_end_date"]) for c in cluster_subset]
        mid = f"{iso3}_{dtype}_movement_{min(dates).strftime('%Y%m%d')}_m{mi:04d}"

        for c in cluster_subset:
            c["parent_event_id"] = mid

        edge_weights = [
            G[i][j]["weight"]
            for i in component for j in component
            if i < j and G.has_edge(i, j)
        ]
        mean_sim = round(float(np.mean(edge_weights)), 4) if edge_weights else 0.0

        movements.append({
            "movement_id":        mid,
            "disruption_type":    dtype,
            "iso3":               iso3,
            "event_date":         min(dates).strftime("%Y-%m-%d"),
            "event_end_date":     max(end_dates).strftime("%Y-%m-%d"),
            "n_clusters":         len(cluster_subset),
            "n_articles":         sum(c["n_articles"] for c in cluster_subset),
            "mean_centroid_sim":  mean_sim,
            "child_cluster_ids":  [c["cluster_id"] for c in cluster_subset],
        })
        n_linked += len(cluster_subset)

    log.info(
        "Movements: %d movements linking %d clusters (%d clusters standalone)",
        len(movements), n_linked, len(grouped) - n_linked,
    )
    return grouped, movements


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_grouping(
    input_path: Path,
    out_dir: Path,
    date_str: str,
    threshold: float = EDGE_THRESHOLD,
) -> list[dict]:

    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_extractions(input_path)
    if not records:
        log.warning("No qualifying records in %s", input_path)
        return []

    compute_embeddings(records)

    G        = build_graph(records, threshold)
    clusters = get_clusters(G)

    by_idx  = {r["idx"]: r for r in records}
    grouped = []
    n_chain_warnings = 0

    for ci, node_ids in enumerate(clusters):
        cluster_recs = [by_idx[n] for n in node_ids]
        ms = _mean_internal_score(G, node_ids)

        if len(node_ids) > 2 and ms < CHAIN_WARN_THRESHOLD:
            n_chain_warnings += 1

        iso3  = cluster_recs[0]["iso3"] or "UNK"
        dtype = cluster_recs[0]["disruption_type"]
        date  = min(r["event_date"] for r in cluster_recs).strftime("%Y%m%d")
        cid   = f"{iso3}_{dtype}_{date}_c{ci:04d}"

        grouped.append(_aggregate_cluster(cluster_recs, cid, ms))

    if n_chain_warnings:
        log.warning("%d clusters may be over-merged (mean_internal_score < %.2f)",
                    n_chain_warnings, CHAIN_WARN_THRESHOLD)

    # Level-2: link related clusters into movements
    grouped, movements = link_movements(grouped)

    # Strip internal centroid arrays before writing
    for g in grouped:
        g.pop("_centroid", None)

    log.info(
        "Grouped: %d records → %d events (%.1f%% reduction)",
        len(records), len(grouped),
        100 * (1 - len(grouped) / len(records)) if records else 0,
    )

    out_jsonl = out_dir / f"{date_str}_grouped.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for ev in grouped:
            fh.write(json.dumps(ev, default=str) + "\n")
    log.info("Written: %s", out_jsonl)

    if movements:
        out_movements = out_dir / f"{date_str}_movements.jsonl"
        with out_movements.open("w", encoding="utf-8") as fh:
            for mv in movements:
                fh.write(json.dumps(mv, default=str) + "\n")
        log.info("Written: %s", out_movements)

    report = {
        "date":             date_str,
        "input":            str(input_path),
        "threshold":        threshold,
        "raw_records":      len(records),
        "canonical_events": len(grouped),
        "movements":        len(movements),
        "reduction_pct":    round(100 * (1 - len(grouped) / len(records)), 1) if records else 0,
        "chain_warnings":   n_chain_warnings,
        "type_breakdown": {
            t: {
                "raw":     sum(1 for r in records if r["disruption_type"] == t),
                "grouped": sum(1 for e in grouped if e["disruption_type"] == t),
            }
            for t in KNOWN_TYPES
        },
    }
    with (out_dir / f"{date_str}_grouped_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    return grouped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _daterange(start: str, end: str):
    from datetime import datetime, timedelta
    cur = datetime.strptime(start, "%Y%m%d")
    fin = datetime.strptime(end,   "%Y%m%d")
    while cur <= fin:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group raw GDELT extractions into canonical events"
    )
    parser.add_argument(
        "--range", nargs=2, metavar=("START", "END"), default=None,
        help="Date range YYYYMMDD YYYYMMDD — processes every day in range",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Single date YYYYMMDD",
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Explicit path to a single extractions.jsonl",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--threshold", type=float, default=EDGE_THRESHOLD,
        help=f"Min pairwise similarity to create an edge (default {EDGE_THRESHOLD})",
    )
    args = parser.parse_args()

    out_dir = args.out

    if args.range:
        dates = list(_daterange(args.range[0], args.range[1]))
        for d in dates:
            input_path = _ROOT / "Builder_GDELT" / "results" / "daily" / d / "extractions.jsonl"
            if not input_path.exists():
                log.warning("[SKIP] No extractions for %s", d)
                continue
            run_grouping(input_path, out_dir, d, threshold=args.threshold)

    elif args.date:
        input_path = _ROOT / "Builder_GDELT" / "results" / "daily" / args.date / "extractions.jsonl"
        if not input_path.exists():
            log.error("Input not found: %s", input_path)
            sys.exit(1)
        run_grouping(input_path, out_dir, args.date, threshold=args.threshold)

    elif args.input:
        date_str = args.input.parent.name
        if not args.input.exists():
            log.error("Input not found: %s", args.input)
            sys.exit(1)
        run_grouping(args.input, out_dir, date_str, threshold=args.threshold)

    else:
        dates = list(_daterange(START_DATE, END_DATE))
        for d in dates:
            input_path = _ROOT / "Builder_GDELT" / "results" / "daily" / d / "extractions.jsonl"
            if not input_path.exists():
                log.warning("[SKIP] No extractions for %s", d)
                continue
            run_grouping(input_path, out_dir, d, threshold=args.threshold)


if __name__ == "__main__":
    main()
