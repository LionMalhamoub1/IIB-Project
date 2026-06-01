# v2 global clustering — pools the full date range into one pass instead of running per-day.
# Fixes cross-day fragmentation from group_gdelt_extractions.py.
# MOVEMENT_THRESHOLD 0.60 and MOVEMENT_WINDOW 30d (vs 0.72/14d in v1) to handle topic drift.

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Reuse shared utilities from parent verification directory
_HERE = Path(__file__).resolve().parent
_VER  = _HERE.parent
_SD   = _VER.parent
_ROOT = _SD.parent
sys.path.insert(0, str(_VER))
from _utils import extract_iso3, extract_subloc, fuzzy_similarity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DEFAULT_OUT_DIR = _HERE / "output"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
KNOWN_TYPES           = frozenset({"protests", "labour_strike"})
CONFIDENCE_THRESHOLD  = 0.6
EDGE_THRESHOLD        = 0.45
CHAIN_WARN_THRESHOLD  = 0.40
DAY_NUMBER_BONUS      = 0.10
SUBLOC_HARD_GATE      = 0.20
EMBEDDING_MODEL       = "all-MiniLM-L6-v2"

TYPE_WINDOW: dict[str, int] = {
    "protests":      3,
    "labour_strike": 7,
    "flood":         7,
}
MAX_WINDOW: dict[str, int] = {
    "protests":      7,
    "labour_strike": 21,
    "flood":         14,
}

# --- V2 changes: relaxed movement linking ---
MOVEMENT_THRESHOLD = 0.60   # was 0.72 — captures topic drift across long events
MOVEMENT_WINDOW    = 30     # was 14  — catches campaigns that pause and resume
MIN_MULTI_CLUSTERS = 1      # was 2   — one strong multi-article cluster is enough

# Weights (unchanged from v1)
BASE_WEIGHTS_SOCIAL_EMB: dict[str, float] = {
    "temporal":  0.15,
    "location":  0.30,
    "embedding": 0.55,
}
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
# Location normalisation (unchanged)
# ---------------------------------------------------------------------------
_LOC_NORM_PATTERNS: list[tuple[str, str]] = [
    (r"\bU\.S\.A\.?\b",          "United States"),
    (r"\bUSA\b",                  "United States"),
    (r"\bU\.S\.\b",               "United States"),
    (r"\bUS\b(?!\s+[A-Z]{2}\b)", "United States"),
    (r"\bU\.K\.?\b",              "United Kingdom"),
    (r"\bUK\b",                   "United Kingdom"),
    (r"\bDR\.?\s*Congo\b",        "Democratic Republic of the Congo"),
    (r"\bDRC\b",                  "Democratic Republic of the Congo"),
]
_LOC_NORM_RE = [
    (re.compile(p, re.IGNORECASE), r) for p, r in _LOC_NORM_PATTERNS
]

def _normalise_location(loc_name: str) -> str:
    for pattern, repl in _LOC_NORM_RE:
        loc_name = pattern.sub(repl, loc_name)
    return loc_name


# ---------------------------------------------------------------------------
# Dynamic temporal window (unchanged)
# ---------------------------------------------------------------------------
def _get_window(dtype: str, dur_a: Optional[float], dur_b: Optional[float]) -> int:
    base = TYPE_WINDOW[dtype]
    cap  = MAX_WINDOW[dtype]
    if dur_a or dur_b:
        max_dur_days = max((dur_a or 0.0), (dur_b or 0.0)) / 24.0
        return min(cap, max(base, int(np.ceil(max_dur_days)) + 1))
    return base


# ---------------------------------------------------------------------------
# Embeddings (unchanged)
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
    extras = record.get("extras") or {}
    def _val(v) -> str:
        if isinstance(v, list):
            return " ".join(str(x) for x in v if x)
        return str(v).strip() if v else ""
    parts = []
    for src in [record.get("event_description"), record.get("issue") or extras.get("issue"),
                extras.get("target_of_protest"), extras.get("protesting_groups"),
                record.get("protest_type") or extras.get("protest_type"),
                record.get("sector") or extras.get("sector")]:
        t = _val(src)
        if t:
            parts.append(t)
    return " | ".join(parts)


def compute_embeddings(records: list[dict]) -> None:
    texts     = [_make_embedding_text(r) for r in records]
    non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
    if not non_empty:
        for r in records:
            r["embedding"] = None
        return
    model  = _get_embedding_model()
    idxs   = [i for i, _ in non_empty]
    chunks = [t for _, t in non_empty]
    vecs   = model.encode(chunks, normalize_embeddings=True, batch_size=64,
                          show_progress_bar=False)
    embed_map = dict(zip(idxs, vecs))
    for i, r in enumerate(records):
        r["embedding"] = embed_map.get(i, None)
    log.info("Embeddings: %d / %d records had extractable content",
             len(non_empty), len(records))


def _cluster_centroid(article_records: list[dict]) -> Optional[np.ndarray]:
    vecs = [r["embedding"] for r in article_records if r.get("embedding") is not None]
    if not vecs:
        return None
    centroid = np.mean(vecs, axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm >= 1e-9 else None


# ---------------------------------------------------------------------------
# Loading — KEY CHANGE: accepts list of files, pools all articles
# ---------------------------------------------------------------------------
def load_extractions(
    paths: list[Path],
    date_min: Optional[pd.Timestamp] = None,
    date_max: Optional[pd.Timestamp] = None,
) -> list[dict]:
    """Load and pool articles from multiple daily extraction files.

    V1 loaded one file at a time, creating artificial per-day boundaries.
    V2 pools all files so the similarity graph sees every article together.
    Articles are deduplicated by URL to avoid double-counting when date ranges
    overlap (e.g. if the same article appeared in two consecutive day files).

    date_min / date_max: if provided, articles whose event_date falls outside
    this window are dropped.  This prevents articles with corrupt future dates
    (e.g. 2023 dates found in a 2018 extraction run) from polluting the pool.
    A small buffer of MAX_WINDOW days is added on each side to allow articles
    reporting on events that started just before the range start.
    """
    records: list[dict] = []
    seen_urls: set[str] = set()
    skipped: dict[str, int] = defaultdict(int)

    # Add a small buffer so articles about events just outside the window
    # aren't clipped (e.g. a Dec 28 event appearing in a Jan file)
    _buffer = pd.Timedelta(days=max(MAX_WINDOW.values()))
    _min = (date_min - _buffer) if date_min is not None else None
    _max = (date_max + _buffer) if date_max is not None else None

    for path in paths:
        if not path.exists():
            log.warning("[SKIP] Not found: %s", path)
            continue
        log.info("Loading: %s", path)
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

                # URL deduplication across files
                url = e.get("url", "")
                if url and url in seen_urls:
                    skipped["url_dup"] += 1
                    continue
                if url:
                    seen_urls.add(url)

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

                # Date range filter — drops articles with corrupt/future dates
                if _min is not None and date < _min:
                    skipped["date_range"] += 1
                    continue
                if _max is not None and date > _max:
                    skipped["date_range"] += 1
                    continue

                extras   = e.get("extras") or {}
                loc_name = _normalise_location((e.get("location_name") or "").strip())

                records.append({
                    "idx":                    len(records),
                    "url":                    url,
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
                    "sector":                 extras.get("sector") or None,
                    "issue":                  extras.get("issue") or None,
                    "protest_type":           extras.get("protest_type") or None,
                    "estimated_participants": extras.get("estimated_participants") or None,
                    "reported_day_number":    extras.get("reported_day_number") or None,
                    "main_cause":             extras.get("main_cause") or None,
                    "extras":                 extras,
                })

    log.info(
        "Pooled %d records from %d files | skipped: type=%d conf=%d date=%d "
        "date_range=%d url_dup=%d parse=%d",
        len(records), len(paths),
        skipped["type"], skipped["conf"], skipped["date"],
        skipped["date_range"], skipped["url_dup"], skipped["parse"],
    )
    return records


# ---------------------------------------------------------------------------
# Similarity (unchanged from v1)
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
    if a["iso3"] is None or b["iso3"] is None or a["iso3"] != b["iso3"]:
        return 0.0
    if a["disruption_type"] != b["disruption_type"]:
        return 0.0

    dtype    = a["disruption_type"]
    max_days = _get_window(dtype, a.get("duration_hours"), b.get("duration_hours"))
    days_gap = abs((a["event_date"] - b["event_date"]).days)
    if days_gap > max_days:
        return 0.0

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
            scores["main_cause"]  = fuzzy_similarity(_str(a["main_cause"]), _str(b["main_cause"]))
            weights["main_cause"] = BASE_WEIGHTS_FLOOD["main_cause"]
    else:
        ea, eb       = a.get("embedding"), b.get("embedding")
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
                scores["protest_type"]  = (
                    1.0 if _str(a["protest_type"]) == _str(b["protest_type"]) else 0.2
                )
                weights["protest_type"] = BASE_WEIGHTS_SOCIAL_FALLBACK["protest_type"]
            psim = _participants_sim(
                a.get("estimated_participants"), b.get("estimated_participants")
            )
            if psim is not None:
                scores["participants"]  = psim
                weights["participants"] = 0.10

    if not weights:
        return 0.0

    score = sum(scores[k] * weights[k] for k in scores) / sum(weights.values())

    da, db = a.get("reported_day_number"), b.get("reported_day_number")
    if da is not None and db is not None:
        try:
            if abs(abs(float(da) - float(db)) - days_gap) <= 1:
                score = min(1.0, score + DAY_NUMBER_BONUS)
        except (TypeError, ValueError):
            pass

    return round(score, 4)


# ---------------------------------------------------------------------------
# Graph construction + clustering (unchanged logic, same buckets)
# ---------------------------------------------------------------------------
def build_graph(records: list[dict], threshold: float):
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
                if (rb["event_date"] - ra["event_date"]).days > MAX_WINDOW[dtype]:
                    break
                sim = _similarity(ra, rb)
                if sim <= 0.0:
                    continue
                conf_weight  = np.sqrt(min(ra["confidence"], rb["confidence"]))
                weighted_sim = round(sim * conf_weight, 4)
                if weighted_sim >= threshold:
                    G.add_edge(ra["idx"], rb["idx"], weight=weighted_sim)
                    n_edges += 1

    log.info("Graph: %d nodes | %d edges (threshold=%.2f)",
             G.number_of_nodes(), n_edges, threshold)
    return G


def get_clusters(G, resolution: float = 1.0) -> list[list[int]]:
    """Cluster articles using Louvain community detection.

    Replaces connected_components (single-linkage) which caused severe
    over-merging: weakly-linked articles would chain together into one giant
    cluster covering an entire country's month of protests.

    Louvain finds *dense* subgraphs rather than just connected ones, so the
    Women's March, a Guantanamo protest, and a school closure protest in the
    USA will end up in separate clusters rather than one 366-article blob.

    resolution controls granularity: higher = smaller, tighter clusters.
    Default 1.0 is standard; try 1.5-2.0 if clusters are still too large.
    Isolated nodes (no edges) each become their own singleton community,
    identical to connected_components behaviour for those nodes.
    """
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    # louvain_communities requires at least one edge; fall back gracefully
    if G.number_of_edges() == 0:
        components = list(nx.connected_components(G))
    else:
        components = louvain_communities(G, weight="weight",
                                         resolution=resolution, seed=42)

    components = sorted(components, key=len, reverse=True)
    n_multi = sum(1 for c in components if len(c) > 1)
    log.info("Clusters (Louvain): %d total | %d multi-article | %d singletons",
             len(components), n_multi, len(components) - n_multi)
    return [sorted(c) for c in components]


def _mean_internal_score(G, node_ids: list[int]) -> float:
    if len(node_ids) < 2:
        return 1.0
    edges = [G[u][v]["weight"] for u in node_ids for v in node_ids
             if u < v and G.has_edge(u, v)]
    return float(np.mean(edges)) if edges else 0.0


# ---------------------------------------------------------------------------
# Level-1 aggregation (unchanged)
# ---------------------------------------------------------------------------
def _aggregate_cluster(records: list[dict], cluster_id: str, mean_score: float) -> dict:
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
        "parent_event_id":     None,
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
    cluster["_centroid"] = _cluster_centroid(records)
    return cluster


# ---------------------------------------------------------------------------
# Level-2 movement linking — RELAXED thresholds vs v1
# ---------------------------------------------------------------------------
def link_movements(grouped: list[dict]) -> tuple[list[dict], list[dict]]:
    """Link Level-1 clusters into movements.

    V2 changes vs v1:
      - MOVEMENT_THRESHOLD lowered 0.72 → 0.60: captures topic drift across
        long events where coverage language shifts over time.
      - MOVEMENT_WINDOW extended 14 → 30 days: catches campaigns that pause
        and resume, or that generate intermittent news coverage.
      - MIN_MULTI_CLUSTERS lowered 2 → 1: a single well-supported cluster
        linking to smaller ones is a genuine movement signal.
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
            continue
        # V2: only require MIN_MULTI_CLUSTERS (=1) multi-article clusters
        if sum(1 for i in component if linkable[i]["n_articles"] > 1) < MIN_MULTI_CLUSTERS:
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
        "Movements: %d movements linking %d clusters (%d standalone)",
        len(movements), n_linked, len(grouped) - n_linked,
    )
    return grouped, movements


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_grouping(
    input_paths: list[Path],
    out_dir: Path,
    range_label: str,
    threshold: float = EDGE_THRESHOLD,
    resolution: float = 1.0,
    date_min: Optional[pd.Timestamp] = None,
    date_max: Optional[pd.Timestamp] = None,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_extractions(input_paths, date_min=date_min, date_max=date_max)
    if not records:
        log.warning("No qualifying records in provided paths.")
        return []

    compute_embeddings(records)

    G        = build_graph(records, threshold)
    clusters = get_clusters(G, resolution=resolution)

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

    grouped, movements = link_movements(grouped)

    for g in grouped:
        g.pop("_centroid", None)

    log.info(
        "Grouped: %d records -> %d events (%.1f%% reduction)",
        len(records), len(grouped),
        100 * (1 - len(grouped) / len(records)) if records else 0,
    )

    out_jsonl = out_dir / f"{range_label}_grouped.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for ev in grouped:
            fh.write(json.dumps(ev, default=str) + "\n")
    log.info("Written: %s", out_jsonl)

    if movements:
        out_mv = out_dir / f"{range_label}_movements.jsonl"
        with out_mv.open("w", encoding="utf-8") as fh:
            for mv in movements:
                fh.write(json.dumps(mv, default=str) + "\n")
        log.info("Written: %s", out_mv)

    report = {
        "range":            range_label,
        "n_input_files":    len(input_paths),
        "threshold":        threshold,
        "raw_records":      len(records),
        "canonical_events": len(grouped),
        "movements":        len(movements),
        "reduction_pct":    round(100 * (1 - len(grouped) / len(records)), 1) if records else 0,
        "chain_warnings":   n_chain_warnings,
        "movement_params": {
            "threshold": MOVEMENT_THRESHOLD,
            "window_days": MOVEMENT_WINDOW,
            "min_multi_clusters": MIN_MULTI_CLUSTERS,
        },
        "type_breakdown": {
            t: {
                "raw":     sum(1 for r in records if r["disruption_type"] == t),
                "grouped": sum(1 for e in grouped if e["disruption_type"] == t),
            }
            for t in KNOWN_TYPES
        },
    }
    with (out_dir / f"{range_label}_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    return grouped


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _daterange(start: str, end: str):
    cur = datetime.strptime(start, "%Y%m%d")
    fin = datetime.strptime(end,   "%Y%m%d")
    while cur <= fin:
        yield cur.strftime("%Y%m%d")
        cur += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Global article clustering across a date range"
    )
    parser.add_argument(
        "--range", nargs=2, metavar=("START", "END"), required=True,
        help="Date range YYYYMMDD YYYYMMDD",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--threshold", type=float, default=EDGE_THRESHOLD,
        help=f"Min edge similarity (default {EDGE_THRESHOLD})",
    )
    args = parser.parse_args()

    start, end = args.range
    dates = list(_daterange(start, end))
    input_paths = [
        _ROOT / "Builder_GDELT" / "results" / "daily" / d / "extractions.jsonl"
        for d in dates
    ]
    range_label = f"{start}_{end}"
    run_grouping(input_paths, args.out, range_label, threshold=args.threshold)


if __name__ == "__main__":
    main()
