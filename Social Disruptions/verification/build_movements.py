"""
build_movements.py
==================
Post-processing script that builds cross-date movements from all Level-1
grouped cluster files produced by group_gdelt_extractions.py.

The daily grouping script only links clusters within a single day's extraction
file, so a movement like the Iranian protests (Dec 25 – Jan 15) would be
re-discovered independently on each daily run.  This script pools all clusters
across all dates, re-computes centroid embeddings from article content, and
runs the Level-2 movement linking pass across the full date range.

Output
------
  verification/grouped/movements_global.jsonl
      One record per movement, with child_cluster_ids spanning multiple dates.

  verification/grouped/YYYYMMDD_grouped.jsonl  (updated in-place)
      parent_event_id field populated for every cluster that belongs to a
      movement; remains null for standalone clusters.

Usage
-----
  python build_movements.py
  python build_movements.py --grouped path/to/grouped/dir
  python build_movements.py --range 20180101 20180115
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters  (should match group_gdelt_extractions.py)
# ---------------------------------------------------------------------------
GROUPED_DIR        = _HERE / "grouped"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
MOVEMENT_THRESHOLD = 0.72   # centroid cosine similarity to form a movement edge
MOVEMENT_WINDOW    = 14     # max days between cluster event_dates
MIN_MULTI_CLUSTERS = 2      # minimum number of multi-article clusters in a movement

# ---------------------------------------------------------------------------
# Embedding model (lazy-loaded)
# ---------------------------------------------------------------------------
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model (%s)...", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("Embedding model ready.")
    return _model


# ---------------------------------------------------------------------------
# Reconstruct embedding text from a cluster's articles
# ---------------------------------------------------------------------------

def _article_embedding_text(article: dict) -> str:
    """Reconstruct embedding text from an article record in grouped output.

    Mirrors the logic in group_gdelt_extractions._make_embedding_text but
    uses the grouped article format (indicators dict instead of extras).
    Location is excluded — same rationale as the original.
    """
    indicators = article.get("indicators") or {}

    def _val(v) -> str:
        if isinstance(v, list):
            return " ".join(str(x) for x in v if x)
        return str(v).strip() if v else ""

    parts = []

    desc = _val(article.get("event_description") or "")
    if desc:
        parts.append(desc)

    issue = _val(indicators.get("issue") or "")
    if issue:
        parts.append(issue)

    target = _val(indicators.get("target_of_protest") or "")
    if target:
        parts.append(target)

    groups = _val(indicators.get("protesting_groups") or "")
    if groups:
        parts.append(groups)

    ptype = _val(indicators.get("protest_type") or "")
    if ptype:
        parts.append(ptype)

    sector = _val(indicators.get("sector") or "")
    if sector:
        parts.append(sector)

    return " | ".join(parts)


def _compute_cluster_centroid(cluster: dict) -> np.ndarray | None:
    """Compute unit-norm centroid for a cluster from its articles."""
    texts = [_article_embedding_text(a) for a in cluster.get("articles", [])]
    non_empty = [t for t in texts if t.strip()]
    if not non_empty:
        return None

    model = _get_model()
    vecs  = model.encode(non_empty, normalize_embeddings=True, batch_size=64,
                         show_progress_bar=False)
    centroid = np.mean(vecs, axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-9:
        return None
    return centroid / norm


# ---------------------------------------------------------------------------
# Load all grouped clusters
# ---------------------------------------------------------------------------

def load_all_clusters(grouped_dir: Path, date_range: tuple[str, str] | None = None
                      ) -> list[dict]:
    """Load every cluster from all YYYYMMDD_grouped.jsonl files.

    Deduplicates by cluster_id — since articles overlap across daily files,
    the same cluster can appear in multiple files with the same ID.  We keep
    the instance with the highest n_articles.
    """
    files = sorted(grouped_dir.glob("*_grouped.jsonl"))
    if date_range:
        start, end = date_range
        files = [f for f in files if start <= f.stem.split("_")[0] <= end]

    seen:    dict[str, dict] = {}   # cluster_id → record
    n_total = 0

    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cid = c.get("cluster_id")
                if not cid:
                    continue
                n_total += 1
                # Keep whichever instance has the most articles
                if cid not in seen or c.get("n_articles", 0) > seen[cid].get("n_articles", 0):
                    c["_source_file"] = path.name
                    seen[cid] = c

    clusters = list(seen.values())
    log.info("Loaded %d unique clusters from %d records across %d files",
             len(clusters), n_total, len(files))
    return clusters


# ---------------------------------------------------------------------------
# Movement linking
# ---------------------------------------------------------------------------

def build_movements(clusters: list[dict]) -> tuple[list[dict], list[dict]]:
    """Run Level-2 movement linking across all clusters.

    Returns (updated clusters, movements).
    """
    import networkx as nx

    log.info("Computing cluster centroids...")
    for c in clusters:
        c["_centroid"] = _compute_cluster_centroid(c)

    linkable = [c for c in clusters if c["_centroid"] is not None]
    log.info("%d / %d clusters have centroids", len(linkable), len(clusters))

    # Build movement graph
    G = nx.Graph()
    for i in range(len(linkable)):
        G.add_node(i)

    n_edges = 0
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
                n_edges += 1

    log.info("Movement graph: %d nodes | %d edges", G.number_of_nodes(), n_edges)

    # Build lookup for updating parent_event_id
    cid_to_cluster = {c["cluster_id"]: c for c in clusters}

    movements  = []
    n_linked   = 0
    mi_counter = 0

    for component in nx.connected_components(G):
        component = sorted(component)
        if len(component) == 1:
            continue

        # Require at least MIN_MULTI_CLUSTERS multi-article clusters
        if sum(1 for i in component if linkable[i]["n_articles"] > 1) < MIN_MULTI_CLUSTERS:
            continue

        cluster_subset = [linkable[i] for i in component]
        iso3  = cluster_subset[0]["iso3"]
        dtype = cluster_subset[0]["disruption_type"]
        dates     = [pd.Timestamp(c["event_date"]) for c in cluster_subset]
        end_dates = [pd.Timestamp(c["event_end_date"]) for c in cluster_subset]

        mid = f"{iso3}_{dtype}_movement_{min(dates).strftime('%Y%m%d')}_m{mi_counter:04d}"
        mi_counter += 1

        edge_weights = [
            G[i][j]["weight"]
            for i in component for j in component
            if i < j and G.has_edge(i, j)
        ]
        mean_sim = round(float(np.mean(edge_weights)), 4) if edge_weights else 0.0

        child_ids = [c["cluster_id"] for c in cluster_subset]
        for cid in child_ids:
            if cid in cid_to_cluster:
                cid_to_cluster[cid]["parent_event_id"] = mid

        movements.append({
            "movement_id":       mid,
            "disruption_type":   dtype,
            "iso3":              iso3,
            "event_date":        min(dates).strftime("%Y-%m-%d"),
            "event_end_date":    max(end_dates).strftime("%Y-%m-%d"),
            "n_clusters":        len(cluster_subset),
            "n_articles":        sum(c["n_articles"] for c in cluster_subset),
            "mean_centroid_sim": mean_sim,
            "child_cluster_ids": child_ids,
        })
        n_linked += len(cluster_subset)

    log.info("Movements: %d movements linking %d clusters (%d standalone)",
             len(movements), n_linked, len(clusters) - n_linked)
    return clusters, movements


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------

def write_movements(movements: list[dict], grouped_dir: Path) -> None:
    out = grouped_dir / "movements_global.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for mv in movements:
            fh.write(json.dumps(mv, default=str) + "\n")
    log.info("Written: %s (%d movements)", out, len(movements))


def update_grouped_files(clusters: list[dict], grouped_dir: Path) -> None:
    """Rewrite grouped JSONL files with updated parent_event_id fields."""
    # Group updated clusters back by source file
    by_file: dict[str, dict[str, dict]] = defaultdict(dict)
    for c in clusters:
        src = c.get("_source_file")
        if src:
            by_file[src][c["cluster_id"]] = c

    for filename, cluster_map in by_file.items():
        path = grouped_dir / filename
        if not path.exists():
            continue

        updated_lines = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    c = json.loads(line)
                except json.JSONDecodeError:
                    updated_lines.append(line)
                    continue
                cid = c.get("cluster_id")
                if cid and cid in cluster_map:
                    c["parent_event_id"] = cluster_map[cid].get("parent_event_id")
                # Strip internal fields before writing
                c.pop("_centroid", None)
                c.pop("_source_file", None)
                updated_lines.append(json.dumps(c, default=str))

        with path.open("w", encoding="utf-8") as fh:
            fh.write("\n".join(updated_lines) + "\n")

    log.info("Updated parent_event_id in %d grouped files", len(by_file))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(grouped_dir: Path, date_range: tuple[str, str] | None = None) -> None:
    clusters           = load_all_clusters(grouped_dir, date_range)
    clusters, movements = build_movements(clusters)
    write_movements(movements, grouped_dir)
    update_grouped_files(clusters, grouped_dir)
    log.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cross-date movements from all grouped cluster files"
    )
    parser.add_argument(
        "--grouped", type=Path, default=GROUPED_DIR,
        help=f"Directory containing *_grouped.jsonl files (default: {GROUPED_DIR})",
    )
    parser.add_argument(
        "--range", nargs=2, metavar=("START", "END"), default=None,
        help="Only process files within this date range (YYYYMMDD YYYYMMDD)",
    )
    args = parser.parse_args()

    date_range = tuple(args.range) if args.range else None
    run(args.grouped, date_range)


if __name__ == "__main__":
    main()
