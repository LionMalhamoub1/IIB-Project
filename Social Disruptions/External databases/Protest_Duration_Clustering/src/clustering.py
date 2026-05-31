from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from similarity import SimConfig, compute_block_distance_matrix

logger = logging.getLogger(__name__)

# ── Protest-type buckets ──────────────────────────────────────────────────────
# Articles in the same bucket can be clustered together even if their
# raw protest_type strings differ.

_TYPE_BUCKETS: Dict[str, str] = {
    # Mass-movement bucket
    "protest":      "mass_movement",
    "protests":     "mass_movement",
    "demonstration":"mass_movement",
    "demonstrations":"mass_movement",
    "march":        "mass_movement",
    "marches":      "mass_movement",
    "rally":        "mass_movement",
    "rallies":      "mass_movement",
    "sit-in":       "mass_movement",
    "sit in":       "mass_movement",
    "vigil":        "mass_movement",
    "gathering":    "mass_movement",
    "demonstration/protest": "mass_movement",
    # Labour bucket
    "strike":       "labour",
    "strikes":      "labour",
    "walkout":      "labour",
    "walk-out":     "labour",
    "work stoppage":"labour",
    "general strike":"labour",
    "industrial action":"labour",
    "labor action": "labour",
    "labour action":"labour",
    "lockout":      "labour",
    # Unrest / riot bucket
    "riot":         "unrest",
    "riots":        "unrest",
    "unrest":       "unrest",
    "clashes":      "unrest",
    "clash":        "unrest",
    "confrontation":"unrest",
    # Occupation bucket
    "occupation":   "occupation",
    "blockade":     "occupation",
    "roadblock":    "occupation",
    "sit-down":     "occupation",
}

_UNKNOWN_BUCKET = "other"


def normalise_protest_type(raw: str) -> str:
    """Map a raw protest_type string to a bucket name.

    Unknown types fall back to ``"other"``.  Articles with ``protest_type=""``
    also return ``"other"`` and can cluster with any bucket (handled in
    :func:`generate_blocks` by adding them to every bucket they share a
    country with).
    """
    return _TYPE_BUCKETS.get(raw.strip().lower(), _UNKNOWN_BUCKET)


# ── Block generation ──────────────────────────────────────────────────────────

BlockKey = Tuple[str, str]  # (country, type_bucket)


def generate_blocks(df: pd.DataFrame) -> Dict[BlockKey, pd.DataFrame]:
    """Partition articles into (country, type_bucket) blocks.

    Articles with an empty country or protest_type are placed in a catch-all
    bucket so they are still clustered.  An article with no protest_type is
    placed in every bucket for its country so it can merge with any episode.
    """
    blocks: Dict[BlockKey, List[int]] = {}

    for idx, row in df.iterrows():
        country = str(row.get("country", "") or "").strip().lower() or "unknown"
        ptype   = str(row.get("protest_type", "") or "").strip().lower()
        bucket  = normalise_protest_type(ptype) if ptype else None

        if bucket is not None:
            key = (country, bucket)
            blocks.setdefault(key, []).append(idx)
        else:
            # No type → try to assign to any existing bucket for this country,
            # or create an "other" bucket.
            matched = False
            for (c, b), idxs in blocks.items():
                if c == country:
                    idxs.append(idx)
                    matched = True
            if not matched:
                blocks.setdefault((country, _UNKNOWN_BUCKET), []).append(idx)

    return {key: df.loc[sorted(set(idxs))].copy() for key, idxs in blocks.items()}


# ── Clustering config ─────────────────────────────────────────────────────────

@dataclass
class ClusterConfig:
    """Parameters for DBSCAN clustering within each block."""
    min_similarity: float = 0.40
    min_samples:    int   = 1
    max_block_warn: int   = 500


# ── Per-block clustering ──────────────────────────────────────────────────────

def _cluster_block(
    dist_matrix: np.ndarray,
    cfg: ClusterConfig,
) -> np.ndarray:
    """Run DBSCAN on a precomputed distance matrix.

    With ``min_samples=1``, all points are core points — no noise labels.
    Returns integer cluster labels starting from 0.
    """
    eps = 1.0 - cfg.min_similarity
    labels = DBSCAN(
        eps=eps,
        min_samples=cfg.min_samples,
        metric="precomputed",
    ).fit_predict(dist_matrix)
    return labels


# ── Main entry point ──────────────────────────────────────────────────────────

def run_clustering(
    df: pd.DataFrame,
    sim_cfg: SimConfig,
    cluster_cfg: ClusterConfig,
) -> pd.DataFrame:
    """Cluster all articles and return the input DataFrame with a
    ``cluster_id`` column added.

    Each article is assigned a globally unique integer cluster ID.
    Singleton clusters (single articles or articles with no high-similarity
    neighbours) receive their own unique ID.

    Parameters
    ----------
    df:
        Input DataFrame after :func:`~parsing.expand_event_fields`.
    sim_cfg:
        Similarity weights and time-window configuration.
    cluster_cfg:
        DBSCAN parameters.

    Returns
    -------
    Copy of ``df`` with columns ``cluster_id`` (int) and
    ``block_key`` (str) appended.
    """
    blocks = generate_blocks(df)
    logger.info("Generated %d blocks across %d articles.", len(blocks), len(df))

    result_map: Dict[int, int] = {}   # article index → global cluster_id
    global_offset = 0

    for (country, bucket), block_df in blocks.items():
        n = len(block_df)

        if n == 0:
            continue

        if n >= cluster_cfg.max_block_warn:
            logger.warning(
                "Block (%s, %s) has %d articles — distance computation may be slow.",
                country, bucket, n,
            )

        if n == 1:
            idx = block_df.index[0]
            result_map[idx] = global_offset
            global_offset += 1
            continue

        dist = compute_block_distance_matrix(block_df, sim_cfg)
        labels = _cluster_block(dist, cluster_cfg)

        for local_label, idx in zip(labels, block_df.index):
            if local_label == -1:
                # Noise point (only possible with min_samples > 1) — singleton
                result_map[idx] = global_offset
                global_offset += 1
            else:
                global_id = global_offset + int(local_label)
                result_map[idx] = global_id

        n_clusters = len(set(labels) - {-1})
        global_offset += n_clusters

        logger.debug(
            "Block (%s, %s): %d articles → %d clusters.",
            country, bucket, n, n_clusters,
        )

    out = df.copy()
    out["cluster_id"] = out.index.map(result_map)
    out["block_key"]  = out.index.map(
        lambda i: f"{out.at[i, 'country']}|{normalise_protest_type(str(out.at[i, 'protest_type']))}"
        if i in out.index else ""
    )

    n_unassigned = out["cluster_id"].isna().sum()
    if n_unassigned:
        # Articles that appeared in multiple blocks may have been assigned
        # multiple IDs; keep the first assignment (lowest cluster_id).
        out["cluster_id"] = out["cluster_id"].fillna(-1).astype(int)

    logger.info(
        "Clustering complete: %d articles → %d clusters.",
        len(out),
        out["cluster_id"].nunique(),
    )
    return out
