"""
Protest-episode clustering and duration-estimation pipeline.

Usage
-----
    python -m src.pipeline --input articles.csv --out-format parquet

Run with --help for full argument list.
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd

# Resolve the src/ directory so sibling imports work regardless of CWD.
_SRC_DIR  = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from clustering import ClusterConfig, run_clustering  # noqa: E402
from duration   import build_cluster_summaries        # noqa: E402
from io_utils   import load_input, save_output        # noqa: E402
from parsing    import expand_event_fields            # noqa: E402
from similarity import SimConfig                      # noqa: E402

_PROC_DIR = _ROOT_DIR / "data" / "processed"

logger = logging.getLogger(__name__)

# ── Pipeline configuration ────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    input_path:            Path
    out_format:            str   = "parquet"
    time_window_days:      int   = 14
    min_similarity:        float = 0.40
    resolve_relative_dates: bool = False
    w_location:            float = 0.40
    w_actor:               float = 0.30
    w_issue:               float = 0.20
    w_time:                float = 0.10
    min_samples:           int   = 1
    out_dir:               Path  = _PROC_DIR


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute all pipeline stages and return (article_clusters, cluster_summaries).

    Stages
    ------
    1. Load input
    2. Parse event_json → event fields
    3. Block articles by (country, protest_type_bucket)
    4. Compute pairwise distances within blocks
    5. DBSCAN clustering within blocks
    6. Estimate duration per cluster
    7. Save outputs

    Parameters
    ----------
    cfg:
        Full pipeline configuration.

    Returns
    -------
    Tuple of ``(article_clusters_df, cluster_summaries_df)``.
    """
    sim_cfg = SimConfig(
        w_location       = cfg.w_location,
        w_actor          = cfg.w_actor,
        w_issue          = cfg.w_issue,
        w_time           = cfg.w_time,
        time_window_days = cfg.time_window_days,
    )
    cluster_cfg = ClusterConfig(
        min_similarity = cfg.min_similarity,
        min_samples    = cfg.min_samples,
    )

    logger.info("=== Protest Duration Clustering Pipeline ===")
    logger.info("Input            : %s", cfg.input_path)
    logger.info("Time window      : %d days", cfg.time_window_days)
    logger.info("Min similarity   : %.2f", cfg.min_similarity)
    logger.info("Resolve relative : %s", cfg.resolve_relative_dates)
    logger.info("Weights (L/A/I/T): %.2f / %.2f / %.2f / %.2f",
                cfg.w_location, cfg.w_actor, cfg.w_issue, cfg.w_time)
    logger.info("============================================")

    # Step 1 — load
    raw_df = load_input(cfg.input_path)

    # Step 2 — parse event fields
    logger.info("Parsing event_json fields …")
    df = expand_event_fields(raw_df)
    logger.info("Event fields expanded. Columns: %d", len(df.columns))

    # Steps 3–5 — block + cluster
    logger.info("Clustering …")
    clustered = run_clustering(df, sim_cfg, cluster_cfg)

    # Step 6 — duration estimation
    logger.info("Estimating episode durations …")
    summaries = build_cluster_summaries(
        clustered, resolve_relative=cfg.resolve_relative_dates
    )

    # Build the article-cluster output table
    keep_cols = [
        "article_id", "published_date", "url", "title",
        "cluster_id", "country", "city", "protest_type",
    ]
    keep_cols = [c for c in keep_cols if c in clustered.columns]
    article_clusters = clustered[keep_cols].copy()

    # Step 7 — save
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    save_output(article_clusters, cfg.out_dir / "article_clusters",   fmt=cfg.out_format)
    save_output(summaries,        cfg.out_dir / "cluster_summaries",  fmt=cfg.out_format)

    _print_summary(article_clusters, summaries)
    return article_clusters, summaries


# ── Summary report ────────────────────────────────────────────────────────────

def _print_summary(
    articles: pd.DataFrame,
    summaries: pd.DataFrame,
) -> None:
    n_articles = len(articles)
    n_clusters = articles["cluster_id"].nunique()
    n_multi    = (summaries["n_articles"] > 1).sum() if not summaries.empty else 0

    lines = [
        "",
        "------------------------------------------",
        "  PIPELINE SUMMARY",
        "------------------------------------------",
        f"  Articles processed    : {n_articles:,}",
        f"  Clusters found        : {n_clusters:,}",
        f"  Multi-article clusters: {n_multi:,}",
    ]

    if not summaries.empty:
        med_dur   = summaries["estimated_duration_days"].median()
        max_dur   = summaries["estimated_duration_days"].max()
        high_conf = (summaries["duration_confidence"] == "high").sum()
        med_conf  = (summaries["duration_confidence"] == "medium").sum()
        low_conf  = (summaries["duration_confidence"] == "low").sum()
        lines += [
            f"  Median est. duration  : {med_dur:.0f} days",
            f"  Max est. duration     : {max_dur:.0f} days",
            f"  Confidence - high     : {high_conf:,}",
            f"             - medium   : {med_conf:,}",
            f"             - low      : {low_conf:,}",
        ]

    lines.append("------------------------------------------")
    print("\n".join(lines))


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Cluster protest articles into episodes and estimate duration.",
    )
    p.add_argument("--input",  required=True,      help="Path to input CSV or Parquet file.")
    p.add_argument("--out-dir", default=str(_PROC_DIR),
                   help=f"Output directory (default: {_PROC_DIR})")
    p.add_argument("--out-format", choices=["parquet", "csv"], default="parquet")
    p.add_argument("--time-window-days", type=int,   default=14,
                   help="Max days between articles to be blocking candidates.")
    p.add_argument("--min-similarity",   type=float, default=0.40,
                   help="Minimum composite similarity to cluster two articles together.")
    p.add_argument("--resolve-relative-dates", type=lambda x: x.lower() == "true",
                   default=False,
                   help="Attempt to resolve relative event_start_reference values.")
    p.add_argument("--w-location", type=float, default=0.40)
    p.add_argument("--w-actor",    type=float, default=0.30)
    p.add_argument("--w-issue",    type=float, default=0.20)
    p.add_argument("--w-time",     type=float, default=0.10)
    p.add_argument("--min-samples", type=int,  default=1,
                   help="DBSCAN min_samples (1 = no noise points; 2+ allows noise).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, args.log_level),
    )

    cfg = PipelineConfig(
        input_path             = Path(args.input).resolve(),
        out_format             = args.out_format,
        time_window_days       = args.time_window_days,
        min_similarity         = args.min_similarity,
        resolve_relative_dates = args.resolve_relative_dates,
        w_location             = args.w_location,
        w_actor                = args.w_actor,
        w_issue                = args.w_issue,
        w_time                 = args.w_time,
        min_samples            = args.min_samples,
        out_dir                = Path(args.out_dir).resolve(),
    )

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
