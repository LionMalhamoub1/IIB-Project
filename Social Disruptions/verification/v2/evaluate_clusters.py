# Internal cluster quality metrics (cohesion, size distribution, singleton rate, etc.).
# No ACLED/MMAD required. Pass --compare to diff two grouped directories, e.g. v1 vs v2.

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DEFAULT_GROUPED_DIR = _HERE / "output"
DEFAULT_OUT_DIR     = _HERE / "eval"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_clusters(grouped_dir: Path, date_range: tuple[str, str] | None = None
                  ) -> list[dict]:
    files = sorted(grouped_dir.glob("*_grouped.jsonl"))
    if date_range:
        start, end = date_range
        files = [f for f in files if _file_overlaps(f.stem, start, end)]

    seen: dict[str, dict] = {}
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
                if cid and (cid not in seen or
                            c.get("n_articles", 0) > seen[cid].get("n_articles", 0)):
                    seen[cid] = c

    log.info("Loaded %d clusters from %s", len(seen), grouped_dir)
    return list(seen.values())


def _file_overlaps(stem: str, start: str, end: str) -> bool:
    parts = stem.replace("_grouped", "").split("_")
    dates = [p for p in parts if p.isdigit() and len(p) == 8]
    if not dates:
        return False
    return dates[-1] >= start and dates[0] <= end


def load_movements(grouped_dir: Path) -> list[dict]:
    movements = []
    for mv_file in grouped_dir.glob("*movements*.jsonl"):
        with mv_file.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    movements.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # Deduplicate by movement_id (same movement may appear in multiple files)
    seen: dict[str, dict] = {}
    for mv in movements:
        mid = mv.get("movement_id")
        if mid and mid not in seen:
            seen[mid] = mv
    return list(seen.values())


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
def compute_metrics(clusters: list[dict], movements: list[dict], label: str) -> dict:
    if not clusters:
        return {"label": label, "n_clusters": 0}

    n_articles_list  = [c.get("n_articles", 1)   for c in clusters]
    cohesion_list    = [c.get("mean_internal_score", 1.0) for c in clusters
                        if c.get("n_articles", 1) > 1]

    # Temporal span (days)
    spans = []
    for c in clusters:
        try:
            s = (pd.Timestamp(c["event_end_date"]) - pd.Timestamp(c["event_date"])).days
            spans.append(s)
        except Exception:
            pass

    n_singletons    = sum(1 for n in n_articles_list if n == 1)
    n_multi         = sum(1 for n in n_articles_list if n > 1)
    total_articles  = sum(n_articles_list)

    by_country = defaultdict(int)
    by_type    = defaultdict(int)
    for c in clusters:
        if c.get("iso3"):
            by_country[c["iso3"]] += 1
        if c.get("disruption_type"):
            by_type[c["disruption_type"]] += 1

    # Movement quality: mean centroid sim within movements
    mv_sims = [mv.get("mean_centroid_sim", 0.0) for mv in movements
               if mv.get("mean_centroid_sim") is not None]

    return {
        "label":                label,
        "n_clusters":           len(clusters),
        "total_articles":       total_articles,
        "reduction_ratio":      round(total_articles / len(clusters), 2) if clusters else None,
        "singleton_count":      n_singletons,
        "singleton_rate":       round(n_singletons / len(clusters), 3) if clusters else None,
        "multi_article_count":  n_multi,
        "n_countries":          len(by_country),
        "n_movements":          len(movements),
        "articles_per_cluster": {
            "mean":   round(float(np.mean(n_articles_list)), 2),
            "median": float(np.median(n_articles_list)),
            "p90":    float(np.percentile(n_articles_list, 90)),
            "max":    int(np.max(n_articles_list)),
        },
        "cohesion": {
            "mean":   round(float(np.mean(cohesion_list)), 4) if cohesion_list else None,
            "median": round(float(np.median(cohesion_list)), 4) if cohesion_list else None,
            "p10":    round(float(np.percentile(cohesion_list, 10)), 4) if cohesion_list else None,
        },
        "temporal_span_days": {
            "mean":           round(float(np.mean(spans)), 2) if spans else None,
            "median":         float(np.median(spans)) if spans else None,
            "max":            int(np.max(spans)) if spans else None,
            "multi_day_pct":  round(sum(1 for s in spans if s > 0) / len(spans), 3)
                              if spans else None,
        },
        "movement_centroid_sim": {
            "mean":   round(float(np.mean(mv_sims)), 4) if mv_sims else None,
            "median": round(float(np.median(mv_sims)), 4) if mv_sims else None,
        },
        "by_disruption_type": dict(by_type),
        "top_10_countries":   dict(
            sorted(by_country.items(), key=lambda x: -x[1])[:10]
        ),
    }


def diff_metrics(m1: dict, m2: dict) -> dict:
    """Compute simple delta between two metric dicts for comparison report."""
    result = {}
    scalar_keys = [
        "n_clusters", "total_articles", "reduction_ratio",
        "singleton_rate", "n_movements", "n_countries",
    ]
    for k in scalar_keys:
        v1, v2 = m1.get(k), m2.get(k)
        if v1 is not None and v2 is not None:
            try:
                result[k] = {"v1": v1, "v2": v2, "delta": round(v2 - v1, 4)}
            except TypeError:
                pass

    # Nested scalar keys
    for parent_k, child_k in [
        ("cohesion", "mean"),
        ("temporal_span_days", "mean"),
        ("temporal_span_days", "multi_day_pct"),
        ("movement_centroid_sim", "mean"),
        ("articles_per_cluster", "mean"),
    ]:
        v1 = (m1.get(parent_k) or {}).get(child_k)
        v2 = (m2.get(parent_k) or {}).get(child_k)
        if v1 is not None and v2 is not None:
            key = f"{parent_k}.{child_k}"
            try:
                result[key] = {"v1": v1, "v2": v2, "delta": round(v2 - v1, 4)}
            except TypeError:
                pass
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Saved: %s", path)


def fig_size_distribution(
    clusters_list: list[tuple[str, list[dict]]],
    out: Path,
) -> None:
    """Histogram of articles-per-cluster, overlaid for each dataset."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    colors = ["#1D3557", "#E63946", "#457B9D"]
    bins = [1, 2, 3, 5, 10, 20, 50, 100, 500]

    for ax_idx, (label, clusters) in enumerate(clusters_list[:2]):
        counts = [c.get("n_articles", 1) for c in clusters]
        ax = axes[ax_idx]
        ax.hist(counts, bins=bins, color=colors[ax_idx], alpha=0.8, edgecolor="white")
        ax.set_xscale("log")
        ax.set_xlabel("Articles per cluster (log scale)")
        ax.set_ylabel("Number of clusters")
        ax.set_title(f"{label}\n"
                     f"n={len(clusters)} clusters | "
                     f"singleton rate={sum(1 for n in counts if n == 1)/len(counts):.0%}")
        ax.axvline(np.median(counts), color="red", lw=1.5, linestyle="--",
                   label=f"median={np.median(counts):.1f}")
        ax.legend()

    fig.suptitle("Cluster size distribution: articles per canonical event", y=1.02)
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_cohesion_distribution(
    clusters_list: list[tuple[str, list[dict]]],
    out: Path,
) -> None:
    """Distribution of mean internal similarity for multi-article clusters."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#1D3557", "#E63946"]

    for i, (label, clusters) in enumerate(clusters_list):
        cohesion = [
            c.get("mean_internal_score", 1.0)
            for c in clusters
            if c.get("n_articles", 1) > 1
        ]
        if not cohesion:
            continue
        ax.hist(cohesion, bins=30, alpha=0.6, color=colors[i % len(colors)],
                label=f"{label} (n={len(cohesion)} multi-article clusters)")
        ax.axvline(np.mean(cohesion), color=colors[i % len(colors)], lw=2,
                   linestyle="--")

    ax.set_xlabel("Mean internal similarity score")
    ax.set_ylabel("Number of clusters")
    ax.set_title("Cluster cohesion: internal similarity of multi-article clusters\n"
                 "(higher = tighter, lower = possible over-merge)")
    ax.axvline(0.45, color="grey", lw=1, linestyle=":", label="edge threshold (0.45)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_temporal_span(
    clusters_list: list[tuple[str, list[dict]]],
    out: Path,
) -> None:
    """Distribution of event duration (event_end_date - event_date in days)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#1D3557", "#E63946"]

    for i, (label, clusters) in enumerate(clusters_list):
        spans = []
        for c in clusters:
            try:
                s = (pd.Timestamp(c["event_end_date"]) - pd.Timestamp(c["event_date"])).days
                spans.append(s)
            except Exception:
                pass
        if not spans:
            continue
        multi_day_pct = sum(1 for s in spans if s > 0) / len(spans)
        ax.hist(spans, bins=range(0, min(max(spans) + 2, 32)),
                alpha=0.6, color=colors[i % len(colors)],
                label=f"{label} | multi-day={multi_day_pct:.0%} | median={np.median(spans):.1f}d")

    ax.set_xlabel("Event span (days)")
    ax.set_ylabel("Number of events")
    ax.set_title("Event temporal span distribution\n"
                 "V2 should show more multi-day events due to global clustering")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_country_coverage(
    clusters_list: list[tuple[str, list[dict]]],
    out: Path,
    top_n: int = 30,
) -> None:
    """Events per country, comparing v1 and v2."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(clusters_list), figsize=(8 * len(clusters_list), 7),
                             sharey=False)
    if len(clusters_list) == 1:
        axes = [axes]
    colors = ["#1D3557", "#E63946"]

    for ax_idx, (label, clusters) in enumerate(clusters_list):
        by_country: dict[str, int] = defaultdict(int)
        for c in clusters:
            if c.get("iso3"):
                by_country[c["iso3"]] += 1
        top = sorted(by_country.items(), key=lambda x: -x[1])[:top_n]
        countries = [t[0] for t in top]
        counts    = [t[1] for t in top]
        ax = axes[ax_idx]
        ax.barh(countries, counts, color=colors[ax_idx % len(colors)], alpha=0.85)
        ax.set_xlabel("Number of events")
        ax.set_title(f"{label}\n{len(by_country)} countries total")
        ax.invert_yaxis()

    fig.suptitle(f"Events per country (top {top_n})", y=1.02)
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_movement_sim_distribution(
    movements_list: list[tuple[str, list[dict]]],
    out: Path,
) -> None:
    """Distribution of mean centroid similarity within movements."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#1D3557", "#E63946"]

    for i, (label, movements) in enumerate(movements_list):
        sims = [mv.get("mean_centroid_sim", 0.0) for mv in movements
                if mv.get("mean_centroid_sim") is not None]
        if not sims:
            continue
        ax.hist(sims, bins=20, alpha=0.6, color=colors[i % len(colors)],
                label=f"{label} (n={len(sims)} movements, mean={np.mean(sims):.2f})")

    ax.set_xlabel("Mean centroid cosine similarity within movement")
    ax.set_ylabel("Count")
    ax.set_title("Movement cohesion: how similar are clusters within the same movement?\n"
                 "V2 threshold is 0.60; V1 was 0.72")
    ax.axvline(0.60, color="orange", lw=1.5, linestyle="--", label="V2 threshold (0.60)")
    ax.axvline(0.72, color="grey",   lw=1.5, linestyle=":",  label="V1 threshold (0.72)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_singleton_rate_by_country(
    clusters_list: list[tuple[str, list[dict]]],
    out: Path,
    top_n: int = 25,
) -> None:
    """Singleton rate (1-article clusters) per country — high rate = noisy or sparse."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(clusters_list), figsize=(8 * len(clusters_list), 7),
                             sharey=False)
    if len(clusters_list) == 1:
        axes = [axes]
    colors = ["#1D3557", "#E63946"]

    for ax_idx, (label, clusters) in enumerate(clusters_list):
        by_country: dict[str, list[int]] = defaultdict(list)
        for c in clusters:
            if c.get("iso3"):
                by_country[c["iso3"]].append(c.get("n_articles", 1))

        rates = {
            iso3: sum(1 for n in ns if n == 1) / len(ns)
            for iso3, ns in by_country.items()
            if len(ns) >= 3   # min 3 events to be meaningful
        }
        sorted_rates = sorted(rates.items(), key=lambda x: -x[1])[:top_n]
        countries = [r[0] for r in sorted_rates]
        vals      = [r[1] for r in sorted_rates]

        ax = axes[ax_idx]
        ax.barh(countries, vals, color=colors[ax_idx % len(colors)], alpha=0.85)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Singleton rate (1-article clusters)")
        ax.set_title(f"{label}\nCountries with highest singleton rate")
        ax.axvline(0.5, color="grey", lw=1, linestyle=":")
        ax.invert_yaxis()

    fig.suptitle("Singleton rate by country (high = sparse coverage or noisy events)",
                 y=1.02)
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(
    grouped_dir:  Path,
    compare_dir:  Path | None,
    out_dir:      Path,
    date_range:   tuple[str, str] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load primary dataset
    clusters_primary   = load_clusters(grouped_dir, date_range)
    movements_primary  = load_movements(grouped_dir)
    metrics_primary    = compute_metrics(clusters_primary, movements_primary,
                                         label=grouped_dir.name)

    datasets         = [(grouped_dir.name, clusters_primary)]
    movement_sets    = [(grouped_dir.name, movements_primary)]
    all_metrics      = {grouped_dir.name: metrics_primary}

    # Load comparison dataset if provided
    if compare_dir is not None:
        clusters_compare   = load_clusters(compare_dir, date_range)
        movements_compare  = load_movements(compare_dir)
        metrics_compare    = compute_metrics(clusters_compare, movements_compare,
                                             label=compare_dir.name)
        datasets.append((compare_dir.name, clusters_compare))
        movement_sets.append((compare_dir.name, movements_compare))
        all_metrics[compare_dir.name] = metrics_compare

    # Print summary
    log.info("=" * 60)
    for label, m in all_metrics.items():
        log.info("  [%s]", label)
        log.info("    Clusters       : %d", m.get("n_clusters", 0))
        log.info("    Total articles : %d", m.get("total_articles", 0))
        log.info("    Reduction ratio: %.2fx", m.get("reduction_ratio") or 0)
        log.info("    Singleton rate : %.1f%%", (m.get("singleton_rate") or 0) * 100)
        log.info("    Multi-day pct  : %.1f%%",
                 ((m.get("temporal_span_days") or {}).get("multi_day_pct") or 0) * 100)
        log.info("    Movements      : %d", m.get("n_movements", 0))
        log.info("    Mean cohesion  : %.3f", (m.get("cohesion") or {}).get("mean") or 0)
    if len(all_metrics) == 2:
        keys = list(all_metrics.keys())
        deltas = diff_metrics(all_metrics[keys[0]], all_metrics[keys[1]])
        log.info("  --- Deltas (v2 - v1) ---")
        for k, v in deltas.items():
            log.info("    %-40s : %+.4f", k, v["delta"])
    log.info("=" * 60)

    # Save metrics
    report = {
        "metrics":   all_metrics,
        "comparison": diff_metrics(
            list(all_metrics.values())[0],
            list(all_metrics.values())[1]
        ) if len(all_metrics) == 2 else None,
    }
    with (out_dir / "eval_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    log.info("Eval report: %s", out_dir / "eval_report.json")

    # Figures
    fig_size_distribution(datasets,     out_dir / "fig1_size_distribution.png")
    fig_cohesion_distribution(datasets, out_dir / "fig2_cohesion.png")
    fig_temporal_span(datasets,         out_dir / "fig3_temporal_span.png")
    fig_country_coverage(datasets,      out_dir / "fig4_country_coverage.png")
    fig_movement_sim_distribution(movement_sets, out_dir / "fig5_movement_cohesion.png")
    fig_singleton_rate_by_country(datasets, out_dir / "fig6_singleton_rate.png")

    log.info("Done. Figures in %s", out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate event cluster quality (no ACLED/MMAD required)"
    )
    parser.add_argument(
        "--grouped", type=Path, default=DEFAULT_GROUPED_DIR,
        help="Directory containing *_grouped.jsonl files to evaluate",
    )
    parser.add_argument(
        "--compare", type=Path, default=None,
        help="Optional second directory to compare against (e.g. v1 vs v2)",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Output directory for figures and report (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--range", nargs=2, metavar=("START", "END"), default=None,
        help="Restrict to date range YYYYMMDD YYYYMMDD",
    )
    args = parser.parse_args()

    date_range = tuple(args.range) if args.range else None
    run(args.grouped, args.compare, args.out, date_range)


if __name__ == "__main__":
    main()
