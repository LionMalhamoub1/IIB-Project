"""
run_matching.py
================
End-to-end runner that wires together the full matching pipeline:
candidate generation → feature vectors → scoring → graph → matching →
richness propagation → evaluation → output.

Design rationale
----------------
The pipeline is structured as a series of pure-function stages with explicit
data passing between them, rather than a monolithic script.  This makes each
stage independently testable and replaceable.  The stages are:

  1. Load     — read enriched reference and GDELT JSONL files
  2. Candidates — spatial + temporal filter to candidate pairs
  3. Vectors  — build normalised hydro-climate feature matrices
  4. Score    — compute weighted similarity for each candidate pair
  5. Graph    — build bipartite networkx graph
  6. Match    — max-weight bipartite matching (1-to-1 assignment)
  7. Propagate — copy reference impact fields to matched GDELT events
  8. Evaluate — precision, recall, breakdowns
  9. Write    — output enriched GDELT JSONL + JSON evaluation report

Input files
-----------
  cache/floods/reference_floods_enriched.jsonl
      Built by Builder_Reference/enrichment/build_official_reference_dataset.py
      Contains all reference events with all hydro-climate enrichment fields.

  cache/gdelt/gdelt_floods_enriched.jsonl
      Built by Builder_Reference/enrichment/run_gdelt_enrichment.py
      Contains all GDELT consolidated flood events with hydro-climate enrichment.

Output files
------------
  cache/matching/gdelt_floods_matched.jsonl
      GDELT events with reference fields propagated for matched events.
      Each record has matched=True/False and, if matched, ref_* fields.

  cache/matching/evaluation_report.json
      Precision, recall, breakdowns by year/source, score distribution.

  cache/matching/match_graph.gpickle
      The full bipartite graph (networkx format) for downstream visualisation
      and analysis.

Usage
-----
    python -m database_matching.run_matching
    python -m database_matching.run_matching --max-km 200 --threshold 0.45
"""

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so Builder_Matching imports work
# whether this file is run directly or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT   = Path(__file__).resolve().parents[2]

REFERENCE_PATH = ROOT / "Builder_Reference" / "outputs" / "reference_floods_enriched.jsonl"
GDELT_PATH     = ROOT / "Builder_GDELT" / "results" / "enriched_floods" / "all_floods_enriched.jsonl"
OUTPUT_DIR     = ROOT / "Builder_Matching" / "outputs"


def _load_jsonl(path: Path) -> list[dict]:
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            events.append(json.loads(line))
    return events


def main(
    max_km: float = 300,
    max_days: int = 14,
    threshold: float = 0.40,
    date_from: str | None = None,
    date_to:   str | None = None,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    log.info(f"Loading reference events: {REFERENCE_PATH}")
    reference_events = _load_jsonl(REFERENCE_PATH)
    log.info(f"  {len(reference_events):,} reference events")

    log.info(f"Loading GDELT events: {GDELT_PATH}")
    gdelt_events = _load_jsonl(GDELT_PATH)
    # Filter to flood events only
    gdelt_events = [e for e in gdelt_events if e.get("disruption_type") == "flood"]
    log.info(f"  {len(gdelt_events):,} GDELT flood events")

    # Optional date window filter (applied to both datasets)
    if date_from or date_to:
        def _date(e: dict) -> str:
            return (e.get("date_start") or e.get("event_date") or "")[:10]

        before_g = len(gdelt_events)
        before_r = len(reference_events)
        if date_from:
            gdelt_events      = [e for e in gdelt_events      if _date(e) >= date_from]
            reference_events  = [e for e in reference_events  if (e.get("date_start") or "")[:10] >= date_from]
        if date_to:
            gdelt_events      = [e for e in gdelt_events      if _date(e) <= date_to]
            reference_events  = [e for e in reference_events  if (e.get("date_start") or "")[:10] <= date_to]
        log.info(
            f"  Date filter {date_from or '*'} -> {date_to or '*'}: "
            f"GDELT {before_g} -> {len(gdelt_events)} | "
            f"Reference {before_r} -> {len(reference_events)}"
        )

    # ------------------------------------------------------------------
    # 2. Candidate generation
    # ------------------------------------------------------------------
    from Builder_Matching.matching.helper_scripts.candidate_generation import generate_candidates
    log.info(f"Generating candidates (max_km={max_km}, max_days={max_days})...")
    candidates = generate_candidates(gdelt_events, reference_events, max_km, max_days)
    log.info(f"  {len(candidates):,} candidate pairs from {len(gdelt_events):,} x {len(reference_events):,}")

    # ------------------------------------------------------------------
    # 3. Feature vectors
    # ------------------------------------------------------------------
    from Builder_Matching.matching.helper_scripts.feature_vectors import build_feature_index, compute_medians
    log.info("Building feature vectors...")
    all_events = gdelt_events + reference_events
    combined_medians = compute_medians(all_events)
    gdelt_matrix, _ = build_feature_index(gdelt_events, combined_medians)
    ref_matrix,   _ = build_feature_index(reference_events, combined_medians)
    log.info(f"  Feature matrix shapes: gdelt={gdelt_matrix.shape}, ref={ref_matrix.shape}")

    # ------------------------------------------------------------------
    # 4. Scoring
    # ------------------------------------------------------------------
    from Builder_Matching.matching.helper_scripts.scoring import score_all_candidates
    log.info("Scoring candidate pairs...")
    scored_pairs = score_all_candidates(
        gdelt_events, reference_events, gdelt_matrix, ref_matrix, candidates
    )
    log.info(f"  {len(scored_pairs):,} scored pairs")

    # ------------------------------------------------------------------
    # 5. Graph
    # ------------------------------------------------------------------
    from Builder_Matching.matching.helper_scripts.graph_builder import build_graph, graph_summary
    log.info("Building bipartite graph...")
    G = build_graph(gdelt_events, reference_events, scored_pairs)
    summary = graph_summary(G)
    log.info(f"  Graph: {summary['edges']:,} edges | "
             f"GDELT connected: {summary['gdelt_connected']}/{summary['gdelt_nodes']} | "
             f"Ref connected: {summary['ref_connected']}/{summary['reference_nodes']}")

    # ------------------------------------------------------------------
    # 6 & 7. Matching + richness propagation
    # ------------------------------------------------------------------
    from Builder_Matching.matching.helper_scripts.matching import run_matching
    log.info(f"Running max-weight matching (threshold={threshold})...")
    enriched_gdelt, match_index = run_matching(G, gdelt_events, reference_events, threshold)
    n_matched = sum(1 for e in enriched_gdelt if e.get("matched"))
    log.info(f"  Matched: {n_matched:,} / {len(gdelt_events):,} GDELT events")

    # ------------------------------------------------------------------
    # 8. Evaluation
    # ------------------------------------------------------------------
    from Builder_Matching.matching.helper_scripts.evaluation import (
        compute_precision_recall, breakdown_by_year,
        breakdown_by_source, score_distribution,
    )
    log.info("Computing evaluation metrics...")
    metrics = {
        "overall":          compute_precision_recall(gdelt_events, reference_events, match_index),
        "by_year":          breakdown_by_year(gdelt_events, reference_events, match_index),
        "by_source":        breakdown_by_source(reference_events, match_index),
        "score_distribution": score_distribution(scored_pairs, match_index),
        "graph_summary":    summary,
        "parameters": {
            "max_km": max_km, "max_days": max_days, "threshold": threshold,
        },
    }

    log.info(
        f"  Precision: {metrics['overall']['precision']:.3f} | "
        f"Recall: {metrics['overall']['recall']:.3f} | "
        f"F1: {metrics['overall']['f1']:.3f}"
    )

    # ------------------------------------------------------------------
    # 9. Write outputs
    # ------------------------------------------------------------------
    matched_path = OUTPUT_DIR / "gdelt_floods_matched.jsonl"
    with matched_path.open("w", encoding="utf-8") as f:
        for event in enriched_gdelt:
            f.write(json.dumps(event, default=str) + "\n")
    log.info(f"Matched GDELT written: {matched_path}")

    eval_path = OUTPUT_DIR / "evaluation_report.json"
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info(f"Evaluation report written: {eval_path}")

    # Write match_index.json — positional mapping gdelt_idx → ref_idx.
    # Builder_Combined/build_combined.py uses this to join the two datasets
    # without re-running the matching algorithm.
    index_path = OUTPUT_DIR / "match_index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in match_index.items()}, f)
    log.info(f"Match index written: {index_path} ({len(match_index)} pairs)")

    try:
        import pickle
        graph_path = OUTPUT_DIR / "match_graph.pkl"
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        log.info(f"Graph written: {graph_path}")
    except Exception as e:
        log.warning(f"Could not write graph file: {e}")

    log.info("Done.")
    return metrics


if __name__ == "__main__":
    # To run directly from the IDE, set parameters here and run the file.
    # To run from the terminal with arguments, use:
    #   python -m Builder_Matching.matching.run_matching --date-from 2018-01-01 --date-to 2018-02-28
    DATE_FROM = "2018-01-01"
    DATE_TO   = "2020-07-31"
    main(max_km=300, max_days=14, threshold=0.40, date_from=DATE_FROM, date_to=DATE_TO)
