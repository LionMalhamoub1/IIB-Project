"""
run_gdelt_graph_matching.py
===========================
Graph-based event-level matching of GDELT social events against ACLED.

Builds a weighted bipartite graph (GDELT ↔ ACLED) and finds the globally
optimal 1-to-1 assignment via max-weight bipartite matching.

Run deduplicate_gdelt_social.py first.  This script defaults to the deduped
JSONL; pass --gdelt to override.

Scoring
-------
Each candidate pair (g, a) with the same ISO-3 and event_date within ±MAX_DAYS:

    score = W_T * temporal(g, a)
          + W_L * location(g, a)
          + W_E * event_type(g, a)

    temporal(g, a)   = 1 - |days_apart| / MAX_DAYS          ∈ [0, 1]
    location(g, a)   = fuzzy(g.subloc, a.admin1 / a.location) ∈ [0, 1]
                       0.3 flat when g has no sub-national info
    event_type(g, a) = TYPE_COMPAT lookup                    ∈ [0, 1]

Weights W_T=0.40, W_L=0.40, W_E=0.20.

Matching
--------
scipy.optimize.linear_sum_assignment (Hungarian algorithm).
Only matches above MATCH_THRESHOLD are kept.

Outputs
-------
  verification/gdelt_graph_matched.parquet
  verification/gdelt_graph_matched.csv
  verification/graph_match_report.json
  verification/figures/
      fig_graph_score_distribution.png
      fig_graph_score_components.png
      fig_graph_match_rate_by_type.png
      fig_graph_match_rate_by_year.png
      fig_graph_match_rate_by_country.png

Usage
-----
  python "Social Disruptions/verification/run_gdelt_graph_matching.py"
  python "Social Disruptions/verification/run_gdelt_graph_matching.py" \\
      --gdelt  path/to/gdelt_social_deduped.jsonl \\
      --acled  path/to/acled_raw_events_dir \\
      --window 7 --threshold 0.35
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
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
# Paths — default to deduped file produced by deduplicate_gdelt_social.py
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SD   = _HERE.parent
_ROOT = _SD.parent

DEFAULT_GDELT_PATH  = _HERE / "gdelt_social_deduped.jsonl"
DEFAULT_ACLED_PANEL = _SD  / "Likelihood_modelling_social" / "data" / "processed" / "acled_country_day_2017_2025.parquet"
DEFAULT_ACLED_RAW   = _SD  / "External databases" / "ACLED" / "data" / "raw" / "events"
DEFAULT_OUT_DIR     = _HERE

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SOCIAL_TYPES:         frozenset[str] = frozenset({"protests", "labour_strike"})
CONFIDENCE_THRESHOLD: float          = 0.6
MAX_DAYS:             int            = 7
MATCH_THRESHOLD:      float          = 0.35
NO_SUBLOC_CREDIT:     float          = 0.30   # location score when GDELT has no subloc

W_TEMPORAL:   float = 0.40
W_LOCATION:   float = 0.40
W_EVENT_TYPE: float = 0.20

TYPE_COMPAT: dict[tuple[str, str], float] = {
    ("protests",      "Protests"):                   1.0,
    ("protests",      "Riots"):                      0.6,
    ("protests",      "Violence against civilians"): 0.2,
    ("protests",      "Strategic developments"):     0.1,
    ("labour_strike", "Protests"):                   0.8,
    ("labour_strike", "Riots"):                      0.4,
    ("labour_strike", "Strategic developments"):     0.2,
}

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_temporal(days_apart: float, max_days: int) -> float:
    return max(0.0, 1.0 - abs(days_apart) / max_days)


def score_location(gdelt_subloc: str, acled_admin1: str, acled_location: str) -> float:
    """
    Fuzzy match GDELT sub-national string against ACLED admin1 and location.

    When GDELT has no sub-national info, returns NO_SUBLOC_CREDIT (not 0).
    This reflects that country+date is still meaningful evidence; we simply
    cannot evaluate the location dimension.  The flat credit is intentionally
    below 0.5 so that a no-subloc GDELT event can only be matched if temporal
    and type scores are also strong.
    """
    if not gdelt_subloc:
        return NO_SUBLOC_CREDIT
    return max(
        fuzzy_similarity(gdelt_subloc, acled_admin1 or ""),
        fuzzy_similarity(gdelt_subloc, acled_location or ""),
    )


def score_event_type(gdelt_type: str, acled_event_type: str) -> float:
    return TYPE_COMPAT.get((gdelt_type, acled_event_type), 0.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gdelt_social(path: Path) -> pd.DataFrame:
    """
    Load GDELT social events from either:
      • the deduped JSONL  (preferred — one record per real-world event)
      • the raw consolidated JSONL (fallback)

    Deduped records have cluster_id and n_source_articles; raw records don't.
    Both formats are handled transparently.
    """
    log.info("Loading GDELT social events: %s", path)
    records = []
    skipped = defaultdict(int)

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

            details = e.get("details") or e.get("extras") or {}
            raw_date = e.get("event_date") or e.get("publish_date")
            try:
                date = pd.Timestamp(raw_date).normalize()
            except Exception:
                skipped["date"] += 1
                continue

            records.append({
                "gdelt_id":        e.get("cluster_id", len(records)),
                "event_date":      date,
                "disruption_type": e["disruption_type"],
                "iso3":            e.get("iso3") or extract_iso3(e),
                "subloc":          e.get("subloc") or extract_subloc(e),
                "confidence":      float(e.get("confidence") or 0.0),
                "num_articles":    int(
                                       e.get("n_source_articles")
                                       or e.get("num_articles") or 1
                                   ),
                "sector":          details.get("sector") or e.get("sector"),
                "issue":           details.get("issue")  or e.get("issue"),
            })

    if not records:
        log.warning("No qualifying GDELT social events found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["event_date"]).reset_index(drop=True)
    df["gdelt_idx"] = df.index   # stable integer index used in matching

    n_res = df["iso3"].notna().sum()
    log.info("GDELT: %d events | country resolved: %d / %d (%.1f%%)",
             len(df), n_res, len(df), 100 * n_res / len(df))
    return df


def load_acled_events(raw_events_dir: Path) -> pd.DataFrame:
    """
    Load ACLED protest/riot events with sub-national location fields.
    Returns: acled_idx (int), acled_id, event_date, iso3, event_type,
             admin1, location
    """
    parquet_files = sorted(raw_events_dir.glob("iso3=*/year=*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No raw ACLED parquet files found in {raw_events_dir}.\n"
            "Run the ACLED pull pipeline first."
        )

    log.info("Loading %d raw ACLED parquet partitions...", len(parquet_files))
    frames = []
    for f in parquet_files:
        try:
            frames.append(pd.read_parquet(
                f,
                columns=["event_id_cnty", "event_date", "iso3",
                         "event_type", "admin1", "location"],
            ))
        except Exception as exc:
            log.warning("Skipping %s: %s", f.name, exc)

    if not frames:
        raise RuntimeError("All ACLED parquet partitions failed to load.")

    events = pd.concat(frames, ignore_index=True)
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce").dt.normalize()
    events["iso3"]       = events["iso3"].str.strip().str.upper()
    events = events.dropna(subset=["event_date", "iso3"])

    relevant = {"Protests", "Riots", "Strategic developments"}
    events   = events[events["event_type"].isin(relevant)].copy()
    events   = events.rename(columns={"event_id_cnty": "acled_id"})
    events["acled_id"]  = events["acled_id"].astype(str)
    events["admin1"]    = events["admin1"].fillna("").str.lower()
    events["location"]  = events["location"].fillna("").str.lower()
    events              = events.reset_index(drop=True)
    events["acled_idx"] = events.index

    log.info("ACLED: %d protest/riot events across %d countries",
             len(events), events["iso3"].nunique())
    return events[["acled_idx", "acled_id", "event_date", "iso3",
                   "event_type", "admin1", "location"]]


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------

def build_candidate_pairs(
    gdelt: pd.DataFrame,
    acled: pd.DataFrame,
    max_days: int,
) -> list[tuple[int, int, float, float, float, float]]:
    """
    Generate (gdelt_idx, acled_idx, t_score, l_score, e_score, composite)
    for all candidate pairs: same iso3, |date_diff| ≤ max_days.

    ACLED index is built with groupby (vectorised) rather than itertuples,
    which is substantially faster for the large ACLED dataset.
    Pre-cached numpy arrays are used for O(1) row lookups in the inner loop.
    """
    log.info("Building candidate pairs (±%d days)...", max_days)

    # Build (iso3, date) → [acled_idx, ...] index via groupby — fast
    acled_index: dict[tuple, list[int]] = {
        key: list(grp["acled_idx"])
        for key, grp in acled.groupby(["iso3", "event_date"])
    }

    # Pre-cache ACLED columns as numpy arrays for O(1) scalar access
    acled_admin1    = acled["admin1"].to_numpy()
    acled_location  = acled["location"].to_numpy()
    acled_event_type = acled["event_type"].to_numpy()
    acled_date      = acled["event_date"].to_numpy()

    offsets = [pd.Timedelta(days=d) for d in range(-max_days, max_days + 1)]
    pairs: list[tuple[int, int, float, float, float, float]] = []

    for row in gdelt.itertuples():
        if pd.isna(row.iso3):
            continue
        gi = row.gdelt_idx
        for off in offsets:
            key = (row.iso3, row.event_date + off)
            for ai in acled_index.get(key, []):
                days_off = (row.event_date - pd.Timestamp(acled_date[ai])).days
                t = score_temporal(days_off, max_days)
                l = score_location(row.subloc, acled_admin1[ai], acled_location[ai])
                e = score_event_type(row.disruption_type, acled_event_type[ai])
                s = W_TEMPORAL * t + W_LOCATION * l + W_EVENT_TYPE * e
                pairs.append((gi, ai, t, l, e, round(s, 4)))

    log.info("Candidate pairs: %d  (%d GDELT × %d ACLED)",
             len(pairs), len(gdelt), len(acled))
    return pairs


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph(
    gdelt: pd.DataFrame,
    acled: pd.DataFrame,
    pairs: list[tuple[int, int, float, float, float, float]],
) -> "nx.Graph":
    import networkx as nx

    G = nx.Graph()
    for row in gdelt.itertuples():
        G.add_node(f"G_{row.gdelt_idx}", bipartite=0,
                   iso3=row.iso3, date=str(row.event_date.date()),
                   dtype=row.disruption_type)
    for row in acled.itertuples():
        G.add_node(f"A_{row.acled_idx}", bipartite=1,
                   iso3=row.iso3, date=str(row.event_date.date()),
                   event_type=row.event_type)
    for gi, ai, t, l, e, s in pairs:
        G.add_edge(f"G_{gi}", f"A_{ai}",
                   temporal=t, location=l, event_type=e, score=s)

    log.info("Graph: %d nodes | %d edges",
             G.number_of_nodes(), G.number_of_edges())
    return G


# ---------------------------------------------------------------------------
# Max-weight bipartite matching
# ---------------------------------------------------------------------------

def run_matching(
    pairs: list[tuple[int, int, float, float, float, float]],
    n_gdelt: int,
    threshold: float,
) -> dict[int, dict]:
    """
    1-to-1 max-weight assignment via the Hungarian algorithm.
    Returns {gdelt_idx: match_info} for matches above threshold.
    """
    from scipy.optimize import linear_sum_assignment

    if not pairs:
        log.warning("No candidate pairs.")
        return {}

    above = [(gi, ai, t, l, e, s) for gi, ai, t, l, e, s in pairs if s >= threshold]
    if not above:
        log.warning("No pairs above threshold %.2f.", threshold)
        return {}

    gdelt_ids = sorted({gi for gi, *_ in above})
    acled_ids = sorted({ai for _, ai, *_ in above})
    g_map     = {g: i for i, g in enumerate(gdelt_ids)}
    a_map     = {a: i for i, a in enumerate(acled_ids)}

    cost         = np.zeros((len(gdelt_ids), len(acled_ids)))
    score_lookup: dict[tuple[int, int], tuple] = {}
    for gi, ai, t, l, e, s in above:
        r, c = g_map[gi], a_map[ai]
        if s > cost[r, c]:              # keep highest score for duplicate pairs
            cost[r, c]            = s
            score_lookup[(gi, ai)] = (t, l, e, s)

    row_ind, col_ind = linear_sum_assignment(-cost)

    matches: dict[int, dict] = {}
    for r, c in zip(row_ind, col_ind):
        gi = gdelt_ids[r]
        ai = acled_ids[c]
        s  = cost[r, c]
        if s < threshold:
            continue
        t, l, e, _ = score_lookup[(gi, ai)]
        matches[gi] = {
            "acled_idx":      ai,
            "score":          round(float(s), 4),
            "temporal_score": round(float(t), 4),
            "location_score": round(float(l), 4),
            "type_score":     round(float(e), 4),
        }

    log.info("Matched %d / %d GDELT events (threshold=%.2f)",
             len(matches), n_gdelt, threshold)
    return matches


# ---------------------------------------------------------------------------
# Annotate
# ---------------------------------------------------------------------------

def annotate(
    gdelt: pd.DataFrame,
    acled: pd.DataFrame,
    matches: dict[int, dict],
) -> pd.DataFrame:
    """
    Append match columns to the GDELT DataFrame.
    Uses vectorised assignment rather than row-by-row iteration.
    """
    # Pre-cache ACLED columns for fast lookup
    acled_id_arr       = acled["acled_id"].to_numpy()
    acled_date_arr     = acled["event_date"].to_numpy()
    acled_admin1_arr   = acled["admin1"].to_numpy()
    acled_location_arr = acled["location"].to_numpy()
    acled_type_arr     = acled["event_type"].to_numpy()

    n = len(gdelt)
    graph_matched        = np.where(gdelt["iso3"].notna(), False, np.nan).astype(object)
    graph_score          = np.full(n, np.nan)
    graph_temporal_score = np.full(n, np.nan)
    graph_location_score = np.full(n, np.nan)
    graph_type_score     = np.full(n, np.nan)
    graph_acled_id       = np.full(n, None, dtype=object)
    graph_acled_date     = np.full(n, pd.NaT, dtype="datetime64[ns]")
    graph_acled_admin1   = np.full(n, None, dtype=object)
    graph_acled_location = np.full(n, None, dtype=object)
    graph_acled_type     = np.full(n, None, dtype=object)

    for gi, m in matches.items():
        ai = m["acled_idx"]
        graph_matched[gi]        = True
        graph_score[gi]          = m["score"]
        graph_temporal_score[gi] = m["temporal_score"]
        graph_location_score[gi] = m["location_score"]
        graph_type_score[gi]     = m["type_score"]
        graph_acled_id[gi]       = acled_id_arr[ai]
        graph_acled_date[gi]     = acled_date_arr[ai]
        graph_acled_admin1[gi]   = acled_admin1_arr[ai]
        graph_acled_location[gi] = acled_location_arr[ai]
        graph_acled_type[gi]     = acled_type_arr[ai]

    gdelt = gdelt.copy()
    gdelt["graph_matched"]        = graph_matched
    gdelt["graph_score"]          = graph_score
    gdelt["graph_temporal_score"] = graph_temporal_score
    gdelt["graph_location_score"] = graph_location_score
    gdelt["graph_type_score"]     = graph_type_score
    gdelt["graph_acled_id"]       = graph_acled_id
    gdelt["graph_acled_date"]     = pd.to_datetime(graph_acled_date)
    gdelt["graph_acled_admin1"]   = graph_acled_admin1
    gdelt["graph_acled_location"] = graph_acled_location
    gdelt["graph_acled_type"]     = graph_acled_type
    return gdelt


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(gdelt: pd.DataFrame, matches: dict, threshold: float) -> dict:
    resolved   = gdelt["iso3"].notna()
    n_resolved = int(resolved.sum())
    sub        = gdelt[resolved]

    def _stats(col: str) -> dict:
        n = int(sub[col].notna().sum())
        m = int(sub[col].dropna().astype(bool).sum())
        return {"n": n, "matched": m,
                "rate": round(m / n, 4) if n > 0 else None}

    by_type: dict = {}
    for dtype, grp in sub.groupby("disruption_type"):
        n = int(grp["graph_matched"].notna().sum())
        m = int(grp["graph_matched"].dropna().astype(bool).sum())
        sc = grp["graph_score"].dropna()
        by_type[dtype] = {
            "n": n, "matched": m,
            "rate":       round(m / n, 4) if n > 0 else None,
            "mean_score": round(float(sc.mean()), 4) if not sc.empty else None,
        }

    by_year: dict = {}
    sub2 = sub.copy()
    sub2["year"] = sub2["event_date"].dt.year
    for year, grp in sub2.groupby("year"):
        n = int(grp["graph_matched"].notna().sum())
        m = int(grp["graph_matched"].dropna().astype(bool).sum())
        by_year[int(year)] = {"n": n, "matched": m,
                               "rate": round(m / n, 4) if n > 0 else None}

    sc_all = gdelt["graph_score"].dropna()
    return {
        "parameters": {
            "max_days": MAX_DAYS, "match_threshold": threshold,
            "weights": {"temporal": W_TEMPORAL, "location": W_LOCATION,
                        "event_type": W_EVENT_TYPE},
            "no_subloc_credit": NO_SUBLOC_CREDIT,
        },
        "total_events":       int(len(gdelt)),
        "country_resolved":   n_resolved,
        "country_resolution_pct": round(n_resolved / len(gdelt) * 100, 1),
        "overall":            {**_stats("graph_matched"),
                               "mean_score":   round(float(sc_all.mean()), 4) if not sc_all.empty else None,
                               "median_score": round(float(sc_all.median()), 4) if not sc_all.empty else None},
        "by_disruption_type": by_type,
        "by_year":            by_year,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Saved: %s", path)


def fig_score_distribution(gdelt: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    scores = gdelt[gdelt["graph_matched"] == True]["graph_score"].dropna()
    if scores.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores, bins=20, color="#457B9D", edgecolor="white", alpha=0.85)
    ax.axvline(MATCH_THRESHOLD, color="#E63946", lw=1.5, linestyle="--",
               label=f"Threshold ({MATCH_THRESHOLD})")
    ax.set_xlabel("Composite match score")
    ax.set_ylabel("Matched GDELT events")
    ax.set_title("Distribution of graph match scores (GDELT ↔ ACLED)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_score_components(gdelt: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    matched = gdelt[gdelt["graph_matched"] == True].copy()
    if matched.empty:
        return
    colors = {"protests": "#457B9D", "labour_strike": "#E63946"}
    fig, ax = plt.subplots(figsize=(6, 5))
    for dtype, grp in matched.groupby("disruption_type"):
        ax.scatter(grp["graph_temporal_score"], grp["graph_location_score"],
                   c=colors.get(dtype, "grey"), label=dtype, alpha=0.6, s=20)
    ax.set_xlabel("Temporal score  (1 = same day)")
    ax.set_ylabel("Location score  (fuzzy sub-national match)")
    ax.set_title("Match score components by disruption type")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_match_rate_by_type(gdelt: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    resolved = gdelt[gdelt["iso3"].notna()]
    types    = sorted(resolved["disruption_type"].unique())
    rates    = [resolved[resolved["disruption_type"] == t]["graph_matched"]
                .dropna().astype(bool).mean() for t in types]
    counts   = [resolved[resolved["disruption_type"] == t]["graph_matched"]
                .notna().sum() for t in types]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(types, rates, color=["#457B9D", "#E63946"], alpha=0.85)
    for bar, r, n in zip(bars, rates, counts):
        if not np.isnan(r):
            ax.text(bar.get_x() + bar.get_width() / 2, r + 0.01,
                    f"{r:.0%}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Graph match rate")
    ax.set_title("GDELT graph match rate by disruption type")
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_match_rate_by_year(gdelt: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    resolved = gdelt[gdelt["iso3"].notna()].copy()
    resolved["year"] = resolved["event_date"].dt.year
    rates  = resolved.groupby("year")["graph_matched"].apply(
                 lambda s: s.dropna().astype(bool).mean())
    counts = resolved.groupby("year")["graph_matched"].apply(
                 lambda s: s.notna().sum())
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rates.index, rates.values, marker="o", color="#457B9D", lw=1.8)
    for x, y, n in zip(rates.index, rates.values, counts.values):
        ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color="grey")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Year")
    ax.set_ylabel("Graph match rate")
    ax.set_title("GDELT graph match rate by year")
    ax.axhline(0.5, color="grey", lw=0.8, linestyle=":", alpha=0.6)
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_match_rate_by_country(gdelt: pd.DataFrame, top_n: int, out: Path) -> None:
    import matplotlib.pyplot as plt
    resolved = gdelt[gdelt["iso3"].notna()]
    top = resolved["iso3"].value_counts()[lambda s: s >= 3].head(top_n).index
    sub = resolved[resolved["iso3"].isin(top)]
    rates = (sub.groupby("iso3")["graph_matched"]
               .apply(lambda s: s.dropna().astype(bool).mean())
               .sort_values())
    colors = ["#2A9D8F" if r >= 0.5 else "#E63946" for r in rates.values]
    fig, ax = plt.subplots(figsize=(9, max(4, len(rates) * 0.3)))
    ax.barh(rates.index, rates.values, color=colors, alpha=0.85)
    ax.set_xlim(0, 1.1)
    ax.axvline(0.5, color="grey", lw=0.8, linestyle=":")
    ax.set_xlabel("Graph match rate")
    ax.set_title(f"GDELT graph match rate by country  (top {top_n}, min 3 events)")
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    gdelt_path:  Path  = DEFAULT_GDELT_PATH,
    acled_raw:   Path  = DEFAULT_ACLED_RAW,
    out_dir:     Path  = DEFAULT_OUT_DIR,
    max_days:    int   = MAX_DAYS,
    threshold:   float = MATCH_THRESHOLD,
) -> pd.DataFrame:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Fallback to raw JSONL if deduped file not yet produced
    if not gdelt_path.exists():
        raw_fallback = _ROOT / "Builder_GDELT" / "results" / "combined" / "all_consolidated.jsonl"
        if raw_fallback.exists():
            log.warning(
                "Deduped file not found at %s.\n"
                "Falling back to raw JSONL: %s\n"
                "Run deduplicate_gdelt_social.py first for best results.",
                gdelt_path, raw_fallback,
            )
            gdelt_path = raw_fallback
        else:
            log.error("No GDELT file found.")
            sys.exit(1)

    gdelt = load_gdelt_social(gdelt_path)
    if gdelt.empty:
        log.error("No qualifying GDELT social events.")
        sys.exit(1)

    try:
        acled = load_acled_events(acled_raw)
    except (FileNotFoundError, RuntimeError) as exc:
        log.error("ACLED unavailable:\n  %s", exc)
        sys.exit(1)

    pairs   = build_candidate_pairs(gdelt, acled, max_days)
    G       = build_graph(gdelt, acled, pairs)
    matches = run_matching(pairs, n_gdelt=len(gdelt), threshold=threshold)
    gdelt   = annotate(gdelt, acled, matches)
    report  = build_report(gdelt, matches, threshold)

    log.info("=" * 60)
    log.info("GRAPH MATCH SUMMARY")
    log.info("  Events             : %d", report["total_events"])
    log.info("  Country resolved   : %d  (%.1f%%)",
             report["country_resolved"], report["country_resolution_pct"])
    o = report["overall"]
    log.info("  Matched            : %d / %d  (%.1f%%)",
             o["matched"], o["n"], (o["rate"] or 0) * 100)
    log.info("  Mean match score   : %.3f", o["mean_score"] or 0)
    log.info("=" * 60)

    gdelt.to_parquet(out_dir / "gdelt_graph_matched.parquet", index=False)
    gdelt.to_csv(out_dir / "gdelt_graph_matched.csv", index=False)
    with (out_dir / "graph_match_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log.info("Outputs written to: %s", out_dir)

    fig_score_distribution(gdelt,   out=fig_dir / "fig_graph_score_distribution.png")
    fig_score_components(gdelt,     out=fig_dir / "fig_graph_score_components.png")
    fig_match_rate_by_type(gdelt,   out=fig_dir / "fig_graph_match_rate_by_type.png")
    fig_match_rate_by_year(gdelt,   out=fig_dir / "fig_graph_match_rate_by_year.png")
    fig_match_rate_by_country(gdelt, top_n=40,
                               out=fig_dir / "fig_graph_match_rate_by_country.png")
    return gdelt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Graph-based GDELT ↔ ACLED event matching"
    )
    parser.add_argument("--gdelt",     type=Path,  default=DEFAULT_GDELT_PATH,
                        help="Deduped JSONL (default) or raw consolidated JSONL")
    parser.add_argument("--acled",     type=Path,  default=DEFAULT_ACLED_RAW)
    parser.add_argument("--out",       type=Path,  default=DEFAULT_OUT_DIR)
    parser.add_argument("--window",    type=int,   default=MAX_DAYS)
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    args = parser.parse_args()
    main(gdelt_path=args.gdelt, acled_raw=args.acled,
         out_dir=args.out, max_days=args.window, threshold=args.threshold)
