"""
matching.py
============
Runs maximum-weight bipartite matching on the event graph and propagates
reference dataset richness to matched GDELT events.

Design rationale
----------------
Maximum-weight bipartite matching vs. greedy best-match
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The naive approach — for each GDELT event, pick the reference event with the
highest score — has a critical flaw: it allows two GDELT events to claim the
same reference event independently.  This creates ghost duplicates in the
output and inflates the apparent GDELT coverage.

Maximum-weight matching (also called the assignment problem) finds the subset
of edges in the bipartite graph that maximises the total similarity score while
enforcing 1-to-1 assignment: each GDELT event matches at most one reference
event, and each reference event is claimed by at most one GDELT event.  This
is the globally optimal assignment.

Implementation: scipy.sparse.csgraph is not used here because the graph is
sparse and networkx's max_weight_matching uses Galil, Micali & Gabow (1986)
with O(n³) worst case but fast in practice for sparse graphs.

Match threshold
~~~~~~~~~~~~~~~
After max-weight matching, pairs with score < MATCH_THRESHOLD are discarded.
The matching algorithm finds the *optimal* assignment subject to graph
structure, but if the best available match is still weak (score < 0.40),
it should be treated as unmatched rather than forced.  0.40 was chosen as a
conservative threshold — it means at least two of the four scoring components
must be non-trivially positive.

Richness propagation
~~~~~~~~~~~~~~~~~~~~
Once a GDELT event is matched to a reference event, the reference event's
impact fields (dead, affected, displaced, damage) are copied to the GDELT
record.  These fields are often missing or under-reported in GDELT (which
extracts from news articles) but present in official databases.  The match
confidence score is also stored so downstream analysis can weight propagated
values appropriately.

Fields propagated: dead, injured, displaced, affected, damage_usd_thousands,
source, source_id, glide_number, date_end, severity, main_cause, event_name,
area_km2, country_iso.
"""

import networkx as nx

MATCH_THRESHOLD = 0.40   # minimum score to accept a matched pair

PROPAGATE_FIELDS = [
    "dead", "injured", "displaced", "affected", "indirectly_affected",
    "houses_destroyed", "houses_damaged", "roads_km", "damage_usd_thousands",
    "severity", "main_cause", "event_name", "area_km2", "glide_number",
    "date_end", "country_iso", "all_country_iso",
    # enrichment fields from reference that GDELT won't have independently
    "jrc_occurrence_pct", "jrc_recurrence_pct",
    "pop_count_25km", "pop_density_km2",
]


def run_matching(
    G: nx.Graph,
    gdelt_events: list[dict],
    reference_events: list[dict],
    threshold: float = MATCH_THRESHOLD,
) -> tuple[list[dict], dict]:
    """
    Run max-weight bipartite matching and return enriched GDELT event records.

    Parameters
    ----------
    G                : bipartite graph from graph_builder.build_graph()
    gdelt_events     : list of enriched GDELT event dicts
    reference_events : list of enriched reference event dicts
    threshold        : minimum edge score to accept a match

    Returns
    -------
    enriched_gdelt : list of GDELT event dicts with reference fields propagated
    match_index    : dict mapping gdelt_idx -> ref_idx for accepted matches
    """
    # Max-weight matching returns a set of (node_a, node_b) edge tuples
    matched_edges = nx.max_weight_matching(G, maxcardinality=False, weight="weight")

    # Build index: gdelt_node -> ref_node
    match_index = {}
    for a, b in matched_edges:
        # Normalise direction (matching returns unordered pairs)
        if str(a).startswith("gdelt_") and str(b).startswith("ref_"):
            gdelt_node, ref_node = a, b
        elif str(b).startswith("gdelt_") and str(a).startswith("ref_"):
            gdelt_node, ref_node = b, a
        else:
            continue  # skip reference-reference or gdelt-gdelt (shouldn't occur)

        edge_data = G.edges[gdelt_node, ref_node]
        if edge_data["weight"] < threshold:
            continue

        gdelt_idx = int(gdelt_node.split("_")[1])
        ref_idx   = int(ref_node.split("_")[1])
        match_index[gdelt_idx] = ref_idx

    # Propagate richness
    enriched_gdelt = []
    for i, event in enumerate(gdelt_events):
        record = dict(event)
        if i in match_index:
            j = match_index[i]
            ref = reference_events[j]
            edge_data = G.edges[f"gdelt_{i}", f"ref_{j}"]

            record["matched"]            = True
            record["match_score"]        = round(edge_data["weight"], 4)
            record["match_ref_source"]   = ref.get("source")
            record["match_ref_id"]       = str(ref.get("source_id", ""))
            record["match_ref_date"]     = str(ref.get("date_start", ""))

            for field in PROPAGATE_FIELDS:
                val = ref.get(field)
                if val is not None:
                    record[f"ref_{field}"] = val
        else:
            record["matched"]     = False
            record["match_score"] = None

        enriched_gdelt.append(record)

    return enriched_gdelt, match_index
