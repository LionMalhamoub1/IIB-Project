"""
graph_builder.py
=================
Builds a weighted bipartite graph from scored candidate pairs, with GDELT
events on one side and reference events on the other.

Design rationale
----------------
Representing the matching problem as a graph, rather than a simple ranked
list, offers several advantages:

  1. Global visibility — the full graph shows all plausible connections
     simultaneously.  A GDELT event might score 0.7 against one reference
     event and 0.65 against another; the graph preserves both rather than
     discarding the second-best option.

  2. Separation of concerns — graph construction (what is connected and how
     strongly) is separated from the matching decision (which connections to
     accept).  This makes it easy to experiment with different matching
     algorithms without rebuilding the graph.

  3. Richness propagation — after matching, the graph structure can be used
     to propagate reference attributes (dead, affected, damage) to matched
     GDELT nodes through weighted edges.  Nodes that are close to matched
     nodes in the graph but not themselves matched can receive partial
     enrichment with a confidence discount.

  4. Interpretability — the graph is a natural artefact for the dissertation.
     Visualising it reveals which floods were well-covered by GDELT (high-
     degree reference nodes), which were missed (isolated reference nodes),
     and which GDELT clusters were noise (isolated GDELT nodes).

Graph structure
---------------
  - Left partition  : GDELT consolidated events  (node type = "gdelt")
  - Right partition : Reference events            (node type = "reference")
  - Edges           : scored pairs with score ≥ MIN_EDGE_SCORE
  - Edge weight     : similarity score (float in [0, 1])
  - Edge attributes : individual component scores (geo, temporal, hydro, text)

MIN_EDGE_SCORE = 0.25 — deliberately permissive.  Weak edges are retained
in the graph so that the matching algorithm (max-weight matching) can make
the global optimisation decision.  Edges below 0.25 are almost certainly
wrong and add noise without value.
"""

import networkx as nx

MIN_EDGE_SCORE = 0.25


def build_graph(
    gdelt_events: list[dict],
    reference_events: list[dict],
    scored_pairs: list[dict],
    min_edge_score: float = MIN_EDGE_SCORE,
) -> nx.Graph:
    """
    Build a weighted bipartite graph from scored candidate pairs.

    Parameters
    ----------
    gdelt_events     : list of enriched GDELT event dicts
    reference_events : list of enriched reference event dicts
    scored_pairs     : output of score_all_candidates()
    min_edge_score   : minimum score to include an edge

    Returns
    -------
    networkx.Graph with bipartite node attributes and weighted edges.
    """
    G = nx.Graph()

    # Add GDELT nodes (left partition, bipartite=0)
    for i, event in enumerate(gdelt_events):
        G.add_node(
            f"gdelt_{i}",
            bipartite=0,
            event_date=str(event.get("event_date") or event.get("date_start", "")),
            location=str(event.get("location_name", "")),
            lat=event.get("lat"),
            lon=event.get("lon"),
            num_articles=event.get("num_articles", 1),
        )

    # Add reference nodes (right partition, bipartite=1)
    for j, event in enumerate(reference_events):
        G.add_node(
            f"ref_{j}",
            bipartite=1,
            source=event.get("source", ""),
            source_id=str(event.get("source_id", "")),
            date_start=str(event.get("date_start", "")),
            location=str(event.get("location_name", "")),
            country=event.get("country", ""),
            dead=event.get("dead"),
            affected=event.get("affected"),
            damage_usd_thousands=event.get("damage_usd_thousands"),
        )

    # Add edges for scored pairs above threshold
    n_edges = 0
    for pair in scored_pairs:
        if pair["score"] < min_edge_score:
            continue
        gdelt_node = f"gdelt_{pair['gdelt_idx']}"
        ref_node   = f"ref_{pair['ref_idx']}"
        G.add_edge(
            gdelt_node,
            ref_node,
            weight=pair["score"],
            geo=pair["geo"],
            temporal=pair["temporal"],
            hydro=pair["hydro"],
            text=pair["text"],
        )
        n_edges += 1

    return G


def graph_summary(G: nx.Graph) -> dict:
    """Return basic statistics about the matching graph."""
    gdelt_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    ref_nodes   = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]

    connected_gdelt = {n for n in gdelt_nodes if G.degree(n) > 0}
    connected_ref   = {n for n in ref_nodes   if G.degree(n) > 0}

    weights = [d["weight"] for _, _, d in G.edges(data=True)]

    return {
        "gdelt_nodes":       len(gdelt_nodes),
        "reference_nodes":   len(ref_nodes),
        "edges":             G.number_of_edges(),
        "gdelt_connected":   len(connected_gdelt),
        "ref_connected":     len(connected_ref),
        "gdelt_isolated":    len(gdelt_nodes) - len(connected_gdelt),
        "ref_isolated":      len(ref_nodes)   - len(connected_ref),
        "mean_edge_weight":  round(sum(weights) / len(weights), 4) if weights else 0,
        "max_edge_weight":   round(max(weights), 4) if weights else 0,
    }
