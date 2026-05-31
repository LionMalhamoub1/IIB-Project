"""
generate_clustering_visual.py
==============================
Creates illustrative visual aids of the two-level clustering algorithm.

Figures
-------
fig_A_level1_clustering.pdf   -- Article nodes → connected components (events)
fig_B_level2_movements.pdf    -- Event clusters → movements across time
fig_C_similarity_gates.pdf    -- How the three similarity components combine
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import networkx as nx
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

BLUE       = "#2166AC"
BLUE_LIGHT = "#92C5DE"
ORANGE     = "#D6604D"
ORANGE_L   = "#F4A582"
GREEN      = "#4DAC26"
GREEN_L    = "#B8E986"
GREY       = "#AAAAAA"
RED        = "#AA1111"
BG         = "#FAFAFA"
LIGHT      = "#EEF4FB"


# =============================================================================
# Figure A — Level-1 clustering: articles → canonical events
# =============================================================================

def fig_A_level1():
    """
    Left panel: raw article nodes (unlabelled, coloured by type).
    Right panel: same nodes drawn with cluster membership colour,
                 cluster hulls drawn, and edges shown.
    """
    np.random.seed(42)

    # ---- Define 3 protest clusters + 1 strike cluster ----
    # Each cluster has articles near a geographic centre
    clusters = [
        # (centre_x, centre_y, n_articles, type, label)
        (1.5,  3.5, 5, "protest",       "Cluster A\n(protest, Iran)"),
        (5.5,  3.8, 4, "protest",       "Cluster B\n(protest, France)"),
        (3.5,  1.2, 3, "protest",       "Cluster C\n(protest, Chile)"),
        (7.5,  1.8, 4, "strike",        "Cluster D\n(strike, India)"),
    ]

    rng = np.random.default_rng(42)

    nodes = []   # (x, y, cluster_id, type)
    for cid, (cx, cy, n, dtype, _) in enumerate(clusters):
        for _ in range(n):
            dx, dy = rng.normal(0, 0.35, 2)
            nodes.append((cx + dx, cy + dy, cid, dtype))

    # Add a few "noise" nodes that are too far from any cluster to link
    noise = [
        (9.0, 3.5, -1, "protest"),
        (0.5, 0.8, -1, "strike"),
    ]
    nodes.extend(noise)

    # Build graph: connect nodes within same cluster if sim > threshold
    # (simulate: intra-cluster always above 0.45, inter-cluster always below)
    G = nx.Graph()
    for i, n1 in enumerate(nodes):
        G.add_node(i, pos=(n1[0], n1[1]), cluster=n1[2], dtype=n1[3])

    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if j <= i:
                continue
            same_cluster = n1[2] == n2[2] and n1[2] >= 0
            if same_cluster:
                # Sim proportional to distance (illustrative)
                dist = np.hypot(n1[0]-n2[0], n1[1]-n2[1])
                sim  = max(0.0, 0.92 - dist * 0.4)
                if sim > 0.45:
                    G.add_edge(i, j, weight=sim)

    pos = nx.get_node_attributes(G, "pos")

    cluster_colors = {
        0: BLUE,
        1: GREEN,
        2: ORANGE,
        3: "#8E44AD",
        -1: GREY,
    }
    type_markers = {"protest": "o", "strike": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), facecolor=BG)

    # ---------- Left panel: raw, uncoloured --------------------------------
    ax = axes[0]
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.3, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Step 1: Raw GDELT articles\n(same country & type — before clustering)",
                 fontsize=11, fontweight="bold", pad=8)

    for i, nd in enumerate(nodes):
        x, y, cid, dtype = nd
        marker = type_markers[dtype]
        ax.scatter(x, y, s=120, c=GREY if cid < 0 else "#AACCEE",
                   marker=marker, edgecolors="#555555", linewidths=0.8, zorder=3)
        ax.text(x, y + 0.28, f"a{i+1}", ha="center", va="bottom",
                fontsize=7, color="#555555")

    # Legend
    ax.scatter([], [], s=90, c="#AACCEE", marker="o",
               edgecolors="#555555", label="Protest article")
    ax.scatter([], [], s=90, c="#AACCEE", marker="s",
               edgecolors="#555555", label="Strike article")
    ax.scatter([], [], s=90, c=GREY,     marker="o",
               edgecolors="#555555", label="Isolated / noise")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5)

    # ---------- Right panel: clustered, with edges and hulls ---------------
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    ax2.set_xlim(-0.5, 10.5)
    ax2.set_ylim(-0.3, 5.5)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Step 2: Connected components → Canonical Events\n"
                  "(edges = cosine similarity > 0.45)",
                  fontsize=11, fontweight="bold", pad=8)

    # Draw edges
    for (u, v, data) in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        sim = data["weight"]
        alpha = 0.3 + 0.5 * (sim - 0.45) / 0.55
        ax2.plot([x1, x2], [y1, y2], color="#999999",
                 lw=1.2, alpha=alpha, zorder=1)
        # Label a few edges with similarity score
        if sim > 0.75 and u < 6:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax2.text(mx, my + 0.12, f"{sim:.2f}", fontsize=6.5,
                     ha="center", color="#666666")

    # Draw cluster convex hulls
    from matplotlib.patches import Polygon as MplPolygon
    from scipy.spatial import ConvexHull

    for cid, (_, _, _, _, clabel) in enumerate(clusters):
        pts = np.array([(nodes[i][0], nodes[i][1])
                        for i in range(len(nodes))
                        if nodes[i][2] == cid])
        if len(pts) < 3:
            # Draw a circle instead
            cx_, cy_ = pts.mean(axis=0)
            circle = plt.Circle((cx_, cy_), 0.55,
                                 color=cluster_colors[cid], alpha=0.12,
                                 zorder=0)
            ax2.add_patch(circle)
        else:
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
                # Expand hull slightly
                centroid = hull_pts.mean(axis=0)
                expanded = centroid + (hull_pts - centroid) * 1.35
                poly = MplPolygon(expanded, closed=True,
                                  facecolor=cluster_colors[cid],
                                  alpha=0.15, edgecolor=cluster_colors[cid],
                                  linewidth=1.5, linestyle="--", zorder=0)
                ax2.add_patch(poly)
            except Exception:
                pass

        # Cluster label
        cx_ = np.mean([nodes[i][0] for i in range(len(nodes)) if nodes[i][2] == cid])
        cy_ = np.max( [nodes[i][1] for i in range(len(nodes)) if nodes[i][2] == cid])
        ax2.text(cx_, cy_ + 0.42, clabel,
                 ha="center", va="bottom", fontsize=8,
                 color=cluster_colors[cid], fontweight="bold")

    # Draw nodes coloured by cluster
    for i, nd in enumerate(nodes):
        x, y, cid, dtype = nd
        col    = cluster_colors[cid]
        marker = type_markers[dtype]
        ax2.scatter(x, y, s=140, c=col, marker=marker,
                    edgecolors="#333333", linewidths=0.8, zorder=3)
        ax2.text(x, y + 0.28, f"a{i+1}", ha="center", va="bottom",
                 fontsize=7, color="#333333")

    # Noise label
    for nd in noise:
        x, y, _, _ = nd
        ax2.text(x, y - 0.35, "isolated\n(no links)", ha="center",
                 fontsize=7, color=GREY, style="italic")

    # Threshold annotation
    ax2.text(0.01, 0.02,
             "Edge drawn if: w·sim_embed + w·sim_loc + w·sim_temp > 0.45",
             transform=ax2.transAxes, fontsize=7.5, color="#555555",
             style="italic")

    # Cluster-colour legend
    handles = [
        mpatches.Patch(facecolor=cluster_colors[c], label=clusters[c][4])
        for c in range(len(clusters))
    ]
    handles.append(mpatches.Patch(facecolor=GREY, label="Isolated"))
    ax2.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=8)

    fig.suptitle("Level-1 Clustering: Articles → Canonical Events",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figA_level1_clustering.pdf")
    fig.savefig(OUT_DIR / "figA_level1_clustering.png")
    plt.close(fig)
    print("  figA saved")


# =============================================================================
# Figure B — Level-2: event clusters → movements across time
# =============================================================================

def fig_B_level2():
    """
    Timeline showing event clusters (circles) across multiple days for two
    countries.  Movement links (dashed arcs) connect related clusters.
    """
    np.random.seed(7)

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-1, 22)
    ax.set_ylim(-1, 6)
    ax.axis("off")

    # -------- Data: (day, y_row, size, color, cluster_label, movement_id) ----
    # Two protest movements in Iran (y=4), one strike in India (y=1.5)
    events = [
        # Iran protest movement 1 (days 1-12)  — BLUE
        dict(day=1,  y=4.0, size=180, col=BLUE,   label="C1\n5 arts", mid=0),
        dict(day=4,  y=4.0, size=250, col=BLUE,   label="C2\n8 arts", mid=0),
        dict(day=7,  y=4.0, size=140, col=BLUE,   label="C3\n4 arts", mid=0),
        dict(day=11, y=4.0, size=200, col=BLUE,   label="C4\n6 arts", mid=0),

        # Iran protest movement 2 (different topic) — starts day 14  GREEN
        dict(day=14, y=4.0, size=160, col=GREEN,  label="C5\n5 arts", mid=1),
        dict(day=18, y=4.0, size=300, col=GREEN,  label="C6\n10 arts", mid=1),

        # India strike movement (days 2-15)  — ORANGE
        dict(day=2,  y=1.8, size=130, col=ORANGE, label="C7\n3 arts", mid=2),
        dict(day=6,  y=1.8, size=220, col=ORANGE, label="C8\n7 arts", mid=2),
        dict(day=9,  y=1.8, size=170, col=ORANGE, label="C9\n5 arts", mid=2),
        dict(day=15, y=1.8, size=140, col=ORANGE, label="C10\n4 arts", mid=2),

        # One isolated protest in Chile — no movement link  GREY
        dict(day=10, y=2.9, size=90,  col=GREY,   label="C11\n2 arts", mid=-1),
    ]

    movement_colors = {0: BLUE, 1: GREEN, 2: ORANGE, -1: GREY}
    movement_labels = {
        0: "Movement M1\n(Iran – protest wave)",
        1: "Movement M2\n(Iran – counter-protest)",
        2: "Movement M3\n(India – general strike)",
    }

    # ---- Draw timeline axes per row ----
    for y_row, country in [(4.0, "Iran  (protests)"),
                           (1.8, "India  (strikes)"),
                           (2.9, "Chile  (protest)")]:
        ax.annotate("", xy=(21, y_row), xytext=(0, y_row),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))
        ax.text(-0.6, y_row, country, ha="right", va="center",
                fontsize=10, fontweight="bold", color="#333333")

    # Day ticks
    for d in range(0, 21, 2):
        ax.text(d, -0.5, f"day {d}", ha="center", fontsize=7.5, color=GREY)

    ax.text(10, -0.9, "Time (days)", ha="center", fontsize=9, color="#555555")

    # ---- Draw movement arcs between same-movement clusters ----
    def draw_arc(x1, y1, x2, y2, col, sim):
        mid_x = (x1 + x2) / 2
        mid_y = y1 + 0.7          # arc height
        ax.annotate("", xy=(x2, y2 + 0.22), xytext=(x1, y1 + 0.22),
                    arrowprops=dict(
                        arrowstyle="-|>",
                        connectionstyle=f"arc3,rad=-0.35",
                        color=col, lw=1.8, ls="--",
                    ))
        ax.text(mid_x, mid_y, f"cosim={sim:.2f}", ha="center",
                fontsize=7.5, color=col, style="italic")

    # Draw arcs for consecutive same-movement clusters
    by_mid: dict[int, list] = {}
    for ev in events:
        by_mid.setdefault(ev["mid"], []).append(ev)

    sim_values = {
        0: [0.81, 0.78, 0.76],   # M1 consecutive sims
        1: [0.74],
        2: [0.83, 0.79, 0.77],
    }

    for mid, evs in by_mid.items():
        if mid < 0:
            continue
        evs_sorted = sorted(evs, key=lambda e: e["day"])
        sims = sim_values.get(mid, [0.75] * len(evs_sorted))
        for i in range(len(evs_sorted) - 1):
            e1, e2 = evs_sorted[i], evs_sorted[i+1]
            draw_arc(e1["day"], e1["y"], e2["day"], e2["y"],
                     movement_colors[mid],
                     sims[i] if i < len(sims) else 0.75)

    # ---- Draw event cluster nodes ----
    for ev in events:
        col = movement_colors[ev["mid"]]
        # Node circle
        ax.scatter(ev["day"], ev["y"], s=ev["size"],
                   c=col, edgecolors="#333333" if col != GREY else "#888888",
                   linewidths=1.2, zorder=4, alpha=0.85)
        # Label inside / below
        ax.text(ev["day"], ev["y"] - 0.55, ev["label"],
                ha="center", va="top", fontsize=7.5, color="#333333")

    # ---- Annotate movements ----
    movement_x = {0: 6, 1: 16, 2: 8.5}
    movement_y = {0: 5.1, 1: 5.1, 2: 0.7}
    for mid, lbl in movement_labels.items():
        ax.text(movement_x[mid], movement_y[mid], lbl,
                ha="center", va="center", fontsize=9,
                color=movement_colors[mid], fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white",
                          edgecolor=movement_colors[mid],
                          linewidth=1.2))

    # Isolated annotation
    iso = [ev for ev in events if ev["mid"] < 0][0]
    ax.text(iso["day"], iso["y"] + 0.6,
            "No movement link\n(gap > 14 days\nor cosim < 0.72)",
            ha="center", fontsize=7.5, color=GREY, style="italic")

    # ---- Threshold box ----
    ax.text(0.5, 0.04,
            "Movement linked if: same country & type  ·  ≤14 day gap  ·  centroid cosim ≥ 0.72",
            transform=ax.transAxes, ha="center",
            fontsize=8, color="#555555", style="italic")

    ax.set_title("Level-2 Clustering: Canonical Events → Sustained Movements",
                 fontsize=13, fontweight="bold", pad=10)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figB_level2_movements.pdf")
    fig.savefig(OUT_DIR / "figB_level2_movements.png")
    plt.close(fig)
    print("  figB saved")


# =============================================================================
# Figure C — Similarity score decomposition (illustrative example)
# =============================================================================

def fig_C_similarity():
    """
    Shows two example article pairs and how embedding / location / temporal
    components combine into a final score, with a bar breakdown.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)

    pairs = [
        {
            "title": "Pair 1 — Linked  (score > 0.45)",
            "a1":    "Article A₁\n\"Protests erupt in Tehran\nover economic conditions\"",
            "a2":    "Article A₂\n\"Demonstrators march in\nIranian capital, clashes\"",
            "embed": 0.82,
            "loc":   0.90,
            "temp":  0.95,
            "linked": True,
        },
        {
            "title": "Pair 2 — Not linked  (score < 0.45)",
            "a1":    "Article A₃\n\"Workers strike at\nMumbai port facility\"",
            "a2":    "Article A₄\n\"Paris fashion week\nprotests by activists\"",
            "embed": 0.38,
            "loc":   0.05,
            "temp":  0.80,
            "linked": False,
        },
    ]

    weights = {"embed": 0.55, "loc": 0.30, "temp": 0.15}

    for ax, pair in zip(axes, pairs):
        ax.set_facecolor(BG)
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title(pair["title"], fontsize=11, fontweight="bold",
                     color=GREEN if pair["linked"] else RED)

        # Draw article boxes
        def article_box(x, y, text, col):
            rect = FancyBboxPatch((x-1.6, y-0.8), 3.2, 1.6,
                                  boxstyle="round,pad=0.1",
                                  facecolor=col, edgecolor="#555555", lw=1.2)
            ax.add_patch(rect)
            ax.text(x, y, text, ha="center", va="center",
                    fontsize=7.8, color="#1A1A2E")

        article_box(2.2, 8.5, pair["a1"], LIGHT if pair["linked"] else "#FFF0EE")
        article_box(7.8, 8.5, pair["a2"], LIGHT if pair["linked"] else "#FFF0EE")

        # Arrow between articles
        link_col = GREEN if pair["linked"] else RED
        ax.annotate("", xy=(6.2, 8.5), xytext=(3.8, 8.5),
                    arrowprops=dict(arrowstyle="<->", color=link_col, lw=1.8))

        # Component bars
        components = [
            ("Embedding\n(55%)", pair["embed"], weights["embed"]),
            ("Location\n(30%)",  pair["loc"],   weights["loc"]),
            ("Temporal\n(15%)",  pair["temp"],  weights["temp"]),
        ]
        bar_colors = [BLUE, ORANGE, GREEN]

        weighted_sum = sum(v * w for _, v, w in components)

        y_base = 6.5
        bar_w  = 2.2
        gap    = 3.0
        for i, (comp_label, raw_sim, wt) in enumerate(components):
            bx = 1.5 + i * gap
            by = y_base

            # Background bar (max)
            rect_bg = mpatches.Rectangle((bx - bar_w/2, by - 1.0),
                                         bar_w, 1.0,
                                         facecolor="#DDDDDD", edgecolor="none")
            ax.add_patch(rect_bg)

            # Filled bar
            rect_fill = mpatches.Rectangle((bx - bar_w/2, by - 1.0),
                                           bar_w * raw_sim, 1.0,
                                           facecolor=bar_colors[i], alpha=0.7,
                                           edgecolor="none")
            ax.add_patch(rect_fill)

            # Outline
            rect_out = mpatches.Rectangle((bx - bar_w/2, by - 1.0),
                                          bar_w, 1.0,
                                          facecolor="none",
                                          edgecolor="#999999", lw=0.8)
            ax.add_patch(rect_out)

            # Labels
            ax.text(bx, by - 0.5, f"{raw_sim:.2f}",
                    ha="center", va="center", fontsize=9.5, fontweight="bold",
                    color="white" if raw_sim > 0.4 else "#555555")
            ax.text(bx, by + 0.25, comp_label,
                    ha="center", va="bottom", fontsize=8, color="#333333")
            ax.text(bx, by - 1.3, f"× {wt}  =  {raw_sim*wt:.3f}",
                    ha="center", va="top", fontsize=7.5, color="#555555")

        # Final score
        threshold = 0.45
        score_color = GREEN if weighted_sum > threshold else RED
        ax.text(5.0, 3.5,
                f"Weighted score = {weighted_sum:.3f}",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color=score_color)
        ax.text(5.0, 2.9,
                f"(0.55×{pair['embed']:.2f} + 0.30×{pair['loc']:.2f} + 0.15×{pair['temp']:.2f})",
                ha="center", va="center", fontsize=8.5, color="#555555")

        verdict = "LINKED  ✓" if pair["linked"] else "NOT LINKED  ✗"
        ax.text(5.0, 2.2, f"Threshold = {threshold}  →  {verdict}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color=score_color,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=score_color, lw=1.5))

    fig.suptitle("Weighted Similarity Decomposition — Two Illustrative Article Pairs",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "figC_similarity_decomposition.pdf")
    fig.savefig(OUT_DIR / "figC_similarity_decomposition.png")
    plt.close(fig)
    print("  figC saved")


# =============================================================================
# Figure D — Combined overview (A + B side by side, compact)
# =============================================================================

def fig_D_overview():
    """
    Single-page summary figure combining the graph clustering concept
    and the timeline movement concept, suitable for a dissertation methods chapter.
    """
    from scipy.spatial import ConvexHull
    from matplotlib.patches import Polygon as MplPolygon

    np.random.seed(42)
    fig = plt.figure(figsize=(14, 11), facecolor=BG)

    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25,
                          left=0.04, right=0.97, top=0.93, bottom=0.04)

    # ---- Panel A (top-left): raw articles ----
    ax_raw  = fig.add_subplot(gs[0, 0])
    # ---- Panel B (top-right): clustered ----
    ax_clus = fig.add_subplot(gs[0, 1])
    # ---- Panel C (bottom, spanning both): movements timeline ----
    ax_mov  = fig.add_subplot(gs[1, :])

    # ----- shared node data (same as fig_A) -----
    clusters_def = [
        (1.8,  3.6, 5, "protest", BLUE,   "Event A  (Iran, protests)"),
        (5.5,  3.8, 4, "protest", GREEN,  "Event B  (France, protests)"),
        (3.8,  1.3, 3, "protest", ORANGE, "Event C  (Chile, protests)"),
        (7.8,  2.0, 4, "strike",  "#8E44AD", "Event D  (India, strike)"),
    ]

    rng = np.random.default_rng(42)
    nodes = []
    for cid, (cx, cy, n, dtype, col, _) in enumerate(clusters_def):
        for _ in range(n):
            dx, dy = rng.normal(0, 0.33, 2)
            nodes.append((cx+dx, cy+dy, cid, dtype))
    nodes.append((9.2, 3.5, -1, "protest"))
    nodes.append((0.6, 0.7, -1, "strike"))

    G = nx.Graph()
    for i, nd in enumerate(nodes):
        G.add_node(i, pos=(nd[0], nd[1]))
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if j <= i or n1[2] != n2[2] or n1[2] < 0:
                continue
            dist = np.hypot(n1[0]-n2[0], n1[1]-n2[1])
            sim  = max(0.0, 0.92 - dist * 0.4)
            if sim > 0.45:
                G.add_edge(i, j, weight=sim)
    pos = nx.get_node_attributes(G, "pos")
    cluster_colors_map = {c: clusters_def[c][4] for c in range(len(clusters_def))}
    cluster_colors_map[-1] = GREY

    for ax, show_clusters in [(ax_raw, False), (ax_clus, True)]:
        ax.set_facecolor(BG)
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.3, 5.3)
        ax.set_aspect("equal")
        ax.axis("off")

        if show_clusters:
            # Edges
            for (u, v, data) in G.edges(data=True):
                x1, y1 = pos[u]; x2, y2 = pos[v]
                ax.plot([x1,x2],[y1,y2], color="#AAAAAA", lw=1.0,
                        alpha=0.55, zorder=1)
            # Hulls
            for cid in range(len(clusters_def)):
                pts = np.array([(nodes[i][0], nodes[i][1])
                                for i in range(len(nodes))
                                if nodes[i][2] == cid])
                if len(pts) < 3:
                    continue
                try:
                    hull = ConvexHull(pts)
                    hp = pts[hull.vertices]
                    centroid = hp.mean(axis=0)
                    expanded = centroid + (hp - centroid) * 1.4
                    poly = MplPolygon(expanded, closed=True,
                                     facecolor=clusters_def[cid][4],
                                     alpha=0.13,
                                     edgecolor=clusters_def[cid][4],
                                     lw=1.5, linestyle="--", zorder=0)
                    ax.add_patch(poly)
                except Exception:
                    pass
                cx_ = pts[:,0].mean()
                cy_ = pts[:,1].max()
                ax.text(cx_, cy_+0.35, clusters_def[cid][5],
                        ha="center", fontsize=7.5,
                        color=clusters_def[cid][4], fontweight="bold")

        # Nodes
        type_markers = {"protest":"o","strike":"s"}
        for i, nd in enumerate(nodes):
            col = cluster_colors_map[nd[2]] if show_clusters else "#AABBCC"
            ax.scatter(nd[0], nd[1], s=100, c=col,
                       marker=type_markers[nd[3]],
                       edgecolors="#444444", linewidths=0.7, zorder=3,
                       alpha=0.9)

    ax_raw.set_title("(a)  Raw articles — same country & type",
                     fontsize=10, fontweight="bold")
    ax_clus.set_title("(b)  Level-1: connected components → canonical events\n"
                      "       (edges = weighted cosine > 0.45)",
                      fontsize=10, fontweight="bold")

    # ----- Movement timeline (bottom) -----
    ax_mov.set_facecolor(BG)
    ax_mov.set_xlim(-0.5, 20.5)
    ax_mov.set_ylim(-0.5, 3.8)
    ax_mov.axis("off")
    ax_mov.set_title("(c)  Level-2: canonical events → sustained movements\n"
                     "       (dashed arc = centroid cosim ≥ 0.72 within 14 days)",
                     fontsize=10, fontweight="bold")

    rows = {
        "Iran  (protests)": (3.1, BLUE),
        "India  (strikes)": (1.5, "#8E44AD"),
        "Chile  (protest)": (0.1, ORANGE),
    }
    for label, (y, col) in rows.items():
        ax_mov.annotate("", xy=(20, y), xytext=(0, y),
                        arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1))
        ax_mov.text(-0.3, y, label, ha="right", va="center",
                    fontsize=9, color="#333333", fontweight="bold")

    # Iran clusters
    iran = [(1,3.1,180),(4,3.1,250),(8,3.1,150),(12,3.1,200)]
    # India
    india = [(2,1.5,130),(7,1.5,210),(13,1.5,160)]
    # Chile (isolated)
    chile = [(9,0.1,90)]

    for evs, col, mid_label, arc_y in [
        (iran,  BLUE,     "Movement M1 (protest wave)", 3.6),
        (india, "#8E44AD","Movement M2 (general strike)", 2.05),
    ]:
        for i, (day, y, sz) in enumerate(evs):
            ax_mov.scatter(day, y, s=sz, c=col, edgecolors="#333",
                           linewidths=0.9, zorder=4, alpha=0.85)
            n_arts = int(sz / 25)
            ax_mov.text(day, y-0.35, f"{n_arts} arts",
                        ha="center", fontsize=7, color="#555")
        for i in range(len(evs)-1):
            d1,y1,_ = evs[i]; d2,y2,_ = evs[i+1]
            ax_mov.annotate("", xy=(d2, y2+0.18), xytext=(d1, y1+0.18),
                            arrowprops=dict(arrowstyle="-|>",
                                           connectionstyle="arc3,rad=-0.3",
                                           color=col, lw=1.6, ls="--"))
        # Movement brace label
        mid_d = (evs[0][0] + evs[-1][0]) / 2
        ax_mov.text(mid_d, arc_y, mid_label,
                    ha="center", fontsize=8.5, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor=col, lw=1))

    # Chile
    day,y,sz = chile[0]
    ax_mov.scatter(day, y, s=sz, c=ORANGE, edgecolors="#333",
                   linewidths=0.9, zorder=4, alpha=0.85)
    ax_mov.text(day, y+0.35, "Isolated event\n(no movement)",
                ha="center", fontsize=7.5, color=GREY, style="italic")

    # Day ticks
    for d in range(0, 21, 2):
        ax_mov.text(d, -0.4, f"d{d}", ha="center", fontsize=7.5, color=GREY)
    ax_mov.text(10, -0.75, "Time →", ha="center", fontsize=9, color="#666")

    fig.suptitle("Two-Level GDELT Event Clustering Pipeline",
                 fontsize=14, fontweight="bold")
    fig.savefig(OUT_DIR / "figD_overview.pdf")
    fig.savefig(OUT_DIR / "figD_overview.png")
    plt.close(fig)
    print("  figD saved")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("Generating clustering visual aids …")
    fig_A_level1()
    fig_B_level2()
    fig_C_similarity()
    fig_D_overview()
    print(f"\nSaved to: {OUT_DIR}")
