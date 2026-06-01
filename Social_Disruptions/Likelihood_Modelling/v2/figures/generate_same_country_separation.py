# Illustrates that same-country articles on the same day are correctly separated by sub-location.

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

BLUE      = "#2166AC"
ORANGE    = "#D6604D"
PURPLE    = "#7B2D8B"
GREY      = "#AAAAAA"
DARK      = "#1A1A2E"
BG        = "#FAFAFA"
THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Article definitions — three real event clusters in India, 1 Jan 2018
# ---------------------------------------------------------------------------
EVENTS = [
    {
        "label":   "Event 1 — Doctors' Protest\n(NMC Bill)",
        "color":   BLUE,
        "articles": [
            {"id": "A1", "conf": 0.90,
             "desc": "Rajasthan MLA resigned over\npoor handling of doctors' strike",
             "loc": "Rajasthan"},
            {"id": "A2", "conf": 0.85,
             "desc": "Indian Medical Association\nprotested over the NMC Bill",
             "loc": "India"},
            {"id": "A3", "conf": 0.88,
             "desc": "Medical students protested\nagainst the NMC Bill",
             "loc": "Thiruvananthapuram"},
            {"id": "A4", "conf": 0.82,
             "desc": "Doctors in Bengaluru rallied\nagainst NMC Bill proposals",
             "loc": "Bengaluru"},
        ],
        # Node positions (will be placed around a centre)
        "centre": np.array([2.5, 6.5]),
    },
    {
        "label":   "Event 2 — Kashmir Conflict\n(Paramilitary clash)",
        "color":   ORANGE,
        "articles": [
            {"id": "B1", "conf": 0.78,
             "desc": "Muslim & Jat communities met\nto restore communal harmony",
             "loc": "Muzaffarnagar"},
            {"id": "B2", "conf": 0.75,
             "desc": "Kashmiri fighters stormed\na paramilitary camp",
             "loc": "J&K"},
            {"id": "B3", "conf": 0.80,
             "desc": "Anti-India protests continued\nafter rebel leader killing",
             "loc": "Kashmir"},
        ],
        "centre": np.array([8.5, 6.5]),
    },
    {
        "label":   "Event 3 — Tribal Rights\n(Visakhapatnam)",
        "color":   PURPLE,
        "articles": [
            {"id": "C1", "conf": 0.85,
             "desc": "Girijana Sangham demonstrated\nagainst govt land decision",
             "loc": "Visakhapatnam"},
            {"id": "C2", "conf": 0.80,
             "desc": "Tribal groups protested in\nAndhra Pradesh over land rights",
             "loc": "Andhra Pradesh"},
            {"id": "C3", "conf": 0.77,
             "desc": "Activists rallied at\nSub-Collector's office",
             "loc": "Rajamahendravaram"},
        ],
        "centre": np.array([5.5, 2.0]),
    },
]

# Illustrative similarity scores
#   Within-cluster: 0.70 – 0.88  (all > threshold)
#   Across-cluster: 0.12 – 0.28  (all < threshold)
WITHIN_SIM = {
    # Event 1
    ("A1","A2"): 0.82, ("A1","A3"): 0.79, ("A1","A4"): 0.76,
    ("A2","A3"): 0.85, ("A2","A4"): 0.81,
    ("A3","A4"): 0.78,
    # Event 2
    ("B1","B2"): 0.74, ("B1","B3"): 0.71,
    ("B2","B3"): 0.77,
    # Event 3
    ("C1","C2"): 0.80, ("C1","C3"): 0.75,
    ("C2","C3"): 0.72,
}
# Only show a couple of cross-cluster attempts (always rejected)
CROSS_SIM = {
    ("A2","B3"): 0.21,   # NMC Bill vs Kashmir conflict
    ("A1","C1"): 0.18,   # doctors vs tribal protest
    ("B2","C2"): 0.15,   # Kashmir vs Andhra Pradesh
}


def scatter_positions(centre, n, radius=0.85, seed=None):
    """Return n positions scattered around a centre."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    r = radius * 0.6 + rng.uniform(0, radius*0.4, n)
    return np.column_stack([
        centre[0] + r * np.cos(angles),
        centre[1] + r * np.sin(angles),
    ])


def draw():
    fig, ax = plt.subplots(figsize=(14, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Map article id -> (x, y) position
    node_pos  = {}
    node_color = {}
    node_conf  = {}
    node_desc  = {}
    node_loc   = {}

    # Assign positions
    seeds = [7, 13, 21]
    for ev, seed in zip(EVENTS, seeds):
        n   = len(ev["articles"])
        pts = scatter_positions(ev["centre"], n, radius=1.15, seed=seed)
        for i, art in enumerate(ev["articles"]):
            node_pos[art["id"]]   = pts[i]
            node_color[art["id"]] = ev["color"]
            node_conf[art["id"]]  = art["conf"]
            node_desc[art["id"]]  = art["desc"]
            node_loc[art["id"]]   = art["loc"]

    # ---- Draw cross-cluster "rejected" edges (dashed red) ----
    for (id1, id2), sim in CROSS_SIM.items():
        x1, y1 = node_pos[id1]
        x2, y2 = node_pos[id2]
        ax.plot([x1, x2], [y1, y2], color="#CC3333", lw=1.2,
                ls="--", alpha=0.55, zorder=1)
        mx, my = (x1+x2)/2, (y1+y2)/2
        # Offset label slightly
        dx, dy = x2-x1, y2-y1
        norm = np.hypot(dx, dy) + 1e-9
        lx = mx + 0.25 * (-dy/norm)
        ly = my + 0.25 * (dx/norm)
        ax.text(lx, ly, f"sim={sim:.2f}\n< {THRESHOLD}",
                ha="center", va="center", fontsize=7.5,
                color="#CC3333",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="#CC3333", alpha=0.85, linewidth=0.8))

    # ---- Draw within-cluster edges (solid, blue-toned by strength) ----
    cmap = plt.cm.Blues
    for (id1, id2), sim in WITHIN_SIM.items():
        x1, y1 = node_pos[id1]
        x2, y2 = node_pos[id2]
        alpha = 0.35 + 0.55 * (sim - 0.45) / 0.55
        lw    = 1.0 + 2.5 * (sim - 0.45) / 0.55
        col   = node_color[id1]   # same for both endpoints
        ax.plot([x1, x2], [y1, y2], color=col, lw=lw, alpha=alpha, zorder=2)
        # Label the strongest edge per cluster
        if sim >= 0.84:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my + 0.18, f"sim={sim:.2f}",
                    ha="center", va="bottom", fontsize=7,
                    color=col)

    # ---- Draw cluster hulls ----
    for ev in EVENTS:
        ids = [a["id"] for a in ev["articles"]]
        pts = np.array([node_pos[i] for i in ids])
        col = ev["color"]
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hp   = pts[hull.vertices]
                centroid = hp.mean(axis=0)
                expanded = centroid + (hp - centroid) * 1.5
                poly = MplPolygon(expanded, closed=True,
                                  facecolor=col, alpha=0.10,
                                  edgecolor=col, lw=2.0,
                                  linestyle="--", zorder=0)
                ax.add_patch(poly)
            except Exception:
                pass
        # Cluster label
        cy_top = max(node_pos[i][1] for i in ids)
        cx_mid = np.mean([node_pos[i][0] for i in ids])
        ax.text(cx_mid, cy_top + 0.55, ev["label"],
                ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=col, linewidth=1.2, alpha=0.9))

    # ---- Draw article nodes ----
    for aid, (x, y) in node_pos.items():
        col  = node_color[aid]
        conf = node_conf[aid]
        size = 200 + 180 * conf

        ax.scatter(x, y, s=size, c=col, edgecolors=DARK,
                   linewidths=1.1, zorder=4, alpha=0.88)
        ax.text(x, y + 0.06, aid, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=5)

        # Description tooltip below node
        ax.text(x, y - 0.38, node_desc[aid],
                ha="center", va="top", fontsize=6.5,
                color="#333333", linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.85, linewidth=0.6))

    # ---- Country badge ----
    ax.text(6.0, 9.1,
            "Country: India  |  Date: 1 January 2018  |  Three simultaneous events",
            ha="center", va="center", fontsize=10,
            color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=DARK,
                      edgecolor="none"))

    # ---- Legend ----
    legend_x, legend_y = 0.0, 0.5
    ax.plot([legend_x, legend_x+0.55], [legend_y, legend_y],
            color=BLUE, lw=2.5, alpha=0.8)
    ax.text(legend_x + 0.65, legend_y, "Within-cluster edge  (sim > 0.45  — linked)",
            va="center", fontsize=8.5, color=DARK)

    legend_y -= 0.45
    ax.plot([legend_x, legend_x+0.55], [legend_y, legend_y],
            color="#CC3333", lw=1.5, ls="--", alpha=0.7)
    ax.text(legend_x + 0.65, legend_y, "Cross-cluster edge   (sim < 0.45  — rejected)",
            va="center", fontsize=8.5, color=DARK)

    # ---- Bottom annotation ----
    ax.text(6.0, -0.35,
            "Articles share the same country (India) but describe unrelated events "
            "— different topics and locations produce low cross-cluster similarity.",
            ha="center", va="center", fontsize=9, color="#555555", style="italic")

    ax.set_title(
        "Same Country, Different Events: Level-1 Clustering Separates "
        "Unrelated Articles into Distinct Canonical Events",
        fontsize=12, fontweight="bold", pad=10,
    )

    fig.savefig(OUT_DIR / "figF_same_country_separation.pdf")
    fig.savefig(OUT_DIR / "figF_same_country_separation.png")
    plt.close(fig)
    print("figF saved")


if __name__ == "__main__":
    draw()
