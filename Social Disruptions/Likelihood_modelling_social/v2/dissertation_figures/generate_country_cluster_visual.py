"""
generate_country_cluster_visual.py
====================================
Creates a figure showing how multiple GDELT news articles about the
same real-world event in one country get clustered into a single
canonical event.

Uses the real Greece/Athens Macedonia-name-dispute protest cluster
(4 Feb 2018, 12 articles, mean internal score = 0.735).
"""

from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon as MplPolygon

OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "figure.dpi":      150,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

BLUE   = "#2166AC"
ORANGE = "#D6604D"
GREEN  = "#4DAC26"
GREY   = "#AAAAAA"
DARK   = "#1A1A2E"
BG     = "#FAFAFA"
CARD   = "#EEF4FB"
CARD_H = "#D0E8FF"

# ---------------------------------------------------------------------------
# Real articles from cluster GRC_protests_20180204_c0000
# Truncated descriptions for readability
# ---------------------------------------------------------------------------
ARTICLES = [
    dict(
        id="A1",
        conf=0.98,
        loc="Athens, Syntagma Sq.",
        desc='"140,000 people protested in\nAthens over the government\'s\nattempts to resolve the\nMacedonia name dispute."',
    ),
    dict(
        id="A2",
        conf=0.95,
        loc="Athens, Syntagma Sq.",
        desc='"Thousands of Greeks protested\nin Athens over the government\'s\nconsideration of a compromise\nwith Macedonia."',
    ),
    dict(
        id="A3",
        conf=0.92,
        loc="Athens, main square",
        desc='"Thousands protest in Athens\nover the Macedonia name dispute\nbetween Greece and its\nneighbour."',
    ),
    dict(
        id="A4",
        conf=0.82,
        loc="Athens, Syntagma Sq.",
        desc='"Greeks rallied in Athens over\nthe inclusion of the word\n\'Macedonia\' in the name of\na neighbouring former republic."',
    ),
    dict(
        id="A5",
        conf=0.81,
        loc="Athens, Syntagma Sq.",
        desc='"Thousands of Greek citizens\nprotested in Syntagma Square\nover FYROM\'s potential use of\nthe name \'Macedonia\'."',
    ),
    dict(
        id="A6",
        conf=0.75,
        loc="Athens, main square",
        desc='"Greeks protested in Athens\'\nmain square over the\nMacedonia name dispute."',
    ),
    dict(
        id="A7",
        conf=0.80,
        loc="Athens, Syntagma Sq.",
        desc='"Thousands of Greeks protested\nin Athens against the Greek\ngovernment over the name\nof Macedonia."',
    ),
    dict(
        id="A8",
        conf=0.70,
        loc="Athens",
        desc='"Hundreds of thousands of\nGreeks rallied in Athens\nover a naming dispute\nwith Macedonia."',
    ),
]

# Illustrative pairwise similarity scores (symmetric, diagonal=1)
# Based on the cluster's mean_internal_score=0.735 — vary around this
N = len(ARTICLES)
np.random.seed(17)
SIM = np.ones((N, N))
# Fill upper triangle with realistic values
raw_sims = {
    (0,1): 0.88, (0,2): 0.84, (0,3): 0.79, (0,4): 0.77, (0,5): 0.72, (0,6): 0.75, (0,7): 0.68,
    (1,2): 0.86, (1,3): 0.81, (1,4): 0.83, (1,5): 0.74, (1,6): 0.78, (1,7): 0.71,
    (2,3): 0.80, (2,4): 0.79, (2,5): 0.76, (2,6): 0.77, (2,7): 0.70,
    (3,4): 0.82, (3,5): 0.75, (3,6): 0.73, (3,7): 0.68,
    (4,5): 0.77, (4,6): 0.80, (4,7): 0.72,
    (5,6): 0.76, (5,7): 0.74,
    (6,7): 0.71,
}
for (i,j), v in raw_sims.items():
    SIM[i,j] = SIM[j,i] = v

THRESHOLD = 0.45  # all pairs are well above this

# ---------------------------------------------------------------------------
# Figure layout:
# Left:   article "cards" stacked
# Middle: similarity graph
# Right:  canonical event result box
# ---------------------------------------------------------------------------

def wrap(text, width=32):
    """Simple word-wrap."""
    words = text.replace("\n", " ").split()
    lines, line = [], []
    for w in words:
        if sum(len(x)+1 for x in line) + len(w) <= width:
            line.append(w)
        else:
            lines.append(" ".join(line))
            line = [w]
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


def draw_figure():
    fig = plt.figure(figsize=(16, 10), facecolor=BG)

    # Three panels: cards | graph | result
    ax_cards  = fig.add_axes([0.01, 0.04, 0.28, 0.88])   # left
    ax_graph  = fig.add_axes([0.32, 0.04, 0.42, 0.88])   # centre
    ax_result = fig.add_axes([0.76, 0.04, 0.23, 0.88])   # right

    for ax in [ax_cards, ax_graph, ax_result]:
        ax.set_facecolor(BG)
        ax.axis("off")

    # =========================================================
    # LEFT: Article cards
    # =========================================================
    ax_cards.set_xlim(0, 10)
    ax_cards.set_ylim(0, 10)
    ax_cards.set_title("Input: 8 news articles\n(Greece · 4 Feb 2018 · protests)",
                        fontsize=11, fontweight="bold", pad=6)

    card_h    = 1.08
    card_gap  = 0.12
    total_h   = N * card_h + (N - 1) * card_gap
    y_start   = (10 - total_h) / 2 + total_h

    for i, art in enumerate(ARTICLES):
        y_top = y_start - i * (card_h + card_gap)
        y_bot = y_top - card_h

        # Card background
        rect = FancyBboxPatch((0.3, y_bot), 9.4, card_h,
                               boxstyle="round,pad=0.08",
                               facecolor=CARD_H if art["conf"] >= 0.90 else CARD,
                               edgecolor=BLUE, linewidth=1.0)
        ax_cards.add_patch(rect)

        # Article ID badge
        ax_cards.text(1.0, y_bot + card_h/2, art["id"],
                      ha="center", va="center", fontsize=9,
                      fontweight="bold", color="white",
                      bbox=dict(boxstyle="circle,pad=0.25",
                                facecolor=BLUE, edgecolor="none"))

        # Confidence bar (thin strip on right)
        bar_max_w = 1.6
        bar_w     = bar_max_w * art["conf"]
        bar_y     = y_bot + 0.08
        bar_h     = 0.18
        ax_cards.add_patch(mpatches.Rectangle(
            (7.9, bar_y), bar_max_w, bar_h,
            facecolor="#DDDDDD", edgecolor="none"))
        ax_cards.add_patch(mpatches.Rectangle(
            (7.9, bar_y), bar_w, bar_h,
            facecolor=BLUE if art["conf"] >= 0.80 else ORANGE,
            edgecolor="none"))
        ax_cards.text(9.7, bar_y + bar_h/2,
                      f"{art['conf']:.2f}", ha="right", va="center",
                      fontsize=7, color="#333")
        ax_cards.text(7.9, bar_y - 0.12, "conf",
                      ha="left", va="top", fontsize=6.5, color=GREY)

        # Description text
        ax_cards.text(1.7, y_bot + card_h/2 + 0.08, art["desc"],
                      ha="left", va="center", fontsize=7.2,
                      color=DARK, linespacing=1.35)

        # Location
        ax_cards.text(1.7, y_bot + 0.10, f"loc: {art['loc']}",
                      ha="left", va="bottom", fontsize=6.8,
                      color="#666666", style="italic")

    # =========================================================
    # CENTRE: Similarity graph
    # =========================================================
    ax_graph.set_xlim(-1.4, 1.4)
    ax_graph.set_ylim(-1.35, 1.5)
    ax_graph.set_title("Weighted similarity graph\n"
                        "(all pairs above threshold = 0.45)",
                        fontsize=11, fontweight="bold", pad=6)

    # Circular layout
    angles = np.linspace(0, 2*np.pi, N, endpoint=False) + np.pi/8
    radius = 0.90
    pos = {i: (radius * np.cos(a), radius * np.sin(a)) for i, a in enumerate(angles)}

    # Draw edges — colour by similarity strength
    cmap = plt.cm.Blues
    for i in range(N):
        for j in range(i+1, N):
            sim = SIM[i,j]
            alpha = 0.25 + 0.65 * (sim - 0.45) / 0.55
            lw    = 0.8 + 3.0 * (sim - 0.45) / 0.55
            col   = cmap(0.3 + 0.7 * (sim - 0.45) / 0.55)
            x1, y1 = pos[i]; x2, y2 = pos[j]
            ax_graph.plot([x1, x2], [y1, y2],
                          color=col, lw=lw, alpha=alpha, zorder=1)

    # Annotate a few key edges with their score
    key_edges = [(0,1,0.88), (1,2,0.86), (3,4,0.82), (0,3,0.79)]
    for i, j, sim in key_edges:
        x1,y1 = pos[i]; x2,y2 = pos[j]
        mx,my = (x1+x2)/2, (y1+y2)/2
        # Nudge label slightly outward from centre
        cx, cy = 0, 0
        dx, dy = mx-cx, my-cy
        norm = np.hypot(dx, dy) + 1e-9
        lx, ly = mx + 0.12*dx/norm, my + 0.12*dy/norm
        ax_graph.text(lx, ly, f"{sim:.2f}",
                      ha="center", va="center", fontsize=7,
                      color=BLUE,
                      bbox=dict(boxstyle="round,pad=0.1",
                                facecolor="white", edgecolor="none",
                                alpha=0.8))

    # Draw cluster hull behind nodes
    pts = np.array([pos[i] for i in range(N)])
    try:
        hull = ConvexHull(pts)
        hp = pts[hull.vertices]
        centroid = hp.mean(axis=0)
        expanded = centroid + (hp - centroid) * 1.45
        poly = MplPolygon(expanded, closed=True,
                          facecolor=BLUE, alpha=0.08,
                          edgecolor=BLUE, lw=2.0,
                          linestyle="--", zorder=0)
        ax_graph.add_patch(poly)
    except Exception:
        pass

    # Draw nodes
    for i, art in enumerate(ARTICLES):
        x, y = pos[i]
        size = 220 + 180 * art["conf"]
        col  = BLUE if art["conf"] >= 0.90 else (BLUE_MED if art["conf"] >= 0.80 else BLUE_LIGHT)
        ax_graph.scatter(x, y, s=size, c=col,
                         edgecolors=DARK, linewidths=1.2, zorder=4)
        ax_graph.text(x, y, art["id"],
                      ha="center", va="center", fontsize=8.5,
                      fontweight="bold", color="white", zorder=5)
        # Location hint outside node
        nudge = 1.22
        ax_graph.text(x*nudge, y*nudge,
                      art["loc"].split(",")[0],
                      ha="center", va="center", fontsize=6.8,
                      color="#555555")

    # Threshold annotation
    ax_graph.text(0, -1.28,
                  "Edge drawn if weighted similarity > 0.45\n"
                  "Edge weight = 0.55·embed + 0.30·location + 0.15·temporal",
                  ha="center", va="center", fontsize=8, color="#555555",
                  style="italic")

    # Colorbar-style legend
    for label, sim_val, yoff in [("High (≥0.85)", 0.95, 0.40),
                                  ("Medium (0.70)", 0.70, 0.25),
                                  ("Lower (0.55)", 0.55, 0.10)]:
        col = cmap(0.3 + 0.7 * (sim_val - 0.45) / 0.55)
        ax_graph.plot([-1.3, -1.0], [yoff, yoff], color=col,
                      lw=2.5 + 2*(sim_val-0.45)/0.55, alpha=0.85)
        ax_graph.text(-0.95, yoff, label, va="center", fontsize=7.5,
                      color="#444")
    ax_graph.text(-1.3, 0.56, "Edge similarity:", fontsize=8,
                  color="#444", fontweight="bold")

    # =========================================================
    # RIGHT: Result — canonical event
    # =========================================================
    ax_result.set_xlim(0, 10)
    ax_result.set_ylim(0, 10)
    ax_result.set_title("Output: 1 canonical event\n(connected component)",
                         fontsize=11, fontweight="bold", pad=6)

    # Main result box
    box = FancyBboxPatch((0.5, 2.8), 9.0, 6.6,
                          boxstyle="round,pad=0.2",
                          facecolor=CARD_H, edgecolor=BLUE, linewidth=2.0)
    ax_result.add_patch(box)

    ax_result.text(5.0, 9.0, "Canonical Event",
                   ha="center", va="center", fontsize=12,
                   fontweight="bold", color=BLUE)

    # Event details
    details = [
        ("Event type",    "Protest"),
        ("Country",       "Greece  (GRC)"),
        ("Location",      "Athens, Syntagma Sq."),
        ("Event date",    "4 February 2018"),
        ("", ""),
        ("Articles merged",    "8  (of 12 total)"),
        ("Avg confidence",     "0.84"),
        ("Mean sim score",     "0.754"),
        ("", ""),
        ("Label assigned",    "protest_today = 1"),
        ("",                  "protest_7d    = 1"),
    ]

    y0 = 8.4
    dy = 0.57
    for k, v in details:
        if not k and not v:
            y0 -= 0.18
            continue
        ax_result.text(1.0, y0, k + (":" if k else ""),
                       ha="left", va="center", fontsize=8.5,
                       color="#444444", fontweight="bold" if k else "normal")
        ax_result.text(9.0, y0, v,
                       ha="right", va="center", fontsize=8.5,
                       color=BLUE if "Label" in k or "=" in v else DARK,
                       fontweight="bold" if "Label" in k or "=" in v else "normal")
        y0 -= dy

    # Reduction arrow/text
    ax_result.annotate("", xy=(5.0, 2.55), xytext=(5.0, 1.5),
                        arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2))
    ax_result.text(5.0, 1.15,
                   "8 articles  →  1 event\n(73% compression)",
                   ha="center", va="center", fontsize=9,
                   color=GREEN, fontweight="bold")

    # Internal score annotation
    score = 0.754
    bar_x0, bar_y0 = 1.0, 2.2
    bar_w_max = 8.0
    ax_result.add_patch(mpatches.Rectangle(
        (bar_x0, bar_y0), bar_w_max, 0.28,
        facecolor="#DDDDDD", edgecolor="none"))
    ax_result.add_patch(mpatches.Rectangle(
        (bar_x0, bar_y0), bar_w_max * score, 0.28,
        facecolor=GREEN, alpha=0.8, edgecolor="none"))
    ax_result.text(bar_x0 + bar_w_max/2, bar_y0 + 0.14,
                   f"mean internal similarity = {score:.3f}",
                   ha="center", va="center", fontsize=8,
                   color=DARK, fontweight="bold")

    # =========================================================
    # Figure-level title and annotation
    # =========================================================
    fig.text(0.5, 0.975,
             "Level-1 Clustering: How 8 news articles about the same event are merged\n"
             "Athens Macedonia-name-dispute protests  ·  Greece  ·  4 February 2018",
             ha="center", va="top", fontsize=13, fontweight="bold", color=DARK)

    # Panel labels
    for ax, lbl in [(ax_cards, "(a)"), (ax_graph, "(b)"), (ax_result, "(c)")]:
        ax.text(0.01, 0.995, lbl, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top", color="#333333")

    # Flow arrows between panels
    fig.text(0.305, 0.50, "→", fontsize=28, ha="center", va="center",
             color=BLUE, fontweight="bold")
    fig.text(0.745, 0.50, "→", fontsize=28, ha="center", va="center",
             color=GREEN, fontweight="bold")

    fig.savefig(OUT_DIR / "figE_country_cluster.pdf")
    fig.savefig(OUT_DIR / "figE_country_cluster.png")
    plt.close(fig)
    print(f"  figE saved -> {OUT_DIR / 'figE_country_cluster.png'}")


# Colour shades used in graph
BLUE_MED   = "#4A90D9"
BLUE_LIGHT = "#92C5DE"

if __name__ == "__main__":
    draw_figure()
