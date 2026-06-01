# Creates a conceptual diagram of gradient boosted trees for the dissertation.

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

_HERE = Path(__file__).resolve().parent
_V2   = _HERE.parent
OUT_DIR = _V2 / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────
C_TREE   = "#3B82F6"   # blue  – tree boxes
C_FEAT   = "#10B981"   # green – feature groups
C_ERR    = "#F59E0B"   # amber – residual / error
C_PRED   = "#8B5CF6"   # purple – final prediction
C_ARROW  = "#6B7280"   # grey
C_BG     = "#F8FAFC"
C_LIGHT  = "#EFF6FF"
C_AMBER_LIGHT = "#FFFBEB"
C_PURPLE_LIGHT = "#F5F3FF"
C_GREEN_LIGHT = "#ECFDF5"

FONT = "DejaVu Sans"

# ── helpers ─────────────────────────────────────────────────────────────────

def rounded_box(ax, x, y, w, h, color, label, sublabel=None,
                fontsize=9, text_color="white", alpha=1.0, zorder=3):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.015",
        facecolor=color, edgecolor="white",
        linewidth=1.5, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x, y + 0.012, label, ha="center", va="center",
                fontsize=fontsize, color=text_color,
                fontweight="bold", zorder=zorder + 1)
        ax.text(x, y - 0.02, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=text_color,
                fontstyle="italic", zorder=zorder + 1)
    else:
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color=text_color,
                fontweight="bold", zorder=zorder + 1)


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, style="->"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle=style, color=color,
                        lw=lw, connectionstyle="arc3,rad=0.0"),
        zorder=2,
    )


def curved_arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, rad=0.3):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="->", color=color,
                        lw=lw, connectionstyle=f"arc3,rad={rad}"),
        zorder=2,
    )


# ── main figure ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_facecolor(C_BG)
fig.patch.set_facecolor(C_BG)
ax.axis("off")

# ───────────────────────────────────────────────────────────────────────────
# SECTION A  — Feature input panel (left strip)
# ───────────────────────────────────────────────────────────────────────────
feat_groups = [
    ("GDELT lags",      "protest_14d, strike_28d,\nregional_spillover",  "#059669"),
    ("Markets",         "FX returns, VIX,\ncommodity indices",           "#0891B2"),
    ("Macro",           "inflation CPI,\nunemployment rate",             "#7C3AED"),
    ("Structural",      "GDP growth,\nCovid stringency",                 "#DC2626"),
    ("Trade / Conflict","GTA measures,\nconflict events",                "#D97706"),
]

feat_x = 0.10
feat_h = 0.095
feat_w = 0.16
feat_gap = 0.115
feat_top = 0.85

for i, (grp, desc, col) in enumerate(feat_groups):
    fy = feat_top - i * feat_gap
    rounded_box(ax, feat_x, fy, feat_w, feat_h, col,
                grp, sublabel=desc, fontsize=7.5, zorder=3)

# Brace label
ax.text(feat_x, 0.07, "91 input features\n(country × day)", ha="center",
        va="center", fontsize=8, color="#374151", style="italic")

# Arrow from feature block to first tree
arrow(ax, feat_x + feat_w / 2 + 0.01, 0.5, 0.255, 0.5,
      color=C_ARROW, lw=2)

# ───────────────────────────────────────────────────────────────────────────
# SECTION B  — Three sequential trees
# ───────────────────────────────────────────────────────────────────────────
tree_xs   = [0.34, 0.52, 0.70]
tree_y    = 0.62
tree_w    = 0.13
tree_h    = 0.22
err_y     = 0.30
err_h     = 0.09
err_w     = 0.13

tree_labels = ["Tree 1", "Tree 2", "Tree 3"]
tree_subs   = ["Base\nlearner", "Corrects\nresiduals", "Refines\nfurther"]
err_labels  = ["Residuals\nr₁ = y − ŷ₁", "Residuals\nr₂ = y − ŷ₂", None]

for i, (tx, tl, ts) in enumerate(zip(tree_xs, tree_labels, tree_subs)):
    # Tree box
    rounded_box(ax, tx, tree_y, tree_w, tree_h, C_TREE,
                tl, sublabel=ts, fontsize=9, zorder=3)

    # Small tree icon inside — draw three simple split lines
    # (purely decorative)
    bx0, by0 = tx - tree_w / 2 + 0.005, tree_y - tree_h / 2 + 0.005
    bx1, by1 = tx + tree_w / 2 - 0.005, tree_y + tree_h / 2 - 0.005

    # residual box below each tree (except last — that leads to sum)
    if i < 2:
        ex = tree_xs[i]
        rounded_box(ax, ex, err_y, err_w, err_h, C_ERR,
                    err_labels[i], fontsize=7.5,
                    text_color="white", zorder=3)
        # Down arrow: tree → residual
        arrow(ax, tx, tree_y - tree_h / 2 - 0.005, ex, err_y + err_h / 2 + 0.005,
              color=C_ERR, lw=1.5)
        # Up/right arrow: residual → next tree
        arrow(ax, ex + err_w / 2 + 0.005, err_y,
              tree_xs[i + 1] - tree_w / 2 - 0.005, tree_y,
              color=C_ERR, lw=1.5)

    # Horizontal arrow between trees
    if i < 2:
        arrow(ax, tx + tree_w / 2 + 0.005, tree_y,
              tree_xs[i + 1] - tree_w / 2 - 0.005, tree_y,
              color=C_ARROW, lw=2)

# Dots indicating more trees
ax.text(0.79, tree_y, "· · ·", ha="center", va="center",
        fontsize=14, color=C_ARROW)

# T trees label
ax.text(sum(tree_xs) / len(tree_xs), 0.93, "Sequential tree building  (T trees, each depth ≤ 6)",
        ha="center", va="center", fontsize=9.5, color="#1E3A5F",
        fontweight="bold")

# ───────────────────────────────────────────────────────────────────────────
# SECTION C  — Summation node
# ───────────────────────────────────────────────────────────────────────────
sum_x, sum_y = 0.845, 0.62
sum_r = 0.038

circle = plt.Circle((sum_x, sum_y), sum_r, color="#1D4ED8",
                     zorder=4, linewidth=1.5, edgecolor="white")
ax.add_patch(circle)
ax.text(sum_x, sum_y, "Σ", ha="center", va="center",
        fontsize=14, color="white", fontweight="bold", zorder=5)
ax.text(sum_x, sum_y - 0.065, "Weighted\nsum", ha="center", va="center",
        fontsize=7.5, color="#1E3A5F")

# Arrow from last tree to sum
arrow(ax, tree_xs[-1] + tree_w / 2 + 0.005, tree_y,
      sum_x - sum_r - 0.005, sum_y, color=C_ARROW, lw=2)

# ───────────────────────────────────────────────────────────────────────────
# SECTION D  — Sigmoid + output
# ───────────────────────────────────────────────────────────────────────────
sig_x, sig_y = 0.845, 0.42
sig_w, sig_h = 0.12, 0.09

rounded_box(ax, sig_x, sig_y, sig_w, sig_h, "#7C3AED",
            "σ(·)", sublabel="Sigmoid", fontsize=9, zorder=3)

arrow(ax, sum_x, sum_y - sum_r - 0.005, sig_x, sig_y + sig_h / 2 + 0.005,
      color=C_ARROW, lw=2)

# Output box
out_x, out_y = 0.845, 0.22
out_w, out_h = 0.14, 0.10

rounded_box(ax, out_x, out_y, out_w, out_h, C_PRED,
            "P(protest) = 0.73", sublabel="Country × day score",
            fontsize=8.5, zorder=3)

arrow(ax, sig_x, sig_y - sig_h / 2 - 0.005, out_x, out_y + out_h / 2 + 0.005,
      color=C_ARROW, lw=2)

# ───────────────────────────────────────────────────────────────────────────
# SECTION E  — Loss / training annotation box (bottom centre)
# ───────────────────────────────────────────────────────────────────────────
ann_x, ann_y = 0.52, 0.12
ann_w, ann_h = 0.30, 0.14

box = FancyBboxPatch(
    (ann_x - ann_w / 2, ann_y - ann_h / 2), ann_w, ann_h,
    boxstyle="round,pad=0.015",
    facecolor=C_AMBER_LIGHT, edgecolor=C_ERR,
    linewidth=1.5, zorder=3,
)
ax.add_patch(box)
ax.text(ann_x, ann_y + 0.025, "Training objective",
        ha="center", va="center", fontsize=8.5,
        fontweight="bold", color="#92400E", zorder=4)
ax.text(ann_x, ann_y - 0.01,
        "Binary cross-entropy  +  L1 (α) + L2 (λ) penalty",
        ha="center", va="center", fontsize=8, color="#78350F", zorder=4)
ax.text(ann_x, ann_y - 0.038,
        "scale_pos_weight handles class imbalance  ·  hyper-params via Optuna",
        ha="center", va="center", fontsize=7.5,
        color="#92400E", style="italic", zorder=4)

# ───────────────────────────────────────────────────────────────────────────
# SECTION F  — Walk-forward annotation (top right)
# ───────────────────────────────────────────────────────────────────────────
wf_x, wf_y = 0.845, 0.80
wf_w, wf_h = 0.23, 0.13

box2 = FancyBboxPatch(
    (wf_x - wf_w / 2, wf_y - wf_h / 2), wf_w, wf_h,
    boxstyle="round,pad=0.015",
    facecolor=C_GREEN_LIGHT, edgecolor="#059669",
    linewidth=1.5, zorder=3,
)
ax.add_patch(box2)
ax.text(wf_x, wf_y + 0.025, "Walk-forward backtest",
        ha="center", va="center", fontsize=8.5,
        fontweight="bold", color="#064E3B", zorder=4)
ax.text(wf_x, wf_y - 0.01,
        "Fold 1:  train 2017–2019  →  test 2020",
        ha="center", va="center", fontsize=7.8, color="#065F46", zorder=4)
ax.text(wf_x, wf_y - 0.038,
        "Fold 2:  train 2017–2020  →  test 2021",
        ha="center", va="center", fontsize=7.8, color="#065F46", zorder=4)

# ───────────────────────────────────────────────────────────────────────────
# Title
# ───────────────────────────────────────────────────────────────────────────
ax.text(0.5, 0.975,
        "Gradient Boosted Trees — Protest / Strike Likelihood Prediction",
        ha="center", va="center", fontsize=12,
        fontweight="bold", color="#111827",
        transform=ax.transAxes)

# ───────────────────────────────────────────────────────────────────────────
# Save
# ───────────────────────────────────────────────────────────────────────────
for ext in ("pdf", "png"):
    fp = OUT_DIR / f"08_gbt_diagram.{ext}"
    fig.savefig(fp, dpi=180, bbox_inches="tight", facecolor=C_BG)
    print(f"Saved: {fp}")

plt.close(fig)
