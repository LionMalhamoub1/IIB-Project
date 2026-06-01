# Generates dissertation-quality figures explaining the GDELT event clustering approach.

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                      # non-interactive backend
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE    = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent.parent / "GDELT_Verification" / "grouped"
OUT_DIR  = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Dissertation style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

BLUE   = "#2166AC"
ORANGE = "#D6604D"
GREEN  = "#4DAC26"
GREY   = "#AAAAAA"
LIGHT  = "#EEF4FB"

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_all_clusters() -> pd.DataFrame:
    """Load all *_grouped.jsonl files into one DataFrame."""
    records = []
    for f in sorted(DATA_DIR.glob("*_grouped.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append({
                    "cluster_id":          obj.get("cluster_id", ""),
                    "disruption_type":     obj.get("disruption_type", ""),
                    "event_date":          obj.get("event_date", ""),
                    "iso3":                obj.get("iso3", ""),
                    "n_articles":          obj.get("n_articles", 1),
                    "mean_internal_score": obj.get("mean_internal_score", np.nan),
                    "confidence_max":      obj.get("confidence_max", np.nan),
                })
    return pd.DataFrame(records)


def load_all_reports() -> pd.DataFrame:
    """Load all *_grouped_report.json files into one DataFrame."""
    records = []
    for f in sorted(DATA_DIR.glob("*_grouped_report.json")):
        with open(f, encoding="utf-8") as fh:
            obj = json.load(fh)
        tb = obj.get("type_breakdown", {})
        records.append({
            "date":              obj.get("date", ""),
            "raw_records":       obj.get("raw_records", 0),
            "canonical_events":  obj.get("canonical_events", 0),
            "reduction_pct":     obj.get("reduction_pct", 0.0),
            "protest_raw":       tb.get("protests",      {}).get("raw",     0),
            "protest_grouped":   tb.get("protests",      {}).get("grouped", 0),
            "strike_raw":        tb.get("labour_strike", {}).get("raw",     0),
            "strike_grouped":    tb.get("labour_strike", {}).get("grouped", 0),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Figure 1 — Pipeline architecture flowchart
# ---------------------------------------------------------------------------

def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # Box drawing helper
    def box(x, y, w, h, label, sublabel="", color=LIGHT, edgecolor=BLUE, fontsize=10):
        rect = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor=edgecolor, linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x, y + (0.15 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize, fontweight="bold",
                color="#1A1A2E")
        if sublabel:
            ax.text(x, y - 0.32, sublabel,
                    ha="center", va="center", fontsize=8, color="#444444",
                    style="italic")

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#333333",
                            lw=1.4, mutation_scale=14),
        )
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.15, my, label, fontsize=8, color="#555555",
                    va="center")

    # Column x-positions
    cx = 5.0   # main column
    lx = 2.5   # left column (level 1)
    rx = 7.5   # right column (level 2)

    # ---- Stage boxes (top → bottom, main column) ----
    box(cx, 13.0, 5.5, 0.9, "Raw GDELT Extractions",
        "articles with entities, location, date, type", color="#E8F4FD")
    arrow(cx, 12.55, cx, 11.95)

    box(cx, 11.6, 5.5, 0.9, "Confidence Filter  (≥ 0.35)",
        "remove low-confidence records", color="#E8F4FD")
    arrow(cx, 11.15, cx, 10.55)

    box(cx, 10.2, 5.5, 0.9, "Sentence-Transformer Embeddings",
        "all-MiniLM-L6-v2  →  384-dim vectors", color="#E8F4FD")
    arrow(cx, 9.75, cx, 9.15)

    # ---- Level 1 block ----
    box(cx, 8.5, 6.0, 1.1, "Level-1 Clustering  (Article → Event)",
        "weighted similarity graph, threshold = 0.45", color="#D0E8FF", edgecolor="#1565C0")

    # Level-1 sub-boxes
    box(lx, 7.0, 3.5, 0.75, "Embedding sim  (55%)",
        "cosine similarity", color="#EEF4FB", edgecolor=GREY, fontsize=9)
    box(cx, 7.0, 2.2, 0.75, "Location (30%)",
        "fuzzy geo match", color="#EEF4FB", edgecolor=GREY, fontsize=9)
    box(rx, 7.0, 2.8, 0.75, "Temporal (15%)",
        "dynamic date window", color="#EEF4FB", edgecolor=GREY, fontsize=9)

    # arrows from level-1 box to sub-boxes
    for tx in [lx, cx, rx]:
        arrow(cx, 7.95, tx, 7.38)

    # aggregate arrow
    arrow(lx, 6.62, cx, 6.0, "")
    arrow(cx, 6.62, cx, 6.0, "")
    arrow(rx, 6.62, cx, 6.0, "")

    box(cx, 5.6, 5.5, 0.75, "Canonical Events  (connected components)",
        "each cluster = one real-world event", color="#C8E6C9", edgecolor=GREEN)

    arrow(cx, 5.22, cx, 4.62)

    # ---- Level 2 block ----
    box(cx, 4.25, 6.0, 0.75, "Level-2 Clustering  (Event → Movement)",
        "centroid cosine ≥ 0.72, same country/type, ≤ 14 days",
        color="#FFE0B2", edgecolor="#E65100")

    arrow(cx, 3.87, cx, 3.27)

    box(cx, 2.9, 5.5, 0.75, "Movements  (sustained campaigns)",
        "linked clusters across multiple days", color="#FFE0B2", edgecolor="#E65100")

    arrow(cx, 2.52, cx, 1.92)

    box(cx, 1.55, 5.5, 0.9,
        "Binary Labels  (protest_7d, strike_7d, strike_30d)",
        "country × day targets for XGBoost", color="#E8F5E9", edgecolor=GREEN)

    ax.set_title("GDELT Event Clustering Pipeline", fontsize=14, fontweight="bold", pad=6)

    fig.savefig(OUT_DIR / "fig1_pipeline_architecture.pdf")
    fig.savefig(OUT_DIR / "fig1_pipeline_architecture.png")
    plt.close(fig)
    print("  fig1 saved")


# ---------------------------------------------------------------------------
# Figure 2 — Similarity weight breakdown
# ---------------------------------------------------------------------------

def fig2_similarity_weights():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # --- Left: Level-1 weights (pie) ---
    ax = axes[0]
    weights = [0.55, 0.30, 0.15]
    labels  = ["Semantic\nembedding", "Geographic\nlocation", "Temporal\nproximity"]
    colors  = [BLUE, ORANGE, GREEN]
    wedges, texts, autotexts = ax.pie(
        weights, labels=labels, colors=colors,
        autopct="%d%%", startangle=120,
        pctdistance=0.65,
        wedgeprops=dict(linewidth=1.2, edgecolor="white"),
        textprops=dict(fontsize=10),
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("Level-1: Article → Event\nSimilarity Weights", fontsize=11, fontweight="bold")

    # --- Right: Level-2 parameters (bar-style legend) ---
    ax2 = axes[1]
    ax2.axis("off")

    params = [
        ("Cosine threshold (L1, edges)",   "≥ 0.45"),
        ("Cosine threshold (L2, movements)","≥ 0.72"),
        ("Protest temporal window",         "3 – 7 days"),
        ("Strike temporal window",          "7 – 21 days"),
        ("Sub-location fuzzy gate",         "≥ 0.20"),
        ("Movement max gap",                "≤ 14 days"),
        ("Confidence weight",               "sim × √min(conf_a, conf_b)"),
        ("Embedding model",                 "all-MiniLM-L6-v2  (384-dim)"),
    ]

    y0 = 0.92
    dy = 0.11
    ax2.text(0.0, y0 + 0.03, "Clustering Hyper-parameters",
             transform=ax2.transAxes, fontsize=11, fontweight="bold")
    for i, (k, v) in enumerate(params):
        y = y0 - i * dy
        ax2.text(0.02, y, f"• {k}:", transform=ax2.transAxes,
                 fontsize=9.5, color="#333333")
        ax2.text(0.55, y, v, transform=ax2.transAxes,
                 fontsize=9.5, color=BLUE, fontweight="bold")

    fig.suptitle("Similarity Scheme and Clustering Parameters",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_similarity_weights.pdf")
    fig.savefig(OUT_DIR / "fig2_similarity_weights.png")
    plt.close(fig)
    print("  fig2 saved")


# ---------------------------------------------------------------------------
# Figure 3 — Cluster size distribution
# ---------------------------------------------------------------------------

def fig3_cluster_size(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    type_map   = {"protests": "Protests", "labour_strike": "Labour Strikes"}
    colors_map = {"protests": BLUE, "labour_strike": ORANGE}

    for ax, dtype in zip(axes, ["protests", "labour_strike"]):
        sub = df[df["disruption_type"] == dtype]["n_articles"]
        bins = np.arange(0.5, min(sub.max() + 2, 52), 1)
        ax.hist(sub, bins=bins, color=colors_map[dtype], edgecolor="white",
                linewidth=0.4, log=True)
        ax.set_xlabel("Articles per canonical event")
        ax.set_ylabel("Count (log scale)")
        ax.set_title(type_map[dtype])
        ax.set_xlim(0.5, 50.5)

        # Annotate median
        med = sub.median()
        ax.axvline(med, color="#333333", ls="--", lw=1.2)
        ax.text(med + 0.5, ax.get_ylim()[1] * 0.6,
                f"median = {med:.0f}", fontsize=8.5, color="#333333")

        frac_multi = (sub > 1).mean() * 100
        ax.text(0.97, 0.97, f"{frac_multi:.0f}% multi-article",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=colors_map[dtype])

    fig.suptitle("Distribution of Articles per Canonical Event Cluster",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_cluster_size_dist.pdf")
    fig.savefig(OUT_DIR / "fig3_cluster_size_dist.png")
    plt.close(fig)
    print("  fig3 saved")


# ---------------------------------------------------------------------------
# Figure 4 — Cluster cohesion (mean_internal_score)
# ---------------------------------------------------------------------------

def fig4_cohesion(df: pd.DataFrame):
    multi = df[df["n_articles"] > 1].copy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    type_map   = {"protests": "Protests", "labour_strike": "Labour Strikes"}
    colors_map = {"protests": BLUE, "labour_strike": ORANGE}

    for ax, dtype in zip(axes, ["protests", "labour_strike"]):
        sub = multi[multi["disruption_type"] == dtype]["mean_internal_score"].dropna()
        ax.hist(sub, bins=30, range=(0, 1), color=colors_map[dtype],
                edgecolor="white", linewidth=0.4)
        ax.axvline(0.45, color="#AA0000", ls="--", lw=1.2, label="Edge threshold (0.45)")
        ax.set_xlabel("Mean intra-cluster similarity")
        ax.set_ylabel("Count")
        ax.set_title(type_map[dtype])
        ax.legend(fontsize=8)

        med = sub.median()
        ax.axvline(med, color="#333333", ls=":", lw=1.2)
        ax.text(med + 0.01, ax.get_ylim()[1] * 0.85,
                f"median={med:.2f}", fontsize=8, color="#333333")

    fig.suptitle("Intra-Cluster Cohesion for Multi-Article Events\n"
                 "(single-article clusters excluded)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_cohesion_dist.pdf")
    fig.savefig(OUT_DIR / "fig4_cohesion_dist.png")
    plt.close(fig)
    print("  fig4 saved")


# ---------------------------------------------------------------------------
# Figure 5 — Article reduction by type
# ---------------------------------------------------------------------------

def fig5_reduction(reports: pd.DataFrame):
    # Aggregate across all days
    total_protest_raw     = reports["protest_raw"].sum()
    total_protest_grouped = reports["protest_grouped"].sum()
    total_strike_raw      = reports["strike_raw"].sum()
    total_strike_grouped  = reports["strike_grouped"].sum()

    categories = ["Protests", "Labour Strikes", "Total"]
    raw_vals   = [total_protest_raw,     total_strike_raw,
                  total_protest_raw + total_strike_raw]
    grp_vals   = [total_protest_grouped, total_strike_grouped,
                  total_protest_grouped + total_strike_grouped]

    x = np.arange(len(categories))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_r = ax.bar(x - w/2, raw_vals, w, label="Raw GDELT articles",
                    color=GREY, edgecolor="white", linewidth=0.5)
    bars_g = ax.bar(x + w/2, grp_vals, w, label="Canonical events",
                    color=BLUE, edgecolor="white", linewidth=0.5)

    # Reduction % labels above grouped bars
    for xp, rv, gv in zip(x, raw_vals, grp_vals):
        red_pct = (1 - gv / rv) * 100 if rv > 0 else 0
        ax.text(xp + w/2, gv + max(raw_vals) * 0.01,
                f"−{red_pct:.0f}%", ha="center", va="bottom",
                fontsize=9, color=BLUE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Count (all days)")
    ax.set_title("Article-to-Event Reduction by Disruption Type\n"
                 "(2018 sample period)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_reduction_by_type.pdf")
    fig.savefig(OUT_DIR / "fig5_reduction_by_type.png")
    plt.close(fig)
    print("  fig5 saved")


# ---------------------------------------------------------------------------
# Figure 6 — Events per country (top 25)
# ---------------------------------------------------------------------------

def fig6_country_events(df: pd.DataFrame):
    counts = (
        df.groupby(["iso3", "disruption_type"])
        .size()
        .reset_index(name="n")
    )
    # Top-25 by total events
    top25 = (
        counts.groupby("iso3")["n"].sum()
        .nlargest(25)
        .index.tolist()
    )
    sub = counts[counts["iso3"].isin(top25)].copy()
    pivot = (
        sub.pivot(index="iso3", columns="disruption_type", values="n")
        .fillna(0)
        .reindex(top25)
        .sort_values("protests", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 9))
    y = np.arange(len(pivot))
    w = 0.4

    ax.barh(y - w/2, pivot.get("protests", 0), w,
            label="Protests", color=BLUE, edgecolor="white")
    ax.barh(y + w/2, pivot.get("labour_strike", 0), w,
            label="Labour strikes", color=ORANGE, edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Number of canonical events")
    ax.set_title("Events per Country  —  Top 25\n(2018 sample period)",
                 fontsize=12, fontweight="bold")
    ax.legend()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_events_per_country.pdf")
    fig.savefig(OUT_DIR / "fig6_events_per_country.png")
    plt.close(fig)
    print("  fig6 saved")


# ---------------------------------------------------------------------------
# Figure 7 — Dynamic temporal windows by event type
# ---------------------------------------------------------------------------

def fig7_temporal_windows():
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")

    # Timeline base
    t_start, t_end = 0, 24
    ax.set_xlim(-1, t_end + 2)
    ax.set_ylim(-1, 5.5)

    def draw_timeline(y, label, anchor_day, windows, colors, type_label):
        # Horizontal baseline
        ax.annotate("", xy=(t_end, y), xytext=(0, y),
                    arrowprops=dict(arrowstyle="-|>", color="#999999", lw=1))

        # Anchor article marker
        ax.plot(anchor_day, y, "D", color="#333333", ms=8, zorder=5)
        ax.text(anchor_day, y + 0.35, "Article\nA₀", ha="center",
                fontsize=8, color="#333333")

        # Draw window spans
        for i, (lo, hi, col, wlabel) in enumerate(windows):
            yoffset = y - 0.55 - i * 0.45
            ax.annotate("", xy=(anchor_day + hi, yoffset),
                        xytext=(anchor_day + lo, yoffset),
                        arrowprops=dict(arrowstyle="<->",
                                        color=col, lw=2))
            ax.text((anchor_day + lo + anchor_day + hi) / 2,
                    yoffset - 0.3, wlabel,
                    ha="center", fontsize=8.5, color=col, fontweight="bold")

        # Type label
        ax.text(-0.5, y, label, ha="right", va="center",
                fontsize=10.5, fontweight="bold")

        # Day tick marks
        for d in range(0, t_end + 1, 3):
            ax.plot([d, d], [y - 0.12, y + 0.12], color="#CCCCCC", lw=0.7)
            ax.text(d, y - 0.35, str(d), ha="center", fontsize=7, color="#888888")
        ax.text(t_end / 2, y - 1.5, "Days relative to A₀",
                ha="center", fontsize=8, color="#666666")

    # Protests row
    draw_timeline(
        y=4.2, label="Protests", anchor_day=6,
        windows=[
            (0, 3, BLUE,   "min window\n(3 days)"),
            (0, 7, "#85C1E9", "max window\n(7 days)"),
        ],
        colors=[BLUE, "#85C1E9"],
        type_label="protests",
    )

    # Labour strikes row
    draw_timeline(
        y=1.5, label="Labour\nStrikes", anchor_day=6,
        windows=[
            (0,  7, ORANGE,    "min window\n(7 days)"),
            (0, 21, "#FAD7A0", "max window\n(21 days)"),
        ],
        colors=[ORANGE, "#FAD7A0"],
        type_label="labour_strike",
    )

    ax.set_title("Dynamic Temporal Linking Windows by Event Type\n"
                 "Articles within the window can be linked to the same cluster",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig7_temporal_windows.pdf")
    fig.savefig(OUT_DIR / "fig7_temporal_windows.png")
    plt.close(fig)
    print("  fig7 saved")


# ---------------------------------------------------------------------------
# Figure 8 — Daily reduction time series
# ---------------------------------------------------------------------------

def fig8_daily_reduction(reports: pd.DataFrame):
    rpt = reports.copy()
    rpt["date_dt"] = pd.to_datetime(rpt["date"], format="%Y%m%d")
    rpt = rpt.sort_values("date_dt")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    # Top: raw vs canonical
    ax = axes[0]
    ax.fill_between(rpt["date_dt"], rpt["raw_records"],
                    alpha=0.25, color=GREY, label="Raw articles")
    ax.fill_between(rpt["date_dt"], rpt["canonical_events"],
                    alpha=0.8, color=BLUE, label="Canonical events")
    ax.set_ylabel("Count")
    ax.set_title("Daily Article Volume and Clustering Reduction", fontsize=11)
    ax.legend(loc="upper right")

    # Bottom: reduction %
    ax2 = axes[1]
    ax2.plot(rpt["date_dt"], rpt["reduction_pct"],
             color=ORANGE, lw=1.2)
    ax2.axhline(rpt["reduction_pct"].mean(), color=ORANGE,
                ls="--", lw=0.9, alpha=0.7)
    ax2.text(rpt["date_dt"].iloc[-1], rpt["reduction_pct"].mean() + 1,
             f"mean = {rpt['reduction_pct'].mean():.0f}%",
             ha="right", fontsize=8.5, color=ORANGE)
    ax2.set_ylabel("Reduction (%)")
    ax2.set_xlabel("Date")
    ax2.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_daily_reduction.pdf")
    fig.savefig(OUT_DIR / "fig8_daily_reduction.png")
    plt.close(fig)
    print("  fig8 saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data …")
    df      = load_all_clusters()
    reports = load_all_reports()
    print(f"  Clusters: {len(df):,}  |  Report days: {len(reports):,}")

    print("Generating figures …")
    fig1_pipeline()
    fig2_similarity_weights()
    fig3_cluster_size(df)
    fig4_cohesion(df)
    fig5_reduction(reports)
    fig6_country_events(df)
    fig7_temporal_windows()
    fig8_daily_reduction(reports)

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
