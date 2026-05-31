"""
generate_plots.py
=================
Generates all exploratory visualisations for the combined flood dataset.

Run from the project root:
    python Builder_Combined/Visualise_Dataset/generate_plots.py

Outputs saved to Builder_Combined/Visualise_Dataset/plots/
  .html  — interactive Plotly figures (open in browser)
  .png   — static Matplotlib / Seaborn figures
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "Builder_Combined" / "outputs" / "combined_floods.jsonl"
EVAL_PATH = ROOT / "Builder_Matching"  / "outputs" / "evaluation_report.json"
OUT_DIR   = Path(__file__).resolve().parent / "plots"
OUT_DIR.mkdir(exist_ok=True)

# ── Style ───────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.dpi"] = 150

COLORS = {
    "matched_gdelt":   "#2ecc71",
    "unmatched_gdelt": "#3498db",
    "unmatched_ref":   "#e74c3c",
}
LABEL = {
    "matched_gdelt":   "Matched (GDELT + Reference)",
    "unmatched_gdelt": "GDELT only",
    "unmatched_ref":   "Reference only (missed by GDELT)",
}

HYDRO_PAIRS = [
    ("chirps_7d_total_mm",      "CHIRPS 7-day total (mm)"),
    ("gpm_7d_total_mm",         "GPM 7-day total (mm)"),
    ("spi_30d",                 "SPI-30"),
    ("era5_soil_moisture_day0", "ERA5 soil moisture (day 0)"),
    ("chirps_7d_anom_pct",      "CHIRPS 7-day anomaly (%)"),
    ("pop_density_km2",         "Population density (km²)"),
]

HYDRO_FIELD_NAMES = [
    "chirps_3d_total_mm", "chirps_7d_total_mm", "chirps_14d_total_mm",
    "chirps_30d_total_mm", "chirps_peak_daily_mm",
    "chirps_7d_baseline_mm", "chirps_7d_anom_mm", "chirps_7d_anom_pct",
    "gpm_1d_total_mm", "gpm_3d_total_mm", "gpm_7d_total_mm",
    "gpm_peak_daily_mm", "gpm_peak_3h_mm",
    "era5_soil_moisture_day0", "era5_soil_moisture_7d_mean",
    "era5_soil_moisture_30d_mean", "era5_soil_moisture_deep_day0",
    "era5_soil_moisture_deep_7d_mean", "era5_precip_7d_mm", "era5_runoff_7d_mm",
    "pop_count_25km", "pop_density_km2",
    "jrc_occurrence_pct", "jrc_recurrence_pct", "terrain_slope_mean",
    "spi_30d", "spi_30d_pct",
]


# ── Data loading ─────────────────────────────────────────────────────────────

def load():
    print("Loading combined dataset...")
    df = pd.read_json(DATA_PATH, lines=True)
    with EVAL_PATH.open() as f:
        report = json.load(f)
    print(f"  {len(df):,} rows | {df.columns.size} columns")
    return df, report


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — World map by row type
# ══════════════════════════════════════════════════════════════════════════════

def plot_01_world_map(df: pd.DataFrame):
    valid = df.dropna(subset=["canonical_lat", "canonical_lon"]).copy()

    # Downsample unmatched_gdelt to keep the map responsive
    ug = valid[valid.row_type == "unmatched_gdelt"].sample(4000, random_state=42)
    rest = valid[valid.row_type != "unmatched_gdelt"]
    plot_df = pd.concat([ug, rest], ignore_index=True)
    plot_df["label"] = plot_df["row_type"].map(LABEL)

    # Marker size: matched events largest so they're always visible
    size_map = {"matched_gdelt": 9, "unmatched_gdelt": 3, "unmatched_ref": 5}
    plot_df["_sz"] = plot_df["row_type"].map(size_map)

    color_map = {v: COLORS[k] for k, v in LABEL.items()}

    fig = px.scatter_geo(
        plot_df,
        lat="canonical_lat", lon="canonical_lon",
        color="label",
        color_discrete_map=color_map,
        size="_sz", size_max=9,
        hover_name="canonical_location_name",
        hover_data={
            "canonical_country":    True,
            "canonical_date_start": True,
            "match_score":          ":.3f",
            "ref_dead":             True,
            "ref_affected":         True,
            "_sz":                  False,
            "label":                False,
        },
        projection="natural earth",
        title="Flood Events — Geographic Coverage by Dataset",
    )
    fig.update_layout(
        legend_title_text="Event type",
        height=620,
        legend=dict(orientation="h", y=-0.05),
    )
    fig.update_geos(showcoastlines=True, coastlinecolor="lightgrey",
                    showland=True, landcolor="WhiteSmoke",
                    showocean=True, oceancolor="LightCyan")
    out = OUT_DIR / "01_world_map_by_type.html"
    fig.write_html(out)
    print(f"  01 -> {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Choropleth: recall by country
# ══════════════════════════════════════════════════════════════════════════════

def plot_02_choropleth_recall(df: pd.DataFrame):
    ref_rows = df[df.has_ref].dropna(subset=["canonical_country_iso"]).copy()

    by_country = (
        ref_rows.groupby("canonical_country_iso")
        .agg(total=("has_ref", "count"), matched=("matched", "sum"),
             country=("canonical_country", "first"))
        .reset_index()
    )
    by_country["recall"] = by_country["matched"] / by_country["total"]
    by_country = by_country[by_country.total >= 2]

    fig = px.choropleth(
        by_country,
        locations="canonical_country_iso",
        color="recall",
        hover_name="country",
        hover_data={"total": True, "matched": True,
                    "recall": ":.1%", "canonical_country_iso": False},
        color_continuous_scale="RdYlGn",
        range_color=(0, 1),
        title="Reference Recall by Country<br><sup>Fraction of reference events captured by at least one GDELT article</sup>",
        labels={"recall": "Recall"},
    )
    fig.update_layout(height=560, coloraxis_colorbar=dict(tickformat=".0%"))
    out = OUT_DIR / "02_choropleth_recall_by_country.html"
    fig.write_html(out)
    print(f"  02 -> {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Hydro signal validation scatter grid
# ══════════════════════════════════════════════════════════════════════════════

def plot_03_hydro_validation(df: pd.DataFrame):
    matched = df[df.row_type == "matched_gdelt"].copy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (field, label) in zip(axes, HYDRO_PAIRS):
        ref_col, gdelt_col = f"ref_{field}", f"gdelt_{field}"
        sub = matched[[ref_col, gdelt_col, "match_score"]].dropna()
        if len(sub) < 5:
            ax.set_visible(False)
            continue

        sc = ax.scatter(
            sub[ref_col], sub[gdelt_col],
            c=sub["match_score"], cmap="RdYlGn",
            alpha=0.75, s=45, vmin=0.4, vmax=1.0,
            edgecolors="grey", linewidths=0.3,
        )
        lo = min(sub[ref_col].min(), sub[gdelt_col].min())
        hi = max(sub[ref_col].max(), sub[gdelt_col].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.45, label="y = x")

        r = sub[[ref_col, gdelt_col]].corr().iloc[0, 1]
        ax.set_title(f"{label}\nPearson r = {r:.2f}  (n={len(sub)})", fontsize=10)
        ax.set_xlabel("Reference value", fontsize=9)
        ax.set_ylabel("GDELT value", fontsize=9)
        plt.colorbar(sc, ax=ax, label="Match score", shrink=0.85)

    fig.suptitle(
        "Hydro Signal Agreement on Matched Events\n"
        "Reference vs GDELT measurements for the same physical flood",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    out = OUT_DIR / "03_hydro_signal_validation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  03 -> {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 — Temporal coverage stacked bar
# ══════════════════════════════════════════════════════════════════════════════

def plot_04_temporal_coverage(df: pd.DataFrame):
    valid = df.dropna(subset=["canonical_date_start"]).copy()
    valid["year"] = pd.to_datetime(valid["canonical_date_start"], errors="coerce").dt.year
    valid = valid.dropna(subset=["year"])
    valid["year"] = valid["year"].astype(int)

    order = ["unmatched_gdelt", "matched_gdelt", "unmatched_ref"]
    pivot = (
        valid.groupby(["year", "row_type"]).size()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), sharex=False)

    # Top: full scale (dominated by unmatched_gdelt)
    pivot.plot(kind="bar", stacked=True, ax=axes[0],
               color=[COLORS[c] for c in order], edgecolor="white", width=0.75)
    axes[0].set_title("Flood Events per Year — All Types (full scale)", fontsize=12)
    axes[0].set_ylabel("Event count")
    axes[0].legend([LABEL[c] for c in order], title="Type", fontsize=9)
    axes[0].set_xticklabels(pivot.index, rotation=45, ha="right")

    # Bottom: zoom in — exclude unmatched_gdelt to show ref/matched detail
    zoom_cols = ["matched_gdelt", "unmatched_ref"]
    pivot_zoom = pivot.reindex(columns=zoom_cols, fill_value=0)
    pivot_zoom.plot(kind="bar", stacked=True, ax=axes[1],
                    color=[COLORS[c] for c in zoom_cols], edgecolor="white", width=0.75)
    axes[1].set_title("Zoom: Matched and Reference-Only Events (GDELT-only excluded)", fontsize=12)
    axes[1].set_ylabel("Event count")
    axes[1].legend([LABEL[c] for c in zoom_cols], title="Type", fontsize=9)
    axes[1].set_xticklabels(pivot.index, rotation=45, ha="right")

    fig.suptitle("Temporal Coverage of the Combined Flood Dataset", fontsize=14, y=1.01)
    plt.tight_layout()
    out = OUT_DIR / "04_temporal_coverage.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  04 -> {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 5 — Match score histogram
# ══════════════════════════════════════════════════════════════════════════════

def plot_05_match_scores(report: dict, df: pd.DataFrame):
    score_dist = report.get("score_distribution", {})
    matched_scores = df[df.row_type == "matched_gdelt"]["match_score"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: all candidate pairs vs accepted (from evaluation report)
    ax = axes[0]
    if score_dist:
        bins    = list(score_dist["all_candidates"].keys())
        all_v   = list(score_dist["all_candidates"].values())
        match_v = list(score_dist["matched_pairs"].values())
        x = np.arange(len(bins))
        ax.bar(x, all_v,   alpha=0.45, color="steelblue", label="All candidate pairs")
        ax.bar(x, match_v, alpha=0.85, color=COLORS["matched_gdelt"], label="Accepted matches")
        # threshold falls between bin index 3 (0.3-0.4) and bin 4 (0.4-0.5)
        ax.axvline(3.5, color="crimson", linestyle="--", lw=1.8, label="Threshold = 0.40")
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=45, ha="right", fontsize=8)
        ax.set_title("Score distribution: all candidates vs accepted\n"
                     f"({score_dist['n_candidates']:,} candidates → {score_dist['n_matched']:,} accepted)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))

    # Right: accepted match score distribution
    ax = axes[1]
    ax.hist(matched_scores, bins=20, color=COLORS["matched_gdelt"],
            edgecolor="white", alpha=0.9)
    ax.axvline(matched_scores.mean(), color="crimson", linestyle="--", lw=1.8,
               label=f"Mean = {matched_scores.mean():.3f}")
    ax.axvline(matched_scores.median(), color="navy", linestyle=":", lw=1.8,
               label=f"Median = {matched_scores.median():.3f}")
    ax.set_title(f"Accepted match score distribution\n(n = {len(matched_scores):,})")
    ax.set_xlabel("Match score")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    fig.suptitle("Matching Score Analysis", fontsize=14)
    plt.tight_layout()
    out = OUT_DIR / "05_match_score_histogram.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  05 -> {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 6 — Recall by reference source
# ══════════════════════════════════════════════════════════════════════════════

def plot_06_recall_by_source(report: dict):
    by_source = report.get("by_source", {})
    if not by_source:
        print("  06 skipped (no by_source data)")
        return

    src_df = (
        pd.DataFrame(by_source).T
        .reset_index()
        .rename(columns={"index": "source"})
    )
    src_df[["recall", "total", "matched"]] = src_df[["recall", "total", "matched"]].apply(
        pd.to_numeric, errors="coerce"
    )
    src_df = src_df.sort_values("recall", ascending=True)

    overall_recall = report["overall"]["recall"]

    fig, ax = plt.subplots(figsize=(10, max(4, len(src_df) * 0.65)))
    bars = ax.barh(
        src_df["source"], src_df["recall"],
        color="steelblue", edgecolor="white", height=0.6,
    )
    for bar, row in zip(bars, src_df.itertuples()):
        ax.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{row.matched}/{row.total}",
            va="center", fontsize=10, color="dimgrey",
        )

    ax.axvline(overall_recall, color="crimson", linestyle="--", lw=1.8,
               label=f"Overall recall = {overall_recall:.1%}")
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xlabel("Recall (fraction of reference events found in GDELT)")
    ax.set_title(
        "GDELT Recall by Reference Source\n"
        "Which databases does GDELT best represent?",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = OUT_DIR / "06_recall_by_source.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  06 -> {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 7 — Media attention vs flood impact
# ══════════════════════════════════════════════════════════════════════════════

def plot_07_media_vs_impact(df: pd.DataFrame):
    matched = df[df.row_type == "matched_gdelt"].copy()

    # One point per reference cluster
    cluster = (
        matched.groupby("cluster_id")
        .agg(
            n_articles=("n_gdelt_in_cluster", "first"),
            ref_affected=("ref_affected", "first"),
            ref_dead=("ref_dead", "first"),
            canonical_country=("canonical_country", "first"),
            match_score=("match_score", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, impact_col, xlabel in [
        (axes[0], "ref_affected", "People affected (reference)"),
        (axes[1], "ref_dead",     "Deaths (reference)"),
    ]:
        sub = cluster.dropna(subset=[impact_col]).copy()
        sub = sub[sub[impact_col] > 0]
        if sub.empty:
            ax.set_visible(False)
            continue

        sc = ax.scatter(
            sub[impact_col], sub["n_articles"],
            c=sub["match_score"],
            cmap="RdYlGn", alpha=0.8, s=65,
            edgecolors="grey", linewidths=0.4,
            vmin=0.4, vmax=1.0,
        )
        plt.colorbar(sc, ax=ax, label="Mean match score", shrink=0.85)
        ax.set_xscale("log")
        ax.set_xlabel(f"{xlabel} (log scale)")
        ax.set_ylabel("Number of GDELT articles in cluster")
        ax.set_title(f"Media attention vs {xlabel.split('(')[0].strip()}")

        # Correlation on log-scale
        log_x = np.log1p(sub[impact_col])
        log_y = np.log1p(sub["n_articles"])
        if log_x.std() > 0 and log_y.std() > 0:
            r = np.corrcoef(log_x, log_y)[0, 1]
            ax.text(0.05, 0.95, f"r(log) = {r:.2f}", transform=ax.transAxes,
                    va="top", fontsize=10, color="navy")

    fig.suptitle(
        "Media Attention vs Flood Impact\n"
        "Each point = one matched reference flood event; dot count = news articles",
        fontsize=13,
    )
    plt.tight_layout()
    out = OUT_DIR / "07_media_attention_vs_impact.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  07 -> {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 8 — Severity bias: matched vs missed reference events
# ══════════════════════════════════════════════════════════════════════════════

def plot_08_severity_bias(df: pd.DataFrame):
    # Use only unique reference events (avoid counting matched_gdelt duplicates)
    ref_rows = df[df.row_type.isin(["matched_gdelt", "unmatched_ref"])].copy()
    # Keep one row per cluster for matched events
    ref_rows = ref_rows.drop_duplicates(subset="cluster_id")
    ref_rows["group"] = ref_rows["matched"].map(
        {True: "Matched by GDELT", False: "Missed by GDELT"}
    )

    palette = {
        "Matched by GDELT": COLORS["matched_gdelt"],
        "Missed by GDELT":  COLORS["unmatched_ref"],
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    for ax, col, label in [
        (axes[0], "ref_dead",     "Deaths"),
        (axes[1], "ref_affected", "People affected"),
    ]:
        sub = ref_rows[["group", col]].dropna()
        sub = sub[sub[col] > 0]
        if sub.empty:
            ax.set_visible(False)
            continue

        order = ["Matched by GDELT", "Missed by GDELT"]
        sns.boxplot(
            data=sub, x="group", y=col, ax=ax,
            hue="group", palette=palette, showfliers=False, width=0.5,
            order=order, legend=False,
        )
        # Strip plot sample (avoid overplotting)
        strip_parts = [
            g.sample(min(150, len(g)), random_state=42)
            for _, g in sub.groupby("group")
        ]
        strip_sub = pd.concat(strip_parts, ignore_index=True)
        sns.stripplot(
            data=strip_sub, x="group", y=col, ax=ax,
            hue="group", palette={"Matched by GDELT": "black", "Missed by GDELT": "black"},
            alpha=0.25, size=3.5, jitter=True, order=order, legend=False,
        )

        # Medians annotation
        for i, grp in enumerate(["Matched by GDELT", "Missed by GDELT"]):
            med = sub[sub.group == grp][col].median()
            ax.text(i, med * 1.5, f"median\n{med:,.0f}", ha="center",
                    fontsize=9, color="navy")

        ax.set_yscale("log")
        ax.set_title(f"{label}", fontsize=12)
        ax.set_xlabel("")
        ax.set_ylabel(f"{label} (log scale)")

    fig.suptitle(
        "Severity Bias: Are Larger Floods More Likely to Appear in GDELT?\n"
        "(One point per unique reference event)",
        fontsize=13,
    )
    plt.tight_layout()
    out = OUT_DIR / "08_severity_bias_boxplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  08 -> {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 9 — Feature completeness heatmap
# ══════════════════════════════════════════════════════════════════════════════

def plot_09_completeness_heatmap(df: pd.DataFrame):
    # Four groups of rows with meaningful completeness to measure
    groups = {
        "ref (matched)":    (df.row_type == "matched_gdelt",  "ref"),
        "gdelt (matched)":  (df.row_type == "matched_gdelt",  "gdelt"),
        "gdelt (unmatched)":(df.row_type == "unmatched_gdelt","gdelt"),
        "ref (unmatched)":  (df.row_type == "unmatched_ref",  "ref"),
    }

    # Short display names for the hydro fields
    short = {
        "chirps_3d_total_mm":           "chirps_3d",
        "chirps_7d_total_mm":           "chirps_7d",
        "chirps_14d_total_mm":          "chirps_14d",
        "chirps_30d_total_mm":          "chirps_30d",
        "chirps_peak_daily_mm":         "chirps_peak",
        "chirps_7d_baseline_mm":        "chirps_base",
        "chirps_7d_anom_mm":            "chirps_anom",
        "chirps_7d_anom_pct":           "chirps_anom%",
        "gpm_1d_total_mm":              "gpm_1d",
        "gpm_3d_total_mm":              "gpm_3d",
        "gpm_7d_total_mm":              "gpm_7d",
        "gpm_peak_daily_mm":            "gpm_peak_d",
        "gpm_peak_3h_mm":               "gpm_peak_3h",
        "era5_soil_moisture_day0":      "era5_sm_d0",
        "era5_soil_moisture_7d_mean":   "era5_sm_7d",
        "era5_soil_moisture_30d_mean":  "era5_sm_30d",
        "era5_soil_moisture_deep_day0": "era5_sm_deep",
        "era5_soil_moisture_deep_7d_mean": "era5_sm_deep7d",
        "era5_precip_7d_mm":            "era5_prec_7d",
        "era5_runoff_7d_mm":            "era5_runoff",
        "pop_count_25km":               "pop_count",
        "pop_density_km2":              "pop_dens",
        "jrc_occurrence_pct":           "jrc_occur",
        "jrc_recurrence_pct":           "jrc_recur",
        "terrain_slope_mean":           "slope",
        "spi_30d":                      "spi_30d",
        "spi_30d_pct":                  "spi_30d%",
    }

    matrix = {}
    for group_label, (mask, prefix) in groups.items():
        sub = df[mask]
        row = {}
        for field in HYDRO_FIELD_NAMES:
            col = f"{prefix}_{field}"
            if col in sub.columns:
                row[short[field]] = sub[col].notna().mean()
            else:
                row[short[field]] = np.nan
        matrix[group_label] = row

    heat = pd.DataFrame(matrix).T

    fig, ax = plt.subplots(figsize=(18, 4))
    sns.heatmap(
        heat, ax=ax, cmap="YlGn", vmin=0, vmax=1,
        annot=True, fmt=".0%", annot_kws={"size": 7},
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Completeness", "shrink": 0.6},
    )
    ax.set_title(
        "Hydro Feature Completeness by Row Group\n"
        "Highlights the richness gap that propagation aims to close",
        fontsize=13,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    out = OUT_DIR / "09_feature_completeness_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  09 -> {out.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df, report = load()

    print("\nGenerating plots...")
    plot_01_world_map(df)
    plot_02_choropleth_recall(df)
    plot_03_hydro_validation(df)
    plot_04_temporal_coverage(df)
    plot_05_match_scores(report, df)
    plot_06_recall_by_source(report)
    plot_07_media_vs_impact(df)
    plot_08_severity_bias(df)
    plot_09_completeness_heatmap(df)

    print(f"\nDone. All plots saved to:\n  {OUT_DIR}")
