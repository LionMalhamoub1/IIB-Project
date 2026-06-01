"""
ICMM Global Mining Dataset — Analysis & Figures
Outputs all figures to Mining Datasets/ICMM/figures/
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import BoundaryNorm, ListedColormap
import seaborn as sns

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
HERE   = Path(__file__).parent
DATA   = HERE / "global-mining-dataset.xlsx"
OUTDIR = HERE / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── palette ────────────────────────────────────────────────────────────────────
PALETTE = sns.color_palette("tab20", 20)
sns.set_theme(style="whitegrid", font_scale=1.05)

# ── load ───────────────────────────────────────────────────────────────────────
df = pd.read_excel(DATA, sheet_name="External")
df.columns = df.columns.str.strip()

# Coerce to numeric — the spreadsheet occasionally has text artefacts in coordinate columns
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")

# Normalise text fields to lowercase/stripped so groupby and set lookups work consistently
df["Primary Commodity"] = df["Primary Commodity"].str.strip().str.lower()
df["Asset Type"]        = df["Asset Type"].str.strip()
df["Country or Region"] = df["Country or Region"].str.strip()

# coal group
COAL = {"coal", "thermal coal", "metallurgical coal"}
CRITICAL = {"copper", "cobalt", "lithium", "nickel", "platinum",
            "palladium", "tungsten", "chromium", "chromite", "manganese"}

df["commodity_group"] = df["Primary Commodity"].apply(
    lambda x: "coal" if x in COAL else ("critical mineral" if x in CRITICAL else x)
)

print(f"Loaded {len(df):,} assets  ->  figures -> {OUTDIR}")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Top-20 countries by asset count (stacked: mines vs processing)
# ═══════════════════════════════════════════════════════════════════════════════
top20_countries = df["Country or Region"].value_counts().head(20).index

is_mine  = df["Asset Type"].str.contains("Mine",    na=False) & ~df["Asset Type"].str.contains("Smelter|Refinery|Plant", na=False)
is_proc  = df["Asset Type"].str.contains("Smelter|Refinery|Plant", na=False)

mine_ct  = df[is_mine].groupby("Country or Region").size().reindex(top20_countries, fill_value=0)
proc_ct  = df[is_proc].groupby("Country or Region").size().reindex(top20_countries, fill_value=0)
other_ct = (df["Country or Region"].value_counts().reindex(top20_countries, fill_value=0)
            - mine_ct - proc_ct)

fig, ax = plt.subplots(figsize=(11, 7))
y = np.arange(len(top20_countries))
ax.barh(y, mine_ct.values,  color="#4C72B0", label="Mine only")
ax.barh(y, other_ct.values, left=mine_ct.values, color="#55A868", label="Mine + processing (combo)")
ax.barh(y, proc_ct.values,  left=mine_ct.values + other_ct.values, color="#C44E52", label="Processing only (smelter/refinery/plant)")
ax.set_yticks(y); ax.set_yticklabels(top20_countries)
ax.invert_yaxis()
ax.set_xlabel("Number of assets")
ax.set_title("Top-20 Countries by Asset Count\n(mining vs processing)")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
fig.savefig(OUTDIR / "fig1_country_asset_type.png", dpi=150)
plt.close(fig)
print("fig1 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Primary commodity breakdown (treemap-style horizontal bars)
# ═══════════════════════════════════════════════════════════════════════════════
comm_counts = df["Primary Commodity"].value_counts().head(20)

fig, ax = plt.subplots(figsize=(11, 7))
colors = [("#C44E52" if c in COAL else ("#55A868" if c in CRITICAL else "#4C72B0"))
          for c in comm_counts.index]
bars = ax.barh(comm_counts.index[::-1], comm_counts.values[::-1], color=colors[::-1])
for bar, val in zip(bars, comm_counts.values[::-1]):
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=8.5)
ax.set_xlabel("Number of assets")
ax.set_title("Top-20 Primary Commodities in ICMM Dataset")
legend_patches = [
    mpatches.Patch(color="#C44E52", label="Coal group"),
    mpatches.Patch(color="#55A868", label="Critical mineral"),
    mpatches.Patch(color="#4C72B0", label="Other"),
]
ax.legend(handles=legend_patches, fontsize=9)
plt.tight_layout()
fig.savefig(OUTDIR / "fig2_primary_commodity.png", dpi=150)
plt.close(fig)
print("fig2 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Confidence factor by commodity (stacked %)
# ═══════════════════════════════════════════════════════════════════════════════
top12 = df["Primary Commodity"].value_counts().head(12).index
conf_order = ["High", "Moderate", "Very Low"]
conf_colors = {"High": "#2ca02c", "Moderate": "#ff7f0e", "Very Low": "#d62728"}

sub = df[df["Primary Commodity"].isin(top12)]
ct = (sub.groupby(["Primary Commodity", "Confidence Factor"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=conf_order, fill_value=0))
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
ct_pct = ct_pct.loc[ct_pct.sum(axis=1).sort_values().index]  # sort by total

fig, ax = plt.subplots(figsize=(10, 6))
left = np.zeros(len(ct_pct))
for col in conf_order:
    ax.barh(ct_pct.index, ct_pct[col].values, left=left,
            color=conf_colors[col], label=col)
    left += ct_pct[col].values
ax.set_xlabel("% of assets")
ax.set_title("Confidence Factor by Primary Commodity (top 12)")
ax.legend(title="Confidence", fontsize=9)
ax.xaxis.set_major_formatter(mticker.PercentFormatter())
plt.tight_layout()
fig.savefig(OUTDIR / "fig3_confidence_by_commodity.png", dpi=150)
plt.close(fig)
print("fig3 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Geographic scatter map (lat/lon)
# ═══════════════════════════════════════════════════════════════════════════════
geo = df.dropna(subset=["Latitude", "Longitude"]).copy()

# assign a plot colour per broad group
def group_color(c):
    if c in COAL:     return "#C44E52"
    if c in CRITICAL: return "#2ca02c"
    if c == "gold":   return "#FFD700"
    if c == "iron ore": return "#8c564b"
    return "#aec7e8"

geo["_color"] = geo["Primary Commodity"].apply(group_color)

fig, ax = plt.subplots(figsize=(16, 8))
# background world outline via scatter density
scatter_groups = {
    "Coal (thermal/met)":     ("#C44E52", geo[geo["Primary Commodity"].isin(COAL)]),
    "Critical minerals":      ("#2ca02c", geo[geo["Primary Commodity"].isin(CRITICAL)]),
    "Gold":                   ("#FFD700", geo[geo["Primary Commodity"] == "gold"]),
    "Iron ore":               ("#8c564b", geo[geo["Primary Commodity"] == "iron ore"]),
    "Other":                  ("#aec7e8", geo[~geo["Primary Commodity"].isin(COAL | CRITICAL | {"gold","iron ore"})]),
}
for label, (color, subset) in scatter_groups.items():
    ax.scatter(subset["Longitude"], subset["Latitude"],
               c=color, s=4, alpha=0.55, linewidths=0, label=f"{label} (n={len(subset):,})")

ax.set_xlim(-180, 180); ax.set_ylim(-75, 85)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Global Distribution of Mining Assets (ICMM)\ncoloured by commodity group")
ax.legend(loc="lower left", fontsize=8.5, markerscale=3)
ax.set_facecolor("#e8f4f8")
plt.tight_layout()
fig.savefig(OUTDIR / "fig4_geo_scatter.png", dpi=150)
plt.close(fig)
print("fig4 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Critical minerals: country heatmap
# ═══════════════════════════════════════════════════════════════════════════════
crit_df = df[df["Primary Commodity"].isin(CRITICAL)]
top_crit_countries = crit_df["Country or Region"].value_counts().head(18).index
top_crit_comms     = crit_df["Primary Commodity"].value_counts().head(8).index

heatmap_data = (crit_df[crit_df["Country or Region"].isin(top_crit_countries) &
                         crit_df["Primary Commodity"].isin(top_crit_comms)]
                .groupby(["Country or Region", "Primary Commodity"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=top_crit_countries, fill_value=0))

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.4, ax=ax, cbar_kws={"label": "Asset count"})
ax.set_title("Critical Mineral Assets: Country × Commodity Heatmap\n(ICMM dataset, primary commodity only)")
ax.set_xlabel("Primary Commodity")
ax.set_ylabel("")
plt.tight_layout()
fig.savefig(OUTDIR / "fig5_critical_mineral_heatmap.png", dpi=150)
plt.close(fig)
print("fig5 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Co-occurrence: primary vs secondary commodity (top 10×10)
# ═══════════════════════════════════════════════════════════════════════════════
has_sec = df.dropna(subset=["Secondary Commodity"]).copy()
has_sec["Secondary Commodity"] = has_sec["Secondary Commodity"].str.strip().str.lower()

top_prim = has_sec["Primary Commodity"].value_counts().head(10).index
top_sec  = has_sec["Secondary Commodity"].value_counts().head(10).index

co = (has_sec[has_sec["Primary Commodity"].isin(top_prim) &
              has_sec["Secondary Commodity"].isin(top_sec)]
      .groupby(["Primary Commodity", "Secondary Commodity"])
      .size()
      .unstack(fill_value=0))

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(co, annot=True, fmt="d", cmap="Blues",
            linewidths=0.4, ax=ax, cbar_kws={"label": "Co-occurrence count"})
ax.set_title("Primary vs Secondary Commodity Co-occurrence\n(top 10 each, assets with both fields populated)")
ax.set_xlabel("Secondary Commodity")
ax.set_ylabel("Primary Commodity")
plt.tight_layout()
fig.savefig(OUTDIR / "fig6_commodity_cooccurrence.png", dpi=150)
plt.close(fig)
print("fig6 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Processing asset concentration (smelters/refineries) by country
# ═══════════════════════════════════════════════════════════════════════════════
proc_df = df[df["Asset Type"].str.contains("Smelter|Refinery|Plant", na=False)]
proc_country = proc_df["Country or Region"].value_counts().head(15)
proc_comm    = proc_df["Primary Commodity"].value_counts().head(10)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].barh(proc_country.index[::-1], proc_country.values[::-1], color="#C44E52")
axes[0].set_xlabel("Processing assets")
axes[0].set_title("Processing Assets by Country\n(smelters, refineries, plants)")

axes[1].barh(proc_comm.index[::-1], proc_comm.values[::-1], color="#4C72B0")
axes[1].set_xlabel("Processing assets")
axes[1].set_title("Processing Assets by Commodity")

plt.suptitle("Supply-Chain Choke Points — Processing Infrastructure", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(OUTDIR / "fig7_processing_assets.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig7 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Cobalt dependency: primary vs secondary classification by country
# ═══════════════════════════════════════════════════════════════════════════════
has_sec["Secondary Commodity"] = has_sec["Secondary Commodity"].str.strip().str.lower()

cobalt_prim = df[df["Primary Commodity"] == "cobalt"]["Country or Region"].value_counts()
cobalt_sec  = has_sec[has_sec["Secondary Commodity"] == "cobalt"]["Country or Region"].value_counts()

all_cobalt_countries = cobalt_prim.index.union(cobalt_sec.index)
cobalt_combined = pd.DataFrame({
    "Primary":   cobalt_prim.reindex(all_cobalt_countries, fill_value=0),
    "Secondary": cobalt_sec.reindex(all_cobalt_countries, fill_value=0),
})
cobalt_combined["Total"] = cobalt_combined.sum(axis=1)
cobalt_combined = cobalt_combined.sort_values("Total", ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(cobalt_combined))
ax.barh(y, cobalt_combined["Primary"].values,  color="#2ca02c", label="Primary cobalt")
ax.barh(y, cobalt_combined["Secondary"].values, left=cobalt_combined["Primary"].values,
        color="#98df8a", label="Secondary cobalt (byproduct of other mine)")
ax.set_yticks(y); ax.set_yticklabels(cobalt_combined.index)
ax.set_xlabel("Number of assets")
ax.set_title("Cobalt — Primary vs Secondary Classification by Country\n"
             "(DRC appears mainly as secondary, revealing byproduct dependency)")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUTDIR / "fig8_cobalt_primary_vs_secondary.png", dpi=150)
plt.close(fig)
print("fig8 done")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary stats table
# ═══════════════════════════════════════════════════════════════════════════════
summary = {
    "Total assets":                 len(df),
    "Countries / regions":          df["Country or Region"].nunique(),
    "Distinct primary commodities": df["Primary Commodity"].nunique(),
    "Pure mines":                   int(is_mine.sum()),
    "Processing-only assets":       int(is_proc.sum()),
    "Critical mineral assets":      int((df["Primary Commodity"].isin(CRITICAL)).sum()),
    "Coal assets":                  int((df["Primary Commodity"].isin(COAL)).sum()),
    "High confidence":              int((df["Confidence Factor"] == "High").sum()),
    "Moderate confidence":          int((df["Confidence Factor"] == "Moderate").sum()),
    "Very Low confidence":          int((df["Confidence Factor"] == "Very Low").sum()),
}
print("\n=== Summary ===")
for k, v in summary.items():
    print(f"  {k:<35} {v:>6,}")

print(f"\nAll figures saved to: {OUTDIR}")
