"""
World Port Index (WPI) — Geospatial & Capability Analysis
Outputs all figures to Port Datasets/World Port Index/figures/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from scipy.spatial import cKDTree

# ── paths ──────────────────────────────────────────────────────────────────────
HERE   = Path(__file__).parent
DATA   = HERE / "WPI.csv"
OUTDIR = HERE / "figures"
OUTDIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)

# ── load & clean ───────────────────────────────────────────────────────────────
df = pd.read_csv(DATA, encoding="latin1")
df.columns = df.columns.str.strip().str.lstrip("\ufeff")

# Drop ports without coordinates — they can't be plotted or used in spatial analysis
df = df.dropna(subset=["Latitude", "Longitude"]).copy()
df = df[(df["Latitude"].between(-90, 90)) & (df["Longitude"].between(-180, 180))]

# standardise Y/N/Unknown to bool where useful
def yn_to_bool(series):
    return series.map({"Yes": True, "No": False}).fillna(False)

# facility and service columns
FAC_COLS = [c for c in df.columns if c.startswith("Facilities")]
SVC_COLS = [c for c in df.columns if c.startswith("Services") or c.startswith("Supplies")]
CRANE_COLS = [c for c in df.columns if "Crane" in c or "Lift" in c]
DEPTH_COLS = ["Channel Depth (m)", "Anchorage Depth (m)",
              "Cargo Pier Depth (m)", "Oil Terminal Depth (m)"]

# Capability score: count the number of "Yes" responses across all key facilities,
# services, and crane columns as a proxy for how well-equipped a port is
cap_cols = FAC_COLS + SVC_COLS + CRANE_COLS
for col in cap_cols:
    df[col] = df[col].astype(str).str.strip()
df["capability_score"] = df[cap_cols].apply(
    lambda row: (row == "Yes").sum(), axis=1
)

# numeric depths — replace 0 with NaN (0 usually means not recorded)
for col in DEPTH_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)

# harbor size ordering
size_order = ["Very Small", "Small", "Medium", "Large"]
df["Harbor Size"] = pd.Categorical(df["Harbor Size"], categories=size_order, ordered=True)

print(f"Loaded {len(df):,} ports  ->  figures -> {OUTDIR}")

# ── WORLD WATER BODY: simplify to primary body ─────────────────────────────────
df["Primary Water Body"] = df["World Water Body"].str.split(";").str[-1].str.strip()

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Global port map coloured by harbor size
# ═══════════════════════════════════════════════════════════════════════════════
size_colors = {
    "Very Small": "#aec7e8",
    "Small":      "#4C72B0",
    "Medium":     "#ff7f0e",
    "Large":      "#d62728",
}
size_alpha = {"Very Small": 0.35, "Small": 0.5, "Medium": 0.75, "Large": 1.0}
size_pts   = {"Very Small": 4,    "Small": 6,   "Medium": 10,   "Large": 18}

fig, ax = plt.subplots(figsize=(17, 9))
ax.set_facecolor("#d0e8f0")
ax.set_xlim(-180, 180); ax.set_ylim(-75, 85)

for size in size_order:
    sub = df[df["Harbor Size"] == size]
    ax.scatter(sub["Longitude"], sub["Latitude"],
               c=size_colors[size], s=size_pts[size],
               alpha=size_alpha[size], linewidths=0,
               label=f"{size} (n={len(sub):,})")

# mark the 20 highest-capability ports
top20 = df.nlargest(20, "capability_score")
ax.scatter(top20["Longitude"], top20["Latitude"],
           c="gold", s=60, marker="*", linewidths=0.4,
           edgecolors="black", zorder=5, label="Top-20 capability")

ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Global Port Distribution — World Port Index\ncoloured by harbour size, stars = top-20 capability score")
ax.legend(loc="lower left", fontsize=9, markerscale=1.5)
plt.tight_layout()
fig.savefig(OUTDIR / "fig1_global_port_map.png", dpi=150)
plt.close(fig)
print("fig1 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Capability score heatmap grid (lat × lon bins)
# ═══════════════════════════════════════════════════════════════════════════════
lat_bins = np.linspace(-75, 85,  33)   # ~5-degree bins
lon_bins = np.linspace(-180, 180, 73)

df["lat_bin"] = pd.cut(df["Latitude"],  bins=lat_bins, labels=False)
df["lon_bin"] = pd.cut(df["Longitude"], bins=lon_bins, labels=False)

grid = df.groupby(["lat_bin", "lon_bin"])["capability_score"].mean().unstack(fill_value=0)
# full grid
full_grid = pd.DataFrame(0.0,
    index=range(len(lat_bins)-1),
    columns=range(len(lon_bins)-1))
full_grid.update(grid)

fig, ax = plt.subplots(figsize=(17, 8))
im = ax.imshow(
    full_grid.values[::-1],        # flip so north is up
    extent=[-180, 180, -75, 85],
    aspect="auto", cmap="YlOrRd", interpolation="nearest",
)
plt.colorbar(im, ax=ax, label="Mean capability score (Yes-responses)", shrink=0.7)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Port Capability Density Heatmap\n(mean capability score per ~5-degree grid cell)")
plt.tight_layout()
fig.savefig(OUTDIR / "fig2_capability_heatmap.png", dpi=150)
plt.close(fig)
print("fig2 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Channel depth map: bubble size = channel depth, colour = harbor size
# ═══════════════════════════════════════════════════════════════════════════════
deep = df.dropna(subset=["Channel Depth (m)"]).copy()
deep = deep[deep["Channel Depth (m)"] > 0]

fig, ax = plt.subplots(figsize=(17, 9))
ax.set_facecolor("#d0e8f0")
ax.set_xlim(-180, 180); ax.set_ylim(-75, 85)

norm = Normalize(vmin=deep["Channel Depth (m)"].quantile(0.05),
                 vmax=deep["Channel Depth (m)"].quantile(0.95))
cmap = plt.cm.plasma
sizes = (deep["Channel Depth (m)"] / deep["Channel Depth (m)"].max() * 80 + 4).clip(4, 80)

sc = ax.scatter(deep["Longitude"], deep["Latitude"],
                c=deep["Channel Depth (m)"], cmap=cmap, norm=norm,
                s=sizes, alpha=0.7, linewidths=0)
plt.colorbar(sc, ax=ax, label="Channel Depth (m)", shrink=0.7)

ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Port Channel Depth — Global Map\n(bubble size and colour proportional to channel depth)")
plt.tight_layout()
fig.savefig(OUTDIR / "fig3_channel_depth_map.png", dpi=150)
plt.close(fig)
print("fig3 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Container & bulk facility ports map
# ═══════════════════════════════════════════════════════════════════════════════
has_container  = df["Facilities - Container"]  == "Yes"
has_solid_bulk = df["Facilities - Solid Bulk"] == "Yes"
has_liquid     = df["Facilities - Liquid Bulk"] == "Yes"
has_oil        = df["Facilities - Oil Terminal"] == "Yes"
has_lng        = df["Facilities - LNG Terminal"] == "Yes"

fig, ax = plt.subplots(figsize=(17, 9))
ax.set_facecolor("#d0e8f0")
ax.set_xlim(-180, 180); ax.set_ylim(-75, 85)

layers = [
    (df[has_solid_bulk],  "#8c564b", 8,  0.6, "Solid bulk"),
    (df[has_liquid],      "#17becf", 8,  0.6, "Liquid bulk"),
    (df[has_oil],         "#ff7f0e", 10, 0.8, "Oil terminal"),
    (df[has_lng],         "#9467bd", 14, 1.0, "LNG terminal"),
    (df[has_container],   "#d62728", 10, 0.9, "Container"),
]
for subset, color, sz, alpha, label in layers:
    ax.scatter(subset["Longitude"], subset["Latitude"],
               c=color, s=sz, alpha=alpha, linewidths=0, label=f"{label} (n={len(subset):,})")

ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Ports with Key Cargo Facilities\n(container, bulk, oil, LNG)")
ax.legend(loc="lower left", fontsize=9, markerscale=2)
plt.tight_layout()
fig.savefig(OUTDIR / "fig4_cargo_facilities_map.png", dpi=150)
plt.close(fig)
print("fig4 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Top-20 countries: port count + avg capability (dual axis)
# ═══════════════════════════════════════════════════════════════════════════════
top20_countries = df["Country Code"].value_counts().head(20).index
country_stats = (df[df["Country Code"].isin(top20_countries)]
                 .groupby("Country Code")
                 .agg(port_count=("Main Port Name", "count"),
                      avg_capability=("capability_score", "mean"))
                 .sort_values("port_count", ascending=True))

fig, ax1 = plt.subplots(figsize=(11, 8))
ax2 = ax1.twiny()

y = np.arange(len(country_stats))
ax1.barh(y, country_stats["port_count"], color="#4C72B0", alpha=0.8, label="Port count")
ax2.plot(country_stats["avg_capability"], y, "o-", color="#d62728",
         markersize=6, linewidth=1.5, label="Avg capability score")

ax1.set_yticks(y); ax1.set_yticklabels(country_stats.index)
ax1.set_xlabel("Number of ports", color="#4C72B0")
ax2.set_xlabel("Avg capability score", color="#d62728")
ax1.tick_params(axis="x", colors="#4C72B0")
ax2.tick_params(axis="x", colors="#d62728")
ax1.set_title("Top-20 Countries: Port Count vs Average Port Capability")

handles = [mpatches.Patch(color="#4C72B0", label="Port count"),
           plt.Line2D([0],[0], color="#d62728", marker="o", label="Avg capability")]
ax1.legend(handles=handles, loc="lower right", fontsize=9)
plt.tight_layout()
fig.savefig(OUTDIR / "fig5_country_count_capability.png", dpi=150)
plt.close(fig)
print("fig5 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Harbor size × harbor type breakdown (heatmap)
# ═══════════════════════════════════════════════════════════════════════════════
type_counts = df["Harbor Type"].value_counts()
top_types = type_counts[type_counts > 30].index

sub = df[df["Harbor Type"].isin(top_types) & df["Harbor Size"].notna()]
ct = (sub.groupby(["Harbor Type", "Harbor Size"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=size_order, fill_value=0))
ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(10, 7))
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues",
            linewidths=0.4, ax=ax, cbar_kws={"label": "Port count"})
ax.set_title("Harbour Type vs Harbour Size\n(port count)")
ax.set_xlabel("Harbour Size"); ax.set_ylabel("")
plt.tight_layout()
fig.savefig(OUTDIR / "fig6_harbor_type_size.png", dpi=150)
plt.close(fig)
print("fig6 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Depth profiles: box plots for large vs small harbors
# ═══════════════════════════════════════════════════════════════════════════════
depth_long = []
for col in DEPTH_COLS:
    tmp = df[["Harbor Size", col]].dropna().copy()
    tmp["Depth Type"] = col.replace(" (m)", "")
    tmp["Depth (m)"] = tmp[col]
    depth_long.append(tmp[["Harbor Size", "Depth Type", "Depth (m)"]])

depth_df = pd.concat(depth_long)
depth_df = depth_df[depth_df["Harbor Size"].notna()]

fig, ax = plt.subplots(figsize=(13, 6))
sns.boxplot(data=depth_df, x="Depth Type", y="Depth (m)",
            hue="Harbor Size", hue_order=size_order,
            palette=["#aec7e8","#4C72B0","#ff7f0e","#d62728"],
            fliersize=2, ax=ax)
ax.set_title("Depth Distributions by Harbour Size\n(channel, anchorage, cargo pier, oil terminal)")
ax.set_xlabel(""); ax.set_ylabel("Depth (m)")
ax.legend(title="Harbour Size", fontsize=9)
plt.tight_layout()
fig.savefig(OUTDIR / "fig7_depth_by_harbor_size.png", dpi=150)
plt.close(fig)
print("fig7 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Strategic chokepoint proximity: ports within 200 km of major straits
# ═══════════════════════════════════════════════════════════════════════════════
STRAITS = {
    "Strait of Malacca": (2.5, 102.0),
    "Strait of Hormuz":  (26.5, 56.5),
    "Suez Canal":        (30.7, 32.3),
    "Panama Canal":      (9.1, -79.7),
    "Bab el-Mandeb":     (12.5, 43.5),
    "Turkish Straits (Bosphorus)": (41.1, 29.0),
    "Strait of Gibraltar": (35.9, -5.6),
    "Dover Strait":      (51.0, 1.5),
    "Lombok Strait":     (-8.5, 115.7),
}
RADIUS_KM = 300

# use approximate equirectangular distance
def haversine_approx(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    mlat = np.radians((lat1 + lat2) / 2)
    return R * np.sqrt(dlat**2 + (np.cos(mlat) * dlon)**2)

df["nearest_strait"] = None
df["nearest_strait_dist_km"] = np.inf

for name, (slat, slon) in STRAITS.items():
    dist = haversine_approx(df["Latitude"].values, df["Longitude"].values, slat, slon)
    mask = dist < df["nearest_strait_dist_km"].values
    df.loc[mask, "nearest_strait"] = name
    df.loc[mask, "nearest_strait_dist_km"] = dist[mask]

near = df[df["nearest_strait_dist_km"] <= RADIUS_KM].copy()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# left: count of ports near each strait
strait_counts = near["nearest_strait"].value_counts()
axes[0].barh(strait_counts.index[::-1], strait_counts.values[::-1], color="#C44E52")
axes[0].set_xlabel("Number of ports within 300 km")
axes[0].set_title(f"Ports Near Strategic Chokepoints\n({len(near):,} ports within 300 km of a strait)")

# right: avg capability score for those ports vs rest
near_cap   = near["capability_score"].mean()
other_cap  = df[df["nearest_strait_dist_km"] > RADIUS_KM]["capability_score"].mean()
strait_cap = near.groupby("nearest_strait")["capability_score"].mean().sort_values()
axes[1].barh(strait_cap.index, strait_cap.values, color="#4C72B0")
axes[1].axvline(df["capability_score"].mean(), color="black", linestyle="--",
                linewidth=1.2, label=f"Global avg ({df['capability_score'].mean():.1f})")
axes[1].set_xlabel("Avg capability score")
axes[1].set_title("Average Port Capability Near Each Strait")
axes[1].legend(fontsize=9)

plt.tight_layout()
fig.savefig(OUTDIR / "fig8_chokepoint_proximity.png", dpi=150)
plt.close(fig)
print("fig8 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 9 — Port isolation map: nearest-neighbour distance
# ═══════════════════════════════════════════════════════════════════════════════
coords_rad = np.radians(df[["Latitude", "Longitude"]].values)
tree = cKDTree(coords_rad)
# query 2nd nearest (first is self)
dists_rad, _ = tree.query(coords_rad, k=2)
# convert radians to km (earth radius 6371 km)
df["nn_dist_km"] = dists_rad[:, 1] * 6371

fig, ax = plt.subplots(figsize=(17, 9))
ax.set_facecolor("#d0e8f0")
ax.set_xlim(-180, 180); ax.set_ylim(-75, 85)

norm = Normalize(vmin=0, vmax=df["nn_dist_km"].quantile(0.97))
sc = ax.scatter(df["Longitude"], df["Latitude"],
                c=df["nn_dist_km"], cmap="RdYlGn_r",
                norm=norm, s=5, alpha=0.7, linewidths=0)
plt.colorbar(sc, ax=ax, label="Distance to nearest port (km)", shrink=0.7)

# annotate the 10 most isolated ports
isolated = df.nlargest(10, "nn_dist_km")
for _, row in isolated.iterrows():
    ax.annotate(row["Main Port Name"],
                xy=(row["Longitude"], row["Latitude"]),
                xytext=(5, 3), textcoords="offset points",
                fontsize=7, color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title("Port Isolation Map — Distance to Nearest Neighbour\n"
             "red = isolated ports, green = dense port clusters")
plt.tight_layout()
fig.savefig(OUTDIR / "fig9_port_isolation.png", dpi=150)
plt.close(fig)
print("fig9 done")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
summary = {
    "Total ports (with coordinates)": len(df),
    "Countries represented":          df["Country Code"].nunique(),
    "Large harbours":                 int((df["Harbor Size"] == "Large").sum()),
    "Container-capable ports":        int(has_container.sum()),
    "LNG terminal ports":             int(has_lng.sum()),
    "Oil terminal ports":             int(has_oil.sum()),
    "Ports near straits (<300 km)":   len(near),
    "Most isolated port":             df.loc[df["nn_dist_km"].idxmax(), "Main Port Name"],
    "Median nn distance (km)":        round(df["nn_dist_km"].median(), 1),
    "Highest capability port":        df.loc[df["capability_score"].idxmax(), "Main Port Name"],
    "Max capability score":           int(df["capability_score"].max()),
}
print("\n=== Summary ===")
for k, v in summary.items():
    print(f"  {k:<40} {v}")

print(f"\nAll figures saved to: {OUTDIR}")
