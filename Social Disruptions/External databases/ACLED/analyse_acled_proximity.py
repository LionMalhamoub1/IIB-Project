"""
ACLED Event Proximity to Supply-Chain Chokepoints
--------------------------------------------------
Loads all cached raw ACLED parquet files (event-level, with lat/lon),
then computes distance from every event to:
  - ICMM mining assets   (Supply Chain Chokepoints/Mining Datasets/ICMM)
  - WPI ports            (Supply Chain Chokepoints/Port Datasets/World Port Index)

Produces figures in Social Disruptions/External databases/ACLED/figures/
"""

from pathlib import Path
import glob, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
import seaborn as sns
from scipy.spatial import cKDTree

# ── paths ──────────────────────────────────────────────────────────────────────
HERE      = Path(__file__).parent
ROOT      = HERE.parent.parent.parent          # IIB-Project root
ACLED_GLOB = HERE / "data/raw/events/**/*.parquet"
ICMM_PATH  = ROOT / "Supply Chain Chokepoints/Mining Datasets/ICMM/global-mining-dataset.xlsx"
WPI_PATH   = ROOT / "Supply Chain Chokepoints/Port Datasets/World Port Index/WPI.csv"
OUTDIR     = HERE / "figures"
OUTDIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)

RADIUS_MINE_KM = 50
RADIUS_PORT_KM = 50

# ── helpers ────────────────────────────────────────────────────────────────────
def nearest_in_tree(event_coords_rad, asset_coords_rad):
    """Return (nearest_index, distance_km) for each event using cKDTree."""
    tree = cKDTree(asset_coords_rad)
    dists_rad, idxs = tree.query(event_coords_rad, k=1, workers=-1)
    return idxs, dists_rad * 6371.0

# ── load ACLED ─────────────────────────────────────────────────────────────────
print("Loading ACLED events ...")
files = glob.glob(str(ACLED_GLOB), recursive=True)
parts = []
for f in files:
    m_iso  = re.search(r'iso3=([A-Z]+)', f)
    m_year = re.search(r'year=(\d+)', f)
    df = pd.read_parquet(f, columns=[
        "event_date", "event_type", "sub_event_type",
        "country", "admin1", "location",
        "latitude", "longitude", "fatalities", "iso3"
    ])
    if m_iso:  df["iso3"] = m_iso.group(1)
    if m_year: df["year"] = int(m_year.group(1))
    parts.append(df)

acled = pd.concat(parts, ignore_index=True)
acled["event_date"] = pd.to_datetime(acled["event_date"], errors="coerce")
acled["year"] = acled["event_date"].dt.year.fillna(acled.get("year", np.nan)).astype("Int64")
acled = acled.dropna(subset=["latitude", "longitude"])
acled = acled[(acled["latitude"].between(-90, 90)) & (acled["longitude"].between(-180, 180))]
print(f"  {len(acled):,} events | {acled['iso3'].nunique()} countries | "
      f"{acled['year'].min()}-{acled['year'].max()}")

# ── load ICMM ──────────────────────────────────────────────────────────────────
print("Loading ICMM mining assets ...")
mines = pd.read_excel(ICMM_PATH, sheet_name="External")
mines.columns = mines.columns.str.strip()
mines["Latitude"]  = pd.to_numeric(mines["Latitude"],  errors="coerce")
mines["Longitude"] = pd.to_numeric(mines["Longitude"], errors="coerce")
mines = mines.dropna(subset=["Latitude", "Longitude"])
mines["Primary Commodity"] = mines["Primary Commodity"].str.strip().str.lower()
COAL     = {"coal", "thermal coal", "metallurgical coal"}
CRITICAL = {"copper", "cobalt", "lithium", "nickel", "platinum",
            "palladium", "tungsten", "chromium", "chromite", "manganese"}
print(f"  {len(mines):,} mining assets")

# ── load WPI ───────────────────────────────────────────────────────────────────
print("Loading WPI ports ...")
ports = pd.read_csv(WPI_PATH, encoding="latin1")
ports.columns = ports.columns.str.strip().str.lstrip("\ufeff")
ports = ports.dropna(subset=["Latitude", "Longitude"])
ports = ports[(ports["Latitude"].between(-90, 90)) & (ports["Longitude"].between(-180, 180))]
ports["Harbor Size"] = ports["Harbor Size"].str.strip()
print(f"  {len(ports):,} ports")

# ── nearest-neighbour distance to each asset class ────────────────────────────
ev_rad   = np.radians(acled[["latitude", "longitude"]].values)
mine_rad = np.radians(mines[["Latitude", "Longitude"]].values)
port_rad = np.radians(ports[["Latitude", "Longitude"]].values)

print("Computing nearest mine per event ...")
mine_idx, mine_dist = nearest_in_tree(ev_rad, mine_rad)
acled["nearest_mine_dist_km"]   = mine_dist
mine_name_col = [c for c in mines.columns if c.strip() == "Mine Name"][0]
acled["nearest_mine_name"]      = mines[mine_name_col].iloc[mine_idx].values
acled["nearest_mine_commodity"] = mines["Primary Commodity"].iloc[mine_idx].values
acled["near_mine"]              = mine_dist <= RADIUS_MINE_KM

print("Computing nearest port per event ...")
port_idx, port_dist = nearest_in_tree(ev_rad, port_rad)
acled["nearest_port_dist_km"] = port_dist
acled["nearest_port_name"]    = ports["Main Port Name"].iloc[port_idx].values
acled["nearest_port_size"]    = ports["Harbor Size"].iloc[port_idx].values
acled["near_port"]            = port_dist <= RADIUS_PORT_KM

acled["near_any"] = acled["near_mine"] | acled["near_port"]

n_near_mine = acled["near_mine"].sum()
n_near_port = acled["near_port"].sum()
n_near_any  = acled["near_any"].sum()
print(f"  Near a mine (<{RADIUS_MINE_KM} km): {n_near_mine:,}  ({100*n_near_mine/len(acled):.1f}%)")
print(f"  Near a port (<{RADIUS_PORT_KM} km): {n_near_port:,}  ({100*n_near_port/len(acled):.1f}%)")
print(f"  Near either           : {n_near_any:,}  ({100*n_near_any/len(acled):.1f}%)")

def comm_group(c):
    if c in COAL:     return "coal"
    if c in CRITICAL: return "critical"
    return "other"

acled["mine_comm_group"] = acled["nearest_mine_commodity"].apply(comm_group)

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Global map: all events, highlight those near chokepoints
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(17, 9))
ax.set_facecolor("#d0e8f0")
ax.set_xlim(-180, 180); ax.set_ylim(-75, 85)

far            = acled[~acled["near_any"]]
near_mine_only = acled[acled["near_mine"] & ~acled["near_port"]]
near_port_only = acled[acled["near_port"] & ~acled["near_mine"]]
near_both      = acled[acled["near_mine"] & acled["near_port"]]

ax.scatter(far["longitude"],            far["latitude"],            c="#cccccc", s=2,  alpha=0.2, linewidths=0, label=f"No chokepoint (n={len(far):,})")
ax.scatter(near_mine_only["longitude"], near_mine_only["latitude"], c="#2ca02c", s=6,  alpha=0.6, linewidths=0, label=f"Near mine only (n={len(near_mine_only):,})")
ax.scatter(near_port_only["longitude"], near_port_only["latitude"], c="#1f77b4", s=6,  alpha=0.6, linewidths=0, label=f"Near port only (n={len(near_port_only):,})")
ax.scatter(near_both["longitude"],      near_both["latitude"],      c="#d62728", s=10, alpha=0.8, linewidths=0, label=f"Near mine + port (n={len(near_both):,})")

ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title(f"ACLED Events Near Supply-Chain Chokepoints\n"
             f"(within {RADIUS_MINE_KM} km of ICMM mine or WPI port, 2017-2025)")
ax.legend(loc="lower left", fontsize=8.5, markerscale=2.5)
plt.tight_layout()
fig.savefig(OUTDIR / "fig1_acled_chokepoint_map.png", dpi=150)
plt.close(fig)
print("fig1 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2 — CDF of distance to nearest mine and nearest port
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
for dist_col, label, color in [
    ("nearest_mine_dist_km", "Nearest ICMM mine", "#2ca02c"),
    ("nearest_port_dist_km", "Nearest WPI port",  "#1f77b4"),
]:
    vals = np.sort(acled[dist_col].dropna().values)
    cdf  = np.arange(1, len(vals) + 1) / len(vals)
    ax.plot(vals, cdf, color=color, lw=2, label=label)

ax.axvline(RADIUS_MINE_KM, color="black", linestyle="--", lw=1,
           label=f"{RADIUS_MINE_KM} km threshold")
ax.set_xlim(0, 500)
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Cumulative fraction of ACLED events")
ax.set_title("CDF: Distance from ACLED Event to Nearest Mine / Port")
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
plt.tight_layout()
fig.savefig(OUTDIR / "fig2_distance_cdf.png", dpi=150)
plt.close(fig)
print("fig2 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Event type breakdown: near vs far
# ═══════════════════════════════════════════════════════════════════════════════
et_near = acled[acled["near_any"]]["event_type"].value_counts(normalize=True) * 100
et_far  = acled[~acled["near_any"]]["event_type"].value_counts(normalize=True) * 100
all_types = et_near.index.union(et_far.index)
et_near = et_near.reindex(all_types, fill_value=0)
et_far  = et_far.reindex(all_types,  fill_value=0)

x = np.arange(len(all_types))
w = 0.35
fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w/2, et_near.values, w, color="#d62728", label="Near chokepoint")
ax.bar(x + w/2, et_far.values,  w, color="#aec7e8", label="Far from chokepoint")
ax.set_xticks(x); ax.set_xticklabels(all_types, rotation=20, ha="right")
ax.set_ylabel("% of events in group")
ax.set_title("Event Type Composition: Near vs Far from Supply-Chain Chokepoints")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUTDIR / "fig3_event_type_near_vs_far.png", dpi=150)
plt.close(fig)
print("fig3 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Events near mines by commodity group, over time
# ═══════════════════════════════════════════════════════════════════════════════
near_mines = acled[acled["near_mine"]].copy()
yearly = (near_mines.groupby(["year", "mine_comm_group"])
          .size().unstack(fill_value=0))
for col in ["coal", "critical", "other"]:
    if col not in yearly.columns:
        yearly[col] = 0

fig, ax = plt.subplots(figsize=(11, 5))
yearly[["coal", "critical", "other"]].plot(
    ax=ax, kind="bar", stacked=True,
    color=["#C44E52", "#2ca02c", "#aec7e8"],
    edgecolor="white", linewidth=0.4)
ax.set_xlabel("Year"); ax.set_ylabel("ACLED events near mine")
ax.set_title(f"ACLED Events Within {RADIUS_MINE_KM} km of a Mine by Commodity Group")
ax.legend(title="Mine commodity", fontsize=9)
ax.set_xticklabels([str(y) for y in yearly.index], rotation=0)
plt.tight_layout()
fig.savefig(OUTDIR / "fig4_events_near_mines_by_commodity.png", dpi=150)
plt.close(fig)
print("fig4 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Top mines and ports most exposed to nearby ACLED events
# ═══════════════════════════════════════════════════════════════════════════════
top_mines = (acled[acled["near_mine"]]
             .groupby("nearest_mine_name")
             .agg(event_count=("event_type", "count"),
                  commodity=("nearest_mine_commodity", "first"))
             .sort_values("event_count", ascending=False).head(15))
top_ports = (acled[acled["near_port"]]
             .groupby("nearest_port_name")
             .agg(event_count=("event_type", "count"),
                  size=("nearest_port_size", "first"))
             .sort_values("event_count", ascending=False).head(15))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

mine_colors = ["#C44E52" if c in COAL else ("#2ca02c" if c in CRITICAL else "#4C72B0")
               for c in top_mines["commodity"]]
axes[0].barh(top_mines.index[::-1], top_mines["event_count"].values[::-1],
             color=mine_colors[::-1])
axes[0].set_xlabel("ACLED events within 50 km")
axes[0].set_title("Top 15 Mines by Nearby ACLED Events")
axes[0].legend(handles=[
    mpatches.Patch(color="#C44E52", label="Coal"),
    mpatches.Patch(color="#2ca02c", label="Critical mineral"),
    mpatches.Patch(color="#4C72B0", label="Other"),
], fontsize=8)

size_color = {"Large": "#d62728", "Medium": "#ff7f0e",
              "Small": "#4C72B0", "Very Small": "#aec7e8"}
port_colors = [size_color.get(s, "#aec7e8") for s in top_ports["size"].values]
axes[1].barh(top_ports.index[::-1], top_ports["event_count"].values[::-1],
             color=port_colors[::-1])
axes[1].set_xlabel("ACLED events within 50 km")
axes[1].set_title("Top 15 Ports by Nearby ACLED Events")
axes[1].legend(handles=[mpatches.Patch(color=c, label=s)
                         for s, c in size_color.items()], fontsize=8, title="Harbour size")

plt.suptitle("Supply-Chain Assets Most Exposed to Nearby Conflict / Protest Activity", fontsize=13)
plt.tight_layout()
fig.savefig(OUTDIR / "fig5_top_exposed_assets.png", dpi=150)
plt.close(fig)
print("fig5 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Country: events near chokepoints (stacked mine vs port)
# ═══════════════════════════════════════════════════════════════════════════════
country_stats = (acled.groupby("iso3")
                 .agg(total=("near_any", "count"),
                      near_mine=("near_mine", "sum"),
                      near_port=("near_port", "sum"),
                      near_any=("near_any", "sum"))
                 .assign(pct_near=lambda d: 100 * d["near_any"] / d["total"])
                 .sort_values("near_any", ascending=False))

top30 = country_stats.head(30)
fig, ax = plt.subplots(figsize=(11, 9))
y = np.arange(len(top30))
ax.barh(y, top30["near_mine"].values, color="#2ca02c", label="Near mine")
ax.barh(y, top30["near_port"].values, left=top30["near_mine"].values,
        color="#1f77b4", label="Near port")
ax.set_yticks(y); ax.set_yticklabels(top30.index)
ax.invert_yaxis()
ax.set_xlabel("ACLED events within 50 km of a chokepoint")
ax.set_title("Countries: ACLED Events Near Supply-Chain Chokepoints")
for i, (_, row) in enumerate(top30.iterrows()):
    ax.text(row["near_any"] + 20, i, f"{row['pct_near']:.0f}%", va="center", fontsize=8)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUTDIR / "fig6_country_exposure.png", dpi=150)
plt.close(fig)
print("fig6 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Fatalities near vs far from chokepoints, by year
# ═══════════════════════════════════════════════════════════════════════════════
fat_year = (acled.groupby(["year", "near_any"])["fatalities"]
            .sum().unstack(fill_value=0))
fat_year.columns = ["Far from chokepoint", "Near chokepoint"]

fig, ax = plt.subplots(figsize=(11, 5))
fat_year.plot(ax=ax, kind="bar", color=["#aec7e8", "#d62728"],
              edgecolor="white", linewidth=0.4)
ax.set_xlabel("Year"); ax.set_ylabel("Total fatalities")
ax.set_title("Fatalities Near vs Far from Supply-Chain Chokepoints by Year")
ax.legend(fontsize=9)
ax.set_xticklabels([str(y) for y in fat_year.index], rotation=0)
plt.tight_layout()
fig.savefig(OUTDIR / "fig7_fatalities_near_vs_far.png", dpi=150)
plt.close(fig)
print("fig7 done")

# ═══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Country x event_type heatmap for events near chokepoints
# ═══════════════════════════════════════════════════════════════════════════════
near_df = acled[acled["near_any"]].copy()
top_iso = near_df["iso3"].value_counts().head(20).index
ct = (near_df[near_df["iso3"].isin(top_iso)]
      .groupby(["iso3", "event_type"])
      .size()
      .unstack(fill_value=0))
ct = ct.loc[ct.sum(axis=1).sort_values(ascending=False).index]

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(ct, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.3, ax=ax, cbar_kws={"label": "Event count"})
ax.set_title("ACLED Event Types Near Chokepoints — Top 20 Countries\n"
             f"(within {RADIUS_MINE_KM} km of an ICMM mine or WPI port)")
ax.set_xlabel("Event type"); ax.set_ylabel("")
plt.tight_layout()
fig.savefig(OUTDIR / "fig8_country_eventtype_heatmap.png", dpi=150)
plt.close(fig)
print("fig8 done")

# ── summary ────────────────────────────────────────────────────────────────────
print(f"""
=== Summary ===
  Total ACLED events loaded              : {len(acled):,}
  Near a mine (<{RADIUS_MINE_KM} km)              : {n_near_mine:,}  ({100*n_near_mine/len(acled):.1f}%)
  Near a port (<{RADIUS_PORT_KM} km)              : {n_near_port:,}  ({100*n_near_port/len(acled):.1f}%)
  Near either                            : {n_near_any:,}  ({100*n_near_any/len(acled):.1f}%)
  Most exposed mine                      : {top_mines.index[0]}
  Most exposed port                      : {top_ports.index[0]}
  Country with most chokepoint events    : {country_stats.index[0]}
""")
print(f"All figures saved to: {OUTDIR}")
