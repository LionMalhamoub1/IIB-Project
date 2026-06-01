"""
Plot data coverage by year for all official datasets in Social Disruptions/External_Databases.
Saves bar charts to Social Disruptions/External_Databases/Coverage_Plots/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE = os.path.dirname(OUTPUT_DIR)

# ── helpers ─────────────────────────────────────────────────────────────────

def save_bar(year_counts: pd.Series, title: str, ylabel: str, filename: str,
             color: str = "#4C72B0"):
    years = sorted(year_counts.index.tolist())
    counts = [year_counts.get(y, 0) for y in years]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(years, counts, color=color, edgecolor="white", linewidth=0.5)

    # label each bar
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=8)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Year", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(years)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.margins(x=0.04)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── 1. ACLED ─────────────────────────────────────────────────────────────────
print("Processing ACLED…")
df = pd.read_csv(os.path.join(BASE, "ACLED", "panels", "acled_country_month.csv"),
                 parse_dates=["month"])
year_counts = df.groupby(df["month"].dt.year)["event_count"].sum()
save_bar(year_counts, "ACLED – Total Events per Year",
         "Total event count (all countries)", "acled_coverage.png", "#DD8452")

# ── 2. GTA (trade interventions) ─────────────────────────────────────────────
print("Processing GTA…")
df = pd.read_csv(os.path.join(BASE, "GTA", "data", "interim",
                               "gta_interventions_clean.csv"),
                 parse_dates=["implementation_date"])
year_counts = df.groupby(df["implementation_date"].dt.year).size()
save_bar(year_counts, "GTA – Trade Interventions per Year",
         "Number of interventions", "gta_coverage.png", "#55A868")

# ── 3. MMAD ──────────────────────────────────────────────────────────────────
print("Processing MMAD…")
df = pd.read_csv(os.path.join(BASE, "MMAD", "data", "processed",
                               "mmad_country_month_2017_2025.csv"))
year_counts = df.groupby("year")["protest_count"].sum()
save_bar(year_counts, "MMAD – Total Protests per Year",
         "Total protest count (all countries)", "mmad_coverage.png", "#C44E52")

# ── 4. ILOSTAT ───────────────────────────────────────────────────────────────
print("Processing ILOSTAT…")
df = pd.read_csv(os.path.join(BASE, "ILOSTAT", "data", "processed",
                               "ilostat_country_month_2017_2025.csv"),
                 parse_dates=["date"])
# count country-months with at least one non-null indicator
valid = df.dropna(subset=["unemployment_rate", "earnings_monthly"], how="all")
year_counts = valid.groupby(valid["date"].dt.year).size()
save_bar(year_counts, "ILOSTAT – Country-Month Records with Data per Year",
         "Country-month records (with data)", "ilostat_coverage.png", "#8172B2")

# ── 5. Google Trends ─────────────────────────────────────────────────────────
print("Processing Google Trends…")
df = pd.read_csv(os.path.join(BASE, "Google_Trends", "data", "processed",
                               "google_trends_country_week_2017_2025.csv"),
                 parse_dates=["week"])
year_counts = df.groupby(df["week"].dt.year).size()
save_bar(year_counts, "Google Trends – Country-Week Records per Year",
         "Country-week records", "google_trends_coverage.png", "#64B5CD")

# ── 6. Inflation / CPI ───────────────────────────────────────────────────────
print("Processing Inflation/CPI…")
df = pd.read_csv(os.path.join(BASE, "Inflation", "data", "processed",
                               "cpi_inflation_monthly_2017_2025.csv"),
                 parse_dates=["date"])
valid = df.dropna(subset=["food_cpi_inflation", "energy_cpi_inflation"], how="all")
year_counts = valid.groupby(valid["date"].dt.year).size()
save_bar(year_counts, "Inflation/CPI – Country-Month Records with Data per Year",
         "Country-month records (with data)", "inflation_coverage.png", "#CCB974")

# ── 7. Markets ───────────────────────────────────────────────────────────────
print("Processing Markets…")
df = pd.read_parquet(os.path.join(BASE, "Markets", "data", "processed",
                                   "markets_country_day_20170101_20251231.parquet"))
df["date"] = pd.to_datetime(df["date"])
valid = df.dropna(subset=["fx_lcu_usd"], how="all")
year_counts = valid.groupby(valid["date"].dt.year).size()
save_bar(year_counts, "Markets – Country-Day Records with FX Data per Year",
         "Country-day records (with data)", "markets_coverage.png", "#4C72B0")

# ── 8. WDI ───────────────────────────────────────────────────────────────────
print("Processing WDI…")
df = pd.read_csv(os.path.join(BASE, "WDI", "data", "processed",
                               "wdi_country_year_2017_2025.csv"))
valid = df.dropna(subset=["gdp_growth"], how="all")
year_counts = valid.groupby("year").size()
save_bar(year_counts, "WDI – Countries with GDP Growth Data per Year",
         "Number of countries", "wdi_coverage.png", "#DA8BC3")

# ── 9. WGI ───────────────────────────────────────────────────────────────────
print("Processing WGI…")
df = pd.read_csv(os.path.join(BASE, "WGI", "data", "processed",
                               "wgi_country_year_2017_2025.csv"))
valid = df.dropna(subset=["political_stability_est"], how="all")
year_counts = valid.groupby("year").size()
save_bar(year_counts, "WGI – Countries with Governance Data per Year",
         "Number of countries", "wgi_coverage.png", "#77C29E")

# ── 10. Combined heatmap ─────────────────────────────────────────────────────
print("Building combined coverage heatmap…")

# Re-collect all normalised series (0-1 scale per dataset)
all_years = list(range(2017, 2026))

def collect(year_counts):
    s = pd.Series({y: year_counts.get(y, 0) for y in all_years}, dtype=float)
    mx = s.max()
    return (s / mx) if mx > 0 else s

datasets = {}

# ACLED
df = pd.read_csv(os.path.join(BASE, "ACLED", "panels", "acled_country_month.csv"),
                 parse_dates=["month"])
datasets["ACLED"] = collect(df.groupby(df["month"].dt.year)["event_count"].sum())

# GTA
df = pd.read_csv(os.path.join(BASE, "GTA", "data", "interim",
                               "gta_interventions_clean.csv"),
                 parse_dates=["implementation_date"])
datasets["GTA"] = collect(df.groupby(df["implementation_date"].dt.year).size())

# MMAD
df = pd.read_csv(os.path.join(BASE, "MMAD", "data", "processed",
                               "mmad_country_month_2017_2025.csv"))
datasets["MMAD"] = collect(df.groupby("year")["protest_count"].sum())

# ILOSTAT
df = pd.read_csv(os.path.join(BASE, "ILOSTAT", "data", "processed",
                               "ilostat_country_month_2017_2025.csv"),
                 parse_dates=["date"])
valid = df.dropna(subset=["unemployment_rate", "earnings_monthly"], how="all")
datasets["ILOSTAT"] = collect(valid.groupby(valid["date"].dt.year).size())

# Google Trends
df = pd.read_csv(os.path.join(BASE, "Google_Trends", "data", "processed",
                               "google_trends_country_week_2017_2025.csv"),
                 parse_dates=["week"])
datasets["Google_Trends"] = collect(df.groupby(df["week"].dt.year).size())

# Inflation
df = pd.read_csv(os.path.join(BASE, "Inflation", "data", "processed",
                               "cpi_inflation_monthly_2017_2025.csv"),
                 parse_dates=["date"])
valid = df.dropna(subset=["food_cpi_inflation", "energy_cpi_inflation"], how="all")
datasets["Inflation/CPI"] = collect(valid.groupby(valid["date"].dt.year).size())

# Markets
df = pd.read_parquet(os.path.join(BASE, "Markets", "data", "processed",
                                   "markets_country_day_20170101_20251231.parquet"))
df["date"] = pd.to_datetime(df["date"])
valid = df.dropna(subset=["fx_lcu_usd"], how="all")
datasets["Markets"] = collect(valid.groupby(valid["date"].dt.year).size())

# WDI
df = pd.read_csv(os.path.join(BASE, "WDI", "data", "processed",
                               "wdi_country_year_2017_2025.csv"))
valid = df.dropna(subset=["gdp_growth"], how="all")
datasets["WDI"] = collect(valid.groupby("year").size())

# WGI
df = pd.read_csv(os.path.join(BASE, "WGI", "data", "processed",
                               "wgi_country_year_2017_2025.csv"))
valid = df.dropna(subset=["political_stability_est"], how="all")
datasets["WGI"] = collect(valid.groupby("year").size())

# Build matrix: rows = datasets, cols = years
matrix = pd.DataFrame(datasets, index=all_years).T   # shape: (9, 9)

# ── plot: heatmap + overall score bar ────────────────────────────────────────
fig, (ax_heat, ax_bar) = plt.subplots(
    2, 1, figsize=(13, 8),
    gridspec_kw={"height_ratios": [6, 2]},
)

im = ax_heat.imshow(matrix.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)

# Annotate cells
for r in range(matrix.shape[0]):
    for c in range(matrix.shape[1]):
        val = matrix.values[r, c]
        txt_color = "black" if val < 0.65 else "white"
        ax_heat.text(c, r, f"{val:.2f}", ha="center", va="center",
                     fontsize=8, color=txt_color)

ax_heat.set_xticks(range(len(all_years)))
ax_heat.set_xticklabels(all_years, fontsize=10)
ax_heat.set_yticks(range(len(matrix.index)))
ax_heat.set_yticklabels(matrix.index, fontsize=10)
ax_heat.set_title("Combined Dataset Coverage by Year\n(cell = fraction of that dataset's peak year)",
                   fontsize=13, fontweight="bold", pad=10)

cbar = fig.colorbar(im, ax=ax_heat, orientation="vertical", fraction=0.02, pad=0.02)
cbar.set_label("Relative coverage (0 = none, 1 = peak year)", fontsize=8)

# Overall score: mean across datasets per year
overall = matrix.mean(axis=0)
bar_colors = plt.cm.YlGn(overall.values)
bars = ax_bar.bar(range(len(all_years)), overall.values, color=bar_colors,
                  edgecolor="grey", linewidth=0.5)
for bar, val in zip(bars, overall.values):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)

ax_bar.set_xticks(range(len(all_years)))
ax_bar.set_xticklabels(all_years, fontsize=10)
ax_bar.set_ylabel("Mean coverage\nscore", fontsize=9)
ax_bar.set_ylim(0, 1.15)
ax_bar.set_title("Overall Mean Coverage Score per Year", fontsize=11, fontweight="bold")
ax_bar.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout(h_pad=2)
out_path = os.path.join(OUTPUT_DIR, "combined_coverage.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  Saved: {out_path}")

print("\nAll plots saved to:", OUTPUT_DIR)
