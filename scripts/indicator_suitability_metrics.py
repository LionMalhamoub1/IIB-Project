import pandas as pd
#once databse numbers are in use to quantify the scoring metrics for the indicators. (like bayesian inference)
# -------------------------------------------------------------
# Indicator Suitability Metric (ISM)
# Likelihood/frequency indicators only
# -------------------------------------------------------------
# Each indicator is scored for:
# - Causal relevance
# - Supply chain relevance
# - Spatial coverage
# - Temporal resolution
# - Temporal coverage
# - Ease of access
# - Data reliability
# - Alignment with GDELT
#
# Scores: 1 (poor) → 5 (excellent)
# Weighted and aggregated into a 0–100 scale.

# -------------------------------Criteria and weights -------------------------------#
criteria = [
    "Causal relevance", 
    "Supply chain relevance", 
    "Spatial coverage", 
    "Temporal resolution",
    "Temporal coverage",
    "Ease of access",
    "Data reliability",
    "Alignment with GDELT",
]
# -------------------------------weighting each criteria by importance (total sums to 1) -------------------------------#
weights = {
    "Causal relevance": 0.10, 
    "Supply chain relevance": 0.10,
    "Spatial coverage": 0.15,
    "Temporal resolution": 0.20,
    "Temporal coverage": 0.10,
    "Ease of access": 0.20,
    "Data reliability": 0.10,
    "Alignment with GDELT": 0.05,
}

# ------------------------------- Indicator scores (likelihood/frequency indicators only) -------------------------------#
indicator_data = {
    # Earthquake
    "Seismic activity frequency": ("Earthquake", {"Causal relevance": 4, "Supply chain relevance": 3, "Spatial coverage": 4, "Temporal resolution": 2, "Temporal coverage": 4, "Ease of access": 4, "Data reliability": 4, "Alignment with GDELT": 2}),
    "Historical earthquake magnitude index": ("Earthquake", {"Causal relevance": 4, "Supply chain relevance": 3, "Spatial coverage": 3, "Temporal resolution": 2, "Temporal coverage": 4, "Ease of access": 4, "Data reliability": 4, "Alignment with GDELT": 2}),
    "Tectonic plate boundary proximity": ("Earthquake", {"Causal relevance": 3, "Supply chain relevance": 2, "Spatial coverage": 4, "Temporal resolution": 1, "Temporal coverage": 4, "Ease of access": 4, "Data reliability": 4, "Alignment with GDELT": 2}),

    # Flooding
    "Rainfall anomaly index": ("Flooding", {"Causal relevance": 5, "Supply chain relevance": 5, "Spatial coverage": 4, "Temporal resolution": 5, "Temporal coverage": 4, "Ease of access": 5, "Data reliability": 5, "Alignment with GDELT": 5}),
    "River discharge / floodplain proximity": ("Flooding", {"Causal relevance": 4, "Supply chain relevance": 5, "Spatial coverage": 4, "Temporal resolution": 4, "Temporal coverage": 4, "Ease of access": 4, "Data reliability": 4, "Alignment with GDELT": 4}),
    "Rainfall intensity": ("Flooding", {"Causal relevance": 5, "Supply chain relevance": 4, "Spatial coverage": 4, "Temporal resolution": 5, "Temporal coverage": 4, "Ease of access": 4, "Data reliability": 4, "Alignment with GDELT": 4}),
    "Climate risk index (precipitation)": ("Flooding", {"Causal relevance": 4, "Supply chain relevance": 4, "Spatial coverage": 4, "Temporal resolution": 3, "Temporal coverage": 4, "Ease of access": 3, "Data reliability": 3, "Alignment with GDELT": 3}),

    # Drought
    "Rainfall deviation (SPI index)": ("Drought", {"Causal relevance": 4, "Supply chain relevance": 4, "Spatial coverage": 4, "Temporal resolution": 3, "Temporal coverage": 4, "Ease of access": 4, "Data reliability": 4, "Alignment with GDELT": 4}),
    "Water reservoir levels": ("Drought", {"Causal relevance": 3, "Supply chain relevance": 3, "Spatial coverage": 3, "Temporal resolution": 2, "Temporal coverage": 3, "Ease of access": 3, "Data reliability": 3, "Alignment with GDELT": 3}),
    "Temperature anomaly": ("Drought", {"Causal relevance": 4, "Supply chain relevance": 3, "Spatial coverage": 4, "Temporal resolution": 4, "Temporal coverage": 4, "Ease of access": 4, "Data reliability": 4, "Alignment with GDELT": 4}),

    # Cyclone / Hurricane
    "Sea surface temperature anomaly": ("Cyclone / Hurricane", {"Causal relevance": 4, "Supply chain relevance": 3, "Spatial coverage": 4, "Temporal resolution": 3, "Temporal coverage": 4, "Ease of access": 3, "Data reliability": 4, "Alignment with GDELT": 3}),
    "Storm frequency index": ("Cyclone / Hurricane", {"Causal relevance": 4, "Supply chain relevance": 3, "Spatial coverage": 3, "Temporal resolution": 2, "Temporal coverage": 3, "Ease of access": 3, "Data reliability": 3, "Alignment with GDELT": 2}),

    # Extreme heat / heatwave
    "Temperature anomaly (heatwave)": ("Extreme heat / heatwave", {"Causal relevance": 5, "Supply chain relevance": 4, "Spatial coverage": 4, "Temporal resolution": 4, "Temporal coverage": 4, "Ease of access": 4, "Data reliability": 4, "Alignment with GDELT": 4}),
    "Days >35°C per year": ("Extreme heat / heatwave", {"Causal relevance": 4, "Supply chain relevance": 4, "Spatial coverage": 4, "Temporal resolution": 3, "Temporal coverage": 4, "Ease of access": 4, "Data reliability": 3, "Alignment with GDELT": 3}),
    "Wet bulb temperature index": ("Extreme heat / heatwave", {"Causal relevance": 5, "Supply chain relevance": 4, "Spatial coverage": 4, "Temporal resolution": 4, "Temporal coverage": 4, "Ease of access": 4, "Data reliability": 4, "Alignment with GDELT": 4}),

    # Landslide
    "Rainfall intensity-duration": ("Landslide", {"Causal relevance": 4, "Supply chain relevance": 3, "Spatial coverage": 3, "Temporal resolution": 4, "Temporal coverage": 3, "Ease of access": 3, "Data reliability": 3, "Alignment with GDELT": 3}),
    "Slope stability index": ("Landslide", {"Causal relevance": 4, "Supply chain relevance": 3, "Spatial coverage": 2, "Temporal resolution": 2, "Temporal coverage": 3, "Ease of access": 2, "Data reliability": 3, "Alignment with GDELT": 2}),
    "Soil moisture anomaly": ("Landslide", {"Causal relevance": 3, "Supply chain relevance": 3, "Spatial coverage": 4, "Temporal resolution": 4, "Temporal coverage": 3, "Ease of access": 3, "Data reliability": 4, "Alignment with GDELT": 3}),

    # Mine collapse / tailings failure
    "Tailings dam stability risk": ("Mine collapse / tailings failure", {"Causal relevance": 4, "Supply chain relevance": 4, "Spatial coverage": 2, "Temporal resolution": 2, "Temporal coverage": 3, "Ease of access": 2, "Data reliability": 3, "Alignment with GDELT": 2}),
    "Precipitation loading anomaly": ("Mine collapse / tailings failure", {"Causal relevance": 4, "Supply chain relevance": 4, "Spatial coverage": 3, "Temporal resolution": 3, "Temporal coverage": 3, "Ease of access": 3, "Data reliability": 3, "Alignment with GDELT": 3}),

    # Labour strikes
    "Unionization rate": ("Labour strikes", {"Causal relevance": 3, "Supply chain relevance": 3, "Spatial coverage": 2, "Temporal resolution": 2, "Temporal coverage": 3, "Ease of access": 3, "Data reliability": 3, "Alignment with GDELT": 2}),
    "Wage growth vs productivity": ("Labour strikes", {"Causal relevance": 3, "Supply chain relevance": 3, "Spatial coverage": 3, "Temporal resolution": 2, "Temporal coverage": 3, "Ease of access": 3, "Data reliability": 3, "Alignment with GDELT": 2}),
    "Unemployment rate": ("Labour strikes", {"Causal relevance": 3, "Supply chain relevance": 3, "Spatial coverage": 4, "Temporal resolution": 3, "Temporal coverage": 4, "Ease of access": 4, "Data reliability": 4, "Alignment with GDELT": 3}),

    # Commodity price shock
    "Market volatility index (VIX)": ("Commodity price shock", {"Causal relevance": 3, "Supply chain relevance": 3, "Spatial coverage": 4, "Temporal resolution": 4, "Temporal coverage": 4, "Ease of access": 4, "Data reliability": 4, "Alignment with GDELT": 3}),
    "Inventory levels": ("Commodity price shock", {"Causal relevance": 3, "Supply chain relevance": 3, "Spatial coverage": 3, "Temporal resolution": 3, "Temporal coverage": 3, "Ease of access": 3, "Data reliability": 3, "Alignment with GDELT": 3}),
    "Demand growth rate": ("Commodity price shock", {"Causal relevance": 3, "Supply chain relevance": 3, "Spatial coverage": 3, "Temporal resolution": 2, "Temporal coverage": 3, "Ease of access": 2, "Data reliability": 3, "Alignment with GDELT": 2}),

    # Trade embargo
    "Trade policy restrictiveness index": ("Trade embargo", {"Causal relevance": 4, "Supply chain relevance": 5, "Spatial coverage": 4, "Temporal resolution": 3, "Temporal coverage": 4, "Ease of access": 4, "Data reliability": 4, "Alignment with GDELT": 3}),
    "Political tension index": ("Trade embargo", {"Causal relevance": 4, "Supply chain relevance": 4, "Spatial coverage": 3, "Temporal resolution": 3, "Temporal coverage": 3, "Ease of access": 3, "Data reliability": 3, "Alignment with GDELT": 3}),
    "Sanction frequency (UN/WTO data)": ("Trade embargo", {"Causal relevance": 5, "Supply chain relevance": 5, "Spatial coverage": 3, "Temporal resolution": 2, "Temporal coverage": 4, "Ease of access": 3, "Data reliability": 4, "Alignment with GDELT": 2}),

    # Pandemic / epidemic
    "Global health security index": ("Pandemic / epidemic", {"Causal relevance": 4, "Supply chain relevance": 4, "Spatial coverage": 4, "Temporal resolution": 2, "Temporal coverage": 3, "Ease of access": 4, "Data reliability": 4, "Alignment with GDELT": 2}),
    "Disease outbreak frequency": ("Pandemic / epidemic", {"Causal relevance": 5, "Supply chain relevance": 4, "Spatial coverage": 4, "Temporal resolution": 3, "Temporal coverage": 4, "Ease of access": 3, "Data reliability": 4, "Alignment with GDELT": 3}),
    "Pathogen surveillance intensity": ("Pandemic / epidemic", {"Causal relevance": 4, "Supply chain relevance": 3, "Spatial coverage": 3, "Temporal resolution": 3, "Temporal coverage": 3, "Ease of access": 3, "Data reliability": 3, "Alignment with GDELT": 3}),

    # Environmental incident / pollution
    "Pollution fine history": ("Environmental incident (pollution)", {"Causal relevance": 3, "Supply chain relevance": 3, "Spatial coverage": 2, "Temporal resolution": 2, "Temporal coverage": 3, "Ease of access": 2, "Data reliability": 3, "Alignment with GDELT": 2}),

    # Resource depletion
    "Static lifetime (R/P)": ("Resource depletion / reserve exhaustion", {"Causal relevance": 4, "Supply chain relevance": 3, "Spatial coverage": 2, "Temporal resolution": 2, "Temporal coverage": 4, "Ease of access": 3, "Data reliability": 3, "Alignment with GDELT": 2}),
    "Exploration success rate": ("Resource depletion / reserve exhaustion", {"Causal relevance": 3, "Supply chain relevance": 3, "Spatial coverage": 2, "Temporal resolution": 2, "Temporal coverage": 3, "Ease of access": 2, "Data reliability": 2, "Alignment with GDELT": 2}),
}

# -------------------------------Compute indicator-level suitability scores -------------------------------#
records = []
for indicator, (disruption, scores) in indicator_data.items():
    total = sum(scores[c] * weights[c] for c in criteria) * 20  # Convert to 0–100 scale
    records.append({"Disruption": disruption, "Indicator": indicator, "Suitability Score": total})

df = pd.DataFrame(records).sort_values("Suitability Score", ascending=False).reset_index(drop=True)

# -------------------------------Compute average suitability per disruption type -------------------------------#
disruption_summary = (
    df.groupby("Disruption")["Suitability Score"]
    .agg(["mean", "max", "count"])
    .rename(columns={"mean": "Average Score", "max": "Best Indicator Score", "count": "Num Indicators"})
    .sort_values("Average Score", ascending=False)
    .reset_index()
)

# ------------------------------- Output ranked results  -------------------------------#
print("\nTop 20 Indicators by Suitability Score:\n")
print(df.head(20).to_string(index=False))

print("\nDisruption Types Ranked by Average Indicator Suitability:\n")
print(disruption_summary.to_string(index=False))

# Save to CSV
df.to_csv("indicator_suitability_likelihood.csv", index=False)
disruption_summary.to_csv("disruption_summary.csv", index=False)
print("\nResults saved to 'indicator_suitability_likelihood.csv' and 'disruption_summary.csv'")
