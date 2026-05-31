"""
export_indicator_importance.py
Generates all indicator importance outputs saved to
final figures/indicator_importance_outputs/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

BASE    = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR = Path(__file__).resolve().parent.parent / "final figures" / "indicator_importance_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [("protest_7d", "Protest 7d"), ("strike_7d", "Strike 7d")]

plt.rcParams.update({
    "font.size": 11, "font.family": "sans-serif",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── Feature block assignment ──────────────────────────────────────────────────

BLOCK_COLORS = {
    "GDELT Lags (M0)":        "#1f77b4",
    "Markets (M1)":           "#ff7f0e",
    "Macro/Governance (M2)":  "#2ca02c",
    "Structural (M3)":        "#d62728",
    "FAO / GTA (M4)":         "#9467bd",
    "Country FE":             "#aaaaaa",
}

FAO_GTA = {"fao_cereals_index_yoy","fao_cereals_index_yoy_above90",
           "fao_cereals_index_yoy_lag1m","fao_cereals_index_yoy_lag3m",
           "fao_cereals_index_yoy_lag6m","fao_cereals_yoy_x_instability",
           "fao_food_index_yoy","fao_food_index_yoy_above90",
           "fao_food_index_yoy_lag1m","fao_food_index_yoy_lag3m",
           "fao_food_index_yoy_lag6m","fao_food_yoy_x_youth_unemp",
           "fao_oils_index_yoy","fao_oils_index_yoy_lag1m",
           "fao_oils_index_yoy_lag3m","fao_oils_index_yoy_lag6m",
           "gta_30d_count","gta_30d_count_z","gta_90d_count","gta_90d_count_z",
           "gta_harmful_events","gta_harmful_events_z",
           "gta_liberalising_events","gta_liberalising_events_z",
           "month_sin","month_cos"}

STRUCTURAL_M3 = {"gini_coef","covid_period","inflation_accel","fx_trend_consistent"}

MARKETS_M1 = {"yield_us10y"}   # plus fx_*, oil_*, gold_*, etc. handled by prefix

MACRO_ENDS = {"_est"}  # governance indicators ending in _est
MACRO_STARTS = ("inflation_cpi","energy_cpi","food_cpi","gdp_","unemployment_")


def get_block(raw_feature: str) -> str:
    f = raw_feature.replace("num__", "").replace("remainder__", "")
    if f.startswith("fe__"):
        return "Country FE"
    if f.startswith("gdelt_"):
        return "GDELT Lags (M0)"
    if f in FAO_GTA:
        return "FAO / GTA (M4)"
    if f in STRUCTURAL_M3 or "_x_" in f:
        return "Structural (M3)"
    if f in MARKETS_M1 or any(f.startswith(p) for p in (
            "fx_","oil_","gold_","silver_","platinum_",
            "copper_","natgas_","vix_","dxy_")):
        return "Markets (M1)"
    if f.endswith("_est") or any(f.startswith(p) for p in MACRO_STARTS):
        return "Macro/Governance (M2)"
    return "Other"


def clean_label(raw: str) -> str:
    f = raw.replace("num__","").replace("remainder__","").replace("fe__","")
    subs = {
        "gdelt_protest_28d_lag":          "GDELT Protest 28d Lag",
        "gdelt_protest_7d_lag":           "GDELT Protest 7d Lag",
        "gdelt_strike_28d_lag":           "GDELT Strike 28d Lag",
        "gdelt_strike_7d_lag":            "GDELT Strike 7d Lag",
        "gdelt_protest_region_14d":       "GDELT Protest Region 14d",
        "gdelt_strike_region_14d":        "GDELT Strike Region 14d",
        "fx_pct_30d":                     "FX Change 30d",
        "fx_pct_7d":                      "FX Change 7d",
        "fx_pct_90d":                     "FX Change 90d",
        "fx_pct_30d_z":                   "FX Change 30d (z)",
        "fx_pct_30d_lag30d":              "FX Change 30d Lag30",
        "fx_pct_30d_lag60d":              "FX Change 30d Lag60",
        "fx_vol_30d":                     "FX Volatility 30d",
        "fx_vol_30d_z":                   "FX Volatility 30d (z)",
        "fx_vol_7d":                      "FX Volatility 7d",
        "fx_trend_consistent":            "FX Trend (Consistent)",
        "fx_pct_30d_x_instability":       "FX×Instability",
        "oil_brent_pct_30d":              "Oil Change 30d",
        "oil_brent_pct_14d":              "Oil Change 14d",
        "oil_brent_pct_30d_z":            "Oil Change 30d (z)",
        "oil_brent_pct_30d_lag30d":       "Oil Change Lag30",
        "oil_brent_pct_30d_lag60d":       "Oil Change Lag60",
        "oil_brent_pct_30d_lag90d":       "Oil Change Lag90",
        "oil_brent_pct_30d_x_inflation":  "Oil×Inflation",
        "oil_brent_pct_30d_x_net_importer": "Oil×Net Importer",
        "gold_pct_30d":                   "Gold Change 30d",
        "gold_vol_30d":                   "Gold Volatility 30d",
        "gold_pct_30d_x_gold_prod":       "Gold×Production",
        "silver_pct_30d":                 "Silver Change 30d",
        "platinum_pct_30d":               "Platinum Change 30d",
        "platinum_pct_30d_x_plat_prod":   "Platinum×Production",
        "copper_pct_30d":                 "Copper Change 30d",
        "copper_pct_90d":                 "Copper Change 90d",
        "copper_vol_30d":                 "Copper Volatility 30d",
        "copper_pct_30d_x_copper_prod":   "Copper×Production",
        "natgas_pct_30d":                 "Nat. Gas Change 30d",
        "vix_level":                      "VIX Level",
        "vix_7d_ma":                      "VIX 7d MA",
        "vix_pct_30d":                    "VIX Change 30d",
        "vix_pct_30d_lag30d":             "VIX Change Lag30",
        "vix_pct_30d_lag60d":             "VIX Change Lag60",
        "dxy_level":                      "DXY Level",
        "dxy_pct_30d":                    "DXY Change 30d",
        "dxy_vol_30d":                    "DXY Volatility 30d",
        "yield_us10y":                    "US 10y Yield",
        "inflation_cpi_yoy":              "CPI Inflation YoY",
        "inflation_cpi_yoy_z":            "CPI Inflation YoY (z)",
        "energy_cpi_inflation":           "Energy CPI Inflation",
        "energy_cpi_inflation_z":         "Energy CPI Inflation (z)",
        "food_cpi_inflation":             "Food CPI Inflation",
        "food_cpi_inflation_z":           "Food CPI Inflation (z)",
        "gdp_growth":                     "GDP Growth",
        "gdp_per_capita_growth":          "GDP per Capita Growth",
        "unemployment_rate":              "Unemployment Rate",
        "unemployment_rate_z":            "Unemployment Rate (z)",
        "unemployment_sa":                "Unemployment SA",
        "unemployment_total":             "Unemployment Total",
        "unemployment_youth":             "Youth Unemployment",
        "government_effectiveness_est":   "Gov. Effectiveness",
        "political_stability_est":        "Political Stability",
        "rule_of_law_est":                "Rule of Law",
        "voice_accountability_est":       "Voice & Accountability",
        "gini_coef":                      "Gini Coefficient",
        "covid_period":                   "COVID Period",
        "inflation_accel":                "Inflation Acceleration",
        "fao_food_index_yoy":             "FAO Food Index YoY",
        "fao_food_index_yoy_above90":     "FAO Food >90th Pct.",
        "fao_food_index_yoy_lag1m":       "FAO Food Lag 1m",
        "fao_food_index_yoy_lag3m":       "FAO Food Lag 3m",
        "fao_food_index_yoy_lag6m":       "FAO Food Lag 6m",
        "fao_cereals_index_yoy":          "FAO Cereals YoY",
        "fao_cereals_index_yoy_above90":  "FAO Cereals >90th Pct.",
        "fao_cereals_index_yoy_lag1m":    "FAO Cereals Lag 1m",
        "fao_cereals_index_yoy_lag3m":    "FAO Cereals Lag 3m",
        "fao_cereals_index_yoy_lag6m":    "FAO Cereals Lag 6m",
        "fao_cereals_yoy_x_instability":  "FAO Cereals×Instability",
        "fao_oils_index_yoy":             "FAO Oils YoY",
        "fao_oils_index_yoy_lag1m":       "FAO Oils Lag 1m",
        "fao_oils_index_yoy_lag3m":       "FAO Oils Lag 3m",
        "fao_oils_index_yoy_lag6m":       "FAO Oils Lag 6m",
        "fao_food_yoy_x_youth_unemp":     "FAO Food×Youth Unemp.",
        "gta_30d_count":                  "GTA Interventions 30d",
        "gta_30d_count_z":                "GTA Interventions 30d (z)",
        "gta_90d_count":                  "GTA Interventions 90d",
        "gta_90d_count_z":                "GTA Interventions 90d (z)",
        "gta_harmful_events":             "GTA Harmful Events",
        "gta_harmful_events_z":           "GTA Harmful Events (z)",
        "gta_liberalising_events":        "GTA Liberalising Events",
        "gta_liberalising_events_z":      "GTA Liberalising Events (z)",
        "month_sin":                      "Month (sin)",
        "month_cos":                      "Month (cos)",
    }
    return subs.get(f, f.replace("_", " ").title())


# ── Load and process LR coefficients ─────────────────────────────────────────

def load_lr_coefs(model_name: str) -> pd.DataFrame:
    dfs = []
    for target, _ in TARGETS:
        df = pd.read_csv(BASE / target / "coefs_lr.csv")
        df = df[(df["model_name"] == model_name) &
                (~df["feature"].str.startswith("fe__"))].copy()
        df["target"] = target
        dfs.append(df)
    coefs = pd.concat(dfs, ignore_index=True)
    # Average across folds
    avg = (coefs.groupby(["target", "feature"])["coefficient"]
                .mean().reset_index())
    avg["block"] = avg["feature"].map(get_block)
    avg["label"] = avg["feature"].map(clean_label)
    avg["abs_coef"] = avg["coefficient"].abs()
    return avg


# ── LR coefficient figure ─────────────────────────────────────────────────────

def fig_lr_coefs(model_name: str, title_prefix: str,
                 out_name: str, top_n: int = 20) -> None:
    coefs = load_lr_coefs(model_name)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle(f"{title_prefix} — LR Coefficients (log-odds, averaged over folds)",
                 fontsize=12, fontweight="bold")

    legend_blocks = set()

    for ax, (target, tlabel) in zip(axes, TARGETS):
        sub = coefs[coefs["target"] == target].copy()
        if top_n is not None:
            sub = sub.nlargest(top_n, "abs_coef")
        sub = sub.sort_values("coefficient")

        colors = [BLOCK_COLORS.get(b, "#555555") for b in sub["block"]]
        bars   = ax.barh(sub["label"], sub["coefficient"],
                         color=colors, edgecolor="white", linewidth=0.3,
                         alpha=0.88, zorder=3)
        ax.axvline(0, color="black", lw=0.9, ls="-", alpha=0.5, zorder=2)
        ax.xaxis.grid(True, color="#cccccc", lw=0.5, alpha=0.6, zorder=0)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_xlabel("Coefficient (log-odds)", fontsize=11)
        ax.set_title(tlabel, fontsize=11, fontweight="bold")
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="y", labelsize=9)

        for b in sub["block"].unique():
            legend_blocks.add(b)

    patches = [mpatches.Patch(color=BLOCK_COLORS[b], alpha=0.88, label=b)
               for b in BLOCK_COLORS if b in legend_blocks and b != "Country FE"]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ── SHAP figures ──────────────────────────────────────────────────────────────

def load_shap(model_name: str, exclude_fe: bool = True) -> pd.DataFrame:
    dfs = []
    for target, _ in TARGETS:
        df = pd.read_csv(BASE / target / "shap_importance.csv")
        df = df[df["model_name"] == model_name].copy()
        if exclude_fe:
            df = df[~df["feature"].str.startswith("fe__")]
        df["target"] = target
        dfs.append(df)
    shap = pd.concat(dfs, ignore_index=True)
    shap["block"] = shap["feature"].map(get_block)
    shap["label"] = shap["feature"].map(clean_label)
    return shap


def fig_shap_values(model_name: str, title_prefix: str,
                    out_name: str, top_n: int = 20) -> None:
    shap = load_shap(model_name)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle(f"{title_prefix} — Mean |SHAP| (averaged over folds)",
                 fontsize=12, fontweight="bold")

    legend_blocks = set()

    for ax, (target, tlabel) in zip(axes, TARGETS):
        sub = (shap[shap["target"] == target]
               .nlargest(top_n, "mean_abs_shap")
               .sort_values("mean_abs_shap"))

        colors = [BLOCK_COLORS.get(b, "#555555") for b in sub["block"]]
        ax.barh(sub["label"], sub["mean_abs_shap"],
                color=colors, edgecolor="white", linewidth=0.3,
                alpha=0.88, zorder=3)
        ax.xaxis.grid(True, color="#cccccc", lw=0.5, alpha=0.6, zorder=0)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(tlabel, fontsize=11, fontweight="bold")
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="y", labelsize=9)

        for b in sub["block"].unique():
            legend_blocks.add(b)

    patches = [mpatches.Patch(color=BLOCK_COLORS[b], alpha=0.88, label=b)
               for b in BLOCK_COLORS if b in legend_blocks and b != "Country FE"]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_shap_comparison(model_name: str, out_name: str, top_n: int = 10) -> None:
    shap = load_shap(model_name)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "M6 XGBoost Full — Top Feature Comparison: Protest vs Strike (Mean |SHAP|)",
        fontsize=12, fontweight="bold",
    )

    legend_blocks = set()

    for ax, (target, tlabel) in zip(axes, TARGETS):
        sub = (shap[shap["target"] == target]
               .nlargest(top_n, "mean_abs_shap")
               .sort_values("mean_abs_shap"))

        colors = [BLOCK_COLORS.get(b, "#555555") for b in sub["block"]]
        ax.barh(sub["label"], sub["mean_abs_shap"],
                color=colors, edgecolor="white", linewidth=0.3,
                alpha=0.88, zorder=3)

        # Annotate rank
        for i, (_, row) in enumerate(sub.iloc[::-1].iterrows()):
            rank = top_n - i
            ax.text(row["mean_abs_shap"] * 0.02, i,
                    f"#{rank}", va="center", fontsize=8, color="white",
                    fontweight="bold")

        ax.xaxis.grid(True, color="#cccccc", lw=0.5, alpha=0.6, zorder=0)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title(tlabel, fontsize=11, fontweight="bold")
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.tick_params(axis="y", labelsize=10)

        for b in sub["block"].unique():
            legend_blocks.add(b)

    patches = [mpatches.Patch(color=BLOCK_COLORS[b], alpha=0.88, label=b)
               for b in BLOCK_COLORS if b in legend_blocks and b != "Country FE"]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = OUT_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


# ── CSV exports ───────────────────────────────────────────────────────────────

def save_lr_coefs_csv(model_name: str, out_name: str) -> pd.DataFrame:
    coefs = load_lr_coefs(model_name)
    pivot = coefs.pivot_table(
        index=["feature", "label", "block"],
        columns="target",
        values="coefficient",
    ).reset_index()
    pivot.columns.name = None
    rename = {}
    for t, _ in TARGETS:
        if t in pivot.columns:
            rename[t] = f"coefficient_{t.replace('_7d','')}"
    pivot = pivot.rename(columns=rename)
    pivot = pivot.sort_values(
        pivot.filter(like="coefficient").abs().sum(axis=1).name
        if False else "feature"
    )
    # Sort by max abs coefficient across targets
    coef_cols = [c for c in pivot.columns if c.startswith("coefficient_")]
    pivot["_max_abs"] = pivot[coef_cols].abs().max(axis=1)
    pivot = pivot.sort_values("_max_abs", ascending=False).drop(columns="_max_abs")
    pivot = pivot[["feature", "label", "block"] + coef_cols]
    for c in coef_cols:
        pivot[c] = pivot[c].round(4)
    out = OUT_DIR / out_name
    pivot.to_csv(out, index=False)
    print(f"Saved: {out}  ({len(pivot)} rows)")
    return pivot


def save_shap_csv(model_name: str, out_name: str) -> pd.DataFrame:
    shap = load_shap(model_name, exclude_fe=False)
    pivot = shap.pivot_table(
        index=["feature", "label", "block"],
        columns="target",
        values="mean_abs_shap",
    ).reset_index()
    pivot.columns.name = None
    rename = {}
    for t, _ in TARGETS:
        if t in pivot.columns:
            rename[t] = f"shap_{t.replace('_7d','')}"
    pivot = pivot.rename(columns=rename)
    shap_cols = [c for c in pivot.columns if c.startswith("shap_")]
    pivot["_max"] = pivot[shap_cols].max(axis=1)
    pivot = pivot.sort_values("_max", ascending=False).drop(columns="_max")
    pivot = pivot[["feature", "label", "block"] + shap_cols]
    for c in shap_cols:
        pivot[c] = pivot[c].round(6)
    out = OUT_DIR / out_name
    pivot.to_csv(out, index=False)
    print(f"Saved: {out}  ({len(pivot)} rows)")
    return pivot


# ── Plain-text summary ────────────────────────────────────────────────────────

THEORETICAL_NOTES = {
    # GDELT lags
    "GDELT Protest 28d Lag":    ("protest", "+", "Expected +: momentum effect — recent protests predict future ones"),
    "GDELT Protest 7d Lag":     ("protest", "+", "Expected +: short-run persistence of protest activity"),
    "GDELT Strike 28d Lag":     ("strike",  "+", "Expected +: persistence of industrial action"),
    "GDELT Strike 7d Lag":      ("strike",  "+", "Expected +: immediate contagion within strike wave"),
    "GDELT Strike Region 14d":  ("both",    "+", "Expected +: regional spill-over from strikes in neighbouring countries"),
    "GDELT Protest Region 14d": ("both",    "+", "Expected +: regional contagion from protests nearby"),
    # Markets
    "FX Change 30d":            ("both", "−", "Expected −: local currency appreciation (positive = stronger) reduces economic grievances"),
    "FX Volatility 30d":        ("both", "+", "Expected +: higher FX volatility signals economic instability, raises unrest risk"),
    "Oil Change 30d":           ("both", "+", "Expected +: rising oil prices raise fuel/food costs, increasing social pressure"),
    "VIX Level":                ("both", "+", "Expected +: elevated global risk appetite associated with weaker EM conditions"),
    "DXY Level":                ("both", "+", "Expected +: stronger USD tightens global financial conditions for EM economies"),
    "Gold Change 30d":          ("both", "?", "Ambiguous: safe-haven demand could reflect either risk-off sentiment (+) or wealth effect (−)"),
    "Copper Change 30d":        ("both", "?", "Ambiguous: copper as industrial demand proxy; price rise could signal growth (+) or cost pressure (+)"),
    "US 10y Yield":             ("both", "+", "Expected +: rising US yields tighten global financial conditions and EM borrowing costs"),
    # Macro/Governance
    "CPI Inflation YoY":        ("both", "+", "Expected +: higher inflation erodes real wages, a leading driver of social unrest"),
    "Unemployment Rate":        ("both", "+", "Expected +: higher unemployment directly increases grievances and strike likelihood"),
    "Youth Unemployment":       ("both", "+", "Expected +: youth joblessness strongly linked to protest activity in literature"),
    "GDP Growth":               ("both", "−", "Expected −: stronger growth reduces economic grievances"),
    "Gov. Effectiveness":       ("both", "−", "Expected −: better governance reduces social conflict"),
    "Political Stability":      ("both", "−", "Expected −: more stable political environment reduces unrest"),
    "Rule of Law":              ("both", "−", "Expected −: stronger institutions dampen strike/protest activity"),
    "Voice & Accountability":   ("both", "?", "Ambiguous: more democratic accountability may enable (legit.) protest (+) or reduce grievances (−)"),
    "Food CPI Inflation":       ("both", "+", "Expected +: food price shocks are one of the strongest empirical triggers of civil unrest"),
    "Energy CPI Inflation":     ("both", "+", "Expected +: energy price increases raise cost of living and protest risk"),
    # Structural
    "Gini Coefficient":         ("both", "+", "Expected +: higher inequality increases relative deprivation and social tensions"),
    "COVID Period":             ("both", "?", "Ambiguous: lockdowns suppressed protests mechanically (−) but economic stress increased latent risk (+)"),
    "Inflation Acceleration":   ("both", "+", "Expected +: accelerating inflation is more salient to households than level; shocks drive protests"),
    "FX Trend (Consistent)":    ("both", "+", "Expected +: sustained depreciation trend signals persistent economic pressure"),
    "Oil×Inflation":            ("both", "+", "Expected +: oil shock amplified by high inflation environment"),
    "Oil×Net Importer":         ("both", "+", "Expected +: oil price rise hurts net importers more through terms-of-trade effect"),
    "FX×Instability":           ("both", "+", "Expected +: currency depreciation is more destabilising in already-fragile political contexts"),
    "Copper×Production":        ("both", "?", "Ambiguous: copper price rise benefits producers but raises input costs; direction depends on net position"),
    # FAO/GTA
    "FAO Food Index YoY":       ("both", "+", "Expected +: global food price increases directly raise food insecurity, a key protest trigger"),
    "FAO Cereals YoY":          ("both", "+", "Expected +: cereal prices affect staple food costs; elevated prices historically precede unrest"),
    "GTA Harmful Events":       ("both", "+", "Expected +: trade-restricting policies signal protectionism and supply disruptions"),
    "FAO Cereals×Instability":  ("both", "+", "Expected +: food price shock amplified in politically fragile contexts"),
}


def save_summary_txt(coef_csv: pd.DataFrame, shap_csv: pd.DataFrame) -> None:
    lines = [
        "=" * 70,
        "TOP FEATURES SUMMARY",
        "Models: LR M3 Structural | LR M0 Persistence | M6 XGBoost Full",
        "=" * 70,
        "",
    ]

    def add_section(title, df, value_col_protest, value_col_strike,
                    value_label, top_n=10):
        lines.append(f"{'─'*70}")
        lines.append(title)
        lines.append(f"{'─'*70}")
        for target_col, target_name in [
            (value_col_protest, "PROTEST 7d"),
            (value_col_strike,  "STRIKE 7d"),
        ]:
            if target_col not in df.columns:
                continue
            lines.append(f"\n  {target_name} — Top {top_n} by absolute value:")
            top = df.dropna(subset=[target_col]).copy()
            top["_abs"] = top[target_col].abs()
            top = top.nlargest(top_n, "_abs")
            for rank, (_, row) in enumerate(top.iterrows(), 1):
                val  = row[target_col]
                sign = "+" if val >= 0 else "−"
                lbl  = row.get("label", row["feature"])
                blk  = row.get("block", "")
                note = THEORETICAL_NOTES.get(lbl, ("both", "?", "No note available"))[2]
                lines.append(
                    f"  {rank:>2}. {lbl:<38}  {sign}{abs(val):.4f}  [{blk}]"
                )
                lines.append(f"      → {note}")
        lines.append("")

    # LR M3
    coef_m3 = save_lr_coefs_csv("model3_structural", "_tmp_m3.csv")
    add_section(
        "LR M3 STRUCTURAL",
        coef_m3,
        "coefficient_protest",
        "coefficient_strike",
        "LR coefficient",
    )

    # LR M0
    coef_m0 = save_lr_coefs_csv("model0_persistence", "_tmp_m0.csv")
    add_section(
        "LR M0 PERSISTENCE",
        coef_m0,
        "coefficient_protest",
        "coefficient_strike",
        "LR coefficient",
    )

    # M6 SHAP
    add_section(
        "M6 XGBOOST FULL — SHAP IMPORTANCE",
        shap_csv,
        "shap_protest",
        "shap_strike",
        "Mean |SHAP|",
    )

    lines += [
        "=" * 70,
        "NOTES",
        "─" * 70,
        "• LR coefficients are log-odds units, averaged across fold 1 (2020)",
        "  and fold 2 (2021) of the static backtest.",
        "• SHAP values are mean absolute SHAP, averaged across both folds.",
        "• FX features: positive direction = local currency strengthening",
        "  (appreciation); negative coefficient = appreciation reduces unrest.",
        "• Governance _est features: higher = better institutional quality.",
        "• Country fixed effects are excluded from all rankings.",
        "=" * 70,
    ]

    out = OUT_DIR / "top_features_summary.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {out}")

    # Remove temp files
    for tmp in ["_tmp_m3.csv", "_tmp_m0.csv"]:
        p = OUT_DIR / tmp
        if p.exists():
            p.unlink()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Saving to {OUT_DIR}\n")

    # Figures
    fig_lr_coefs("model3_structural", "M3 LR Structural",
                 "lr_coefficients_m3.png", top_n=20)
    fig_lr_coefs("model0_persistence", "M0 LR Persistence",
                 "lr_coefficients_m0.png", top_n=None)   # show all
    fig_shap_values("model5_xgb", "M6 XGBoost Full",
                    "shap_values_m6.png", top_n=20)
    fig_shap_comparison("model5_xgb", "shap_comparison.png", top_n=10)
    fig_shap_values("model6_xgb_nolag", "M7 XGBoost No Lags",
                    "shap_values_m7.png", top_n=20)
    fig_shap_comparison("model6_xgb_nolag", "shap_comparison_m7.png", top_n=10)

    # CSVs
    coef_m3_df = save_lr_coefs_csv("model3_structural", "lr_coefficients_m3.csv")
    shap_m6_df = save_shap_csv("model5_xgb", "shap_values_m6.csv")
    save_shap_csv("model6_xgb_nolag", "shap_values_m7.csv")

    # Text summary
    save_summary_txt(coef_m3_df, shap_m6_df)

    print("\nDone.")
