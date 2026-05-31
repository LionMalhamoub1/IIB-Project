#!/usr/bin/env python3
"""
Plotting utilities for consolidated disruption data.

Creates:
1) Bar chart of disruption type counts (known only)
2) Confidence score distribution

Saves plots to:
    project_root/plots/
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


# ------------------ PUBLIC ENTRY POINT ------------------ #

def run_plots(
    df_consolidated: pd.DataFrame,
    *,
    project_root: Path,
):
    """
    Generate and save disruption plots.

    Parameters
    ----------
    df_consolidated : pd.DataFrame
        Consolidated disruptions DataFrame.
    project_root : Path
        Root directory of project (used to locate /plots folder).
    """

    plots_dir = project_root / "plots"
    plots_dir.mkdir(exist_ok=True)

    # ---- Filter to known disruptions only ---- #
    df = df_consolidated.copy()
    df = df[df["disruption_type"].fillna("unknown") != "unknown"]

    if len(df) == 0:
        print("No known disruptions to plot.")
        return

    # ==============================
    # 1) BAR CHART – TYPE COUNTS
    # ==============================

    type_counts = (
        df["disruption_type"]
        .value_counts()
        .sort_values(ascending=False)
    )

    plt.figure()
    type_counts.plot(kind="bar")
    plt.title("Disruption Type Counts (Consolidated, Known Only)")
    plt.xlabel("Disruption Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    type_path = plots_dir / "disruption_type_counts.png"
    plt.savefig(type_path)
    plt.close()

    print(f"Saved: {type_path}")


    # ======================================
    # 2) CONFIDENCE SCORE DISTRIBUTION
    # ======================================

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    plt.figure()
    plt.hist(df["confidence"].dropna(), bins=20)
    plt.title("Confidence Score Distribution (Consolidated, Known Only)")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.tight_layout()

    conf_path = plots_dir / "confidence_distribution.png"
    plt.savefig(conf_path)
    plt.close()

    print(f"Saved: {conf_path}")