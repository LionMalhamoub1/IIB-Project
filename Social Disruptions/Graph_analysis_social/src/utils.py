"""
utils.py
--------
Shared utilities for graph analysis scripts.

Provides:
    load_panel_data()          – load and date-filter the modelling panel
    select_features()          – filter to available columns
    safe_correlation()         – pooled Pearson correlation matrix
    country_average_correlation() – per-country correlation then average
    build_network_from_matrix() – NetworkX graph from a square matrix
    save_network_plot()        – draw and save a network figure
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must precede pyplot import
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_panel_data(
    path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Load the modelling panel parquet and optionally filter by date range."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    df = df.sort_values(["country_iso3", "date"]).reset_index(drop=True)
    logger.info(
        "Loaded panel: %d rows | %d countries | %s to %s",
        len(df),
        df["country_iso3"].nunique(),
        str(df["date"].min().date()),
        str(df["date"].max().date()),
    )
    return df


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_features(df: pd.DataFrame, feature_list: list[str]) -> pd.DataFrame:
    """Return a sub-DataFrame containing only the columns in feature_list that
    exist in df.  Warns about any missing columns."""
    present = [c for c in feature_list if c in df.columns]
    missing = [c for c in feature_list if c not in df.columns]
    if missing:
        logger.warning("Feature(s) not found in panel and will be skipped: %s", missing)
    return df[present]


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def safe_correlation(df: pd.DataFrame, min_obs: int = 30) -> pd.DataFrame:
    """Pearson correlation matrix on a pooled (wide) DataFrame.

    Column pairs with fewer than `min_obs` joint non-NaN observations receive
    NaN rather than a potentially unreliable estimate.
    """
    return df.corr(method="pearson", min_periods=min_obs)


def country_average_correlation(
    df: pd.DataFrame,
    feature_cols: list[str],
    min_obs: int = 30,
) -> pd.DataFrame:
    """Compute per-country Pearson correlation matrices then average them.

    Countries with fewer than `min_obs` non-missing rows are skipped.
    The returned matrix is re-indexed to `feature_cols` for consistency.
    """
    mats: list[pd.DataFrame] = []
    for iso, grp in df.groupby("country_iso3"):
        sub = grp[feature_cols].dropna(how="all")
        if len(sub) < min_obs:
            continue
        mats.append(sub.corr(method="pearson", min_periods=min_obs))

    if not mats:
        raise ValueError(
            "No country had sufficient observations to compute correlations. "
            "Try lowering MIN_OBS or using USE_POOLED_DATA=True."
        )

    avg = pd.concat(mats).groupby(level=0).mean()
    avg = avg.reindex(index=feature_cols, columns=feature_cols)
    return avg


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------

def build_network_from_matrix(
    matrix: pd.DataFrame,
    threshold: float,
    directed: bool = False,
) -> nx.Graph | nx.DiGraph:
    """Build a NetworkX (di)graph from a square DataFrame.

    Only entries with |value| >= threshold are added as edges.
    Self-loops are excluded.  NaN entries are skipped.

    Parameters
    ----------
    matrix:    Square DataFrame (index == columns).
    threshold: Minimum |value| required to add an edge.
    directed:  If True returns a DiGraph; otherwise an undirected Graph.
    """
    G: nx.Graph | nx.DiGraph = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(matrix.columns)

    for i in matrix.index:
        for j in matrix.columns:
            if i == j:
                continue
            val = matrix.loc[i, j]
            if pd.isna(val):
                continue
            if abs(val) >= threshold:
                G.add_edge(i, j, weight=float(val))

    return G


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def save_network_plot(
    G: nx.Graph | nx.DiGraph,
    out_path: Path,
    title: str = "",
    figsize: tuple[int, int] = (14, 10),
    directed: bool = False,
) -> None:
    """Draw a network and save a high-resolution PNG.

    Visual encoding:
    - Node size  : proportional to degree.
    - Edge width : proportional to |weight|.
    - Edge color : blue = positive weight, red = negative weight.
    - Arrow heads: shown for directed graphs.
    """
    fig, ax = plt.subplots(figsize=figsize)

    n = max(len(G.nodes()), 1)
    pos = nx.spring_layout(G, seed=42, k=2.5 / n ** 0.5)

    degrees = dict(G.degree())
    node_sizes = [300 + 150 * degrees.get(nd, 0) for nd in G.nodes()]

    edges = list(G.edges(data=True))
    weights = [e[2].get("weight", 0.0) for e in edges]
    edge_colors = ["#2166ac" if w >= 0 else "#d6604d" for w in weights]
    edge_widths = [max(0.5, abs(w) * 4) for w in weights]

    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color="#f7f7f7",
        edgecolors="#333333",
        linewidths=0.8,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="#222222", ax=ax)
    edge_kwargs: dict = dict(
        edge_color=edge_colors,
        width=edge_widths,
        ax=ax,
    )
    if directed:
        edge_kwargs.update(
            arrows=True,
            arrowstyle="-|>",
            arrowsize=15,
            connectionstyle="arc3,rad=0.1",
        )
    nx.draw_networkx_edges(G, pos, **edge_kwargs)

    if title:
        ax.set_title(title, fontsize=13, pad=12)

    legend_elements = [
        mpatches.Patch(facecolor="#2166ac", label="Positive association"),
        mpatches.Patch(facecolor="#d6604d", label="Negative association"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved network figure -> %s", out_path)
