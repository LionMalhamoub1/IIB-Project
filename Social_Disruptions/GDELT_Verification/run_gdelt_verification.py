# Annotates GDELT protest/strike events with ACLED and MMAD match flags.
# ACLED: same country ±window days. MMAD: same country-month.
# Unresolved countries kept in output with null match columns.

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Shared utilities (ISO-3 geocoding, fuzzy matching)
from _utils import extract_iso3 as _extract_iso3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # Social_Disruptions/verification/
_SD   = _HERE.parent                             # Social_Disruptions/
_ROOT = _SD.parent                               # repo root

# Prefer the deduplicated output; fall back to the raw consolidated file.
_DEDUPED_PATH      = _HERE / "gdelt_social_deduped.jsonl"
_RAW_FALLBACK_PATH = _ROOT / "Builder_GDELT" / "results" / "combined" / "all_consolidated.jsonl"
DEFAULT_GDELT_PATH  = _DEDUPED_PATH

DEFAULT_ACLED_PANEL = _SD  / "Likelihood_Modelling" / "data" / "processed" / "acled_country_day_2017_2025.parquet"
DEFAULT_ACLED_RAW   = _SD  / "Data_Sources" / "ACLED" / "data" / "raw" / "events"
DEFAULT_MMAD_PANEL  = _SD  / "Data_Sources" / "MMAD" / "data" / "processed" / "mmad_country_month_2017_2025.parquet"
DEFAULT_OUT_DIR     = _HERE

SOCIAL_TYPES: frozenset[str] = frozenset({"protests", "labour_strike"})
CONFIDENCE_THRESHOLD: float  = 0.6
DEFAULT_ACLED_WINDOW: int    = 0   # exact day match by default

# ---------------------------------------------------------------------------
# GDELT loading
# ---------------------------------------------------------------------------

def load_gdelt_social(path: Path) -> pd.DataFrame:
    """
    Load ALL protests and labour-strike events from a GDELT JSONL file.

    Handles two formats transparently:
      Deduped  : output of deduplicate_gdelt_social.py — has cluster_id,
                 n_source_articles, has_conflict.  Confidence already filtered.
      Raw      : output of GDELT extraction pipeline — one record per article.

    Confidence filter (>= 0.6) is applied to the raw format only (the dedup
    step already enforces it).  Every event — including those with no resolvable
    country — is kept; unresolved events get iso3=NaN.
    """
    # Fallback: if the preferred deduped file is missing, try the raw path.
    if not path.exists() and path == _DEDUPED_PATH and _RAW_FALLBACK_PATH.exists():
        log.warning(
            "Deduped GDELT file not found at %s — falling back to raw file %s.\n"
            "  Run deduplicate_gdelt_social.py first for best results.",
            path, _RAW_FALLBACK_PATH,
        )
        path = _RAW_FALLBACK_PATH

    log.info("Loading GDELT social events: %s", path)
    records = []
    skipped_type = skipped_conf = skipped_parse = 0
    is_deduped = False

    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                skipped_parse += 1
                continue

            if e.get("disruption_type") not in SOCIAL_TYPES:
                skipped_type += 1
                continue

            # Detect deduped format on first qualifying record
            if not records:
                is_deduped = "cluster_id" in e or "n_source_articles" in e

            # Raw format: apply confidence filter; deduped already filtered
            if not is_deduped and (e.get("confidence") or 0.0) < CONFIDENCE_THRESHOLD:
                skipped_conf += 1
                continue

            det = e.get("details") or {}
            records.append({
                "event_date":      e.get("event_date") or e.get("publish_date"),
                "disruption_type": e["disruption_type"],
                "iso3":            _extract_iso3(e),
                "confidence":      float(e.get("confidence") or 0.0),
                # Deduped: n_source_articles; raw: num_articles or 1
                "num_articles":    int(
                    e.get("n_source_articles") or e.get("num_articles") or 1
                ),
                "location_raw":    json.dumps(
                    e.get("location") or e.get("location_name") or ""
                ),
                # Deduplication provenance (None for raw format)
                "cluster_id":      e.get("cluster_id"),
                "has_conflict":    e.get("has_conflict"),
                # Flatten details fields for easy inspection
                "protest_type":           det.get("protest_type"),
                "sector":                 det.get("sector"),
                "issue":                  det.get("issue"),
                "estimated_participants": det.get("estimated_participants"),
                "protesting_groups":      det.get("protesting_groups"),
                "organizations":          det.get("organizations_or_companies"),
                "target_of_protest":      det.get("target_of_protest"),
            })

    df = pd.DataFrame(records)
    if df.empty:
        log.warning("No qualifying GDELT social events found.")
        return df

    df["event_date"] = (
        pd.to_datetime(df["event_date"], errors="coerce", utc=False)
        .dt.tz_localize(None)
        .dt.normalize()          # strip time component → date only
    )
    df = df.dropna(subset=["event_date"])
    df["year_month"] = df["event_date"].dt.to_period("M")

    n_resolved = df["iso3"].notna().sum()
    log.info(
        "Loaded %d events (format=%s) | skipped: type=%d conf=%d parse=%d",
        len(df), "deduped" if is_deduped else "raw",
        skipped_type, skipped_conf, skipped_parse,
    )
    log.info("  Types:            %s", df["disruption_type"].value_counts().to_dict())
    log.info("  Date range:       %s → %s",
             df["event_date"].min().date(), df["event_date"].max().date())
    log.info("  Country resolved: %d / %d (%.1f%%)",
             n_resolved, len(df), 100 * n_resolved / len(df))
    return df


# ---------------------------------------------------------------------------
# Reference lookups
# ---------------------------------------------------------------------------

def build_acled_lookup(panel_path: Path, raw_events_dir: Path) -> pd.DataFrame:
    """
    Build a country-day lookup table from ACLED protest events.

    Returns a DataFrame with columns [iso3, date, acled_n_events].
    Only rows where acled_n_events > 0 are kept (sparse representation).
    """
    if panel_path.exists():
        log.info("Loading ACLED daily panel: %s", panel_path)
        panel = pd.read_parquet(panel_path)
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        panel = panel.rename(columns={"country_iso3": "iso3", "acled_events": "acled_n_events"})
        active = panel[panel["acled_n_events"] > 0][["iso3", "date", "acled_n_events"]].copy()
        log.info("ACLED lookup: %d active country-day cells", len(active))
        return active

    parquet_files = sorted(raw_events_dir.glob("iso3=*/year=*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No ACLED data found.\n"
            f"  Checked panel: {panel_path}\n"
            f"  Checked raw:   {raw_events_dir}\n"
            "Run build_acled_country_day.py first."
        )

    log.info("Loading %d raw ACLED parquet partitions...", len(parquet_files))
    frames = []
    for f in parquet_files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as exc:
            log.warning("Skipping %s: %s", f.name, exc)

    if not frames:
        raise RuntimeError(f"All {len(parquet_files)} ACLED partitions failed to load.")

    events = pd.concat(frames, ignore_index=True)
    events["date"] = pd.to_datetime(events["event_date"], errors="coerce").dt.normalize()
    events["iso3"] = events["iso3"].str.strip().str.upper()
    events = events.dropna(subset=["date", "iso3"])

    protests = events[events["event_type"] == "Protests"]
    active = (
        protests.groupby(["iso3", "date"])
        .size()
        .reset_index(name="acled_n_events")
    )
    log.info("ACLED lookup: %d active country-day cells from raw partitions", len(active))
    return active


def build_mmad_lookup(path: Path) -> dict[tuple[str, object], int]:
    """
    Build a (iso3, Period('M')) → protest_count dict from the MMAD panel.
    Only entries with protest_count > 0 are included.
    """
    log.info("Loading MMAD monthly panel: %s", path)
    df = pd.read_parquet(path)
    df["iso3"] = df["iso3"].str.strip().str.upper()
    df["year_month"] = pd.PeriodIndex(
        pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)),
        freq="M",
    )
    active = df[df["protest_count"] > 0]
    # Vectorised dict construction — faster than itertuples for large panels
    lookup: dict[tuple[str, object], int] = dict(
        zip(zip(active["iso3"], active["year_month"]), active["protest_count"].astype(int))
    )
    log.info("MMAD lookup: %d active country-month cells", len(lookup))
    return lookup


# ---------------------------------------------------------------------------
# Event-level matching
# ---------------------------------------------------------------------------

def match_against_acled(
    gdelt: pd.DataFrame,
    acled_active: pd.DataFrame,
    window_days: int,
    suffix: str | None = None,
) -> pd.DataFrame:
    """
    For each GDELT event with a resolved country, check whether ACLED recorded
    any protest in the same country within ±window_days of the event date.

    suffix controls the output column names so the function can be called
    multiple times with different windows without overwriting results:
        acled_match_<suffix>    : True / False / NaN
        acled_n_events_<suffix> : total ACLED events in the window

    If suffix is None it is derived from window_days:
        window_days=0  → suffix "exact"
        window_days=N  → suffix "±Nd"

    Implementation note: uses vectorised merge-per-offset rather than Python
    itertuples, which is ~20–100× faster for large event sets.
    """
    if suffix is None:
        suffix = "exact" if window_days == 0 else f"±{window_days}d"

    col_match = f"acled_match_{suffix}"
    col_count = f"acled_n_events_{suffix}"

    gdelt = gdelt.copy()

    if acled_active.empty:
        gdelt[col_match] = np.nan
        gdelt[col_count] = np.nan
        return gdelt

    # Work only on rows where iso3 is resolved; unresolved stay NaN.
    resolved_mask = gdelt["iso3"].notna()
    gdelt_r = gdelt.loc[resolved_mask, ["iso3", "event_date"]].copy()

    # Accumulate ACLED counts across all date offsets using merges.
    # For each offset d we shift ACLED dates by +d and join on (iso3, shifted_date).
    totals = pd.Series(0, index=gdelt_r.index, dtype=np.int64)

    acled_base = acled_active[["iso3", "date", "acled_n_events"]].copy()
    for d in range(-window_days, window_days + 1):
        shifted = acled_base.copy()
        shifted["date"] = shifted["date"] + pd.Timedelta(days=d)
        merged = gdelt_r.merge(
            shifted.rename(columns={"date": "event_date"}),
            on=["iso3", "event_date"],
            how="left",
        )["acled_n_events"].fillna(0).astype(np.int64)
        merged.index = gdelt_r.index
        totals += merged

    # Write results back; unresolved rows remain NaN.
    gdelt[col_count] = np.nan
    gdelt.loc[resolved_mask, col_count] = totals
    gdelt[col_match] = np.nan
    gdelt.loc[resolved_mask, col_match] = totals > 0
    return gdelt


def match_against_mmad(
    gdelt: pd.DataFrame,
    mmad_lookup: dict[tuple[str, object], int],
) -> pd.DataFrame:
    """
    For each GDELT event with a resolved country, check whether MMAD recorded
    any protest in the same country-month.

    New columns added to gdelt:
        mmad_match         : True / False / NaN (NaN = country unresolved)
        mmad_protest_count : MMAD protest count for that country-month (0 if no match)
    """
    gdelt = gdelt.copy()
    resolved_mask = gdelt["iso3"].notna()
    gdelt_r = gdelt.loc[resolved_mask, ["iso3", "year_month"]]

    # Vectorised lookup via map
    keys = list(zip(gdelt_r["iso3"], gdelt_r["year_month"]))
    counts_r = pd.array(
        [mmad_lookup.get(k, 0) for k in keys], dtype=np.int64
    )

    gdelt["mmad_protest_count"] = np.nan
    gdelt["mmad_match"]         = np.nan
    gdelt.loc[resolved_mask, "mmad_protest_count"] = counts_r
    gdelt.loc[resolved_mask, "mmad_match"]         = counts_r > 0
    return gdelt


# ---------------------------------------------------------------------------
# ACLED-side recall  (what fraction of ACLED events does GDELT capture?)
# ---------------------------------------------------------------------------

def compute_acled_recall(
    gdelt: pd.DataFrame,
    acled_active: pd.DataFrame,
    window_days: int = 0,
) -> dict:
    """
    For each ACLED protest country-day, check whether GDELT recorded at least
    one event in the same country within ±window_days.

    This is the complement of GDELT→ACLED precision: it answers
    "of the events ACLED knows about, how many did GDELT pick up?"

    The comparison is scoped to the GDELT date range and the countries present
    in the GDELT data to make the denominator fair (ACLED covers countries and
    periods where GDELT may not have been run).

    Returns a dict with overall recall and breakdowns by year and country.
    """
    if acled_active.empty or gdelt.empty:
        return {"n_acled": 0, "n_covered": 0, "recall": None, "by_year": {}, "by_country": {}}

    # Scope ACLED to GDELT's date range and resolved countries
    gdelt_r = gdelt[gdelt["iso3"].notna()]
    if gdelt_r.empty:
        return {"n_acled": 0, "n_covered": 0, "recall": None, "by_year": {}, "by_country": {}}

    date_min = gdelt_r["event_date"].min()
    date_max = gdelt_r["event_date"].max()
    gdelt_countries = set(gdelt_r["iso3"].unique())

    acled_scoped = acled_active[
        acled_active["iso3"].isin(gdelt_countries)
        & (acled_active["date"] >= date_min)
        & (acled_active["date"] <= date_max)
    ].copy()

    if acled_scoped.empty:
        return {"n_acled": 0, "n_covered": 0, "recall": None, "by_year": {}, "by_country": {}}

    # Build a set of GDELT (iso3, date) pairs for fast lookup
    gdelt_pairs = set(zip(gdelt_r["iso3"], gdelt_r["event_date"]))

    # For each ACLED country-day check if any GDELT event falls within ±window_days
    def _covered(iso3: str, date: pd.Timestamp) -> bool:
        return any(
            (iso3, date + pd.Timedelta(days=d)) in gdelt_pairs
            for d in range(-window_days, window_days + 1)
        )

    # Vectorised over offsets: for each offset, mark covered rows via merge
    covered = pd.Series(False, index=acled_scoped.index)
    gdelt_index = gdelt_r.groupby(["iso3", "event_date"]).size().reset_index()[["iso3", "event_date"]]
    for d in range(-window_days, window_days + 1):
        shifted = gdelt_index.copy()
        shifted["event_date"] = shifted["event_date"] + pd.Timedelta(days=d)
        merged = acled_scoped[["iso3", "date"]].merge(
            shifted.rename(columns={"event_date": "date"}),
            on=["iso3", "date"],
            how="left",
            indicator=True,
        )["_merge"] == "both"
        merged.index = acled_scoped.index
        covered |= merged

    acled_scoped = acled_scoped.copy()
    acled_scoped["covered"] = covered
    acled_scoped["year"] = acled_scoped["date"].dt.year

    n_total   = len(acled_scoped)
    n_covered = int(covered.sum())

    by_year: dict = {}
    for year, grp in acled_scoped.groupby("year"):
        by_year[int(year)] = {
            "n_acled":   int(len(grp)),
            "n_covered": int(grp["covered"].sum()),
            "recall":    round(grp["covered"].mean(), 4),
        }

    by_country: dict = {}
    for iso3, grp in acled_scoped.groupby("iso3"):
        by_country[str(iso3)] = {
            "n_acled":   int(len(grp)),
            "n_covered": int(grp["covered"].sum()),
            "recall":    round(grp["covered"].mean(), 4),
        }

    return {
        "window_days": window_days,
        "n_acled":     n_total,
        "n_covered":   n_covered,
        "recall":      round(n_covered / n_total, 4) if n_total > 0 else None,
        "by_year":     by_year,
        "by_country":  by_country,
    }


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _acled_match_cols(gdelt: pd.DataFrame) -> list[str]:
    """Return all acled_match_* columns present, exact first."""
    cols = [c for c in gdelt.columns if c.startswith("acled_match_")]
    # put exact first
    exact = [c for c in cols if c.endswith("_exact")]
    rest  = [c for c in cols if not c.endswith("_exact")]
    return exact + rest


def compute_summary(gdelt: pd.DataFrame) -> dict:
    """
    Compute match-rate statistics over the verified GDELT event table.

    Rates are computed only over events where the country was resolved
    (unresolved events cannot be verified either way).
    """
    total      = len(gdelt)
    resolved   = gdelt["iso3"].notna()
    n_resolved = resolved.sum()

    def _rate(col: str) -> dict:
        base = gdelt[resolved & gdelt[col].notna()]
        if len(base) == 0:
            return {"n": 0, "matched": 0, "rate": None}
        matched = base[col].astype(bool).sum()
        return {"n": int(len(base)), "matched": int(matched),
                "rate": round(matched / len(base), 4)}

    acled_cols = _acled_match_cols(gdelt)
    has_mmad   = "mmad_match" in gdelt.columns

    # Per disruption_type breakdown
    by_type: dict = {}
    for dtype, grp in gdelt[resolved].groupby("disruption_type"):
        entry: dict = {"n": int(len(grp))}
        for col in acled_cols:
            label = col.replace("acled_match_", "acled_match_rate_")
            entry[label] = (
                round(grp[col].dropna().astype(bool).mean(), 4)
                if grp[col].notna().any() else None
            )
        if has_mmad:
            entry["mmad_match_rate"] = (
                round(grp["mmad_match"].dropna().astype(bool).mean(), 4)
                if grp["mmad_match"].notna().any() else None
            )
        by_type[dtype] = entry

    # Per year breakdown
    by_year: dict = {}
    gdelt_r = gdelt[resolved].copy()
    gdelt_r["year"] = gdelt_r["event_date"].dt.year
    for year, grp in gdelt_r.groupby("year"):
        entry = {"n": int(len(grp))}
        for col in acled_cols:
            label = col.replace("acled_match_", "acled_match_rate_")
            entry[label] = (
                round(grp[col].dropna().astype(bool).mean(), 4)
                if grp[col].notna().any() else None
            )
        if has_mmad:
            entry["mmad_match_rate"] = (
                round(grp["mmad_match"].dropna().astype(bool).mean(), 4)
                if grp["mmad_match"].notna().any() else None
            )
        by_year[int(year)] = entry

    overall: dict = {}
    for col in acled_cols:
        overall[col] = _rate(col)
    if has_mmad:
        overall["mmad_match"] = _rate("mmad_match")

    return {
        "total_gdelt_events":     int(total),
        "country_resolved":       int(n_resolved),
        "country_unresolved":     int(total - n_resolved),
        "country_resolution_pct": round(n_resolved / total * 100, 1) if total > 0 else None,
        "overall":                overall,
        "by_disruption_type":     by_type,
        "by_year":                by_year,
        # Populated later in main() once acled_active is available
        "acled_recall":           None,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Saved: %s", path)


def _col_label(col: str) -> str:
    """Human-readable label for a match column."""
    if col == "mmad_match":
        return "MMAD (month)"
    # acled_match_exact  →  "ACLED exact day"
    # acled_match_±7d    →  "ACLED ±7d"
    suffix = col.replace("acled_match_", "")
    return f"ACLED {suffix.replace('exact', 'exact day')}"


# Colour palette: exact day gets a darker shade than windowed
_PALETTE = [
    "#1D3557",   # dark blue  (ACLED exact)
    "#457B9D",   # mid blue   (ACLED ±Nd)
    "#A8DADC",   # light blue (extra ACLED windows if any)
    "#2A9D8F",   # teal       (MMAD)
]


def _all_ref_cols(gdelt: pd.DataFrame) -> list[str]:
    """Ordered list of all reference match columns: exact first, then windowed, then MMAD."""
    return _acled_match_cols(gdelt) + (["mmad_match"] if "mmad_match" in gdelt.columns else [])


def fig_match_rate_by_type(gdelt: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    resolved = gdelt[gdelt["iso3"].notna()]
    types    = sorted(resolved["disruption_type"].unique())
    refs     = _all_ref_cols(resolved)
    if not refs:
        return

    x = np.arange(len(types))
    w = 0.8 / len(refs)
    fig, ax = plt.subplots(figsize=(max(7, len(types) * 2), 4))

    for i, col in enumerate(refs):
        rates = [
            resolved[resolved["disruption_type"] == t][col].dropna().astype(bool).mean()
            for t in types
        ]
        offset = (i - len(refs) / 2 + 0.5) * w
        bars = ax.bar(x + offset, rates, w, label=_col_label(col),
                      color=_PALETTE[i % len(_PALETTE)], alpha=0.85)
        for bar, r in zip(bars, rates):
            if not np.isnan(r):
                ax.text(bar.get_x() + bar.get_width() / 2, r + 0.01,
                        f"{r:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(types)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction of events matched")
    ax.set_title("GDELT event match rate by disruption type")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_match_rate_by_year(gdelt: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt

    resolved = gdelt[gdelt["iso3"].notna()].copy()
    resolved["year"] = resolved["event_date"].dt.year
    refs = _all_ref_cols(resolved)
    if not refs:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, col in enumerate(refs):
        rates = (
            resolved.groupby("year")[col]
            .apply(lambda s: s.dropna().astype(bool).mean())
        )
        ls = "-" if "exact" in col else ("--" if col.startswith("acled") else ":")
        ax.plot(rates.index, rates.values, marker="o", label=_col_label(col),
                color=_PALETTE[i % len(_PALETTE)], lw=1.8, linestyle=ls)

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction of events matched")
    ax.set_title("GDELT event match rate by year")
    ax.axhline(0.5, color="grey", lw=0.8, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_match_rate_by_country(gdelt: pd.DataFrame, top_n: int, out: Path) -> None:
    import matplotlib.pyplot as plt

    resolved = gdelt[gdelt["iso3"].notna()]
    counts   = resolved["iso3"].value_counts()
    top_countries = counts[counts >= 3].head(top_n).index
    sub  = resolved[resolved["iso3"].isin(top_countries)]
    refs = _all_ref_cols(sub)
    if not refs:
        return

    # Sort by exact-day ACLED rate if available, else first col
    sort_col = refs[0]
    country_rates = (
        sub.groupby("iso3")[sort_col]
        .apply(lambda s: s.dropna().astype(bool).mean())
        .sort_values()
    )
    ordered = country_rates.index.tolist()

    y = np.arange(len(ordered))
    w = 0.8 / len(refs)
    fig, ax = plt.subplots(figsize=(10, max(4, len(ordered) * 0.3)))

    for i, col in enumerate(refs):
        rates = [
            sub[sub["iso3"] == c][col].dropna().astype(bool).mean()
            for c in ordered
        ]
        offset = (i - len(refs) / 2 + 0.5) * w
        ax.barh(y + offset, rates, w, label=_col_label(col),
                color=_PALETTE[i % len(_PALETTE)], alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(ordered, fontsize=8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Fraction of events matched")
    ax.set_title(f"GDELT event match rate by country  (min 3 events, top {top_n})")
    ax.axvline(0.5, color="grey", lw=0.8, linestyle=":")
    ax.legend()
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_acled_window_sensitivity(
    gdelt: pd.DataFrame,
    acled_active: pd.DataFrame,
    windows: list[int],
    out: Path,
) -> None:
    """
    Show how the ACLED match rate changes as the time window ±N days grows.
    Useful for understanding how tight the temporal matching needs to be.
    """
    import matplotlib.pyplot as plt

    resolved = gdelt[gdelt["iso3"].notna()].copy()
    if resolved.empty or acled_active.empty:
        return

    # For each window size, count matched events using vectorised offset merges.
    acled_base = acled_active[["iso3", "date", "acled_n_events"]].copy()
    gdelt_r = resolved[["iso3", "event_date"]].copy()

    rates = []
    for w in windows:
        totals = pd.Series(0, index=gdelt_r.index, dtype=np.int64)
        for d in range(-w, w + 1):
            shifted = acled_base.copy()
            shifted["date"] = shifted["date"] + pd.Timedelta(days=d)
            merged = gdelt_r.merge(
                shifted.rename(columns={"date": "event_date"}),
                on=["iso3", "event_date"],
                how="left",
            )["acled_n_events"].fillna(0).astype(np.int64)
            merged.index = gdelt_r.index
            totals += merged
        rates.append((totals > 0).mean())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(windows, rates, marker="o", color="#457B9D", lw=1.8)
    ax.set_xlabel("±window (days)")
    ax.set_ylabel("ACLED match rate")
    ax.set_title("GDELT–ACLED match rate vs temporal window size")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="grey", lw=0.8, linestyle=":")
    for x, y_val in zip(windows, rates):
        ax.annotate(f"{y_val:.0%}", (x, y_val), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8)
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


def fig_acled_recall_by_country(
    recall: dict | None,
    top_n: int,
    out: Path,
) -> None:
    """
    Horizontal bar chart: ACLED recall by country (what fraction of ACLED
    protest country-days did GDELT capture?).  Countries sorted by recall.
    Only countries with >= 3 ACLED country-days shown; top_n displayed.
    """
    import matplotlib.pyplot as plt

    if not recall or recall.get("recall") is None:
        return

    by_country = recall["by_country"]
    rows = [
        (iso3, v["recall"], v["n_acled"])
        for iso3, v in by_country.items()
        if v["n_acled"] >= 3
    ]
    if not rows:
        return

    rows.sort(key=lambda x: x[1])
    rows = rows[-top_n:]  # take top_n highest-recall countries

    countries = [r[0] for r in rows]
    recalls   = [r[1] for r in rows]
    counts    = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(10, max(4, len(countries) * 0.3)))
    bars = ax.barh(countries, recalls, color="#1D3557", alpha=0.85)

    # Annotate with ACLED event count
    for bar, n in zip(bars, counts):
        ax.text(
            min(bar.get_width() + 0.01, 1.02), bar.get_y() + bar.get_height() / 2,
            f"n={n}", va="center", fontsize=7, color="#555",
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("ACLED recall  (fraction of ACLED country-days covered by GDELT)")
    ax.set_title(
        f"ACLED recall by country  (±{recall['window_days']}d window, "
        f"min 3 ACLED events, top {top_n})\n"
        f"Overall recall: {recall['recall']:.1%}  "
        f"({recall['n_covered']} / {recall['n_acled']} ACLED country-days)"
    )
    ax.axvline(0.5, color="grey", lw=0.8, linestyle=":")
    fig.tight_layout()
    _save(fig, out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    gdelt_path:   Path = DEFAULT_GDELT_PATH,
    acled_panel:  Path = DEFAULT_ACLED_PANEL,
    acled_raw:    Path = DEFAULT_ACLED_RAW,
    mmad_panel:   Path = DEFAULT_MMAD_PANEL,
    out_dir:      Path = DEFAULT_OUT_DIR,
    window_days:  int  = DEFAULT_ACLED_WINDOW,
) -> pd.DataFrame:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load GDELT — keep all events
    # ------------------------------------------------------------------
    # Resolve fallback: if default deduped path is missing, try raw file.
    if not gdelt_path.exists() and gdelt_path == _DEDUPED_PATH:
        if _RAW_FALLBACK_PATH.exists():
            gdelt_path = _RAW_FALLBACK_PATH
        else:
            log.error("GDELT file not found at either:")
            log.error("  %s", _DEDUPED_PATH)
            log.error("  %s", _RAW_FALLBACK_PATH)
            log.error("Run deduplicate_gdelt_social.py (or the GDELT extraction pipeline) first.")
            sys.exit(1)
    elif not gdelt_path.exists():
        log.error("GDELT file not found: %s", gdelt_path)
        sys.exit(1)

    gdelt = load_gdelt_social(gdelt_path)
    if gdelt.empty:
        log.error("No GDELT social events with confidence >= %.1f found.", CONFIDENCE_THRESHOLD)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Load reference lookups
    # ------------------------------------------------------------------
    acled_active: Optional[pd.DataFrame] = None
    try:
        acled_active = build_acled_lookup(acled_panel, acled_raw)
    except FileNotFoundError as exc:
        log.warning("ACLED unavailable — skipping ACLED matching.\n  %s", exc)

    mmad_lookup: Optional[dict] = None
    if mmad_panel.exists():
        mmad_lookup = build_mmad_lookup(mmad_panel)
    else:
        log.warning("MMAD panel not found at %s — skipping.", mmad_panel)

    # ------------------------------------------------------------------
    # 3. Match each GDELT event individually
    # ------------------------------------------------------------------
    if acled_active is not None:
        log.info("Matching against ACLED (exact day)...")
        gdelt = match_against_acled(gdelt, acled_active, window_days=0)
        if window_days > 0:
            log.info("Matching against ACLED (±%d days)...", window_days)
            gdelt = match_against_acled(gdelt, acled_active, window_days=window_days)

    if mmad_lookup is not None:
        log.info("Matching against MMAD (same country-month)...")
        gdelt = match_against_mmad(gdelt, mmad_lookup)

    # ------------------------------------------------------------------
    # 4. Summary statistics
    # ------------------------------------------------------------------
    summary = compute_summary(gdelt)

    # ACLED-side recall: what fraction of ACLED events does GDELT capture?
    if acled_active is not None:
        log.info("Computing ACLED recall (exact day)...")
        recall_exact = compute_acled_recall(gdelt, acled_active, window_days=0)
        summary["acled_recall"] = recall_exact
        if window_days > 0:
            log.info("Computing ACLED recall (±%d days)...", window_days)
            recall_windowed = compute_acled_recall(gdelt, acled_active, window_days=window_days)
            summary["acled_recall_windowed"] = recall_windowed

    log.info("=" * 60)
    log.info("MATCH SUMMARY  (country-resolved events only)")
    log.info("  Total GDELT events  : %d", summary["total_gdelt_events"])
    log.info("  Country resolved    : %d / %d  (%.1f%%)",
             summary["country_resolved"], summary["total_gdelt_events"],
             summary["country_resolution_pct"] or 0)
    log.info("  --- GDELT precision (does ACLED confirm this event?) ---")
    for col, stats in summary["overall"].items():
        if stats and stats["n"] > 0:
            log.info("  %-28s : %d / %d  (%.1f%%)",
                     _col_label(col), stats["matched"], stats["n"],
                     (stats["rate"] or 0) * 100)
    log.info("  --- ACLED recall (does GDELT cover this ACLED event?) ---")
    for key in ("acled_recall", "acled_recall_windowed"):
        rec = summary.get(key)
        if rec and rec.get("recall") is not None:
            log.info("  ±%-3dd window : %d / %d ACLED country-days  (%.1f%%)",
                     rec["window_days"], rec["n_covered"], rec["n_acled"],
                     rec["recall"] * 100)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 5. Write outputs
    # ------------------------------------------------------------------
    out_parquet = out_dir / "gdelt_verified.parquet"
    out_csv     = out_dir / "gdelt_verified.csv"
    gdelt.to_parquet(out_parquet, index=False)
    gdelt.to_csv(out_csv, index=False)
    log.info("Verified events written: %s", out_parquet)

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Summary written: %s", summary_path)

    # ------------------------------------------------------------------
    # 6. Figures
    # ------------------------------------------------------------------
    fig_match_rate_by_type(gdelt, out=fig_dir / "fig1_match_rate_by_type.png")
    fig_match_rate_by_year(gdelt, out=fig_dir / "fig2_match_rate_by_year.png")
    fig_match_rate_by_country(gdelt, top_n=40, out=fig_dir / "fig3_match_rate_by_country.png")
    if acled_active is not None:
        fig_acled_window_sensitivity(
            gdelt, acled_active,
            windows=[0, 1, 3, 7, 14, 30],
            out=fig_dir / "fig4_acled_window_sensitivity.png",
        )
        fig_acled_recall_by_country(
            summary["acled_recall"],
            top_n=40,
            out=fig_dir / "fig5_acled_recall_by_country.png",
        )

    return gdelt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tag each GDELT social event with ACLED and MMAD match status"
    )
    parser.add_argument("--gdelt",  type=Path, default=DEFAULT_GDELT_PATH)
    parser.add_argument("--acled",  type=Path, default=DEFAULT_ACLED_PANEL)
    parser.add_argument("--mmad",   type=Path, default=DEFAULT_MMAD_PANEL)
    parser.add_argument("--out",    type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--window", type=int,  default=DEFAULT_ACLED_WINDOW,
                        help="±N days for windowed ACLED match (default 0 = exact day only; "
                             "set e.g. --window 7 to add a ±7d column alongside exact)")
    args = parser.parse_args()
    main(
        gdelt_path=args.gdelt,
        acled_panel=args.acled,
        mmad_panel=args.mmad,
        out_dir=args.out,
        window_days=args.window,
    )
