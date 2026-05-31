from __future__ import annotations

from typing import Dict, Optional, Set

import pandas as pd


DEFAULT_SOCIAL_EVENT_TYPES: Set[str] = {
    "Protests",
    "Riots",
    "Violence against civilians",
    "Explosions/Remote violence",
    "Strategic developments",
}


def filter_social_disruption(events: pd.DataFrame, event_types: Optional[Set[str]] = None) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    if "event_type" not in events.columns:
        return events.copy()

    keep = event_types or DEFAULT_SOCIAL_EVENT_TYPES
    return events[events["event_type"].isin(keep)].copy()


def add_month(events: pd.DataFrame, date_col: str = "event_date") -> pd.DataFrame:
    df = events.copy()
    df["month"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return df


def country_month_panel(
    events: pd.DataFrame,
    *,
    use_iso3: bool = True,
    severity: str = "count_plus_fatalities",
) -> pd.DataFrame:
    if events.empty:
        cols = ["iso3", "month", "event_count", "fatalities", "severity_score"] if use_iso3 else \
               ["country", "month", "event_count", "fatalities", "severity_score"]
        return pd.DataFrame(columns=cols)

    df = add_month(events)

    geo = "iso3" if (use_iso3 and "iso3" in df.columns) else "country"
    if geo not in df.columns:
        raise ValueError("Expected 'iso3' or 'country' column for aggregation.")

    df["event_count"] = 1
    df["fatalities"] = pd.to_numeric(df.get("fatalities", 0), errors="coerce").fillna(0)

    out = df.groupby([geo, "month"], as_index=False).agg(
        event_count=("event_count", "sum"),
        fatalities=("fatalities", "sum"),
    )

    if severity == "fatalities_only":
        out["severity_score"] = out["fatalities"]
    elif severity == "count_only":
        out["severity_score"] = out["event_count"]
    else:
        out["severity_score"] = out["event_count"] + out["fatalities"]

    return out.sort_values([geo, "month"])


def admin1_month_panel(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["iso3", "admin1", "month", "event_count", "fatalities"])

    df = add_month(events)

    iso = "iso3" if "iso3" in df.columns else "country"
    if "admin1" not in df.columns:
        raise ValueError("No 'admin1' column found. Include it in 'fields' when fetching ACLED events.")

    df["event_count"] = 1
    df["fatalities"] = pd.to_numeric(df.get("fatalities", 0), errors="coerce").fillna(0)

    out = df.groupby([iso, "admin1", "month"], as_index=False).agg(
        event_count=("event_count", "sum"),
        fatalities=("fatalities", "sum"),
    )
    return out.sort_values([iso, "admin1", "month"])
