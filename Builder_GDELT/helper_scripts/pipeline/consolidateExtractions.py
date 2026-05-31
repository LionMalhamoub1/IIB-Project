#!/usr/bin/env python3
"""
Event-level deduplication for extracted disruption events.
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd


# ------------------ CONFIG ------------------ #

EVENT_EVENT_TOLERANCE_DAYS = 1
EVENT_PUBLISH_TOLERANCE_DAYS = 2
PUBLISH_PUBLISH_TOLERANCE_DAYS = 3


# ------------------ LOAD ------------------ #

def load_extractions(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    if input_path.suffix.lower() == ".jsonl":
        records = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        df = pd.DataFrame(records)

    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)

    else:
        raise ValueError("Input must be .jsonl or .csv")

    df["event_date"] = pd.to_datetime(
        df.get("event_date"), errors="coerce", utc=True
    ).dt.tz_convert(None)

    df["publish_date"] = pd.to_datetime(
        df.get("publish_date"), errors="coerce", utc=True
    ).dt.tz_convert(None)

    df["disruption_type"] = (
        df.get("disruption_type", "")
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    return df


# ------------------ HELPERS ------------------ #

def location_tokens(location: str) -> set[str]:
    if not location:
        return set()

    location = location.lower()
    location = re.sub(r"\(.*?\)", "", location)
    location = re.sub(r"[^a-z\s]", " ", location)

    return {t for t in location.split() if len(t) > 2}


def choose_match_date(record: Dict[str, Any]) -> Optional[Tuple[pd.Timestamp, str]]:
    ed = record.get("event_date")
    if isinstance(ed, pd.Timestamp) and not pd.isna(ed):
        return ed, "event"

    pd_ = record.get("publish_date")
    if isinstance(pd_, pd.Timestamp) and not pd.isna(pd_):
        return pd_, "publish"

    return None


def dates_close_asymmetric(
    d1: pd.Timestamp, src1: str,
    d2: pd.Timestamp, src2: str
) -> bool:
    delta_days = abs((d1 - d2).days)

    if src1 == "event" and src2 == "event":
        tol = EVENT_EVENT_TOLERANCE_DAYS
    elif src1 != src2:
        tol = EVENT_PUBLISH_TOLERANCE_DAYS
    else:
        tol = PUBLISH_PUBLISH_TOLERANCE_DAYS

    return delta_days <= tol


# ------------------ MERGING ------------------ #

def merge_cluster(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}

    merged["disruption_type"] = cluster[0]["disruption_type"]

    event_dates = [r["event_date"] for r in cluster if pd.notna(r.get("event_date"))]
    merged["event_date"] = min(event_dates) if event_dates else None

    publish_dates = [r["publish_date"] for r in cluster if pd.notna(r.get("publish_date"))]
    merged["publish_date"] = min(publish_dates) if publish_dates else None

    merged["location_name"] = max(
        (r.get("location_name", "") for r in cluster if r.get("location_name")),
        key=len,
        default=""
    )

    merged["urls"] = sorted({r.get("url") for r in cluster if r.get("url")})
    merged["num_articles"] = len(cluster)

    merged["source_title"] = max(
        (r.get("source_title", "") for r in cluster if r.get("source_title")),
        key=len,
        default=""
    )

    merged["duration_hours"] = next(
        (r.get("duration_hours") for r in cluster if r.get("duration_hours") is not None),
        None
    )

    extras: Dict[str, List[Any]] = {}
    for r in cluster:
        if isinstance(r.get("extras"), dict):
            for k, v in r["extras"].items():
                if v is not None:
                    extras.setdefault(k, []).append(v)

    merged["extras"] = {
        k: vals[0] if len(vals) == 1 else vals
        for k, vals in extras.items()
    }

    merged["confidence"] = max(
        (r.get("confidence", 0.0) for r in cluster),
        default=0.0
    )

    # Prefer Nominatim-geocoded coords (higher accuracy) over raw GDELT actiongeo.
    # If any article in the cluster was geocoded from its location_name, use
    # the median of those; only fall back to all GDELT coords when none exist.
    nominatim_lats = [r["lat"] for r in cluster
                      if r.get("lat") is not None
                      and r.get("geo_source") == "nominatim_location_name"]
    nominatim_lons = [r["lon"] for r in cluster
                      if r.get("lon") is not None
                      and r.get("geo_source") == "nominatim_location_name"]

    if nominatim_lats and nominatim_lons:
        merged["lat"]          = round(float(pd.Series(nominatim_lats).median()), 5)
        merged["lon"]          = round(float(pd.Series(nominatim_lons).median()), 5)
        merged["geo_n_coords"] = len(nominatim_lats)
        merged["geo_source"]   = "nominatim_location_name"
    else:
        actiongeo_lats = [r["lat"] for r in cluster if r.get("lat") is not None]
        actiongeo_lons = [r["lon"] for r in cluster if r.get("lon") is not None]
        if actiongeo_lats and actiongeo_lons:
            merged["lat"]          = round(float(pd.Series(actiongeo_lats).median()), 5)
            merged["lon"]          = round(float(pd.Series(actiongeo_lons).median()), 5)
            merged["geo_n_coords"] = len(actiongeo_lats)
            merged["geo_source"]   = "actiongeo_median"
        else:
            merged["lat"]          = None
            merged["lon"]          = None
            merged["geo_n_coords"] = 0
            merged["geo_source"]   = None

    return merged


# ------------------ DEDUPLICATION ------------------ #

def dedupe_events(df: pd.DataFrame) -> pd.DataFrame:
    records = df.to_dict(orient="records")

    clusters: List[List[Dict[str, Any]]] = []

    for record in records:
        rec_tokens = location_tokens(record.get("location_name", ""))
        rec_match = choose_match_date(record)

        matched = False

        for cluster in clusters:
            rep = cluster[0]

            if record["disruption_type"] != rep["disruption_type"]:
                continue

            rep_match = choose_match_date(rep)
            if rec_match is None or rep_match is None:
                continue

            rec_date, rec_src = rec_match
            rep_date, rep_src = rep_match

            if not dates_close_asymmetric(rec_date, rec_src, rep_date, rep_src):
                continue

            if rec_tokens & location_tokens(rep.get("location_name", "")):
                cluster.append(record)
                matched = True
                break

        if not matched:
            clusters.append([record])

    merged_events = [merge_cluster(c) for c in clusters]
    return pd.DataFrame(merged_events)


# ------------------ SAVE ------------------ #

def save_outputs(df: pd.DataFrame, output_csv: Path, output_jsonl: Path):
    df.to_csv(output_csv, index=False)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = row.to_dict()
            for k in ("event_date", "publish_date"):
                if isinstance(record.get(k), pd.Timestamp):
                    record[k] = record[k].isoformat()
                elif record.get(k) is None or pd.isna(record.get(k)):
                    record[k] = None
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------ PUBLIC ENTRY POINT ------------------ #

def run_consolidation(input_path: Path) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="Degrees of freedom <= 0", category=RuntimeWarning)
        df_before = load_extractions(input_path)
        df_after  = dedupe_events(df_before)



    #no need to save here anymore as this is done within the pipeline runner stage

    #output_base = input_path.with_name(input_path.stem + "Consolidated")

    #output_csv = output_base.with_suffix(".csv")
    #output_jsonl = output_base.with_suffix(".jsonl")

    #save_outputs(df_after, output_csv, output_jsonl)

    #print(f"Saved {output_csv.name} and {output_jsonl.name}")

    return df_after