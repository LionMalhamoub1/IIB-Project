"""
Flood reference dataset loaders.

This file loads flood events from previously smoke-tested reference datasets:
- DFO
- GDACS
- EM-DAT
- ReliefWeb

It assumes reference data has already been downloaded and cached.
No API calls are made here.

Flowchart role:
- 'Reference Loading' stage on the reference-data stream

Two loading modes
-----------------
1. Individual loaders (load_dfo, load_gdacs, etc.) + load_all_flood_references
   Used by the existing validation pipeline.

2. load_combined_references
   Used by the enrichment pipeline. Reads the single unified JSONL produced
   by combine.py, which is the preferred approach for enrichment because all
   source fields are preserved in each record.
"""

from pathlib import Path
from typing import List
from datetime import datetime, date
import json

from database_validation.helper_scripts.models import RefEvent


# ------------------ HELPERS ------------------ #

def _parse_date(s) -> date | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s[:19]).date()
    except Exception:
        return None


# ------------------ DFO ------------------ #

def load_dfo(cache_path: Path) -> List[RefEvent]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    out: List[RefEvent] = []

    for row in data:
        out.append(
            RefEvent(
                ref_id=f"DFO_{row.get('id')}",
                dataset="DFO",
                ref_type="flood",
                date_start=_parse_date(row.get("start_date")),
                date_end=_parse_date(row.get("end_date")),
                location_name=row.get("other_country") or row.get("country"),
                country=row.get("country"),
                lat=row.get("lat"),
                lon=row.get("lon"),
                text=row.get("main_cause") or "",
                raw=row,
            )
        )
    return out


# ------------------ GDACS ------------------ #

def load_gdacs(cache_path: Path) -> List[RefEvent]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    out: List[RefEvent] = []

    for row in data:
        out.append(
            RefEvent(
                ref_id=f"GDACS_{row.get('id')}",
                dataset="GDACS",
                ref_type="flood",
                date_start=_parse_date(row.get("fromdate")),
                date_end=_parse_date(row.get("todate")),
                location_name=row.get("country"),
                country=row.get("country"),
                lat=row.get("lat"),
                lon=row.get("lon"),
                text=row.get("name"),
                raw=row,
            )
        )
    return out


# ------------------ EM-DAT ------------------ #

def load_emdat(cache_path: Path) -> List[RefEvent]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    out: List[RefEvent] = []

    for row in data:
        out.append(
            RefEvent(
                ref_id=f"EMDAT_{row.get('disaster_no')}",
                dataset="EM-DAT",
                ref_type="flood",
                date_start=_parse_date(row.get("start_date")),
                date_end=_parse_date(row.get("end_date")),
                location_name=row.get("location"),
                country=row.get("country"),
                lat=None,
                lon=None,
                text=row.get("event_name"),
                raw=row,
            )
        )
    return out


# ------------------ COPERNICUS EMS ------------------ #

def load_copernicus(cache_path: Path) -> List[RefEvent]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    out: List[RefEvent] = []

    for row in data:
        out.append(
            RefEvent(
                ref_id=f"Copernicus_{row.get('id')}",
                dataset="Copernicus",
                ref_type="flood",
                date_start=_parse_date(row.get("date_start")),
                date_end=None,
                location_name=row.get("country"),
                country=row.get("country"),
                lat=row.get("lat"),
                lon=row.get("lon"),
                text=row.get("event_name") or "",
                raw=row,
            )
        )
    return out


# ------------------ RELIEFWEB ------------------ #

def load_reliefweb(cache_path: Path) -> List[RefEvent]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    out: List[RefEvent] = []

    for row in data:
        out.append(
            RefEvent(
                ref_id=f"RELIEFWEB_{row.get('id')}",
                dataset="ReliefWeb",
                ref_type="flood",
                date_start=_parse_date(row.get("date")),
                date_end=None,
                location_name=row.get("country"),
                country=row.get("country"),
                lat=None,
                lon=None,
                text=row.get("name"),
                raw=row,
            )
        )
    return out


# ------------------ DESINVENTAR ------------------ #

def load_desinventar(cache_path: Path) -> List[RefEvent]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    out: List[RefEvent] = []

    for row in data:
        out.append(
            RefEvent(
                ref_id=f"Desinventar_{row.get('id')}",
                dataset="Desinventar",
                ref_type="flood",
                date_start=_parse_date(row.get("date_start")),
                date_end=None,
                location_name=row.get("location_name"),
                country=row.get("country"),
                lat=None,
                lon=None,
                text=row.get("main_cause") or "",
                raw=row,
            )
        )
    return out


# ------------------ HANZE ------------------ #

def load_hanze(cache_path: Path) -> List[RefEvent]:
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    out: List[RefEvent] = []

    for row in data:
        country = row.get("country")
        out.append(
            RefEvent(
                ref_id=f"HANZE_{row.get('id')}",
                dataset="HANZE",
                ref_type="flood",
                date_start=_parse_date(row.get("date_start")),
                date_end=_parse_date(row.get("date_end")),
                location_name=row.get("location_name"),
                country=country,
                lat=None,
                lon=None,
                text=row.get("main_cause") or "",
                raw=row,
            )
        )
    return out


# ------------------ AGGREGATOR ------------------ #

def load_all_flood_references(
    dfo_path: Path,
    gdacs_path: Path,
    emdat_path: Path,
    reliefweb_path: Path,
    copernicus_path: Path | None = None,
    desinventar_path: Path | None = None,
    hanze_path: Path | None = None,
) -> List[RefEvent]:
    """
    Load and concatenate all flood reference datasets.
    Used by the existing validation pipeline.
    """
    refs: List[RefEvent] = []
    refs.extend(load_dfo(dfo_path))
    refs.extend(load_gdacs(gdacs_path))
    refs.extend(load_emdat(emdat_path))
    refs.extend(load_reliefweb(reliefweb_path))
    if copernicus_path and copernicus_path.exists():
        refs.extend(load_copernicus(copernicus_path))
    if desinventar_path and desinventar_path.exists():
        refs.extend(load_desinventar(desinventar_path))
    if hanze_path and hanze_path.exists():
        refs.extend(load_hanze(hanze_path))
    return refs


# ------------------ COMBINED LOADER (enrichment pipeline) ------------------ #

def load_combined_references(combined_path: Path) -> List[RefEvent]:
    """
    Load the unified combined JSONL produced by combine.py.

    Each record already uses the unified schema, so all enrichment fields
    are preserved in the RefEvent.raw dict for downstream use.
    Used by the enrichment pipeline instead of load_all_flood_references.
    """
    out: List[RefEvent] = []

    with combined_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            source = row.get("source", "unknown")
            source_id = row.get("source_id", "")
            ref_id = f"{source}_{source_id}"

            out.append(
                RefEvent(
                    ref_id=ref_id,
                    dataset=source,
                    ref_type="flood",
                    date_start=_parse_date(row.get("date_start")),
                    date_end=_parse_date(row.get("date_end")),
                    location_name=row.get("location_name"),
                    country=row.get("country"),
                    lat=row.get("lat"),
                    lon=row.get("lon"),
                    text=row.get("event_name") or row.get("main_cause") or "",
                    raw=row,
                )
            )

    return out
