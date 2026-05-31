"""
Input/output utilities for the validation pipeline.

This file is responsible for:
- Loading extracted disruption events from CSV or JSONL files
- Writing validation outputs (summaries, matches, diagnostics) to disk

Flowchart role:
- Entry point for extracted-event data
- Exit point for validation results

This module performs no validation or matching logic.
"""

from pathlib import Path
from typing import List, Dict, Any
import csv
import json

from Builder_Matching.helper_scripts.models import ExtractedEvent


# ------------------ LOAD EXTRACTED EVENTS ------------------ #

def load_extracted_events(path: Path) -> List[ExtractedEvent]:
    """
    Load LLM-extracted events from a CSV or JSONL file and convert them
    into ExtractedEvent objects.
    """
    if not path.exists():
        raise FileNotFoundError(path)

    events: List[ExtractedEvent] = []

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                events.append(_row_to_extracted_event(row, i))

    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                events.append(_row_to_extracted_event(row, i))

    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return events


def _row_to_extracted_event(row: Dict[str, Any], idx: int) -> ExtractedEvent:
    """
    Convert a raw row dict into an ExtractedEvent.
    This function assumes minimal schema consistency.
    """
    return ExtractedEvent(
        event_id=row.get("event_id") or f"extracted_{idx}",
        disruption_type=(row.get("disruption_type") or "unknown").lower(),
        event_date_raw=row.get("event_date"),
        location_raw=row.get("location_name") or row.get("location"),
        title=row.get("source_title") or row.get("title"),
        text=row.get("text") or row.get("article_text"),
        published_at_raw=row.get("published_at") or row.get("publication_date"),
        url=row.get("url"),
        extras=row.get("extras") if isinstance(row.get("extras"), dict) else {},
    )


# ------------------ WRITE OUTPUT FILES ------------------ #

def write_json(path: Path, data: Any) -> None:
    """
    Write a Python object to disk as formatted JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Write a list of dictionaries to disk as a CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
