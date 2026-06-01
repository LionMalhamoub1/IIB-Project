from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
from dateutil import parser as _du

logger = logging.getLogger(__name__)

# ── Word-number tables ────────────────────────────────────────────────────────

_CARDINALS: Dict[str, int] = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}

_ORDINALS: Dict[str, int] = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
    "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20,
    "twenty-first": 21, "twenty-second": 22, "twenty-third": 23,
    "twenty-fourth": 24, "twenty-fifth": 25, "twenty-sixth": 26,
    "twenty-seventh": 27, "twenty-eighth": 28, "twenty-ninth": 29,
    "thirtieth": 30,
}

_UNIT_DAYS: Dict[str, int] = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
    "year": 365, "years": 365,
}

# Patterns built once at import time.
_CARD_PAT = "|".join(sorted(_CARDINALS, key=len, reverse=True))
_ORD_PAT  = "|".join(sorted(_ORDINALS,  key=len, reverse=True))
_UNIT_PAT = r"day|days|week|weeks|month|months|year|years"

# Ordinal suffix: "1st", "2nd", "3rd", "10th", …
_ORD_SUFFIX = r"\d+(?:st|nd|rd|th)"
# Anything that looks like an ordinal (word or digit-suffix)
_ANY_ORD   = rf"(?:{_ORD_SUFFIX}|{_ORD_PAT})"
# Anything that looks like a count (digit or cardinal word)
_ANY_NUM   = rf"(?:\d+|{_CARD_PAT})"

# Relative-date markers that make event_start_reference unreliable
_RELATIVE_RE = re.compile(
    r"\b(?:last|this|next|since|ago|yesterday|today|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"recently|earlier|previous|few days)\b",
    re.I,
)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _str_to_cardinal(s: str) -> Optional[int]:
    s = s.strip().lower()
    if s.isdigit():
        return int(s)
    return _CARDINALS.get(s)


def _str_to_ordinal(s: str) -> Optional[int]:
    """Parse ordinal word ('third') or suffix ('10th') to int."""
    s = s.strip().lower()
    m = re.fullmatch(r"(\d+)(?:st|nd|rd|th)", s)
    if m:
        return int(m.group(1))
    return _ORDINALS.get(s)


def _any_to_int(s: str) -> Optional[int]:
    return _str_to_ordinal(s) or _str_to_cardinal(s)


# ── Public parsers ────────────────────────────────────────────────────────────

# "14 days", "two weeks", "a month-long strike", "3-week walkout"
_DUR_RE = re.compile(
    rf"\b(?P<qty>{_ANY_NUM})\s*-?\s*(?P<unit>{_UNIT_PAT})(?:\s*-\s*long)?\b",
    re.I,
)


def parse_explicit_duration(text: str) -> Optional[int]:
    """Parse a free-text duration phrase into a number of days.

    Examples
    --------
    "14 days"            → 14
    "two weeks"          → 14
    "a week-long strike" → 7
    "three-month period" → 90
    "2-day walkout"      → 2

    Returns ``None`` if no parseable duration is found.
    """
    if not text:
        return None
    for m in _DUR_RE.finditer(text.lower()):
        qty = _any_to_int(m.group("qty"))
        if qty and qty > 0:
            return qty * _UNIT_DAYS.get(m.group("unit").lower(), 1)
    return None


# Patterns for "day N", "Nth day", "entered its Nth day", "Nth week"
_DAY_NUM_RE   = re.compile(rf"\bday\s+({_ANY_ORD}|{_ANY_NUM})\b", re.I)
_NUM_DAY_RE   = re.compile(rf"\b({_ANY_ORD})\s+day\b", re.I)
_ENTERED_RE   = re.compile(
    rf"\b(?:entered|reached|into|now\s+in|in)\s+(?:its\s+)?({_ANY_ORD})\s+day\b",
    re.I,
)
_WEEK_ORD_RE  = re.compile(rf"\b({_ANY_ORD})\s+week\b", re.I)


def parse_reported_day_number(text: str) -> Optional[int]:
    """Parse the day-count from a phrase describing an ongoing event.

    Examples
    --------
    "third day of protests"  → 3
    "entered its 10th day"   → 10
    "day 5"                  → 5
    "on the 8th day"         → 8
    "in its second week"     → 14

    Returns ``None`` if no match.
    """
    if not text:
        return None
    t = text.lower()

    m = _ENTERED_RE.search(t)
    if m:
        v = _any_to_int(m.group(1))
        if v:
            return v

    m = _DAY_NUM_RE.search(t)
    if m:
        v = _any_to_int(m.group(1))
        if v:
            return v

    m = _NUM_DAY_RE.search(t)
    if m:
        v = _any_to_int(m.group(1))
        if v:
            return v

    m = _WEEK_ORD_RE.search(t)
    if m:
        v = _any_to_int(m.group(1))
        if v:
            return v * 7

    return None


def parse_event_start_reference(
    text: str,
    pub_date: Optional[date] = None,
    resolve_relative: bool = False,
) -> Optional[date]:
    """Parse an absolute date from ``event_start_reference``.

    Relative phrases ("since last Monday", "last week") are skipped by
    default.  Set ``resolve_relative=True`` to attempt resolution using
    ``pub_date``.

    Returns a :class:`datetime.date` or ``None``.
    """
    if not text or not text.strip():
        return None

    if _RELATIVE_RE.search(text) and not resolve_relative:
        return None

    default_dt = None
    if pub_date is not None:
        import datetime
        default_dt = datetime.datetime(pub_date.year, pub_date.month, pub_date.day)

    try:
        parsed = _du.parse(text, default=default_dt, fuzzy=True)
        today = date.today()
        if 1990 <= parsed.year <= today.year + 1:
            return parsed.date()
    except Exception:
        pass

    return None


# ── Event JSON schema & expansion ─────────────────────────────────────────────

_EMPTY_EVENT: Dict[str, Any] = {
    "country": "",
    "region_or_state": "",
    "city": "",
    "specific_location": "",
    "protest_type": "",
    "protesting_groups": [],
    "organizations_or_companies": [],
    "target_of_protest": "",
    "issue": "",
    "sector": "",
    "estimated_participants": "",
    "event_start_reference": "",
    "reported_day_number": "",
    "explicit_duration": "",
    "continuation_indicator": None,
    "resolution_indicator": None,
}

_STR_FIELDS = (
    "country", "region_or_state", "city", "specific_location",
    "protest_type", "target_of_protest", "issue", "sector",
    "event_start_reference", "reported_day_number", "explicit_duration",
)
_LIST_FIELDS = ("protesting_groups", "organizations_or_companies")


def parse_event_json(raw: Any) -> Dict[str, Any]:
    """Safely parse a raw event_json value into a normalised dict.

    Handles str (JSON), dict, and missing/null values.
    """
    if isinstance(raw, dict):
        data: Dict[str, Any] = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("event_json parse failed: %r", raw[:120])
            return dict(_EMPTY_EVENT)
    else:
        return dict(_EMPTY_EVENT)

    out = dict(_EMPTY_EVENT)
    for k in _EMPTY_EVENT:
        v = data.get(k)
        if v is not None:
            out[k] = v
    return out


def _norm_str(s: Any) -> str:
    return str(s).strip().lower() if s else ""


def _norm_list(lst: Any) -> List[str]:
    if not isinstance(lst, list):
        return []
    return [s.strip().lower() for s in lst if isinstance(s, str) and s.strip()]


def expand_event_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Parse ``event_json`` and append event fields as new columns.

    Existing columns with the same names are overwritten.  All string fields
    are lower-cased and stripped; list fields are normalised to lowercase lists.
    """
    records = df["event_json"].map(parse_event_json)
    event_df = pd.DataFrame(list(records), index=df.index)

    for col in _STR_FIELDS:
        event_df[col] = event_df[col].map(_norm_str)

    for col in _LIST_FIELDS:
        event_df[col] = event_df[col].map(_norm_list)

    overlap = [c for c in event_df.columns if c in df.columns]
    base = df.drop(columns=overlap)
    return pd.concat([base, event_df], axis=1)
