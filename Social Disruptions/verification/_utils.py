"""
_utils.py
=========
Shared utilities for the GDELT social event verification pipeline.

Centralising these here means a fix in the alias table or fuzzy matcher
propagates to deduplicate_gdelt_social.py, run_gdelt_graph_matching.py, and
run_gdelt_verification.py automatically.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Country name → ISO-3166-1 alpha-3
# ---------------------------------------------------------------------------

# Common aliases and sub-national entities that pycountry won't resolve alone.
# Keys must be lowercase.
COUNTRY_ALIAS: dict[str, str] = {
    "england": "GBR", "scotland": "GBR", "wales": "GBR",
    "northern ireland": "GBR", "britain": "GBR", "great britain": "GBR",
    "uk": "GBR", "united kingdom": "GBR",
    "usa": "USA", "us": "USA", "united states": "USA",
    "united states of america": "USA",
    "russia": "RUS", "iran": "IRN",
    "south korea": "KOR", "north korea": "PRK",
    "democratic republic of the congo": "COD",
    "democratic republic of congo": "COD",
    "dr congo": "COD", "drc": "COD", "dr. congo": "COD",
    "kinshasa": "COD",   # Kinshasa is only in DRC, not Republic of Congo
    "republic of the congo": "COG", "republic of congo": "COG",
    "congo-brazzaville": "COG", "brazzaville": "COG",
    "congo": "COD",   # bare "Congo" in context almost always means DRC
    "ivory coast": "CIV", "côte d'ivoire": "CIV", "cote d'ivoire": "CIV",
    "eswatini": "SWZ", "swaziland": "SWZ",
    "taiwan": "TWN", "hong kong": "HKG",
    "palestine": "PSE", "west bank": "PSE", "gaza": "PSE",
    "kosovo": "XKX",
    "czech republic": "CZE", "czechia": "CZE",
    "bolivia": "BOL", "venezuela": "VEN", "vietnam": "VNM", "laos": "LAO",
    "south africa": "ZAF",
    "north macedonia": "MKD", "macedonia": "MKD",
    "moldova": "MDA",
    "cape verde": "CPV", "cabo verde": "CPV",
    "trinidad and tobago": "TTO", "trinidad": "TTO",
    "micronesia": "FSM",
    "myanmar": "MMR", "burma": "MMR",
    "turkey": "TUR", "turkiye": "TUR",
    "brunei": "BRN",
    "tanzania": "TZA", "syria": "SYR",
    "timor-leste": "TLS", "east timor": "TLS",
    "gambia": "GMB", "the gambia": "GMB",
    "namibia": "NAM",
    "bahamas": "BHS", "the bahamas": "BHS",
    "comoros": "COM", "the comoros": "COM",
    # Sub-national → parent country
    "manitoba": "CAN", "ontario": "CAN", "quebec": "CAN",
    "alberta": "CAN", "british columbia": "CAN",
    "california": "USA", "texas": "USA", "new york": "USA", "florida": "USA",
    "kashmir": "IND", "tamil nadu": "IND",
}

_iso3_cache: dict[str, Optional[str]] = {}


def name_to_iso3(name: str) -> Optional[str]:
    """
    Map a country / territory name to ISO-3166-1 alpha-3.
    Returns None if the name cannot be resolved.
    Results are cached in-process.
    """
    if not name:
        return None
    key = name.strip().lower()
    if key in _iso3_cache:
        return _iso3_cache[key]

    # 1. Direct alias lookup
    result = COUNTRY_ALIAS.get(key)
    if result:
        _iso3_cache[key] = result
        return result

    # 2. pycountry — name, common_name, official_name
    try:
        import pycountry
        c = (
            pycountry.countries.get(name=name.strip())
            or pycountry.countries.get(common_name=name.strip())
            or pycountry.countries.get(official_name=name.strip())
        )
        if c:
            _iso3_cache[key] = c.alpha_3
            return c.alpha_3
    except ImportError:
        pass  # pycountry optional; alias table covers common cases

    _iso3_cache[key] = None
    return None


def extract_iso3(event: dict) -> Optional[str]:
    """
    Extract ISO-3 country code from a GDELT event dict.

    Handles two formats:
      Raw extraction : event['location'] = ["country", region, city, specific]
                       → read element 0 directly.
      Consolidated   : event['location_name'] = joined free-text string
                       → scan tokens right-to-left (country is last).
    """
    # Raw extraction format
    loc = event.get("location")
    if isinstance(loc, list) and len(loc) >= 1:
        country_str = (loc[0] or "").strip()
        if country_str:
            return name_to_iso3(country_str)

    # Consolidated / flattened format
    loc_name: str = (event.get("location_name") or "").strip()
    if loc_name:
        cleaned = re.sub(r"\s*\(.*?\)", "", loc_name)
        for token in reversed(re.split(r"[;,]", cleaned)):
            token = token.strip()
            if token:
                iso3 = name_to_iso3(token)
                if iso3:
                    return iso3
        return name_to_iso3(loc_name)

    return None


def extract_subloc(event: dict) -> str:
    """
    Return the most specific sub-national string available from a GDELT event.

    For raw format: tries city (location[2]) then region (location[1]).
    For consolidated format: takes the first (most specific) comma-separated
    token, skipping the country (last token).

    Always returns lowercase; returns "" if nothing is available.
    """
    loc = event.get("location")
    if isinstance(loc, list):
        city   = (loc[2] if len(loc) > 2 else "") or ""
        region = (loc[1] if len(loc) > 1 else "") or ""
        if city.strip():
            return city.strip().lower()
        if region.strip():
            return region.strip().lower()

    loc_name: str = (event.get("location_name") or "").strip()
    if loc_name:
        cleaned = re.sub(r"\s*\(.*?\)", "", loc_name)
        parts = [p.strip() for p in re.split(r"[;,]", cleaned) if p.strip()]
        # Format is [country, region, city, specific_location] — country is first.
        # Return the most specific sub-national level available.
        if len(parts) >= 3:
            return parts[2].lower()   # city level
        if len(parts) >= 2:
            return parts[1].lower()   # region / state level

    return ""


# ---------------------------------------------------------------------------
# Fuzzy string similarity
# ---------------------------------------------------------------------------

def fuzzy_similarity(a: str, b: str) -> float:
    """
    Normalised string similarity in [0, 1].

    Uses rapidfuzz.fuzz.token_set_ratio if available (significantly faster
    for short geographic strings), falls back to difflib.SequenceMatcher.
    token_set_ratio is chosen over simple ratio because it handles word-order
    variation and partial overlaps well (e.g. "Tamil Nadu" vs "Nadu, Tamil").
    """
    if not a or not b:
        return 0.0
    try:
        from rapidfuzz import fuzz
        return fuzz.token_set_ratio(a, b) / 100.0
    except ImportError:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, a, b).ratio()
