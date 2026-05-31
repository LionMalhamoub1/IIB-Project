"""
Combine cached flood reference datasets into a single unified file.

Reads the four individual cache files (DFO, EM-DAT, GDACS, ReliefWeb),
maps each record to a common schema, and writes one combined JSONL file.

Unified schema fields
---------------------
source               : str   — originating dataset (DFO / EM-DAT / GDACS / ReliefWeb / Copernicus / Desinventar / HANZE)
source_id            : str   — original ID within that dataset
glide_number         : str?  — GLIDE disaster identifier (DFO; EM-DAT via External IDs)
date_start           : str?  — ISO date
date_end             : str?  — ISO date
country              : str?  — country name
country_iso          : str?  — ISO-3 code (EM-DAT, ReliefWeb)
region               : str?  — sub-region (EM-DAT, Desinventar)
lat                  : float? — centroid latitude
lon                  : float? — centroid longitude
location_name        : str?  — free-text location description
area_km2             : float? — flooded / affected area (DFO; Copernicus: flooded ha ÷ 100)
dead                 : int?  — confirmed deaths (DFO, EM-DAT, Desinventar)
injured              : int?  — confirmed injuries (Desinventar)
displaced            : int?  — displaced persons (DFO, Desinventar: relocated + evacuated)
affected             : int?  — total directly affected (EM-DAT, GDACS, Desinventar)
indirectly_affected  : int?  — indirectly affected persons (Desinventar)
houses_destroyed     : int?  — housing units destroyed (Desinventar)
houses_damaged       : int?  — housing units damaged (Desinventar)
roads_km             : float? — road length damaged/affected in km (Desinventar: m÷1000; Copernicus)
damage_usd_thousands : float? — economic damage in thousands USD (EM-DAT, Desinventar)
damage_eur2020_thousands : float? — economic damage in thousands EUR 2020 (HANZE)
severity             : str?  — severity label (DFO: 1/2/3; GDACS: Green/Orange/Red)
main_cause           : str?  — cause description (DFO MainCause, EM-DAT Origin)
event_name           : str?  — official event name (EM-DAT, GDACS, ReliefWeb)
"""

import json
from pathlib import Path
from typing import Any

import pycountry


# ------------------ COUNTRY NORMALISATION ------------------ #

# pycountry.countries.lookup() handles most names but fails on informal/short forms.
# This fallback maps known problem names directly to ISO-3.
_COUNTRY_FALLBACK: dict[str, str] = {
    "tanzania":     "TZA",
    "iran":         "IRN",
    "bolivia":      "BOL",
    "venezuela":    "VEN",
    "syria":        "SYR",
    "south korea":  "KOR",
    "north korea":  "PRK",
    "russia":       "RUS",
    "vietnam":      "VNM",
    "laos":         "LAO",
    "moldova":      "MDA",
    "congo":        "COG",
    "dr congo":     "COD",
    "democratic republic of the congo": "COD",
    "democratic republic of congo":     "COD",
    "democratic republic congo":        "COD",
    "democratic  republic of the congo": "COD",  # DFO double-space typo
    "republic of congo":                "COG",
    "ivory coast":  "CIV",
    "cote d'iavoir": "CIV",   # DFO typo
    "cape verde":   "CPV",
    "micronesia":   "FSM",
    "palestine":    "PSE",
    "taiwan":       "TWN",
    "brunei":       "BRN",
    "czechia":      "CZE",
    "czech republic": "CZE",
    "uk":           "GBR",
    "united kingdom": "GBR",
    "northern ireland": "GBR",  # DFO uses this
    "unitd kingdom": "GBR",     # DFO typo
    "usa":          "USA",
    "united states": "USA",
    "turkey":       "TUR",
    # DFO typos
    "guatamala":    "GTM",
    "phillipines":  "PHL",
    "zimbawe":      "ZWE",
    "madascar":     "MDG",
}


def _to_iso3(name: str | None) -> str | None:
    """Convert a single country name to ISO-3, returning None on failure."""
    if not name:
        return None
    key = name.strip().lower()
    if key in _COUNTRY_FALLBACK:
        return _COUNTRY_FALLBACK[key]
    try:
        return pycountry.countries.lookup(name.strip()).alpha_3
    except LookupError:
        return None


def _to_all_iso3(name: str | None) -> list[str]:
    """
    Convert a country name (or comma-separated list of country names) to a
    list of ISO-3 codes.  GDACS often records multi-country events as a single
    comma-separated string (e.g. "Mozambique, South Africa, Zimbabwe").
    Each country is looked up independently; unresolved names are dropped.
    Returns a list of one or more ISO-3 codes, or an empty list.
    """
    if not name:
        return []
    parts = [p.strip() for p in name.split(",") if p.strip()]
    codes = []
    for part in parts:
        iso = _to_iso3(part)
        if iso and iso not in codes:
            codes.append(iso)
    return codes


# ------------------ PER-SOURCE MAPPERS ------------------ #

def _from_dfo(record: dict) -> dict:
    iso = _to_iso3(record.get("country"))
    return {
        "source":               "DFO",
        "source_id":            record.get("id"),
        "glide_number":         record.get("glide_number"),
        "date_start":           record.get("start_date"),
        "date_end":             record.get("end_date"),
        "country":              record.get("country"),
        "country_iso":          iso,
        "all_country_iso":      [iso] if iso else [],
        "region":               None,
        "lat":                  record.get("lat"),
        "lon":                  record.get("lon"),
        "location_name":        record.get("other_country") or record.get("country"),
        "area_km2":             record.get("area_km2"),
        "dead":                 record.get("dead"),
        "injured":              None,
        "displaced":            record.get("displaced"),
        "affected":             None,
        "indirectly_affected":  None,
        "houses_destroyed":     None,
        "houses_damaged":       None,
        "roads_km":             None,
        "damage_usd_thousands": None,
        "severity":             str(record["severity"]) if record.get("severity") is not None else None,
        "main_cause":           record.get("main_cause"),
        "event_name":           None,
    }


def _from_emdat(record: dict) -> dict:
    iso = record.get("country_iso")
    return {
        "source":               "EM-DAT",
        "source_id":            record.get("id"),
        "glide_number":         record.get("glide_number"),
        "date_start":           record.get("start_date"),
        "date_end":             record.get("end_date"),
        "country":              record.get("country"),
        "country_iso":          iso,
        "all_country_iso":      [iso] if iso else [],
        "region":               record.get("region"),
        "lat":                  record.get("lat"),
        "lon":                  record.get("lon"),
        "location_name":        record.get("location"),
        "area_km2":             None,
        "dead":                 record.get("dead"),
        "injured":              None,
        "displaced":            None,
        "affected":             record.get("affected"),
        "indirectly_affected":  None,
        "houses_destroyed":     None,
        "houses_damaged":       None,
        "roads_km":             None,
        "damage_usd_thousands": record.get("damage_usd_thousands"),
        "severity":             None,
        "main_cause":           record.get("main_cause"),
        "event_name":           record.get("event_name"),
    }


def _from_gdacs(record: dict) -> dict:
    all_iso = _to_all_iso3(record.get("country"))
    primary_iso = all_iso[0] if all_iso else None
    return {
        "source":               "GDACS",
        "source_id":            str(record.get("id", "")),
        "glide_number":         None,
        "date_start":           record.get("fromdate"),
        "date_end":             record.get("todate"),
        "country":              record.get("country"),
        "country_iso":          primary_iso,
        "all_country_iso":      all_iso,
        "region":               None,
        "lat":                  record.get("lat"),
        "lon":                  record.get("lon"),
        "location_name":        record.get("country"),
        "area_km2":             None,
        "dead":                 None,
        "injured":              None,
        "displaced":            None,
        "affected":             record.get("population"),
        "indirectly_affected":  None,
        "houses_destroyed":     None,
        "houses_damaged":       None,
        "roads_km":             None,
        "damage_usd_thousands": None,
        "severity":             record.get("alertlevel"),
        "main_cause":           None,
        "event_name":           record.get("name"),
    }


def _from_reliefweb(record: dict) -> dict:
    iso = record.get("country_iso")
    return {
        "source":               "ReliefWeb",
        "source_id":            str(record.get("id", "")),
        "glide_number":         record.get("glide") or None,
        "date_start":           record.get("date"),
        "date_end":             None,
        "country":              record.get("country"),
        "country_iso":          iso,
        "all_country_iso":      [iso] if iso else [],
        "region":               None,
        "lat":                  None,
        "lon":                  None,
        "location_name":        record.get("country"),
        "area_km2":             None,
        "dead":                 None,
        "injured":              None,
        "displaced":            None,
        "affected":             None,
        "indirectly_affected":  None,
        "houses_destroyed":     None,
        "houses_damaged":       None,
        "roads_km":             None,
        "damage_usd_thousands": None,
        "severity":             None,
        "main_cause":           None,
        "event_name":           record.get("name"),
    }


def _from_ifrc(record: dict) -> dict:
    iso = record.get("country_iso")
    return {
        "source":               "IFRC",
        "source_id":            str(record.get("id", "")),
        "glide_number":         record.get("glide_number"),
        "date_start":           record.get("date_start"),
        "date_end":             record.get("date_end"),
        "country":              record.get("country"),
        "country_iso":          iso,
        "all_country_iso":      [iso] if iso else [],
        "region":               None,
        "lat":                  None,
        "lon":                  None,
        "location_name":        record.get("country"),
        "area_km2":             None,
        "dead":                 None,
        "injured":              None,
        "displaced":            None,
        "affected":             record.get("num_beneficiaries"),
        "indirectly_affected":  None,
        "houses_destroyed":     None,
        "houses_damaged":       None,
        "roads_km":             None,
        "damage_usd_thousands": None,
        "severity":             record.get("appeal_type"),   # EA or DREF
        "main_cause":           None,
        "event_name":           record.get("name"),
    }


def _from_copernicus(record: dict) -> dict:
    iso = record.get("country_iso")
    all_iso = []
    for name in (record.get("all_countries") or []):
        c = _to_iso3(name)
        if c and c not in all_iso:
            all_iso.append(c)
    if not all_iso and iso:
        all_iso = [iso]

    # Population affected: prefer per-AOI sum, fall back to agg field
    population = record.get("agg_population")

    # Flooded area in ha -> km²
    flooded_ha = record.get("agg_flooded_ha")
    area_km2 = round(flooded_ha / 100.0, 2) if flooded_ha else None

    return {
        "source":               "Copernicus",
        "source_id":            str(record.get("id", "")),
        "glide_number":         None,
        "date_start":           record.get("event_time") or record.get("date_start"),
        "date_end":             None,
        "country":              record.get("country"),
        "country_iso":          iso,
        "all_country_iso":      all_iso,
        "region":               record.get("continent"),
        "lat":                  record.get("lat"),
        "lon":                  record.get("lon"),
        "location_name":        record.get("country"),
        "area_km2":             area_km2,
        "dead":                 None,
        "injured":              None,
        "displaced":            None,
        "affected":             int(population) if population else None,
        "indirectly_affected":  None,
        "houses_destroyed":     None,
        "houses_damaged":       None,
        "roads_km":             record.get("agg_roads_km"),
        "damage_usd_thousands": None,
        "severity":             record.get("sub_category"),
        "main_cause":           record.get("sub_category"),
        "event_name":           record.get("event_name"),
        # Extended Copernicus-specific fields (preserved in combined output)
        "emsr_code":            record.get("id"),
        "activation_time":      record.get("date_start"),
        "gdacs_id":             record.get("gdacs_id"),
        "charter_number":       record.get("charter_number"),
        "report_link":          record.get("report_link"),
        "extent_wkt":           record.get("extent_wkt"),
        "agg_buildup_ha":       record.get("agg_buildup_ha"),
        "detail_available":     record.get("detail_available", False),
    }


def _from_desinventar(record: dict) -> dict:
    iso = record.get("country_iso")
    return {
        "source":               "Desinventar",
        "source_id":            str(record.get("id", "")),
        "glide_number":         record.get("glide_number"),
        "date_start":           record.get("date_start"),
        "date_end":             None,
        "country":              record.get("country"),
        "country_iso":          iso,
        "all_country_iso":      [iso] if iso else [],
        "region":               record.get("region"),
        "lat":                  None,
        "lon":                  None,
        "location_name":        record.get("location_name"),
        "area_km2":             None,
        "dead":                 record.get("dead"),
        "injured":              record.get("injured"),
        "displaced":            record.get("displaced"),
        "affected":             record.get("affected"),
        "indirectly_affected":  record.get("indirectly_affected"),
        "houses_destroyed":     record.get("houses_destroyed"),
        "houses_damaged":       record.get("houses_damaged"),
        "roads_km":             record.get("roads_km"),
        "damage_usd_thousands": record.get("damage_usd_thousands"),
        "severity":             None,
        "main_cause":           record.get("main_cause"),
        "event_name":           None,
    }


def _from_hanze(record: dict) -> dict:
    iso = record.get("country_iso")
    return {
        "source":                   "HANZE",
        "source_id":                str(record.get("id", "")),
        "glide_number":             None,
        "date_start":               record.get("date_start"),
        "date_end":                 record.get("date_end"),
        "country":                  record.get("country"),
        "country_iso":              iso,
        "all_country_iso":          [iso] if iso else [],
        "region":                   None,
        "lat":                      None,
        "lon":                      None,
        "location_name":            record.get("location_name"),
        "area_km2":                 record.get("area_km2"),
        "dead":                     record.get("dead"),
        "injured":                  None,
        "displaced":                None,
        "affected":                 record.get("affected"),
        "indirectly_affected":      None,
        "houses_destroyed":         None,
        "houses_damaged":           None,
        "roads_km":                 None,
        "damage_usd_thousands":     None,
        "damage_eur2020_thousands": record.get("damage_eur2020_thousands"),
        "severity":                 record.get("flood_type"),
        "main_cause":               record.get("main_cause"),
        "event_name":               None,
        # HANZE-specific
        "flood_type":               record.get("flood_type"),
        "source_notes":             record.get("source_notes"),
    }


_MAPPERS = {
    "dfo":         _from_dfo,
    "emdat":       _from_emdat,
    "gdacs":       _from_gdacs,
    "reliefweb":   _from_reliefweb,
    "ifrc":        _from_ifrc,
    "copernicus":  _from_copernicus,
    "desinventar": _from_desinventar,
    "hanze":       _from_hanze,
}


# ------------------ COMBINER ------------------ #

def combine_flood_references(
    dfo_path: Path,
    emdat_path: Path,
    gdacs_path: Path,
    reliefweb_path: Path,
    output_path: Path,
    ifrc_path: Path | None = None,
    copernicus_path: Path | None = None,
    desinventar_path: Path | None = None,
    hanze_path: Path | None = None,
) -> None:
    """
    Load all cache files, map to unified schema, write combined JSONL.
    Optional sources (ifrc, copernicus, desinventar, hanze) are silently
    skipped if their paths are None or the file does not exist.
    """
    sources = {
        "dfo":         dfo_path,
        "emdat":       emdat_path,
        "gdacs":       gdacs_path,
        "reliefweb":   reliefweb_path,
        "ifrc":        ifrc_path,
        "copernicus":  copernicus_path,
        "desinventar": desinventar_path,
        "hanze":       hanze_path,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for key, path in sources.items():
            if path is None or not path.exists():
                if path is not None:
                    print(f"[combine] WARNING: {path} not found — skipping {key}")
                continue

            records = json.loads(path.read_text(encoding="utf-8"))
            mapper = _MAPPERS[key]

            for record in records:
                unified = mapper(record)
                fout.write(json.dumps(unified) + "\n")

    print(f"[combine] Written: {output_path}")


# ------------------ CLI ------------------ #

if __name__ == "__main__":
    # python -m Builder_Reference.helper_scripts.reference.combine
    base = Path("cache/floods")
    # Use the web-scraped DesInventar file if available, fall back to the
    # manually-supplied one (desinventar.json from cache/dfo.py).
    desinventar_web = base / "desinventar.json"
    combine_flood_references(
        dfo_path=base / "dfo.json",
        emdat_path=base / "emdat.json",
        gdacs_path=base / "gdacs.json",
        reliefweb_path=base / "reliefweb.json",
        copernicus_path=base / "copernicus.json",
        desinventar_path=desinventar_web if desinventar_web.exists() else None,
        hanze_path=base / "hanze.json",
        output_path=base / "reference_floods_combined.jsonl",
    )
