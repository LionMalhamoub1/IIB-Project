"""
Consolidate the combined flood reference dataset by merging records from
different sources that refer to the same real-world event.

Input : reference_floods_combined.jsonl  (output of combine.py)
Output: reference_floods_consolidated.jsonl

Algorithm
---------
1. Load all records from the combined JSONL into a flat list.

2. Pass 0 — Desinventar pre-aggregation
   Desinventar records are sub-national (one row per district/municipality).
   A single EM-DAT/DFO country-level event may correspond to many Desinventar
   district entries.  Before cross-source matching, cluster Desinventar records
   by (country_iso, date_window) using a rolling DESINVENTAR_CLUSTER_DAYS
   window.  Each cluster is replaced by one synthetic "Desinventar-AGG" record
   that sums impact fields and lists the constituent record IDs.  The original
   district records are kept as children and written to the output with a
   cluster_id pointer; they are NOT subject to further cross-source merging.

3. Pass 1 — GLIDE exact match
   Records from different sources that share the same GLIDE number are
   definitively the same event. Add a graph edge between them.

4. Pass 2 — ISO-3 country + date-range proximity
   For each pair of records from different sources in the same country:
   - Require date-range overlap with DATE_TOLERANCE_DAYS padding.
   - Enforce a one-to-one constraint: only add an edge if neither record
     already has a candidate from the other's source, which would indicate
     ambiguity (two floods in the same country at the same time).
   - Sub-national location tiebreaker: if BOTH records have location tokens
     that go beyond just the country name, and those tokens share nothing,
     skip the merge (likely different sub-events in the same country).
     If either record only has country-level location, the tiebreaker does
     NOT fire — we cannot penalise a source for lacking sub-national detail.
   - Desinventar-AGG records use a relaxed tolerance (DESINVENTAR_TOLERANCE_DAYS)
     and skip the one-to-one constraint on the Desinventar side (many district
     clusters may all link to the same country-level event).

5. Find connected components using union-find.
   Each component is a group of records that all refer to the same event.

6. Merge each component into one consolidated record using field priority.
   Fields are taken from the most authoritative source that has a value.

7. Write the consolidated records to JSONL.
   Each record carries matched_sources and source_ids for traceability.
   Desinventar child records are appended with their cluster_id.
"""

import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Optional


# ------------------ CONFIG ------------------ #

# How many days beyond true date-range overlap to still consider a match.
# Accounts for reporting lag between sources (e.g. EM-DAT registers later).
DATE_TOLERANCE_DAYS = 7

# Maximum ratio of event durations for two records to be considered the same
# event.  A 342-day EM-DAT umbrella entry and a 2-day GDACS alert are clearly
# not the same discrete flood, so we cap the ratio at 5x.
MAX_DURATION_RATIO = 5

# Desinventar pre-aggregation: district records within this many days of each
# other (same country) are grouped into one synthetic cluster record.
DESINVENTAR_CLUSTER_DAYS = 7

# Relaxed date tolerance used when matching Desinventar-AGG records against
# country-level sources (EM-DAT often records the event start several days
# after Desinventar's precise district-affected date).
DESINVENTAR_TOLERANCE_DAYS = 14

# Field priority: for each field, list sources in order of preference.
# The first non-None value encountered wins.
FIELD_PRIORITY: dict[str, list[str]] = {
    "dead":                     ["DFO", "EM-DAT", "HANZE", "Desinventar"],
    "injured":                  ["Desinventar"],
    "displaced":                ["DFO", "Desinventar"],
    "affected":                 ["EM-DAT", "GDACS", "HANZE", "IFRC", "Desinventar"],
    "indirectly_affected":      ["Desinventar"],
    "houses_destroyed":         ["Desinventar"],
    "houses_damaged":           ["Desinventar"],
    "roads_km":                 ["Copernicus", "Desinventar"],
    "damage_usd_thousands":     ["EM-DAT", "Desinventar"],
    "damage_eur2020_thousands": ["HANZE"],
    "lat":                      ["DFO", "GDACS", "Copernicus", "EM-DAT"],
    "lon":                      ["DFO", "GDACS", "Copernicus", "EM-DAT"],
    "date_start":               ["DFO", "EM-DAT", "HANZE", "GDACS", "ReliefWeb", "IFRC", "Copernicus", "Desinventar"],
    "date_end":                 ["DFO", "EM-DAT", "HANZE", "GDACS", "IFRC"],
    "country_iso":              ["EM-DAT", "ReliefWeb", "IFRC", "DFO", "GDACS", "HANZE", "Copernicus", "Desinventar"],
    "country":                  ["DFO", "EM-DAT", "GDACS", "ReliefWeb", "IFRC", "HANZE"],
    "region":                   ["EM-DAT", "Desinventar"],
    "location_name":            ["Desinventar", "HANZE", "EM-DAT", "DFO", "GDACS", "ReliefWeb"],
    "event_name":               ["ReliefWeb", "GDACS", "EM-DAT", "IFRC", "Copernicus"],
    "area_km2":                 ["DFO", "HANZE", "Copernicus"],
    "severity":                 ["DFO", "GDACS", "HANZE"],
    "main_cause":               ["DFO", "EM-DAT", "HANZE", "Desinventar"],
    "glide_number":             ["DFO", "EM-DAT", "ReliefWeb", "IFRC", "Desinventar"],
}


# ------------------ UNION-FIND ------------------ #

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[px] = py

    def same(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)


# ------------------ HELPERS ------------------ #

def _parse_date(s: Optional[str]):
    if not s:
        return None
    try:
        from datetime import date
        parts = s[:10].split("-")
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return None


def _ranges_overlap(s1, e1, s2, e2, tolerance_days: int = DATE_TOLERANCE_DAYS) -> bool:
    """Date-range overlap with tolerance padding on both ends."""
    d1s = _parse_date(s1)
    d2s = _parse_date(s2)
    if not d1s or not d2s:
        return False
    d1e = _parse_date(e1) or d1s
    d2e = _parse_date(e2) or d2s
    tol = timedelta(days=tolerance_days)
    return d1s <= d2e + tol and d2s <= d1e + tol


def _duration_ratio_ok(s1, e1, s2, e2) -> bool:
    """Return False if one event's duration is more than MAX_DURATION_RATIO times
    the other's.  This prevents long umbrella entries (e.g. a 300-day EM-DAT
    country summary) from being treated as the same discrete event as a 2-day
    GDACS alert."""
    d1s = _parse_date(s1)
    d2s = _parse_date(s2)
    if not d1s or not d2s:
        return True  # cannot judge — let other checks decide
    d1 = max((_parse_date(e1) - d1s).days if _parse_date(e1) else 0, 0)
    d2 = max((_parse_date(e2) - d2s).days if _parse_date(e2) else 0, 0)
    shorter = min(d1, d2)
    longer  = max(d1, d2)
    if shorter == 0:
        return longer <= DATE_TOLERANCE_DAYS  # point event: only ok if other is also very short
    return (longer / shorter) <= MAX_DURATION_RATIO


def _subnational_tokens(record: dict) -> set[str]:
    """
    Location tokens that go beyond the country name.

    Returns an empty set if the location_name contains nothing more than
    what's already in the country field — i.e. this record has no
    sub-national location detail, so the tiebreaker should not fire.
    """
    loc_text    = record.get("location_name") or ""
    country_text = record.get("country") or ""

    loc_tokens     = {t.lower() for t in loc_text.split()     if len(t) > 2}
    country_tokens = {t.lower() for t in country_text.split() if len(t) > 2}

    return loc_tokens - country_tokens


# ------------------ PASS 0: DESINVENTAR PRE-AGGREGATION ------------------ #

def _common_region(cluster: list[dict]) -> str | None:
    """Return the shared region if all districts agree, else None."""
    regions = {(r.get("region") or "").strip() for r in cluster}
    regions.discard("")
    return regions.pop() if len(regions) == 1 else None


def _cluster_location_name(cluster: list[dict]) -> str:
    """
    Best available location name for a multi-district AGG record.
    Uses 'Region, Country' if all districts share a region, else just country.
    """
    country = (cluster[0].get("country") or "").strip()
    region  = _common_region(cluster)
    if region:
        return f"{region}, {country}"
    return country


def _aggregate_desinventar(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Cluster Desinventar district records by (country_iso, rolling date window)
    and replace them with one synthetic "Desinventar-AGG" record per cluster.

    Returns:
        agg_records  — the synthetic cluster records (participate in cross-source matching)
        child_records — the original district records (written to output with cluster_id)
    """
    from datetime import date as _date

    desinv = [(i, r) for i, r in enumerate(records) if r.get("source") == "Desinventar"]
    other  = [r for r in records if r.get("source") != "Desinventar"]

    if not desinv:
        return [], []

    # Sort by country then date
    def sort_key(ir):
        _, r = ir
        iso  = r.get("country_iso") or ""
        ds   = r.get("date_start") or "9999-99-99"
        return (iso, ds)

    desinv.sort(key=sort_key)

    clusters: list[list[dict]] = []
    current_cluster: list[dict] = []
    current_iso = None
    cluster_end: _date | None = None

    tol = timedelta(days=DESINVENTAR_CLUSTER_DAYS)

    for _, rec in desinv:
        iso = rec.get("country_iso") or ""
        ds  = _parse_date(rec.get("date_start"))
        if ds is None:
            # No date — put in its own singleton cluster
            clusters.append([rec])
            continue
        if iso != current_iso or cluster_end is None or ds > cluster_end + tol:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = [rec]
            current_iso = iso
            cluster_end = ds
        else:
            current_cluster.append(rec)
            de = _parse_date(rec.get("date_end") or rec.get("date_start"))
            if de and de > cluster_end:
                cluster_end = de

    if current_cluster:
        clusters.append(current_cluster)

    agg_records: list[dict] = []
    child_records: list[dict] = []

    for cluster_idx, cluster in enumerate(clusters):
        cluster_id = f"DESAGG_{cluster_idx}"

        # Annotate children
        for rec in cluster:
            rec["cluster_id"] = cluster_id
            child_records.append(rec)

        if len(cluster) == 1:
            # Singleton — promote directly as agg (no summing needed)
            agg = dict(cluster[0])
            agg["source"]    = "Desinventar-AGG"
            agg["source_id"] = cluster_id
            agg["cluster_id"] = cluster_id
            agg["cluster_size"] = 1
        else:
            # Sum impact fields across the cluster
            def _sum(field):
                vals = [r.get(field) for r in cluster if r.get(field) is not None]
                return sum(vals) if vals else None

            first = cluster[0]
            dates = sorted(
                d for d in (_parse_date(r.get("date_start")) for r in cluster) if d
            )
            end_dates = sorted(
                d for d in (_parse_date(r.get("date_end") or r.get("date_start")) for r in cluster) if d
            )

            agg = {
                "source":                   "Desinventar-AGG",
                "source_id":                cluster_id,
                "cluster_id":               cluster_id,
                "cluster_size":             len(cluster),
                "glide_number":             next((r.get("glide_number") for r in cluster if r.get("glide_number")), None),
                "date_start":               dates[0].isoformat() if dates else first.get("date_start"),
                "date_end":                 end_dates[-1].isoformat() if end_dates else None,
                "country":                  first.get("country"),
                "country_iso":              first.get("country_iso"),
                "all_country_iso":          first.get("all_country_iso") or [],
                # If all districts share the same region, carry it through
                # so geocoding can resolve a sub-national point rather than
                # falling back to a country centroid.
                "region":                   _common_region(cluster),
                "lat":                      None,
                "lon":                      None,
                "location_name":            _cluster_location_name(cluster),
                "area_km2":                 None,
                "dead":                     _sum("dead"),
                "injured":                  _sum("injured"),
                "displaced":                _sum("displaced"),
                "affected":                 _sum("affected"),
                "indirectly_affected":      _sum("indirectly_affected"),
                "houses_destroyed":         _sum("houses_destroyed"),
                "houses_damaged":           _sum("houses_damaged"),
                "roads_km":                 _sum("roads_km"),
                "damage_usd_thousands":     _sum("damage_usd_thousands"),
                "damage_eur2020_thousands": None,
                "severity":                 None,
                "main_cause":               next((r.get("main_cause") for r in cluster if r.get("main_cause")), None),
                "event_name":               None,
            }

        agg_records.append(agg)

    return agg_records, child_records


def _glide_is_valid(g) -> bool:
    return bool(g) and str(g).strip() not in ("", "0", "None", "nan")


# ------------------ PASS 1: GLIDE MATCHING ------------------ #

def _glide_edges(records: list[dict]) -> list[tuple[int, int]]:
    glide_index: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        g = rec.get("glide_number")
        if _glide_is_valid(g):
            glide_index[str(g)].append(i)

    edges = []
    for nodes in glide_index.values():
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                i, j = nodes[a], nodes[b]
                if records[i]["source"] != records[j]["source"]:
                    edges.append((i, j))
    return edges


# ------------------ PASS 2: COUNTRY + DATE MATCHING ------------------ #

def _country_date_edges(
    records: list[dict],
    uf: UnionFind,
) -> list[tuple[int, int]]:
    # Index by every country ISO code the record covers.
    # Single-country records have all_country_iso = [country_iso].
    # Multi-country records (e.g. a GDACS cyclone spanning three countries)
    # are indexed under each country so they can match records from any of them.
    iso_index: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        all_iso = rec.get("all_country_iso") or []
        if not all_iso:
            # Fallback for records produced before all_country_iso was added
            iso = rec.get("country_iso")
            if iso:
                all_iso = [iso]
        for iso in all_iso:
            if i not in iso_index[iso]:  # avoid duplicate entries
                iso_index[iso].append(i)

    # For each node, track which sources it already has a candidate from
    source_candidates: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    # First pass: collect all valid candidates without committing to edges
    valid_pairs: list[tuple[int, int]] = []

    for iso, nodes in iso_index.items():
        for a in range(len(nodes)):
            for b in range(a + 1, len(nodes)):
                i, j = nodes[a], nodes[b]
                ri, rj = records[i], records[j]

                if ri["source"] == rj["source"]:
                    continue
                if uf.same(i, j):
                    continue  # Already merged via GLIDE

                # Use relaxed tolerance if either side is a Desinventar-AGG cluster
                is_desagg = (ri["source"] == "Desinventar-AGG" or
                             rj["source"] == "Desinventar-AGG")
                tol = DESINVENTAR_TOLERANCE_DAYS if is_desagg else DATE_TOLERANCE_DAYS

                if not _ranges_overlap(
                    ri.get("date_start"), ri.get("date_end"),
                    rj.get("date_start"), rj.get("date_end"),
                    tolerance_days=tol,
                ):
                    continue

                if not _duration_ratio_ok(
                    ri.get("date_start"), ri.get("date_end"),
                    rj.get("date_start"), rj.get("date_end"),
                ):
                    continue  # One event is an umbrella entry, not the same discrete flood

                # Sub-national tiebreaker: only fires when BOTH records have
                # location detail beyond the country name.  If either record
                # is country-level only, we cannot use location to distinguish.
                # Desinventar-AGG always presents country-level location, so
                # the tiebreaker will not fire against it.
                ti = _subnational_tokens(ri)
                tj = _subnational_tokens(rj)
                if ti and tj and not (ti & tj):
                    continue  # Different sub-events in the same country

                valid_pairs.append((i, j))
                source_candidates[i][rj["source"]].append(j)
                source_candidates[j][ri["source"]].append(i)

    # Second pass: enforce one-to-one — only keep pairs where each node
    # has exactly one candidate from the other's source.
    # Exception: Desinventar-AGG records may link to one country-level event
    # even if that event has multiple AGG clusters (many-to-one allowed on
    # the Desinventar-AGG side only).
    edges = []
    for i, j in valid_pairs:
        ri, rj = records[i], records[j]
        i_is_desagg = ri["source"] == "Desinventar-AGG"
        j_is_desagg = rj["source"] == "Desinventar-AGG"

        i_to_j = source_candidates[i].get(rj["source"], [])
        j_to_i = source_candidates[j].get(ri["source"], [])

        # Standard one-to-one for non-Desinventar pairs
        if not i_is_desagg and not j_is_desagg:
            if len(i_to_j) == 1 and len(j_to_i) == 1:
                edges.append((i, j))
        # Desinventar-AGG side: allow many clusters per country event,
        # but the country-level record must still be unambiguous (one candidate)
        elif i_is_desagg and not j_is_desagg:
            if len(j_to_i) == 1:  # j has only one Desinventar-AGG candidate
                edges.append((i, j))
        elif j_is_desagg and not i_is_desagg:
            if len(i_to_j) == 1:  # i has only one Desinventar-AGG candidate
                edges.append((i, j))
        else:
            # Both Desinventar-AGG — standard one-to-one
            if len(i_to_j) == 1 and len(j_to_i) == 1:
                edges.append((i, j))

    return edges


# ------------------ MERGING ------------------ #

def _merge_group(group: list[dict]) -> dict:
    # Treat Desinventar-AGG as "Desinventar" for field priority lookups
    by_source = {}
    for rec in group:
        src = rec["source"]
        by_source[src] = rec
        if src == "Desinventar-AGG":
            by_source.setdefault("Desinventar", rec)

    def pick(field: str):
        priority = FIELD_PRIORITY.get(field, list(by_source.keys()))
        for src in priority:
            val = by_source.get(src, {}).get(field)
            if val is not None:
                return val
        for rec in group:
            val = rec.get(field)
            if val is not None:
                return val
        return None

    merged = {
        "source":               "CONSOLIDATED",
        "matched_sources":      sorted({rec["source"] for rec in group}),
        "source_ids":           {rec["source"]: rec["source_id"] for rec in group},
        "glide_number":         pick("glide_number"),
        "date_start":           pick("date_start"),
        "date_end":             pick("date_end"),
        "country":              pick("country"),
        "country_iso":          pick("country_iso"),
        "region":               pick("region"),
        "lat":                  pick("lat"),
        "lon":                  pick("lon"),
        "location_name":        pick("location_name"),
        "area_km2":             pick("area_km2"),
        "dead":                 pick("dead"),
        "injured":              pick("injured"),
        "displaced":            pick("displaced"),
        "affected":             pick("affected"),
        "indirectly_affected":  pick("indirectly_affected"),
        "houses_destroyed":     pick("houses_destroyed"),
        "houses_damaged":       pick("houses_damaged"),
        "roads_km":                 pick("roads_km"),
        "damage_usd_thousands":     pick("damage_usd_thousands"),
        "damage_eur2020_thousands": pick("damage_eur2020_thousands"),
        "severity":                 pick("severity"),
        "main_cause":           pick("main_cause"),
        "event_name":           pick("event_name"),
    }
    return merged


# ------------------ MAIN ------------------ #

def consolidate_flood_references(
    combined_path: Path,
    output_path: Path,
) -> None:
    raw_records: list[dict] = []
    with combined_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    # Pass 0: pre-aggregate Desinventar district records into country-level clusters
    agg_records, child_records = _aggregate_desinventar(raw_records)
    non_desinventar = [r for r in raw_records if r.get("source") != "Desinventar"]
    records = non_desinventar + agg_records
    print(f"[consolidate] Pass 0 (Desinventar-AGG): {len(agg_records)} clusters "
          f"from {len(child_records)} district records")

    n = len(records)
    uf = UnionFind(n)

    # Pass 1: GLIDE edges
    glide_edges = _glide_edges(records)
    for i, j in glide_edges:
        uf.union(i, j)
    print(f"[consolidate] Pass 1 (GLIDE): {len(glide_edges)} edges")

    # Pass 2: country + date edges (Desinventar-AGG uses relaxed tolerance)
    cd_edges = _country_date_edges(records, uf)
    for i, j in cd_edges:
        uf.union(i, j)
    print(f"[consolidate] Pass 2 (country+date): {len(cd_edges)} edges")

    # Find connected components
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        components[uf.find(i)].append(i)

    multi = sum(1 for members in components.values() if len(members) > 1)
    print(f"[consolidate] Components: {len(components)} total, {multi} merged, "
          f"{len(components) - multi} standalone")

    # Merge and write — consolidated events first, then Desinventar children
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for members in components.values():
            group = [records[i] for i in members]
            if len(group) == 1:
                rec = group[0]
                rec["matched_sources"] = [rec["source"]]
                rec["source_ids"] = {rec["source"]: rec["source_id"]}
            else:
                rec = _merge_group(group)
            fout.write(json.dumps(rec) + "\n")

        # Append Desinventar district children (linked via cluster_id)
        for rec in child_records:
            rec["matched_sources"] = ["Desinventar"]
            rec["source_ids"] = {"Desinventar": rec.get("source_id")}
            fout.write(json.dumps(rec) + "\n")

    total_out = len(components) + len(child_records)
    print(f"[consolidate] Written: {output_path} ({total_out} records: "
          f"{len(components)} consolidated + {len(child_records)} Desinventar children)")


# ------------------ CLI ------------------ #

if __name__ == "__main__":
    # python -m Builder_Reference.helper_scripts.reference.consolidate
    base = Path("cache/floods")
    consolidate_flood_references(
        combined_path=base / "reference_floods_combined.jsonl",
        output_path=base / "reference_floods_consolidated.jsonl",
    )
