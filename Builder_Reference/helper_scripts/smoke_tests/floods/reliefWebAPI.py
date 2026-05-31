#!/usr/bin/env python3
"""
Smoke test for the ReliefWeb API (disasters endpoint).

Fixes:
- ReliefWeb v2 filters use filter[field]/filter[value] (not query[filter]...)
- Adds verbose=1 to help debug malformed requests
- Uses date.event for disaster event date (where available)
"""

import requests

APPNAME = "individual-4thYearProjectDisruptionLikelihoodsForSupplyChains-4!6H"
API_BASE = "https://api.reliefweb.int/v2"


def _first_name(value) -> str:
    if isinstance(value, list):
        if not value:
            return ""
        first = value[0]
        return first.get("name", "") if isinstance(first, dict) else ""
    if isinstance(value, dict):
        return value.get("name", "")
    return ""


def _join_names(value) -> str:
    if isinstance(value, list):
        return ", ".join(v.get("name", "") for v in value if isinstance(v, dict) and v.get("name"))
    if isinstance(value, dict):
        return value.get("name", "")
    return ""


def main():
    url = f"{API_BASE}/disasters"

    params = {
        "appname": APPNAME,
        "limit": 5,
        # ✅ Correct way to filter in v2 (GET)
        "filter[field]": "country",
        "filter[value]": "Pakistan",
        # Helpful while debugging: API returns a 'details' section showing parsed query
        "verbose": 1,
        # Fields
        "fields[include][]": [
            "name",
            "type",
            "primary_type",
            "status",
            "country",
            "primary_country",
            "date",
            "glide",
            "url",
        ],
        "sort[]": "date.event:desc",  # event date is usually what you want for disasters
    }

    resp = requests.get(url, params=params, timeout=20)

    if resp.status_code != 200:
        print("ReliefWeb request failed.")
        print(f"Status: {resp.status_code}")
        print(f"URL: {resp.url}")
        try:
            err = resp.json()
            print("Response JSON (truncated):")
            print(str(err)[:2500])
            # If verbose=1, sometimes useful details are in err.get("details")
            if isinstance(err, dict) and "details" in err:
                print("\nParsed query details (from verbose=1):")
                print(str(err["details"])[:2500])
        except Exception:
            print("Response text (truncated):")
            print(resp.text[:2500])

        resp.raise_for_status()

    payload = resp.json()
    items = payload.get("data", [])

    print(f"ReliefWeb API reachable. Returned {len(items)} records.\n")

    for item in items:
        fields = item.get("fields", {}) or {}

        name = fields.get("name", "")

        # disasters: type is a list of dicts; primary_type exists too
        dtype = _first_name(fields.get("primary_type")) or _first_name(fields.get("type"))

        # Prefer date.event for disasters (falls back to date.created)
        date_obj = fields.get("date", {}) if isinstance(fields.get("date", {}), dict) else {}
        date = date_obj.get("event") or date_obj.get("created") or ""

        country = _join_names(fields.get("primary_country")) or _join_names(fields.get("country"))

        print(f"- {name}")
        print(f"    type: {dtype}")
        print(f"    date: {date}")
        print(f"    country: {country}")
        print()

    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
