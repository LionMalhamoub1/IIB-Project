#!/usr/bin/env python3

#
#ONLY WORKS FOR LIVE FLOODS - NOT REALLY RELEVANT
#
#

"""
Smoke test for NASA EONET flood events API.

This script:
  - queries the EONET API for flood events,
  - prints a small sample of returned events.

NASA EONET (Earth Observatory Natural Event Tracker) provides open access to
global natural hazard event metadata. No API key is required.
"""

import requests

# ---------------------------------------------------------------------
# NASA EONET floods endpoint
# ---------------------------------------------------------------------

EONET_FLOODS_URL = "https://eonet.gsfc.nasa.gov/api/v3/events?category=floods&limit=10"


def main():
    print("Fetching NASA EONET flood events...")
    response = requests.get(EONET_FLOODS_URL, timeout=20)

    # Raise error if HTTP request failed
    response.raise_for_status()

    data = response.json()

    events = data.get("events", [])

    print(f"NASA EONET reachable. Returned {len(events)} flood events.\n")

    # Print a few sample events
    for event in events[:5]:
        event_id = event.get("id", "")
        title = event.get("title", "")
        start = event.get("geometry", [{}])[0].get("date", "")
        categories = [c.get("title", "") for c in event.get("categories", [])]

        print(f"- {title}")
        print(f"    id: {event_id}")
        print(f"    category: {', '.join(categories)}")
        print(f"    first geometry date: {start}")
        print()

    print("NASA EONET flood smoke test completed successfully.")


if __name__ == "__main__":
    main()
