#!/usr/bin/env python3
"""
Smoke test for the GDACS flood alert feed.

This script:
  - downloads the GDACS RSS feed for flood alerts,
  - parses the XML,
  - prints a small sample of flood events.

GDACS = Global Disaster Alert and Coordination System (run by JRC / EU).
No API key or registration is required for RSS access.
"""

import requests
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------
# GDACS flood RSS feed URL
# ---------------------------------------------------------------------

# Flood alerts from the last 7 days
GDACS_FLOOD_FEED_URL = "https://www.gdacs.org/xml/rss_fl_7d.xml"


# ---------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------

def main():
    print("Fetching GDACS flood RSS feed...")
    response = requests.get(GDACS_FLOOD_FEED_URL, timeout=20)

    # Raise an error if HTTP request failed
    response.raise_for_status()

    # Parse XML
    root = ET.fromstring(response.content)

    # RSS structure: <rss><channel><item>...</item></channel></rss>
    channel = root.find("channel")
    if channel is None:
        raise RuntimeError("Invalid RSS format: <channel> not found")

    items = channel.findall("item")

    print(f"GDACS RSS reachable. Found {len(items)} flood alert items.\n")

    # Print a few sample events
    for item in items[:5]:
        title = (item.findtext("title") or "").strip()
        pubdate = (item.findtext("pubDate") or "").strip()
        link = (item.findtext("link") or "").strip()

        print(f"- {title}")
        if pubdate:
            print(f"    published: {pubdate}")
        if link:
            print(f"    link: {link}")
        print()

    print("GDACS flood smoke test completed successfully.")


if __name__ == "__main__":
    main()
