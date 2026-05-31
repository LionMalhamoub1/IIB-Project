from __future__ import annotations

import json
import time
import random
import requests
import trafilatura
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from typing import Optional, Dict

# Optional Newspaper3k
try:
    from newspaper import Article as _NPArticle
    _HAS_NEWSPAPER = True
except Exception:
    _HAS_NEWSPAPER = False


def extract_article_text(url: str, timeout: int = 20) -> Dict[str, Optional[str]]:
    """
    Extract article text, title, and publication date from a URL.

    Returns:
        {
            "url": str,
            "title": str,
            "text": str,
            "publish_date": str | None,   # ISO 8601 if available
            # "html": str                # optional (see below)
        }
    """

    # ------------------ HELPERS ------------------ #

    def _prep(s: str) -> str:
        if not s:
            return ""
        s = s.replace("\u00a0", " ").replace("\r", " ")
        return " ".join(s.split()).strip()

    def _parse_date(val: str) -> Optional[str]:
        try:
            return dateparser.parse(val).isoformat()
        except Exception:
            return None

    # ------------------ CONFIG ------------------ #

    HEADERS_PRIMARY = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    HEADERS_RETRY = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Version/17.0 Safari/605.1.15"
        )
    }

    META_DATE_TAGS = [
        ("property", "article:published_time"),
        ("property", "article:modified_time"),
        ("name", "pubdate"),
        ("name", "publish-date"),
        ("name", "publication_date"),
        ("itemprop", "datePublished"),
        ("itemprop", "dateModified"),
    ]

    title = ""
    text = ""
    publish_date = None
    html = None

    # Optional: gentle jitter to avoid soft blocking under concurrency
    time.sleep(random.uniform(0.15, 0.45))

    # ------------------ FETCH HTML ------------------ #

    try:
        resp = requests.get(url, headers=HEADERS_PRIMARY, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
    except Exception:
        html = None

    if not html:
        return {
            "url": url,
            "title": "",
            "text": "",
            "publish_date": None,
        }

    soup = BeautifulSoup(html, "html.parser")

    # ------------------ 1) META TAGS ------------------ #

    for attr, key in META_DATE_TAGS:
        tag = soup.find("meta", attrs={attr: key})
        if tag and tag.get("content"):
            parsed = _parse_date(tag["content"])
            if parsed:
                publish_date = parsed
                break

    # ------------------ 2) <time datetime="..."> ------------------ #

    if publish_date is None:
        time_tag = soup.find("time", datetime=True)
        if time_tag:
            publish_date = _parse_date(time_tag["datetime"])

    # ------------------ 3) JSON-LD STRUCTURED DATA ------------------ #

    if publish_date is None:
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(script.string)
                if isinstance(data, dict):
                    for key in ("datePublished", "dateModified"):
                        if key in data:
                            publish_date = _parse_date(data[key])
                            if publish_date:
                                break
                if publish_date:
                    break
            except Exception:
                pass

    # ------------------ 4) TRAFILATURA ------------------ #

    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=False
    ) or ""

    meta = trafilatura.bare_extraction(html)
    if isinstance(meta, dict):
        title = meta.get("title") or title

        if publish_date is None:
            raw = meta.get("date") or meta.get("published")
            if raw:
                publish_date = _parse_date(raw)

    # ------------------ 5) NEWSPAPER3K (ALWAYS TRY FOR DATE) ------------------ #

    if _HAS_NEWSPAPER:
        try:
            art = _NPArticle(url)
            art.download()
            art.parse()

            if not title:
                title = art.title or ""

            if not text:
                text = art.text or ""

            if publish_date is None and art.publish_date:
                publish_date = art.publish_date.isoformat()

        except Exception:
            pass

    # ------------------ 6) RETRY FETCH IF DATE STILL MISSING ------------------ #

    if publish_date is None:
        try:
            resp2 = requests.get(url, headers=HEADERS_RETRY, timeout=timeout)
            if resp2.ok:
                soup2 = BeautifulSoup(resp2.text, "html.parser")
                for attr, key in META_DATE_TAGS:
                    tag = soup2.find("meta", attrs={attr: key})
                    if tag and tag.get("content"):
                        parsed = _parse_date(tag["content"])
                        if parsed:
                            publish_date = parsed
                            break
        except Exception:
            pass

    # ------------------ FINAL CLEANUP ------------------ #

    return {
        "url": url,
        "title": _prep(title),
        "text": _prep(text),
        "publish_date": publish_date,
        # Optional: enable if you want HTML caching
        # "html": html,
    }


if __name__ == "__main__":
    # Quick manual test
    # art = extract_article_text("https://www.bbc.co.uk/news/articles/c07m2v1z4evo")
    # print(art["publish_date"])
    pass
