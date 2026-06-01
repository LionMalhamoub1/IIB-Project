from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.globaltradealert.org/api/v1/data/"


@dataclass(frozen=True)
class GTAClientConfig:
    raw_dir: Path
    api_key: str = ""
    base_url: str = _BASE_URL
    use_cache: bool = True
    page_size: int = 1000
    request_delay_s: float = 1.0
    max_retries: int = 3
    retry_backoff_s: float = 2.0


class GTAClient:

    def __init__(self, cfg: GTAClientConfig) -> None:
        self.cfg = cfg
        self._api_key = cfg.api_key or os.environ.get("GTA_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "GTA API key not found.  Pass api_key= to GTAClientConfig "
                "or set the GTA_API_KEY environment variable."
            )
        cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"APIKey {self._api_key}",
            "Content-Type": "application/json",
        }

    def _cache_path(self, start: str, end: str) -> Path:
        start_s = start.replace("-", "")
        end_s   = end.replace("-", "")
        return self.cfg.raw_dir / f"gta_interventions_{start_s}_{end_s}.json"

    def _post_page(self, body: Dict[str, Any]) -> List[dict]:
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                time.sleep(self.cfg.request_delay_s)
                resp = requests.post(
                    self.cfg.base_url,
                    headers=self._headers(),
                    json=body,
                    timeout=60,
                )
                resp.raise_for_status()

                data = resp.json()
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return data.get("results", data.get("data", []))
                return []

            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                if status == 401:
                    raise RuntimeError(
                        "GTA API authentication failed — check your API key."
                    ) from exc
                if status == 429:
                    wait = self.cfg.retry_backoff_s * (2 ** (attempt - 1))
                    logger.warning(
                        "[GTA] Rate limited (429). Waiting %.1fs  (retry %d/%d).",
                        wait, attempt, self.cfg.max_retries,
                    )
                    time.sleep(wait)
                    continue
                logger.error("[GTA] HTTP %s on attempt %d/%d: %s",
                             status, attempt, self.cfg.max_retries, exc)

            except requests.RequestException as exc:
                wait = self.cfg.retry_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "[GTA] Request error on attempt %d/%d: %s. Retrying in %.1fs.",
                    attempt, self.cfg.max_retries, exc, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"GTA API request failed after {self.cfg.max_retries} attempts.  "
            "Check connectivity and API key."
        )

    def fetch_interventions(
        self,
        start_date: str,
        end_date: str,
        countries: List[str] | str = "all",
        page_size: Optional[int] = None,
    ) -> List[dict]:
        cache = self._cache_path(start_date, end_date)

        if self.cfg.use_cache and cache.exists():
            logger.info("[GTA] Cache hit — loading %s", cache.name)
            with open(cache, "r", encoding="utf-8") as fh:
                records = json.load(fh)
            logger.info("[GTA] Loaded %d records from cache.", len(records))
            return records

        size = min(page_size or self.cfg.page_size, 1000)

        request_data: Dict[str, Any] = {
            "implementation_period": [start_date, end_date],
            "keep_implementation_na": False,
        }

        if countries != "all" and isinstance(countries, list) and countries:
            from utils import iso3_to_un_codes  # noqa: PLC0415
            un_codes = iso3_to_un_codes(countries)
            if un_codes:
                request_data["implementer"] = un_codes
                request_data["keep_implementer"] = True
                logger.info("[GTA] Filtering to %d countries (%d UN codes).",
                            len(countries), len(un_codes))
            else:
                logger.warning(
                    "[GTA] No valid UN codes for %s — fetching all countries.", countries
                )

        all_records: List[dict] = []
        offset     = 0
        page_num   = 0

        logger.info("[GTA] Fetching interventions  %s → %s", start_date, end_date)
        logger.info("[GTA] Page size: %d", size)

        with tqdm(unit=" records", desc="GTA fetch", dynamic_ncols=True) as pbar:
            while True:
                page_num += 1
                body = {
                    "limit":        size,
                    "offset":       offset,
                    "sorting":      "date_implemented",
                    "request_data": request_data,
                }

                pbar.set_postfix(page=page_num)
                batch = self._post_page(body)

                if not batch:
                    logger.info("[GTA]   Empty page — pagination complete.")
                    break

                all_records.extend(batch)
                pbar.update(len(batch))
                logger.info("[GTA]   +%d records  (running total: %d)",
                            len(batch), len(all_records))

                if len(batch) < size:
                    break

                offset += size

        logger.info("[GTA] Total records fetched: %d", len(all_records))

        try:
            with open(cache, "w", encoding="utf-8") as fh:
                json.dump(all_records, fh)
            logger.info("[GTA] Raw cache saved → %s", cache)
        except OSError as exc:
            logger.warning("[GTA] Cache write failed: %s", exc)

        return all_records
