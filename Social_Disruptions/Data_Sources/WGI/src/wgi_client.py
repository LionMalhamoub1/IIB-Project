from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

WB_BASE = "https://api.worldbank.org/v2"


@dataclass(frozen=True)
class WGIClientConfig:
    cache_dir: Path
    timeout_s: int = 60
    per_page: int = 20000


class WGIClient:

    def __init__(self, cfg: WGIClientConfig):
        self.cfg = cfg
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, url: str, params: Dict[str, Any]) -> Path:
        blob = url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        h = hashlib.md5(blob.encode("utf-8")).hexdigest()
        return self.cfg.cache_dir / f"{h}.json"

    def _get_json(self, url: str, params: Dict[str, Any], use_cache: bool = True) -> Any:
        cache_path = self._cache_key(url, params)
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))

        r = requests.get(url, params=params, timeout=self.cfg.timeout_s)
        r.raise_for_status()
        data = r.json()

        cache_path.write_text(json.dumps(data), encoding="utf-8")
        return data

    def fetch_indicator_series(
        self,
        indicator_code: str,
        countries: Iterable[str] = ("all",),
        source_id: int = 3,   # 3 = WGI data source in the World Bank API
        start_year: Optional[int] = None,
        end_year: Optional[int] = None,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        country_str = ";".join(countries)
        url = f"{WB_BASE}/country/{country_str}/indicator/{indicator_code}"

        params: Dict[str, Any] = {
            "format": "json",
            "source": source_id,
            "per_page": self.cfg.per_page,
            "page": 1,
        }

        if start_year and end_year:
            params["date"] = f"{start_year}:{end_year}"
        elif start_year:
            params["date"] = f"{start_year}:"
        elif end_year:
            params["date"] = f":{end_year}"

        data = self._get_json(url, params, use_cache=use_cache)

        if not isinstance(data, list) or len(data) < 2 or data[1] is None:
            return []

        meta = data[0] or {}
        rows = data[1] or []
        pages = int(meta.get("pages", 1))

        for p in range(2, pages + 1):
            params_p = dict(params)
            params_p["page"] = p
            data_p = self._get_json(url, params_p, use_cache=use_cache)
            if isinstance(data_p, list) and len(data_p) >= 2 and data_p[1]:
                rows.extend(data_p[1])

        return rows
