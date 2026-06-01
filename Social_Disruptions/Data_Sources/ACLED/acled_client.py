from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

from acled_auth import ACLEDAuth

_DEFAULT_CACHE_DIR = str(Path(__file__).resolve().parent / "data")


@dataclass
class ACLEDClientConfig:
    base_url: str = "https://acleddata.com/api/acled/read"
    cache_dir: str = _DEFAULT_CACHE_DIR
    timeout_s: int = 90
    sleep_s: float = 0.25
    page_size: int = 5000


def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sha1_dict(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


def _normalise_events(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "event_date" in df.columns:
        df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    for col in ("fatalities", "latitude", "longitude", "year"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "iso3" in df.columns:
        df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    return df


class ACLEDClient:

    def __init__(self, auth: ACLEDAuth, cfg: Optional[ACLEDClientConfig] = None):
        self.auth = auth
        self.cfg = cfg or ACLEDClientConfig()
        _mkdir(self.cfg.cache_dir)
        self.sess = requests.Session()
        self.sess.headers.update({"User-Agent": "lion-lithium-risk/1.0"})

    def _get_headers(self) -> Dict[str, str]:
        token = self.auth.get_access_token()
        return {"Authorization": f"Bearer {token}"}

    def fetch_events(
        self,
        *,
        countries: List[str],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None,
        extra_params: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Pull events for the given countries/dates, paginating automatically. Results cached to parquet."""
        country_param = ":OR:country=".join([c.replace(":", "") for c in countries])
        params_base = {
            "_format": "json",
            "country": country_param,
            "event_date": f"{start_date}|{end_date}",
            "event_date_where": "BETWEEN",
            "limit": str(self.cfg.page_size),
            "page": "1",
        }
        if fields:
            params_base["fields"] = "|".join(fields)
        if extra_params:
            params_base.update(extra_params)

        cache_key = _sha1_dict(params_base)
        cache_path = os.path.join(self.cfg.cache_dir, f"{cache_key}.parquet")

        if use_cache and (not force_refresh) and os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        rows: List[dict] = []
        page = 1

        pbar = tqdm(disable=not show_progress, desc="ACLED pages")
        while True:
            params = dict(params_base)
            params["page"] = str(page)

            r = self.sess.get(
                self.cfg.base_url,
                params=params,
                headers=self._get_headers(),
                timeout=self.cfg.timeout_s,
            )

            if r.status_code == 401:
                # Token may have expired mid-pagination; refresh and retry once
                r = self.sess.get(
                    self.cfg.base_url,
                    params=params,
                    headers=self._get_headers(),
                    timeout=self.cfg.timeout_s,
                )

            r.raise_for_status()
            js = r.json()

            data = js.get("data", []) or []
            if not data:
                break

            rows.extend(data)
            page += 1
            pbar.update(1)
            time.sleep(self.cfg.sleep_s)

        pbar.close()

        df = _normalise_events(pd.DataFrame(rows))
        df.to_parquet(cache_path, index=False)
        csv_path = cache_path.replace(".parquet", ".csv")
        df.to_csv(csv_path, index=False)

        return df
