from __future__ import annotations

import hashlib
import io
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger(__name__)

_MMAD_REPORTS_URL = "https://mmadatabase.org/download/505/"
_TIMEOUT_S        = 120
_MAX_RETRIES      = 5
_BACKOFF          = 2.0
_RETRY_STATUS     = (429, 500, 502, 503, 504)


@dataclass
class MMADClientConfig:
    cache_dir:      Path
    reports_url:    str   = _MMAD_REPORTS_URL
    timeout:        int   = _TIMEOUT_S
    max_retries:    int   = _MAX_RETRIES
    backoff_factor: float = _BACKOFF


class MMADClient:

    def __init__(self, cfg: MMADClientConfig) -> None:
        self.cfg = cfg
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = self._build_session()

    def _build_session(self) -> requests.Session:
        retry = Retry(
            total=self.cfg.max_retries,
            backoff_factor=self.cfg.backoff_factor,
            status_forcelist=list(_RETRY_STATUS),
            allowed_methods={"GET"},
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers["User-Agent"] = "MMADPipeline/1.0 (research)"
        return session

    def _cache_path(self, name: str) -> Path:
        key = hashlib.md5(name.encode()).hexdigest()
        return self.cfg.cache_dir / f"{key}_{name}"

    def fetch_reports(self, use_cache: bool = True) -> pd.DataFrame:
        cache = self._cache_path("mmad_reports.csv")

        if use_cache and cache.exists():
            logger.debug("[cache] MMAD reports.csv")
            return pd.read_csv(cache, low_memory=False)

        logger.info("Downloading MMAD reports zip from %s …", self.cfg.reports_url)
        resp = self._session.get(self.cfg.reports_url, timeout=self.cfg.timeout)
        resp.raise_for_status()

        ct = resp.headers.get("content-type", "")
        if "html" in ct:
            raise RuntimeError(
                f"Expected zip download but got HTML from {self.cfg.reports_url}. "
                "The MMAD website may be unavailable."
            )

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise RuntimeError(
                    f"No CSV found inside zip. Contents: {zf.namelist()}"
                )
            target = next(
                (n for n in csv_names if "report" in n.lower()),
                csv_names[0],
            )
            logger.info("Extracting %s from zip", target)
            csv_bytes = zf.read(target)

        cache.write_bytes(csv_bytes)
        logger.info("Saved → %s", cache)

        return pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "MMADClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
