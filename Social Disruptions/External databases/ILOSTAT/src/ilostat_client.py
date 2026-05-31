from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger(__name__)

_TOC_URL       = "https://rplumber.ilo.org/metadata/toc/indicator/"
_DATA_URL      = "https://rplumber.ilo.org/data/indicator/"
_TIMEOUT_S     = 120
_MAX_RETRIES   = 5
_BACKOFF       = 2.0
_RETRY_STATUS  = (429, 500, 502, 503, 504)


@dataclass
class ILOSTATClientConfig:
    cache_dir:        Path
    timeout:          int   = _TIMEOUT_S
    max_retries:      int   = _MAX_RETRIES
    backoff_factor:   float = _BACKOFF
    pool_connections: int   = 20
    pool_maxsize:     int   = 20


class ILOSTATClient:

    def __init__(self, cfg: ILOSTATClientConfig) -> None:
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
        adapter = HTTPAdapter(
            max_retries=retry,
            pool_connections=self.cfg.pool_connections,
            pool_maxsize=self.cfg.pool_maxsize,
        )
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        session.headers["User-Agent"] = "ILOSTATPipeline/2.0 (research)"
        return session

    @staticmethod
    def _cache_key(url: str, params: dict) -> str:
        qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        canonical = f"{url}?{qs}" if qs else url
        return hashlib.md5(canonical.encode()).hexdigest()

    def _cache_path(self, key: str, suffix: str) -> Path:
        return self.cfg.cache_dir / f"{key}{suffix}"

    def _get_bytes(self, url: str, params: dict) -> bytes:
        resp = self._session.get(url, params=params, timeout=self.cfg.timeout)
        resp.raise_for_status()
        return resp.content

    def fetch_toc(self, use_cache: bool = True) -> pd.DataFrame:
        params = {"lang": "en", "format": "csv"}
        key    = self._cache_key(_TOC_URL, params)
        cache  = self._cache_path(key, ".csv")

        if use_cache and cache.exists():
            logger.debug("[cache] TOC")
        else:
            logger.info("Fetching ILOSTAT TOC …")
            cache.write_bytes(self._get_bytes(_TOC_URL, params))

        toc = pd.read_csv(cache, low_memory=False)

        toc = toc[~toc["id"].astype(str).str.upper().str.startswith("SDG_")].copy()
        for db_col in ("database", "database.label"):
            if db_col in toc.columns:
                toc = toc[
                    ~toc[db_col].astype(str).str.upper().str.contains("SDG", na=False)
                ]
                break

        return toc.reset_index(drop=True)

    def fetch_indicator(
        self,
        dataset_id: str,
        use_cache:  bool = True,
    ) -> pd.DataFrame:
        for fmt, suffix, reader in [
            (".parquet", ".parquet", pd.read_parquet),
            (".csv",     ".csv",     lambda p: pd.read_csv(p, low_memory=False)),
        ]:
            params = {"id": dataset_id, "format": fmt}
            key    = self._cache_key(_DATA_URL, params)
            cache  = self._cache_path(key, suffix)

            if use_cache and cache.exists():
                logger.debug("[cache] %s (%s)", dataset_id, fmt)
                try:
                    return reader(cache)
                except Exception:
                    cache.unlink(missing_ok=True)

            try:
                data = self._get_bytes(_DATA_URL, params)
                cache.write_bytes(data)
                return reader(cache)
            except requests.HTTPError as exc:
                code = exc.response.status_code if exc.response is not None else "?"
                if fmt == ".parquet":
                    logger.debug(
                        "%s: parquet unavailable (HTTP %s); trying CSV …",
                        dataset_id, code,
                    )
                    continue
                raise RuntimeError(
                    f"{dataset_id}: CSV download also failed (HTTP {code})."
                ) from exc
            except Exception as exc:
                cache.unlink(missing_ok=True)
                if fmt == ".parquet":
                    logger.debug(
                        "%s: parquet read failed (%s); trying CSV …",
                        dataset_id, exc,
                    )
                    continue
                raise RuntimeError(
                    f"{dataset_id}: failed to load {fmt} response: {exc}"
                ) from exc

        raise RuntimeError(f"{dataset_id}: could not be downloaded in any format.")

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "ILOSTATClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
