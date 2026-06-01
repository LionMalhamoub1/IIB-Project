from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from pytrends.request import TrendReq

try:
    import pycountry
except ImportError as exc:
    raise ImportError("pycountry is required: pip install pycountry") from exc


log = logging.getLogger(__name__)

_ISO3_TO_ISO2: Dict[str, str] = {c.alpha_3: c.alpha_2 for c in pycountry.countries}


@dataclass(frozen=True)
class TrendsClientConfig:
    cache_dir: Path
    hl: str = "en-US"
    tz: int = 0
    sleep_s: float = 6.0
    max_retries: int = 4
    retry_backoff_base: float = 60.0


class TrendsClient:

    def __init__(self, cfg: TrendsClientConfig) -> None:
        self.cfg = cfg
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pt = TrendReq(hl=cfg.hl, tz=cfg.tz)

    def _cache_path(
        self,
        country_iso3: str,
        kw_batch: List[str],
        win_start: pd.Timestamp,
        win_end: pd.Timestamp,
    ) -> Path:
        key = f"{country_iso3}|{'|'.join(sorted(kw_batch))}|{win_start.date()}|{win_end.date()}"
        return self.cfg.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.parquet"

    def fetch_window(
        self,
        country_iso3: str,
        kw_batch: List[str],
        win_start: pd.Timestamp,
        win_end: pd.Timestamp,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch weekly Trends interest for kw_batch in country/window. Returns weekly DataFrame or empty if nothing came back. Cached."""
        iso2 = _ISO3_TO_ISO2.get(country_iso3)
        if iso2 is None:
            log.warning("No ISO2 mapping for %s — skipping window", country_iso3)
            return pd.DataFrame()

        cache = self._cache_path(country_iso3, kw_batch, win_start, win_end)
        if use_cache and cache.exists():
            return pd.read_parquet(cache)

        timeframe = f"{win_start.strftime('%Y-%m-%d')} {win_end.strftime('%Y-%m-%d')}"
        df: Optional[pd.DataFrame] = None

        for attempt in range(self.cfg.max_retries):
            try:
                self._pt.build_payload(
                    kw_list=kw_batch, timeframe=timeframe, geo=iso2
                )
                df = self._pt.interest_over_time()
                break
            except Exception as exc:
                if attempt < self.cfg.max_retries - 1:
                    wait = self.cfg.retry_backoff_base * (2 ** attempt)
                    log.warning(
                        "Attempt %d/%d failed [%s %s]: %s — retrying in %.0fs",
                        attempt + 1, self.cfg.max_retries,
                        country_iso3, timeframe, exc, wait,
                    )
                    time.sleep(wait)
                else:
                    log.error(
                        "All %d retries exhausted [%s %s]: %s",
                        self.cfg.max_retries, country_iso3, timeframe, exc,
                    )

        time.sleep(self.cfg.sleep_s)

        if df is None or df.empty:
            return pd.DataFrame()

        if "isPartial" in df.columns:
            # Drop the final partial week — its index is incomplete so the 0–100 score is not normalised correctly
            df = df[~df["isPartial"].astype(bool)].drop(columns=["isPartial"])

        df.index = pd.to_datetime(df.index)
        df = df[[c for c in df.columns if c in kw_batch]].astype(float)

        if df.empty:
            return pd.DataFrame()

        df.to_parquet(cache)
        return df
