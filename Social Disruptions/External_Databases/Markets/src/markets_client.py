from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketsClientConfig:
    raw_dir: Path
    use_cache: bool = True
    max_age_days: float = 1.0
    request_delay_s: float = 0.5


class MarketsClient:

    def __init__(self, cfg: MarketsClientConfig) -> None:
        self.cfg = cfg
        cfg.raw_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(self, ticker: str) -> str:
        return ticker.replace("=", "X").replace("^", "").replace("/", "-")

    def _cache_path(self, subdir: str, filename: str) -> Path:
        p = self.cfg.raw_dir / subdir
        p.mkdir(parents=True, exist_ok=True)
        return p / filename

    def _is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        if self.cfg.max_age_days <= 0:
            return False
        age_days = (time.time() - path.stat().st_mtime) / 86400.0
        return age_days < self.cfg.max_age_days

    def _fetch_ticker(
        self,
        ticker: str,
        start: str,
        end: str,
        cache_path: Path,
        col_name: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        col_name = col_name or ticker

        if self.cfg.use_cache and self._is_fresh(cache_path):
            logger.info("  [cache] %s", cache_path.name)
            try:
                return pd.read_parquet(cache_path)
            except Exception as exc:
                logger.warning("  Cache read failed (%s); re-downloading.", exc)

        logger.info("  [download] %s  %s → %s", ticker, start, end)
        try:
            time.sleep(self.cfg.request_delay_s)
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                multi_level_index=False,
            )
        except TypeError:
            try:
                time.sleep(self.cfg.request_delay_s)
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )
            except Exception as exc:
                logger.error("  Download failed for %s: %s", ticker, exc)
                return None
        except Exception as exc:
            logger.error("  Download failed for %s: %s", ticker, exc)
            return None

        if raw is None or raw.empty:
            logger.warning("  No data returned for ticker: %s", ticker)
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        if "Close" not in raw.columns:
            logger.warning("  'Close' column absent for %s (got: %s)", ticker, list(raw.columns))
            return None

        df = raw[["Close"]].copy()
        df.columns = [col_name]
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        df = df.loc[start:end]

        if df.empty:
            logger.warning("  After date filtering, no rows remain for %s", ticker)
            return None

        logger.info("  → %d rows  (%s to %s)", len(df), df.index[0].date(), df.index[-1].date())

        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
        except Exception as exc:
            logger.warning("  Cache write failed: %s", exc)

        return df

    def fetch_fx(
        self,
        iso3: str,
        currency: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        ticker = f"USD{currency}=X"
        safe = self._safe_name(ticker)
        start_s = start.replace("-", "")
        end_s = end.replace("-", "")
        cache = self._cache_path(
            f"countries/{iso3}",
            f"fx_{safe}_{start_s}_{end_s}.parquet",
        )
        df = self._fetch_ticker(ticker, start, end, cache, col_name="fx_lcu_usd")
        return df

    def fetch_oil_brent(
        self,
        start: str,
        end: str,
        fallback_ticker: str = "CL=F",
    ) -> Optional[pd.DataFrame]:
        primary = "BZ=F"
        start_s = start.replace("-", "")
        end_s = end.replace("-", "")
        cache = self._cache_path("global", f"brent_{start_s}_{end_s}.parquet")
        df = self._fetch_ticker(primary, start, end, cache, col_name="oil_brent_usd")

        if df is None or df.empty:
            logger.warning("Brent (%s) returned no data; trying fallback: %s", primary, fallback_ticker)
            safe = self._safe_name(fallback_ticker)
            cache_fb = self._cache_path("global", f"oil_{safe}_{start_s}_{end_s}.parquet")
            df = self._fetch_ticker(fallback_ticker, start, end, cache_fb, col_name="oil_brent_usd")

        return df

    def fetch_us10y(self, start: str, end: str) -> Optional[pd.DataFrame]:
        ticker = "^TNX"
        start_s = start.replace("-", "")
        end_s = end.replace("-", "")
        cache = self._cache_path("global", f"us10y_{start_s}_{end_s}.parquet")
        return self._fetch_ticker(ticker, start, end, cache, col_name="yield_us10y")

    def fetch_named_series(
        self,
        name: str,
        ticker: str,
        start: str,
        end: str,
        subdir: str = "global",
    ) -> Optional[pd.DataFrame]:
        """Fetch a single named series by ticker and cache it in subdir."""
        safe = self._safe_name(ticker)
        start_s = start.replace("-", "")
        end_s = end.replace("-", "")
        cache = self._cache_path(subdir, f"{name}_{safe}_{start_s}_{end_s}.parquet")
        return self._fetch_ticker(ticker, start, end, cache, col_name=name)

    def fetch_global_indices(
        self,
        tickers: Dict[str, str],
        start: str,
        end: str,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Fetch global risk indicators (VIX, DXY, SP500, Gold)."""
        return {
            name: self.fetch_named_series(name, ticker, start, end, subdir="global")
            for name, ticker in tickers.items()
        }

    def fetch_commodities(
        self,
        tickers: Dict[str, str],
        start: str,
        end: str,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Fetch food & energy commodity prices."""
        return {
            name: self.fetch_named_series(name, ticker, start, end, subdir="global")
            for name, ticker in tickers.items()
        }

    def fetch_local_yield(
        self,
        iso3: str,
        ticker: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        if not ticker or not ticker.strip():
            logger.debug("  Local yield ticker empty for %s — skipping.", iso3)
            return None

        safe = self._safe_name(ticker)
        start_s = start.replace("-", "")
        end_s = end.replace("-", "")
        cache = self._cache_path(
            f"countries/{iso3}",
            f"yield10y_{safe}_{start_s}_{end_s}.parquet",
        )
        df = self._fetch_ticker(ticker, start, end, cache, col_name="yield_local10y")
        if df is None:
            logger.warning("  Local yield unavailable for %s (ticker: %s)", iso3, ticker)
        return df
