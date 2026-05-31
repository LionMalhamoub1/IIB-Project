from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from markets_client import MarketsClient, MarketsClientConfig  # noqa: E402
from markets_features import (  # noqa: E402
    build_fx_features,
    build_global_series_features,
    build_oil_features,
    build_yield_features,
    reindex_to_daily,
    missingness_report,
)

MARKETS_ROOT = _SRC_DIR.parent
DEFAULT_CONFIG = _SRC_DIR / "markets_config.yaml"
RAW_DIR = MARKETS_ROOT / "data" / "raw"
PROCESSED_DIR = MARKETS_ROOT / "data" / "processed"


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def _resolve(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def process_country_fx(
    iso3: str,
    currency: str,
    client: MarketsClient,
    start: str,
    end: str,
    pct_windows: List[int],
    vol_windows: List[int],
    ffill_limit: int,
) -> Optional[pd.DataFrame]:
    logger.info("[FX] %s  (USD/%s)", iso3, currency)
    raw_df = client.fetch_fx(iso3, currency, start, end)

    if raw_df is None or raw_df.empty:
        logger.warning("[FX] %s: no data returned — FX columns will be NaN.", iso3)
        return None

    raw_series = raw_df["fx_lcu_usd"].dropna()
    if raw_series.empty:
        logger.warning("[FX] %s: all Close values are NaN.", iso3)
        return None

    features = build_fx_features(raw_series, pct_windows, vol_windows)

    features = reindex_to_daily(features, start, end, ffill_limit)
    return features


def process_oil(
    client: MarketsClient,
    start: str,
    end: str,
    pct_windows: List[int],
    ffill_limit: int,
    fallback_ticker: str = "CL=F",
) -> Optional[pd.DataFrame]:
    logger.info("[Oil] Brent crude (BZ=F)")
    raw_df = client.fetch_oil_brent(start, end, fallback_ticker=fallback_ticker)

    if raw_df is None or raw_df.empty:
        logger.warning("[Oil] No data returned — oil columns will be NaN.")
        return None

    raw_series = raw_df["oil_brent_usd"].dropna()
    features = build_oil_features(raw_series, pct_windows)
    features = reindex_to_daily(features, start, end, ffill_limit)
    return features


def process_us10y(
    client: MarketsClient,
    start: str,
    end: str,
    ffill_limit: int,
) -> Optional[pd.Series]:
    logger.info("[Yield] US 10-Year Treasury (^TNX)")
    raw_df = client.fetch_us10y(start, end)

    if raw_df is None or raw_df.empty:
        logger.warning("[Yield] US 10Y not available.")
        return None

    series = raw_df["yield_us10y"].dropna()
    full_index = pd.date_range(start=start, end=end, freq="D", name="date")
    series = series.reindex(full_index).ffill(limit=ffill_limit)
    return series


def process_local_yield(
    iso3: str,
    ticker: str,
    client: MarketsClient,
    start: str,
    end: str,
    ffill_limit: int,
) -> Optional[pd.Series]:
    if not ticker or not ticker.strip():
        return None

    logger.info("[Yield] %s local 10Y  (%s)", iso3, ticker)
    raw_df = client.fetch_local_yield(iso3, ticker, start, end)

    if raw_df is None or raw_df.empty:
        return None

    series = raw_df["yield_local10y"].dropna()
    full_index = pd.date_range(start=start, end=end, freq="D", name="date")
    series = series.reindex(full_index).ffill(limit=ffill_limit)
    return series


def process_global_series(
    label: str,
    fetch_result: Dict[str, Optional[pd.DataFrame]],
    pct_windows: List[int],
    ffill_limit: int,
    start: str,
    end: str,
) -> Optional[pd.DataFrame]:
    """Combine per-indicator DataFrames into a single features DataFrame."""
    parts: List[pd.Series] = []
    for name, df in fetch_result.items():
        if df is not None and not df.empty:
            parts.append(df.iloc[:, 0].rename(name))
        else:
            logger.warning("[%s] No data for '%s' — column will be NaN.", label, name)

    if not parts:
        logger.warning("[%s] All tickers failed — no features produced.", label)
        return None

    raw_df = pd.concat(parts, axis=1)
    raw_df.index = pd.to_datetime(raw_df.index)

    features = build_global_series_features(raw_df, pct_windows)
    features = reindex_to_daily(features, start, end, ffill_limit)
    return features


def _global_series_column_names(tickers: Dict[str, str], pct_windows: List[int]) -> List[str]:
    cols: List[str] = []
    for name in tickers:
        cols.append(name)
        for w in sorted(pct_windows):
            cols.append(f"{name}_pct_{w}d")
    return cols


def build_panel(
    cfg: Dict[str, Any],
    client: MarketsClient,
    start: str,
    end: str,
) -> pd.DataFrame:
    countries: Dict[str, str] = _resolve(cfg, "countries", default={})
    fx_cfg = _resolve(cfg, "fx", default={})
    oil_cfg = _resolve(cfg, "oil", default={})
    yield_cfg = _resolve(cfg, "yields", default={})
    gidx_cfg = _resolve(cfg, "global_indices", default={})
    comm_cfg = _resolve(cfg, "commodities", default={})

    pct_windows_fx: List[int] = fx_cfg.get("pct_change_windows", [7, 30, 90])
    vol_windows: List[int] = fx_cfg.get("vol_windows", [7, 30])
    ffill_fx: int = fx_cfg.get("ffill_limit", 7)

    pct_windows_oil: List[int] = oil_cfg.get("pct_change_windows", [14, 30])
    oil_fallback: str = oil_cfg.get("fallback_ticker", "CL=F")
    ffill_oil: int = oil_cfg.get("ffill_limit", 7)

    yields_enabled: bool = bool(yield_cfg.get("enabled", True))
    country_yield_tickers: Dict[str, str] = yield_cfg.get("country_tickers", {})
    ffill_yield: int = ffill_fx

    gidx_enabled: bool = bool(gidx_cfg.get("enabled", True))
    gidx_tickers: Dict[str, str] = gidx_cfg.get("tickers", {})
    pct_windows_gidx: List[int] = gidx_cfg.get("pct_change_windows", [7, 30])
    ffill_gidx: int = gidx_cfg.get("ffill_limit", 7)

    comm_enabled: bool = bool(comm_cfg.get("enabled", True))
    comm_tickers: Dict[str, str] = comm_cfg.get("tickers", {})
    pct_windows_comm: List[int] = comm_cfg.get("pct_change_windows", [14, 30])
    ffill_comm: int = comm_cfg.get("ffill_limit", 7)

    full_dates = pd.date_range(start=start, end=end, freq="D", name="date")

    oil_features = process_oil(
        client, start, end, pct_windows_oil, ffill_oil, fallback_ticker=oil_fallback
    )

    us10y_series: Optional[pd.Series] = None
    if yields_enabled:
        us10y_series = process_us10y(client, start, end, ffill_yield)

    gidx_features: Optional[pd.DataFrame] = None
    if gidx_enabled and gidx_tickers:
        logger.info("[Global Indices] Fetching %d tickers …", len(gidx_tickers))
        gidx_features = process_global_series(
            "GlobalIndices",
            client.fetch_global_indices(gidx_tickers, start, end),
            pct_windows_gidx, ffill_gidx, start, end,
        )

    comm_features: Optional[pd.DataFrame] = None
    if comm_enabled and comm_tickers:
        logger.info("[Commodities] Fetching %d tickers …", len(comm_tickers))
        comm_features = process_global_series(
            "Commodities",
            client.fetch_commodities(comm_tickers, start, end),
            pct_windows_comm, ffill_comm, start, end,
        )

    country_frames: List[pd.DataFrame] = []
    n_total = len(countries)
    n_fx_ok = 0

    for idx, (iso3, currency) in enumerate(countries.items(), start=1):
        logger.info("─" * 60)
        logger.info("Country %d/%d: %s  (currency: %s)", idx, n_total, iso3, currency)

        scaffold = pd.DataFrame(index=full_dates)
        scaffold.index.name = "date"

        fx_df = process_country_fx(
            iso3, currency, client,
            start, end, pct_windows_fx, vol_windows, ffill_fx,
        )
        if fx_df is not None:
            scaffold = scaffold.join(fx_df, how="left")
            n_fx_ok += 1
        else:
            for col in _fx_column_names(pct_windows_fx, vol_windows):
                scaffold[col] = float("nan")

        if oil_features is not None:
            scaffold = scaffold.join(oil_features, how="left")
        else:
            for col in _oil_column_names(pct_windows_oil):
                scaffold[col] = float("nan")

        if gidx_features is not None:
            scaffold = scaffold.join(gidx_features, how="left")
        elif gidx_enabled and gidx_tickers:
            for col in _global_series_column_names(gidx_tickers, pct_windows_gidx):
                scaffold[col] = float("nan")

        if comm_features is not None:
            scaffold = scaffold.join(comm_features, how="left")
        elif comm_enabled and comm_tickers:
            for col in _global_series_column_names(comm_tickers, pct_windows_comm):
                scaffold[col] = float("nan")

        if yields_enabled:
            local10y: Optional[pd.Series] = None
            local_ticker = country_yield_tickers.get(iso3, "")
            if local_ticker:
                local10y = process_local_yield(
                    iso3, local_ticker, client, start, end, ffill_yield
                )

            yield_df = build_yield_features(us10y_series, local10y)
            if not yield_df.empty:
                scaffold = scaffold.join(yield_df, how="left")
            else:
                for col in ["yield_us10y", "yield_local10y", "yield_spread_vs_us"]:
                    if col not in scaffold.columns:
                        scaffold[col] = float("nan")

        scaffold.insert(0, "country_iso3", iso3)
        scaffold = scaffold.reset_index()
        country_frames.append(scaffold)

    logger.info("=" * 60)
    logger.info("Stacking %d country frames …", len(country_frames))

    if not country_frames:
        raise RuntimeError("No country data was produced. Check config and connectivity.")

    panel = pd.concat(country_frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["country_iso3", "date"]).reset_index(drop=True)

    logger.info(
        "Panel shape: %d rows × %d cols  (%d countries, %d FX series OK)",
        len(panel), len(panel.columns), n_total, n_fx_ok,
    )

    return panel


def _fx_column_names(pct_windows: List[int], vol_windows: List[int]) -> List[str]:
    cols = ["fx_lcu_usd", "fx_log_return"]
    cols += [f"fx_pct_{w}d" for w in sorted(pct_windows)]
    cols += [f"fx_vol_{w}d" for w in sorted(vol_windows)]
    return cols


def _oil_column_names(pct_windows: List[int]) -> List[str]:
    cols = ["oil_brent_usd"]
    cols += [f"oil_brent_pct_{w}d" for w in sorted(pct_windows)]
    return cols


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pull daily markets data (FX, oil, yields) and build a country-day panel."
    )
    p.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG,
        help=f"Path to YAML config file (default: {DEFAULT_CONFIG})",
    )
    p.add_argument(
        "--start", type=str, default=None,
        help="Override start_date from config (YYYY-MM-DD)",
    )
    p.add_argument(
        "--end", type=str, default=None,
        help="Override end_date from config (YYYY-MM-DD)",
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help="Ignore cached files and re-download everything",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("Markets Pipeline")
    logger.info("Config: %s", args.config)

    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    cfg = load_config(args.config)

    start: str = args.start or str(_resolve(cfg, "start_date", default="2017-01-01"))
    end: str = args.end or str(_resolve(cfg, "end_date", default="2025-12-31"))
    logger.info("Date range: %s → %s", start, end)

    cache_cfg = _resolve(cfg, "cache", default={})
    use_cache = False if args.no_cache else bool(cache_cfg.get("use_cache", True))
    max_age = 0.0 if args.no_cache else float(cache_cfg.get("max_age_days", 1.0))
    delay = float(cache_cfg.get("request_delay_s", 0.5))

    logger.info("Cache: %s  (max_age_days=%.1f)", "enabled" if use_cache else "DISABLED", max_age)

    client = MarketsClient(
        MarketsClientConfig(
            raw_dir=RAW_DIR,
            use_cache=use_cache,
            max_age_days=max_age,
            request_delay_s=delay,
        )
    )

    panel = build_panel(cfg, client, start, end)

    logger.info("\n%s", missingness_report(panel.drop(columns=["country_iso3", "date"])))

    start_s = start.replace("-", "")
    end_s = end.replace("-", "")
    out_cfg = _resolve(cfg, "output", default={})
    stem = out_cfg.get("processed_filename", "markets_country_day_<START>_<END>")
    stem = stem.replace("<START>", start_s).replace("<END>", end_s)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_parquet = PROCESSED_DIR / f"{stem}.parquet"
    out_csv = PROCESSED_DIR / f"{stem}.csv"

    panel.to_parquet(out_parquet, index=False)
    logger.info("Saved parquet → %s", out_parquet)

    if bool(out_cfg.get("save_csv", True)):
        panel.to_csv(out_csv, index=False)
        logger.info("Saved CSV     → %s", out_csv)

    logger.info("Done.")


if __name__ == "__main__":
    main()
