from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

try:
    from dotenv import load_dotenv as _load_dotenv
    def _load_env_file(env_path: Path) -> None:
        _load_dotenv(env_path, override=True)
except ImportError:
    def _load_env_file(env_path: Path) -> None:  # type: ignore[misc]
        with open(env_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key:
                    os.environ[key] = val

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from gta_client import GTAClient, GTAClientConfig   # noqa: E402
from utils import (                                  # noqa: E402
    eval_to_harmful,
    eval_to_liberalising,
    missingness_report,
    setup_logging,
)

GTA_ROOT      = _SRC_DIR.parent
DEFAULT_CONFIG = GTA_ROOT / "config" / "gta_config.yaml"
RAW_DIR       = GTA_ROOT / "data" / "raw"
INTERIM_DIR   = GTA_ROOT / "data" / "interim"
PROCESSED_DIR = GTA_ROOT / "data" / "processed"

logger = logging.getLogger(__name__)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _resolve(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def clean_interventions(records: List[dict]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for rec in records:
        intervention_id   = rec.get("intervention_id")
        gta_eval          = str(rec.get("gta_evaluation", ""))
        harmful           = eval_to_harmful(gta_eval)
        liberalising      = eval_to_liberalising(gta_eval)
        date_implemented  = rec.get("date_implemented")
        intervention_type = rec.get("intervention_type", "") or ""

        sectors_raw = rec.get("affected_sectors") or []
        if sectors_raw and isinstance(sectors_raw[0], dict):
            sector_codes = [str(s.get("product_id", "")) for s in sectors_raw]
        else:
            sector_codes = [str(s) for s in sectors_raw]
        affected_sector = ";".join(sector_codes) if sector_codes else None

        implementers = rec.get("implementing_jurisdictions") or []

        if not implementers:
            rows.append({
                "intervention_id":     intervention_id,
                "implementing_country": None,
                "implementation_date": date_implemented,
                "intervention_type":   intervention_type,
                "affected_sector":     affected_sector,
                "harmful":             harmful,
                "liberalising":        liberalising,
            })
        else:
            for impl in implementers:
                rows.append({
                    "intervention_id":     intervention_id,
                    "implementing_country": impl.get("iso"),
                    "implementation_date": date_implemented,
                    "intervention_type":   intervention_type,
                    "affected_sector":     affected_sector,
                    "harmful":             harmful,
                    "liberalising":        liberalising,
                })

    df = pd.DataFrame(rows, columns=[
        "intervention_id", "implementing_country", "implementation_date",
        "intervention_type", "affected_sector", "harmful", "liberalising",
    ])

    n_raw = len(df)
    logger.info("[Clean] Raw rows (after explode): %d", n_raw)

    df["implementation_date"] = pd.to_datetime(
        df["implementation_date"], errors="coerce"
    )

    n_before = len(df)
    df = df.dropna(subset=["implementation_date"])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("[Clean] Dropped %d rows with missing implementation_date.", n_dropped)

    n_before = len(df)
    df = df.drop_duplicates(subset=["intervention_id", "implementing_country"])
    n_dupes = n_before - len(df)
    if n_dupes:
        logger.info("[Clean] Dropped %d duplicate (intervention_id, country) rows.", n_dupes)

    df = df.sort_values(["implementing_country", "implementation_date"]).reset_index(drop=True)

    logger.info("[Clean] Final clean rows: %d", len(df))
    logger.info("\n%s", missingness_report(df))

    return df


def build_country_day_panel(
    df: pd.DataFrame,
    start: str,
    end: str,
    harmful_events: bool = True,
    liberalising_events: bool = True,
    rolling_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    work = df.dropna(subset=["implementing_country", "implementation_date"]).copy()
    work = work.rename(columns={
        "implementing_country": "country_iso3",
        "implementation_date":  "date",
    })

    countries  = sorted(work["country_iso3"].unique())
    full_dates = pd.date_range(start=start, end=end, freq="D", name="date")
    full_mi    = pd.MultiIndex.from_product(
        [countries, full_dates], names=["country_iso3", "date"]
    )

    agg: Dict[str, Any] = {"gta_policy_events": ("intervention_id", "count")}
    if harmful_events:
        agg["gta_harmful_events"]      = ("harmful",      "sum")
    if liberalising_events:
        agg["gta_liberalising_events"] = ("liberalising", "sum")

    counts = (
        work.groupby(["country_iso3", "date"])
        .agg(**agg)
        .reindex(full_mi, fill_value=0)
        .reset_index()
    )

    for col in ["gta_policy_events", "gta_harmful_events", "gta_liberalising_events"]:
        if col in counts.columns:
            counts[col] = counts[col].astype(int)

    if rolling_windows:
        counts = counts.sort_values(["country_iso3", "date"])
        for window in sorted(rolling_windows):
            counts[f"gta_{window}d_count"] = (
                counts
                .groupby("country_iso3")["gta_policy_events"]
                .transform(lambda x: x.rolling(window, min_periods=1).sum())
                .astype(int)
            )

    counts = counts.sort_values(["country_iso3", "date"]).reset_index(drop=True)

    logger.info(
        "[Panel] Shape: %d rows × %d cols  (%d countries, %d date range)",
        len(counts), len(counts.columns),
        counts["country_iso3"].nunique(),
        (pd.to_datetime(end) - pd.to_datetime(start)).days + 1,
    )

    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pull GTA intervention data and build a country-day panel."
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
        help="Ignore cached files and re-fetch from the API",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    _env_path = Path(__file__).resolve().parents[3] / ".env"
    if _env_path.exists():
        _load_env_file(_env_path)
        logger.info(".env loaded from %s  (GTA_API_KEY present: %s)",
                    _env_path, "GTA_API_KEY" in os.environ)
    else:
        logger.warning(".env not found at %s — relying on shell environment.", _env_path)

    logger.info("=" * 60)
    logger.info("GTA Pipeline")
    logger.info("Config: %s", args.config)

    if not args.config.exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    cfg = load_config(args.config)

    start: str = args.start or str(_resolve(cfg, "start_date", default="2017-01-01"))
    end:   str = args.end   or str(_resolve(cfg, "end_date",   default="2025-12-31"))
    logger.info("Date range: %s → %s", start, end)

    countries_cfg = _resolve(cfg, "countries", default="all")

    api_cfg   = _resolve(cfg, "api",   default={})
    cache_cfg = _resolve(cfg, "cache", default={})

    use_cache  = False if args.no_cache else bool(cache_cfg.get("use_cache", True))
    page_size  = int(api_cfg.get("page_size",       1000))
    delay      = float(api_cfg.get("request_delay_s", 1.0))
    max_ret    = int(api_cfg.get("max_retries",        3))
    backoff    = float(api_cfg.get("retry_backoff_s",  2.0))
    base_url   = str(api_cfg.get("base_url",
                                  "https://api.globaltradealert.org/api/v1/data/"))

    logger.info("Cache: %s", "enabled" if use_cache else "DISABLED")

    feat_cfg   = _resolve(cfg, "features", default={})
    harmful_ev = bool(feat_cfg.get("harmful_events",      True))
    liberal_ev = bool(feat_cfg.get("liberalising_events", True))
    rolling_w: List[int] = feat_cfg.get("rolling_windows", []) or []

    out_cfg  = _resolve(cfg, "output", default={})
    save_csv = bool(out_cfg.get("save_csv", True))

    logger.info("=" * 60)
    logger.info("Step 1 — Fetching interventions from GTA API")

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    client = GTAClient(
        GTAClientConfig(
            raw_dir          = RAW_DIR,
            use_cache        = use_cache,
            base_url         = base_url,
            page_size        = page_size,
            request_delay_s  = delay,
            max_retries      = max_ret,
            retry_backoff_s  = backoff,
        )
    )

    records = client.fetch_interventions(
        start_date = start,
        end_date   = end,
        countries  = countries_cfg,
        page_size  = page_size,
    )

    start_s = start.replace("-", "")
    end_s   = end.replace("-", "")
    raw_path = RAW_DIR / f"gta_interventions_{start_s}_{end_s}.json"
    logger.info("Step 2 — Raw JSON: %s  (%d records)", raw_path.name, len(records))

    logger.info("=" * 60)
    logger.info("Step 3 — Cleaning and normalising")

    clean_df = clean_interventions(records)

    logger.info("=" * 60)
    logger.info("Step 4 — Saving interim files")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    interim_parquet = INTERIM_DIR / "gta_interventions_clean.parquet"
    interim_csv     = INTERIM_DIR / "gta_interventions_clean.csv"

    clean_df.to_parquet(interim_parquet, index=False)
    logger.info("Saved parquet → %s", interim_parquet)

    if save_csv:
        clean_df.to_csv(interim_csv, index=False)
        logger.info("Saved CSV     → %s", interim_csv)

    logger.info("=" * 60)
    logger.info("Step 5 — Building country-day panel")
    logger.info(
        "  Options: harmful_events=%s  liberalising_events=%s  rolling_windows=%s",
        harmful_ev, liberal_ev, rolling_w or "none",
    )

    panel = build_country_day_panel(
        df                  = clean_df,
        start               = start,
        end                 = end,
        harmful_events      = harmful_ev,
        liberalising_events = liberal_ev,
        rolling_windows     = rolling_w,
    )

    total_events = int(panel["gta_policy_events"].sum())
    active_days  = int((panel["gta_policy_events"] > 0).sum())
    logger.info(
        "Panel summary: %d total events, %d country-days with ≥1 event",
        total_events, active_days,
    )

    logger.info("=" * 60)
    logger.info("Step 6 — Saving processed panel")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stem = out_cfg.get(
        "processed_filename", "gta_country_day_<START>_<END>"
    ).replace("<START>", start_s).replace("<END>", end_s)

    out_parquet = PROCESSED_DIR / f"{stem}.parquet"
    out_csv     = PROCESSED_DIR / f"{stem}.csv"

    panel.to_parquet(out_parquet, index=False)
    logger.info("Saved parquet → %s", out_parquet)

    if save_csv:
        panel.to_csv(out_csv, index=False)
        logger.info("Saved CSV     → %s", out_csv)

    logger.info("=" * 60)
    logger.info("Done.")


if __name__ == "__main__":
    main()
