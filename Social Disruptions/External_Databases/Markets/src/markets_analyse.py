from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

_SRC_DIR = Path(__file__).resolve().parent
MARKETS_ROOT = _SRC_DIR.parent
PROCESSED_DIR = MARKETS_ROOT / "data" / "processed"

logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_panel(path: Optional[Path] = None) -> pd.DataFrame:
    if path is not None:
        logger.info("Loading panel from: %s", path)
        return pd.read_parquet(path)

    candidates = sorted(
        PROCESSED_DIR.glob("markets_country_day_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No markets_country_day_*.parquet found in {PROCESSED_DIR}.\n"
            "Run markets_pipeline.py first."
        )
    chosen = candidates[0]
    logger.info("Auto-detected panel: %s", chosen.name)
    return pd.read_parquet(chosen)


def _annual_summary(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    sub["year"] = sub["date"].dt.year

    def first_valid(s: pd.Series) -> float:
        v = s.dropna()
        return float(v.iloc[0]) if not v.empty else float("nan")

    def last_valid(s: pd.Series) -> float:
        v = s.dropna()
        return float(v.iloc[-1]) if not v.empty else float("nan")

    rows = []
    for year, grp in sub.groupby("year"):
        grp = grp.sort_values("date")
        fx_start = first_valid(grp["fx_lcu_usd"])
        fx_end = last_valid(grp["fx_lcu_usd"])

        if np.isfinite(fx_start) and np.isfinite(fx_end) and fx_start > 0:
            fx_depr = (fx_end / fx_start - 1.0) * 100.0
        else:
            fx_depr = float("nan")

        crisis_days = int(
            (grp["fx_pct_30d"] > grp.attrs.get("crisis_threshold", 5.0)).sum()
        ) if "fx_pct_30d" in grp.columns else 0

        rows.append(
            {
                "year": int(year),
                "fx_start": fx_start,
                "fx_end": fx_end,
                "fx_depreciation_pct": round(fx_depr, 4) if np.isfinite(fx_depr) else None,
                "fx_vol_30d_mean": _safe_mean(grp["fx_vol_30d"]),
                "fx_vol_30d_max": _safe_max(grp["fx_vol_30d"]),
                "fx_pct_30d_worst": _safe_max(grp["fx_pct_30d"]),
                "fx_pct_30d_best": _safe_min(grp["fx_pct_30d"]),
                "oil_mean_usd": _safe_mean(grp["oil_brent_usd"]),
                "oil_pct_30d_worst": _safe_min(grp["oil_brent_pct_30d"]),
                "yield_us10y_mean": _safe_mean(grp["yield_us10y"]),
                "n_trading_days": int(grp["fx_lcu_usd"].notna().sum()),
                "n_crisis_days_30d": crisis_days,
            }
        )

    return pd.DataFrame(rows)


def _summary_stats(
    sub: pd.DataFrame,
    iso3: str,
    crisis_threshold: float,
) -> Dict[str, Any]:
    sub_di = sub.set_index("date")

    fx = sub_di["fx_lcu_usd"].dropna()
    lr = sub_di["fx_log_return"].dropna()
    pct30 = sub_di["fx_pct_30d"].dropna()
    vol30 = sub_di["fx_vol_30d"].dropna()

    if len(fx) >= 2 and fx.iloc[0] > 0:
        total_depr = (fx.iloc[-1] / fx.iloc[0] - 1.0) * 100.0
    else:
        total_depr = None

    ann_vol = float(lr.std() * np.sqrt(252) * 100) if len(lr) > 1 else None

    all_fx = sub_di["fx_lcu_usd"]
    is_nan = all_fx.isna()
    max_gap = int(
        is_nan.groupby((is_nan != is_nan.shift()).cumsum()).cumsum().max()
    ) if is_nan.any() else 0

    stats: Dict[str, Any] = {
        "country_iso3": iso3,
        "date_start": str(sub["date"].min().date()),
        "date_end": str(sub["date"].max().date()),
        "n_calendar_days": int(len(sub)),
        "n_trading_days_fx": int(len(fx)),
        "data_coverage_pct": round(len(fx) / len(sub) * 100, 2) if len(sub) else None,
        "first_date_with_fx": str(fx.index[0].date()) if not fx.empty else None,
        "last_date_with_fx": str(fx.index[-1].date()) if not fx.empty else None,
        "max_consecutive_gap_days": max_gap,
        "fx_first": round(float(fx.iloc[0]), 6) if not fx.empty else None,
        "fx_last": round(float(fx.iloc[-1]), 6) if not fx.empty else None,
        "fx_min": round(float(fx.min()), 6) if not fx.empty else None,
        "fx_max": round(float(fx.max()), 6) if not fx.empty else None,
        "total_depreciation_pct": round(total_depr, 4) if total_depr is not None else None,
        "max_30d_depreciation_pct": round(float(pct30.max()), 4) if not pct30.empty else None,
        "max_30d_appreciation_pct": round(float(pct30.min()), 4) if not pct30.empty else None,
        "mean_vol_30d": round(float(vol30.mean()), 6) if not vol30.empty else None,
        "max_vol_30d": round(float(vol30.max()), 6) if not vol30.empty else None,
        "annualised_vol_pct": round(ann_vol, 4) if ann_vol is not None else None,
        "crisis_threshold_pct": crisis_threshold,
        "n_crisis_days_30d": int((pct30 > crisis_threshold).sum()),
        "crisis_coverage_pct": round(
            (pct30 > crisis_threshold).sum() / len(pct30) * 100, 2
        ) if not pct30.empty else None,
        "oil_mean_usd": round(float(sub["oil_brent_usd"].dropna().mean()), 4)
        if not sub["oil_brent_usd"].dropna().empty else None,
        "yield_us10y_mean": round(float(sub["yield_us10y"].dropna().mean()), 4)
        if not sub["yield_us10y"].dropna().empty else None,
    }

    if not pct30.empty:
        stats["date_max_30d_depreciation"] = str(pct30.idxmax().date())
        stats["date_max_30d_appreciation"] = str(pct30.idxmin().date())

    return stats


def _safe_mean(s: pd.Series) -> Optional[float]:
    v = s.dropna()
    return round(float(v.mean()), 6) if not v.empty else None


def _safe_max(s: pd.Series) -> Optional[float]:
    v = s.dropna()
    return round(float(v.max()), 6) if not v.empty else None


def _safe_min(s: pd.Series) -> Optional[float]:
    v = s.dropna()
    return round(float(v.min()), 6) if not v.empty else None


def analyse_country(
    iso3: str,
    sub: pd.DataFrame,
    crisis_threshold: float,
    save_csv: bool,
) -> None:
    out_dir = PROCESSED_DIR / iso3
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = out_dir / f"{iso3}_daily.parquet"
    sub.to_parquet(parquet_path, index=False)
    if save_csv:
        sub.to_csv(out_dir / f"{iso3}_daily.csv", index=False)

    sub_with_attr = sub.copy()
    for _, grp in sub_with_attr.groupby(sub_with_attr["date"].dt.year):
        grp.attrs["crisis_threshold"] = crisis_threshold

    ann = _annual_summary(sub)
    if "fx_pct_30d" in sub.columns:
        crisis_by_year = (
            sub.assign(year=sub["date"].dt.year)
            .groupby("year")["fx_pct_30d"]
            .apply(lambda s: int((s > crisis_threshold).sum()))
            .rename("n_crisis_days_30d")
        )
        ann = ann.drop(columns=["n_crisis_days_30d"], errors="ignore")
        ann = ann.merge(crisis_by_year.reset_index(), on="year", how="left")

    ann_path = out_dir / f"{iso3}_annual_summary.csv"
    ann.to_csv(ann_path, index=False)

    stats = _summary_stats(sub, iso3, crisis_threshold)
    json_path = out_dir / f"{iso3}_summary_stats.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, default=str)

    logger.info(
        "  [%s] daily=%d rows | coverage=%.0f%% | crisis_days=%d | saved to %s",
        iso3,
        len(sub),
        stats.get("data_coverage_pct") or 0,
        stats.get("n_crisis_days_30d") or 0,
        out_dir.relative_to(MARKETS_ROOT),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-country analysis of the markets daily panel."
    )
    p.add_argument(
        "--panel", type=Path, default=None,
        help="Path to processed panel parquet (auto-detected if omitted)",
    )
    p.add_argument(
        "--crisis-threshold", type=float, default=5.0, metavar="PCT",
        help="30d FX depreciation %% above which a day is flagged as 'crisis' (default: 5)",
    )
    p.add_argument(
        "--no-csv", action="store_true",
        help="Skip writing CSV files (write parquet + JSON only)",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("Markets per-country analyser")
    logger.info("Crisis threshold: %.1f%%  30d FX depreciation", args.crisis_threshold)

    panel = load_panel(args.panel)
    panel["date"] = pd.to_datetime(panel["date"])

    countries = sorted(panel["country_iso3"].unique())
    logger.info("Countries found: %d", len(countries))
    logger.info("=" * 60)

    for iso3 in countries:
        sub = panel[panel["country_iso3"] == iso3].copy().reset_index(drop=True)
        analyse_country(iso3, sub, args.crisis_threshold, save_csv=not args.no_csv)

    logger.info("=" * 60)
    logger.info(
        "Done. Per-country outputs written to: %s",
        (PROCESSED_DIR).relative_to(MARKETS_ROOT),
    )


if __name__ == "__main__":
    main()
