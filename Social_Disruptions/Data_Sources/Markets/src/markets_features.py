from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods) * 100.0


def _log_return(series: pd.Series) -> pd.Series:
    shifted = series.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(series / shifted)
    return pd.Series(lr, index=series.index, name="log_return")


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    # Require at least half the window to compute a value — avoids NaN for most of the series
    # while still being a bit conservative at the very start
    min_periods = max(2, window // 2)
    return series.rolling(window=window, min_periods=min_periods).std()


def build_fx_features(
    raw: pd.Series,
    pct_windows: List[int],
    vol_windows: List[int],
) -> pd.DataFrame:
    if raw.empty:
        logger.warning("build_fx_features received empty series")
        return pd.DataFrame()

    out = pd.DataFrame(index=raw.index)
    out["fx_lcu_usd"] = raw.values

    log_ret = _log_return(raw)
    out["fx_log_return"] = log_ret.values

    for w in sorted(pct_windows):
        out[f"fx_pct_{w}d"] = _pct_change(raw, w).values

    for w in sorted(vol_windows):
        out[f"fx_vol_{w}d"] = _rolling_std(log_ret, w).values

    return out


def build_oil_features(
    raw: pd.Series,
    pct_windows: List[int],
) -> pd.DataFrame:
    if raw.empty:
        logger.warning("build_oil_features received empty series")
        return pd.DataFrame()

    out = pd.DataFrame(index=raw.index)
    out["oil_brent_usd"] = raw.values

    for w in sorted(pct_windows):
        out[f"oil_brent_pct_{w}d"] = _pct_change(raw, w).values

    return out


def build_yield_features(
    us10y: Optional[pd.Series],
    local10y: Optional[pd.Series],
) -> pd.DataFrame:
    parts: List[pd.Series] = []

    if us10y is not None and not us10y.empty:
        parts.append(us10y.rename("yield_us10y"))

    if local10y is not None and not local10y.empty:
        parts.append(local10y.rename("yield_local10y"))

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, axis=1)

    if "yield_us10y" in out.columns and "yield_local10y" in out.columns:
        out["yield_spread_vs_us"] = out["yield_local10y"] - out["yield_us10y"]

    return out


def build_global_series_features(
    raw: pd.DataFrame,
    pct_windows: List[int],
) -> pd.DataFrame:
    """Build level + pct-change features for a multi-column global series DataFrame.

    Each column in *raw* becomes a level column plus ``{col}_pct_{w}d`` columns.
    """
    if raw.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=raw.index)
    for col in raw.columns:
        out[col] = raw[col].values
        series = raw[col]
        for w in sorted(pct_windows):
            out[f"{col}_pct_{w}d"] = _pct_change(series, w).values

    return out


def reindex_to_daily(
    df: pd.DataFrame,
    start: str,
    end: str,
    ffill_limit: int = 7,
) -> pd.DataFrame:
    full_index = pd.date_range(start=start, end=end, freq="D", name="date")
    df = df.reindex(full_index)
    if ffill_limit > 0:
        df = df.ffill(limit=ffill_limit)
    return df


def missingness_report(panel: pd.DataFrame) -> str:
    frac = panel.isna().mean().sort_values(ascending=False)
    lines = ["Missingness (fraction NaN per column):"]
    for col, val in frac.items():
        marker = "  ***" if val > 0.5 else ("  *" if val > 0.1 else "")
        lines.append(f"  {col:<35s} {val:.3f}{marker}")
    return "\n".join(lines)
