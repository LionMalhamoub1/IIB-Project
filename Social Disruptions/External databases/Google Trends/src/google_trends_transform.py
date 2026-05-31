from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def merge_batches(batch_dfs: List[pd.DataFrame], anchor: str) -> pd.DataFrame:
    if not batch_dfs:
        return pd.DataFrame()
    if len(batch_dfs) == 1:
        return batch_dfs[0].copy().astype(float)

    reference = batch_dfs[0].copy().astype(float)

    for raw in batch_dfs[1:]:
        curr = raw.copy().astype(float)
        common_dates = reference.index.intersection(curr.index)

        if anchor in reference.columns and anchor in curr.columns and len(common_dates) > 0:
            ref_sum  = reference.loc[common_dates, anchor].sum()
            curr_sum = curr.loc[common_dates, anchor].sum()
            if curr_sum > 1e-9:
                scale = ref_sum / curr_sum
                for col in curr.columns:
                    if col != anchor:
                        curr[col] *= scale

        for col in curr.columns:
            if col != anchor and col not in reference.columns:
                reference[col] = curr[col].reindex(reference.index)

    return reference


def stitch_windows(
    windows: List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]],
) -> pd.DataFrame:
    if not windows:
        return pd.DataFrame()

    windows = sorted(windows, key=lambda x: x[0])
    result = windows[0][2].copy().astype(float)

    for _, _, raw in windows[1:]:
        curr = raw.copy().astype(float)
        overlap = result.index.intersection(curr.index)

        if len(overlap) >= 3:
            shared_cols = [c for c in result.columns if c in curr.columns]
            ref_means  = result.loc[overlap, shared_cols].mean()
            curr_means = curr.loc[overlap, shared_cols].mean()

            for col in shared_cols:
                if curr_means[col] > 1e-9:
                    curr[col] *= ref_means[col] / curr_means[col]

            result.loc[overlap, shared_cols] = (
                result.loc[overlap, shared_cols].values
                + curr.loc[overlap, shared_cols].values
            ) / 2

        new_dates = curr.index.difference(result.index)
        if len(new_dates):
            result = pd.concat([result, curr.loc[new_dates]]).sort_index()

    return result


def add_anomaly_scores(
    df: pd.DataFrame,
    keyword_cols: List[str],
    window: int = 12,
) -> pd.DataFrame:
    out = df.copy()
    min_periods = max(1, window // 2)

    for col in keyword_cols:
        series = out[col].astype(float)
        roll_mean = series.rolling(window=window, min_periods=min_periods).mean()
        roll_std  = series.rolling(window=window, min_periods=min_periods).std()
        roll_std  = roll_std.replace(0, np.nan)
        out[f"{col}_zscore"] = (series - roll_mean) / roll_std

    return out
