from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = ("article_id", "published_date", "event_json")


def load_input(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV or Parquet input file and validate required columns.

    Parameters
    ----------
    path:
        Absolute or relative path to a .csv, .tsv, or .parquet file.

    Returns
    -------
    DataFrame with at least the required columns; ``published_date`` coerced
    to :class:`datetime.date` and ``article_id`` cast to str.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", low_memory=False)
    else:
        raise ValueError(
            f"Unsupported format '{suffix}'. Use .csv, .tsv, or .parquet."
        )

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input missing required column(s): {missing}. "
            f"Available: {list(df.columns)}"
        )

    df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce").dt.date
    n_bad = df["published_date"].isna().sum()
    if n_bad:
        logger.warning("%d rows dropped: unparseable published_date.", n_bad)
        df = df.dropna(subset=["published_date"]).reset_index(drop=True)

    df["article_id"] = df["article_id"].astype(str)
    logger.info("Loaded %d articles from %s", len(df), path.name)
    return df


def save_output(df: pd.DataFrame, path: Path, fmt: str = "parquet") -> None:
    """Write a DataFrame to ``path`` as parquet or CSV.

    The parent directory is created if it does not exist.  The file
    extension is always forced to match ``fmt``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        out = path.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    else:
        out = path.with_suffix(".csv")
        df.to_csv(out, index=False)
    logger.info("Saved %d rows → %s", len(df), out.name)
