from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_SRC      = Path(__file__).resolve().parent
MOD_ROOT  = _SRC.parent
REPO_ROOT = MOD_ROOT.parent

ACLED_EVENTS_DIR: Path = (
    REPO_ROOT / "External_Databases" / "ACLED" / "data" / "raw" / "events"
)

START_DATE: str = "2017-01-01"
END_DATE:   str = "2025-12-31"

PROTEST_TYPES:  frozenset[str] = frozenset({"Protests"})
RIOT_TYPES:     frozenset[str] = frozenset({"Riots"})
VIOLENCE_TYPES: frozenset[str] = frozenset({"Violence against civilians", "Battles", "Explosions/Remote violence"})

OUT_PARQUET: Path = MOD_ROOT / "data" / "processed" / "acled_country_day_2017_2025.parquet"
OUT_CSV:     Path = MOD_ROOT / "data" / "processed" / "acled_country_day_2017_2025.csv"


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)


def load_events(events_dir: Path) -> pd.DataFrame:
    files = sorted(events_dir.glob("iso3=*/year=*.parquet"))
    if not files:
        raise FileNotFoundError(f"No ACLED event parquet files found in: {events_dir}")
    frames = [pd.read_parquet(f) for f in files]
    events = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d raw events from %d partition files.", len(events), len(files))
    return events


def build_country_day(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["date"] = pd.to_datetime(events["event_date"], errors="coerce")
    events = events.dropna(subset=["date", "iso3"])
    events = events.rename(columns={"iso3": "country_iso3"})
    events["country_iso3"] = events["country_iso3"].str.strip().str.upper()

    start = pd.Timestamp(START_DATE)
    end   = pd.Timestamp(END_DATE)
    events = events[(events["date"] >= start) & (events["date"] <= end)]

    protests = events[events["event_type"].isin(PROTEST_TYPES)]

    def _count(df: pd.DataFrame, col: str) -> pd.Series:
        return df.groupby(["country_iso3", "date"]).size().rename(col)

    agg = (
        _count(protests, "acled_events")
        .to_frame()
        .fillna(0)
        .astype(int)
        .reset_index()
    )

    full_index = pd.date_range(START_DATE, END_DATE, freq="D", name="date")
    countries  = sorted(agg["country_iso3"].unique())
    full_mi    = pd.MultiIndex.from_product(
        [countries, full_index], names=["country_iso3", "date"]
    )
    panel = (
        agg.set_index(["country_iso3", "date"])
        .reindex(full_mi)
        .fillna(0)
        .astype(int)
        .reset_index()
        .sort_values(["country_iso3", "date"])
        .reset_index(drop=True)
    )
    logger.info(
        "Panel: %d rows | %d countries | columns: %s",
        len(panel), panel["country_iso3"].nunique(), panel.columns.tolist(),
    )
    return panel


def main() -> None:
    _setup_logging()

    logger.info("=" * 60)
    logger.info("ACLED Country-Day Panel Builder")
    logger.info("Protest types  : %s", sorted(PROTEST_TYPES))
    logger.info("Riot types     : %s", sorted(RIOT_TYPES))
    logger.info("Violence types : %s", sorted(VIOLENCE_TYPES))
    logger.info("Period      : %s to %s", START_DATE, END_DATE)
    logger.info("=" * 60)

    events = load_events(ACLED_EVENTS_DIR)
    panel  = build_country_day(events)

    n_countries = panel["country_iso3"].nunique()
    n_days      = panel["date"].nunique()
    density     = (panel["acled_events"] > 0).mean() * 100

    logger.info(
        "Panel: %d rows | %d countries | %d calendar days each.",
        len(panel), n_countries, n_days,
    )
    logger.info("Event density: %.2f%% of country-days have >= 1 event.", density)

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PARQUET, index=False)
    panel.to_csv(OUT_CSV, index=False)

    logger.info("Saved -> %s", OUT_PARQUET)
    logger.info("Done.")


if __name__ == "__main__":
    main()
