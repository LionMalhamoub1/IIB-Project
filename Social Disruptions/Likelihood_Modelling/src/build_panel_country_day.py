from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import missingness_report, zscore_by_country  # noqa: E402


LAG_ANNUAL_BY_1YR: bool          = True
USE_WITHIN_COUNTRY_ZSCORES: bool = True
USE_GOOGLE_TRENDS: bool          = True
CREATE_INTERACTIONS: bool        = True
USE_CPI: bool                    = True
USE_FAO: bool                    = True
USE_ILOSTAT: bool                = True
USE_GTA: bool                    = True


MOD_ROOT  = _SRC.parent
REPO_ROOT = MOD_ROOT.parent
EXT       = REPO_ROOT / "External_Databases"

ACLED_DAY_FILE = MOD_ROOT / "data" / "processed" / "acled_country_day_2017_2025.parquet"
MARKETS_FILE   = EXT / "Markets"       / "data" / "processed" / "markets_country_day_20170101_20251231.parquet"
WDI_FILE       = EXT / "WDI"           / "data" / "processed" / "wdi_country_year_2017_2025.parquet"
WGI_FILE       = EXT / "WGI"           / "data" / "processed" / "wgi_country_year_2017_2025.parquet"
GTRENDS_FILE   = EXT / "Google_Trends" / "data" / "processed" / "google_trends_country_week_2017_2025.parquet"
CPI_FILE       = EXT / "Inflation"     / "data" / "processed" / "cpi_inflation_monthly_2017_2025.parquet"
FAO_FILE       = EXT / "FAO"           / "data" / "processed" / "fao_food_price_monthly_2017_2025.parquet"
ILOSTAT_FILE   = EXT / "ILOSTAT"       / "data" / "processed" / "ilostat_country_month_2017_2025.parquet"
GTA_FILE       = EXT / "GTA"           / "data" / "processed" / "gta_country_day_20170101_20251231.parquet"

OUT_PANEL = MOD_ROOT / "data" / "interim" / "modelling_panel.parquet"


WDI_ZSCORE_COLS: list[str] = [
    "gdp_growth",
    "gdp_per_capita_growth",
    "inflation_cpi_yoy",
    "unemployment_total",
    "unemployment_youth",
]

WGI_ZSCORE_COLS: list[str] = [
    "political_stability_est",
    "voice_accountability_est",
    "government_effectiveness_est",
    "rule_of_law_est",
]

MKT_ZSCORE_COLS: list[str] = ["fx_pct_30d", "fx_vol_30d", "oil_brent_pct_30d"]

GTRENDS_INDEX_COLS: list[str] = [
    "economic_stress_index",
    "labour_conflict_index",
    "protest_mobilisation_index",
]

CPI_ZSCORE_COLS: list[str] = ["food_cpi_inflation", "energy_cpi_inflation"]

FAO_YOY_COLS: list[str] = [
    "fao_food_index_yoy",
    "fao_cereals_index_yoy",
    "fao_oils_index_yoy",
]

FAO_LAG_MONTHS: list[int] = [1, 3, 6]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


logger = logging.getLogger(__name__)


def load_acled_day() -> pd.DataFrame:
    df = pd.read_parquet(ACLED_DAY_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country_iso3", "date"]).reset_index(drop=True)
    logger.info("ACLED: %d rows | %d countries.", len(df), df["country_iso3"].nunique())
    return df


def add_acled_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["country_iso3", "date"])

    def _roll(s: pd.Series, w: int) -> pd.Series:
        return s.rolling(w, min_periods=1).sum()

    for col, windows in [
        ("acled_events",      [7, 28]),
        ("riot_events",       [7, 28]),
        ("violence_events",   [7, 28]),
        ("protest_fatalities",[7, 28]),
    ]:
        if col not in df.columns:
            continue
        base = col.replace("_events", "").replace("_fatalities", "_fat")
        for w in windows:
            label = f"{base}_{w}d_lag"
            df[label] = df.groupby("country_iso3")[col].transform(
                lambda s, w=w: _roll(s, w).shift(1)
            )
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    month = df["date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    return df


def load_markets() -> pd.DataFrame:
    df = pd.read_parquet(MARKETS_FILE)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country_iso3", "date"])
    # Shift all market features by 1 day per country so day T only sees day T-1 data
    value_cols = [c for c in df.columns if c not in ("country_iso3", "date")]
    df[value_cols] = df.groupby("country_iso3")[value_cols].shift(1)
    logger.info("Markets: %d rows | %d countries.", len(df), df["country_iso3"].nunique())
    return df


def add_markets_zscores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["country_iso3", "date"])
    for col in MKT_ZSCORE_COLS:
        if col in df.columns:
            df[f"{col}_z"] = zscore_by_country(df, col, min_obs=20)
    return df


def _prep_annual(
    df: pd.DataFrame,
    zscore_cols: list[str],
    lag: bool,
) -> pd.DataFrame:
    df = df.sort_values(["country_iso3", "year"]).copy()

    if USE_WITHIN_COUNTRY_ZSCORES:
        for col in zscore_cols:
            if col in df.columns:
                df[f"{col}_z"] = zscore_by_country(df, col, min_obs=3)

    if lag:
        df["year"] = df["year"] + 1

    return df


def _expand_to_daily(df: pd.DataFrame, date_spine: pd.DatetimeIndex) -> pd.DataFrame:
    countries = df["country_iso3"].unique()
    full_mi = pd.MultiIndex.from_product(
        [countries, date_spine], names=["country_iso3", "date"]
    )
    return (
        df.set_index(["country_iso3", "date"])
        .reindex(full_mi)
        .groupby(level="country_iso3")
        .ffill()
        .reset_index()
    )


def _expand_annual_to_daily(
    annual: pd.DataFrame,
    date_spine: pd.DatetimeIndex,
) -> pd.DataFrame:
    annual = annual.copy()
    annual["date"] = pd.to_datetime(annual["year"].astype(str) + "-01-01")
    annual = annual.drop(columns="year")
    return _expand_to_daily(annual, date_spine)


def load_wdi_wgi(date_spine: pd.DatetimeIndex) -> pd.DataFrame:
    wdi = pd.read_parquet(WDI_FILE)
    wgi = pd.read_parquet(WGI_FILE)
    logger.info("WDI: %d rows | WGI: %d rows.", len(wdi), len(wgi))

    wdi = _prep_annual(wdi, WDI_ZSCORE_COLS, lag=LAG_ANNUAL_BY_1YR)
    wgi = _prep_annual(wgi, WGI_ZSCORE_COLS, lag=LAG_ANNUAL_BY_1YR)

    annual = wdi.merge(wgi, on=["country_iso3", "year"], how="outer")
    return _expand_annual_to_daily(annual, date_spine)


def load_google_trends() -> pd.DataFrame | None:
    if not GTRENDS_FILE.exists():
        logger.warning("Google Trends file not found: %s — skipping.", GTRENDS_FILE)
        return None

    gt = pd.read_parquet(GTRENDS_FILE)
    gt = gt.rename(columns={"week": "date"})
    gt["date"] = pd.to_datetime(gt["date"])

    keep_cols = [c for c in GTRENDS_INDEX_COLS if c in gt.columns]
    gt = gt[["country_iso3", "date"] + keep_cols].sort_values(["country_iso3", "date"])
    logger.info(
        "Google Trends: %d rows | %d countries | cols: %s.",
        len(gt), gt["country_iso3"].nunique(), keep_cols,
    )
    return gt


def _expand_trends_to_daily(
    gt: pd.DataFrame,
    date_spine: pd.DatetimeIndex,
) -> pd.DataFrame:
    gt = gt.sort_values(["country_iso3", "date"])
    keep_cols = [c for c in GTRENDS_INDEX_COLS if c in gt.columns]

    if USE_WITHIN_COUNTRY_ZSCORES:
        for col in keep_cols:
            gt[f"{col}_z"] = zscore_by_country(gt, col, min_obs=10)

    # Shift by 1 week per country so day T only sees last week's trends
    all_value_cols = keep_cols + [f"{c}_z" for c in keep_cols if f"{c}_z" in gt.columns]
    gt[all_value_cols] = gt.groupby("country_iso3")[all_value_cols].shift(1)

    return _expand_to_daily(gt, date_spine)


def load_cpi(date_spine: pd.DatetimeIndex) -> pd.DataFrame | None:
    if not CPI_FILE.exists():
        logger.warning("CPI file not found: %s — skipping.", CPI_FILE)
        return None

    cpi = pd.read_parquet(CPI_FILE)
    cpi["date"] = pd.to_datetime(cpi["date"])
    cpi = cpi.sort_values(["country_iso3", "date"])
    logger.info("CPI: %d rows | %d countries.", len(cpi), cpi["country_iso3"].nunique())

    if USE_WITHIN_COUNTRY_ZSCORES:
        for col in CPI_ZSCORE_COLS:
            if col in cpi.columns:
                cpi[f"{col}_z"] = zscore_by_country(cpi, col, min_obs=6)

    # Shift by 1 month per country so day T only sees last month's CPI
    value_cols = [c for c in cpi.columns if c not in ("country_iso3", "date")]
    cpi[value_cols] = cpi.groupby("country_iso3")[value_cols].shift(1)

    return _expand_to_daily(cpi, date_spine)


def load_ilostat(date_spine: pd.DatetimeIndex) -> pd.DataFrame | None:
    if not ILOSTAT_FILE.exists():
        logger.warning("ILOSTAT file not found: %s — skipping.", ILOSTAT_FILE)
        return None

    ilo = pd.read_parquet(ILOSTAT_FILE)
    ilo["date"] = pd.to_datetime(ilo["date"])
    ilo = ilo.sort_values(["country_iso3", "date"])

    keep = ["unemployment_sa", "unemployment_rate", "earnings_monthly"]
    keep = [c for c in keep if c in ilo.columns]
    ilo  = ilo[["country_iso3", "date"] + keep]

    if USE_WITHIN_COUNTRY_ZSCORES:
        for col in keep:
            ilo[f"{col}_z"] = zscore_by_country(ilo, col, min_obs=6)

    # Shift by 1 month per country so day T only sees last month's ILOSTAT data
    value_cols = [c for c in ilo.columns if c not in ("country_iso3", "date")]
    ilo[value_cols] = ilo.groupby("country_iso3")[value_cols].shift(1)

    logger.info("ILOSTAT: %d rows | %d countries.", len(ilo), ilo["country_iso3"].nunique())
    return _expand_to_daily(ilo, date_spine)


def load_fao(date_spine: pd.DatetimeIndex) -> pd.DataFrame | None:
    if not FAO_FILE.exists():
        logger.warning("FAO file not found: %s — skipping.", FAO_FILE)
        return None

    fao = pd.read_parquet(FAO_FILE)
    fao["date"] = pd.to_datetime(fao["date"])
    fao = fao.sort_values("date").reset_index(drop=True)

    present_yoy = [c for c in FAO_YOY_COLS if c in fao.columns]

    # Threshold indicators: 1 if YoY above expanding 90th-percentile (no look-ahead)
    for col in present_yoy:
        fao[f"{col}_above90"] = (
            fao[col] > fao[col].expanding(min_periods=12).quantile(0.90)
        ).astype(float)

    # Lagged YoY columns
    for lag in FAO_LAG_MONTHS:
        for col in present_yoy:
            fao[f"{col}_lag{lag}m"] = fao[col].shift(lag)

    # Shift all FAO columns by 1 month so day T only sees last month's data
    value_cols = [c for c in fao.columns if c != "date"]
    fao[value_cols] = fao[value_cols].shift(1)

    # Expand monthly to daily by date (no country dimension — broadcast to all)
    date_df   = pd.DataFrame({"date": date_spine})
    fao_daily = date_df.merge(fao, on="date", how="left")
    fao_cols  = [c for c in fao_daily.columns if c != "date"]
    fao_daily[fao_cols] = fao_daily[fao_cols].ffill()

    logger.info(
        "FAO: %d daily rows | %d series (incl. lags & thresholds).",
        len(fao_daily), len(fao_cols),
    )
    return fao_daily


def load_gta() -> pd.DataFrame | None:
    if not GTA_FILE.exists():
        logger.warning("GTA file not found: %s — skipping.", GTA_FILE)
        return None

    gta = pd.read_parquet(GTA_FILE)
    gta["date"] = pd.to_datetime(gta["date"])
    gta = gta.sort_values(["country_iso3", "date"])

    keep = ["gta_harmful_events", "gta_liberalising_events", "gta_30d_count", "gta_90d_count"]
    keep = [c for c in keep if c in gta.columns]
    gta = gta[["country_iso3", "date"] + keep]

    if USE_WITHIN_COUNTRY_ZSCORES:
        for col in keep:
            gta[f"{col}_z"] = zscore_by_country(gta, col, min_obs=10)

    # Shift by 1 day per country so day T only sees day T-1 data
    value_cols = [c for c in gta.columns if c not in ("country_iso3", "date")]
    gta[value_cols] = gta.groupby("country_iso3")[value_cols].shift(1)

    logger.info("GTA: %d rows | %d countries.", len(gta), gta["country_iso3"].nunique())
    return gta


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    if "fx_pct_30d" in df.columns and "political_stability_est" in df.columns:
        instability = 1.0 - df["political_stability_est"].clip(-3.0, 3.0)
        df["fx_pct_30d_x_instability"] = df["fx_pct_30d"] * instability

    if "oil_brent_pct_30d" in df.columns and "inflation_cpi_yoy" in df.columns:
        df["oil_brent_pct_30d_x_inflation"] = (
            df["oil_brent_pct_30d"] * df["inflation_cpi_yoy"]
        )

    # Cereals price stress interacted with governance (Hendrix & Haggard 2015)
    if "fao_cereals_index_yoy" in df.columns and "political_stability_est" in df.columns:
        instability = 1.0 - df["political_stability_est"].clip(-3.0, 3.0)
        df["fao_cereals_yoy_x_instability"] = df["fao_cereals_index_yoy"] * instability

    # Food price stress interacted with youth unemployment
    if "fao_food_index_yoy" in df.columns and "unemployment_youth" in df.columns:
        df["fao_food_yoy_x_youth_unemp"] = (
            df["fao_food_index_yoy"] * df["unemployment_youth"]
        )

    return df


def main() -> None:
    _setup_logging()

    logger.info("=" * 60)
    logger.info("Modelling Panel Builder  v4")
    logger.info("  LAG_ANNUAL_BY_1YR          : %s", LAG_ANNUAL_BY_1YR)
    logger.info("  USE_WITHIN_COUNTRY_ZSCORES : %s", USE_WITHIN_COUNTRY_ZSCORES)
    logger.info("  USE_GOOGLE_TRENDS          : %s", USE_GOOGLE_TRENDS)
    logger.info("  CREATE_INTERACTIONS        : %s", CREATE_INTERACTIONS)
    logger.info("  USE_CPI                    : %s", USE_CPI)
    logger.info("  USE_FAO                    : %s", USE_FAO)
    logger.info("  USE_ILOSTAT                : %s", USE_ILOSTAT)
    logger.info("  USE_GTA                    : %s", USE_GTA)
    logger.info("=" * 60)

    panel = load_acled_day()
    panel = add_acled_lags(panel)
    panel = add_temporal_features(panel)

    date_spine = pd.DatetimeIndex(panel["date"].unique(), name="date").sort_values()

    markets = load_markets()
    if USE_WITHIN_COUNTRY_ZSCORES:
        markets = add_markets_zscores(markets)
    panel = panel.merge(markets, on=["country_iso3", "date"], how="left")
    logger.info("After Markets merge  : %d rows.", len(panel))

    annual_daily = load_wdi_wgi(date_spine)
    panel = panel.merge(annual_daily, on=["country_iso3", "date"], how="left")
    logger.info("After WDI/WGI merge  : %d rows.", len(panel))

    if USE_GOOGLE_TRENDS:
        gt_raw = load_google_trends()
        if gt_raw is not None:
            gt_daily = _expand_trends_to_daily(gt_raw, date_spine)
            panel = panel.merge(gt_daily, on=["country_iso3", "date"], how="left")
            logger.info("After Trends merge   : %d rows.", len(panel))

    if USE_CPI:
        cpi_daily = load_cpi(date_spine)
        if cpi_daily is not None:
            panel = panel.merge(cpi_daily, on=["country_iso3", "date"], how="left")
            logger.info("After CPI merge      : %d rows.", len(panel))

    if USE_FAO:
        fao_daily = load_fao(date_spine)
        if fao_daily is not None:
            panel = panel.merge(fao_daily, on="date", how="left")
            logger.info("After FAO merge      : %d rows.", len(panel))

    if USE_ILOSTAT:
        ilo_daily = load_ilostat(date_spine)
        if ilo_daily is not None:
            panel = panel.merge(ilo_daily, on=["country_iso3", "date"], how="left")
            logger.info("After ILOSTAT merge  : %d rows.", len(panel))

    if USE_GTA:
        gta = load_gta()
        if gta is not None:
            panel = panel.merge(gta, on=["country_iso3", "date"], how="left")
            logger.info("After GTA merge      : %d rows.", len(panel))

    if CREATE_INTERACTIONS:
        panel = add_interactions(panel)

    panel = panel.sort_values(["country_iso3", "date"]).reset_index(drop=True)

    logger.info(
        "\nMissingness report (top 15 columns by %% missing):\n%s",
        missingness_report(panel).head(15).to_string(),
    )

    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PANEL, index=False)

    logger.info("Saved -> %s", OUT_PANEL)
    logger.info(
        "Panel: %d rows x %d cols | %d countries | %s to %s",
        len(panel), len(panel.columns),
        panel["country_iso3"].nunique(),
        str(panel["date"].min().date()),
        str(panel["date"].max().date()),
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
