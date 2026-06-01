from __future__ import annotations

import csv
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

_SRC_DIR      = Path(__file__).resolve().parent
_ILOSTAT_ROOT = _SRC_DIR.parent

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ilostat_client import ILOSTATClient, ILOSTATClientConfig  # noqa: E402

START_YEAR  = 2017
END_YEAR    = 2025
COUNTRIES   = "ALL"
MAX_WORKERS = 6
OUT_FORMAT  = "parquet"
USE_CACHE   = True
LOG_LEVEL   = "INFO"

# ---------------------------------------------------------------------------
# Sub-annual target indicators: feature_name -> (monthly_id, quarterly_id)
# For each feature the pipeline uses the highest-frequency series available
# for a given country (monthly > quarterly > annual forward-fill).
# ---------------------------------------------------------------------------
SUBANNUAL_INDICATORS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "cpi_yoy":           ("CPI_NCYR_COI_RT_M",     "CPI_NCYR_COI_RT_Q"),
    "cpi_mom":           ("CPI_NCPD_COI_RT_M",     "CPI_NCPD_COI_RT_Q"),
    "unemployment_rate": ("UNE_DEAP_SEX_AGE_RT_M", "UNE_DEAP_SEX_AGE_RT_Q"),
    "unemployment_sa":   ("UNE_DEA1_SEX_AGE_RT_M", "UNE_DEA1_SEX_AGE_RT_Q"),
    "earnings_monthly":  ("EAR_EMTA_SEX_ECO_NB_M", None),
}

KEYWORDS_FILE  = str(_ILOSTAT_ROOT / "config" / "keywords.txt")
ALLOWLIST_FILE = None
DENYLIST_FILE  = str(_ILOSTAT_ROOT / "config" / "denylist.txt")

_CACHE_DIR = _ILOSTAT_ROOT / ".cache"
_RAW_DIR   = _ILOSTAT_ROOT / "data" / "raw"
_REF_DIR   = _ILOSTAT_ROOT / "data" / "reference"
_PROC_DIR  = _ILOSTAT_ROOT / "data" / "processed"

for _d in (_CACHE_DIR, _RAW_DIR, _REF_DIR, _PROC_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )

logger = logging.getLogger(__name__)

DEFAULT_KEYWORDS: List[str] = [
    "strike",
    "lockout",
    "industrial dispute",
    "union",
    "collective bargaining",
    "unemployment",
    "youth unemployment",
    "underemployment",
    "labour force participation",
    "informal",
    "hours worked",
    "employment by sector",
    "employment rate",
    "inactivity",
    "wage",
    "earnings",
    "real wage",
    "working poverty",
    "minimum wage",
    "vulnerable employment",
    "temporary",
    "self-employed",
    "neet",
    "part-time",
    "precarious",
    "social protection",
    "social security",
]


def filter_indicators(
    toc:       pd.DataFrame,
    keywords:  List[str],
    allowlist: Optional[Set[str]] = None,
    denylist:  Optional[Set[str]] = None,
) -> pd.DataFrame:
    allowlist = allowlist or set()
    denylist  = denylist  or set()

    label_col  = "indicator.label" if "indicator.label" in toc.columns else "label"
    extra_cols = [c for c in ("description", "indicator.description") if c in toc.columns]
    search = toc[label_col].astype(str)
    for col in extra_cols:
        search = search + " " + toc[col].astype(str)

    patterns      = [re.compile(re.escape(kw), re.IGNORECASE) for kw in keywords]
    ids           = toc["id"].astype(str)

    exact_deny  = {d for d in denylist if not d.endswith("*")}
    prefix_deny = [d[:-1] for d in denylist if d.endswith("*")]
    mask_deny   = ids.isin(exact_deny)
    for pfx in prefix_deny:
        mask_deny = mask_deny | ids.str.startswith(pfx)

    mask_allow    = ids.isin(allowlist)
    mask_keywords = search.apply(lambda t: any(p.search(t) for p in patterns))

    selected = toc[(~mask_deny) & (mask_allow | mask_keywords)].copy()

    logger.info(
        "Indicator filter: %d total → %d selected  "
        "(keyword: %d, allowlisted: %d, denied: %d)",
        len(toc), len(selected),
        int((mask_keywords & ~mask_deny & ~mask_allow).sum()),
        int((mask_allow & ~mask_deny).sum()),
        int(mask_deny.sum()),
    )
    return selected.reset_index(drop=True)


def _fetch_one(
    client:     ILOSTATClient,
    dataset_id: str,
    use_cache:  bool,
) -> Tuple[str, Optional[pd.DataFrame]]:
    try:
        return dataset_id, client.fetch_indicator(dataset_id, use_cache=use_cache)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping %s — download failed: %s", dataset_id, exc)
        return dataset_id, None


def download_all(
    client:      ILOSTATClient,
    dataset_ids: List[str],
    max_workers: int  = 8,
    use_cache:   bool = True,
) -> Dict[str, pd.DataFrame]:
    results: Dict[str, pd.DataFrame] = {}
    n_failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_fetch_one, client, iid, use_cache): iid
            for iid in dataset_ids
        }
        with tqdm(
            total=len(futures),
            unit=" datasets",
            desc="Downloading indicators",
            dynamic_ncols=True,
        ) as pbar:
            for fut in as_completed(futures):
                iid, df = fut.result()
                if df is not None:
                    results[iid] = df
                else:
                    n_failed += 1
                pbar.update(1)
                pbar.set_postfix(ok=len(results), failed=n_failed)

    logger.info(
        "Download complete: %d succeeded, %d failed.", len(results), n_failed
    )
    return results


_COL_AREA   = ("ref_area",   "REF_AREA")
_COL_TIME   = ("time",       "TIME_PERIOD")
_COL_VALUE  = ("obs_value",  "OBS_VALUE")
_COL_SEX    = ("sex",        "SEX")
_COL_CL1    = ("classif1",   "CLASSIF1")
_COL_CL2    = ("classif2",   "CLASSIF2")
_COL_STATUS = ("obs_status", "OBS_STATUS")
_COL_UNIT   = ("unit",       "UNIT_MEASURE", "unit_measure")
_COL_FREQ   = ("freq",       "FREQ")


def _find_col(df: pd.DataFrame, candidates: tuple) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_col(df: pd.DataFrame, candidates: tuple) -> pd.Series:
    col = _find_col(df, candidates)
    if col:
        return df[col].astype(str).reset_index(drop=True)
    return pd.Series("", index=range(len(df)))


def build_long_table(
    raw:        Dict[str, pd.DataFrame],
    toc:        pd.DataFrame,
    start_year: int,
    end_year:   int,
    countries:  Optional[Set[str]] = None,
) -> pd.DataFrame:
    label_col   = "indicator.label" if "indicator.label" in toc.columns else "label"
    id_to_label = dict(zip(toc["id"].astype(str), toc[label_col].astype(str)))

    frames: List[pd.DataFrame] = []

    for iid, df in raw.items():
        if df.empty:
            continue

        area_col  = _find_col(df, _COL_AREA)
        time_col  = _find_col(df, _COL_TIME)
        value_col = _find_col(df, _COL_VALUE)

        if not (area_col and time_col and value_col):
            logger.warning(
                "%s: missing area/time/value column — skipping. Found: %s",
                iid, list(df.columns[:15]),
            )
            continue

        df = df.copy().reset_index(drop=True)

        time_str  = df[time_col].astype(str).str.slice(0, 4)
        is_annual = time_str.str.match(r"^\d{4}$", na=False)
        df        = df[is_annual].copy().reset_index(drop=True)
        if df.empty:
            continue

        df["_year"] = time_str[is_annual].astype(int).values

        df = df[(df["_year"] >= start_year) & (df["_year"] <= end_year)].reset_index(drop=True)
        if df.empty:
            continue

        if countries:
            df = df[df[area_col].astype(str).isin(countries)].reset_index(drop=True)
        if df.empty:
            continue

        df["_value"] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["_value"]).reset_index(drop=True)
        if df.empty:
            continue

        n = len(df)
        out = pd.DataFrame(
            {
                "indicator_id":    [iid] * n,
                "indicator_label": [id_to_label.get(iid, "")] * n,
                "ref_area":        df[area_col].astype(str).values,
                "year":            df["_year"].values,
                "sex":             _get_col(df, _COL_SEX).values,
                "classif1":        _get_col(df, _COL_CL1).values,
                "classif2":        _get_col(df, _COL_CL2).values,
                "value":           df["_value"].values,
                "unit":            _get_col(df, _COL_UNIT).values,
                "freq":            _get_col(df, _COL_FREQ).values,
                "obs_status":      _get_col(df, _COL_STATUS).values,
            }
        )
        frames.append(out)

    if not frames:
        logger.warning("Long table: no records survived cleaning.")
        return pd.DataFrame(
            columns=[
                "indicator_id", "indicator_label", "ref_area", "year",
                "sex", "classif1", "classif2", "value", "unit",
                "freq", "obs_status",
            ]
        )

    long = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["ref_area", "indicator_id", "year"])
        .reset_index(drop=True)
    )
    logger.info("[Long table] %d rows × %d cols", *long.shape)
    return long


def build_wide_panel(long: pd.DataFrame) -> pd.DataFrame:
    if long.empty:
        return pd.DataFrame()

    df = long.copy()

    sex_total = (df["sex"] == "SEX_T").astype(int)
    cl1_total = (
        df["classif1"].str.upper().str.contains("TOTAL", na=False)
        | (df["classif1"].fillna("") == "")
    ).astype(int)
    df["_priority"] = sex_total * 2 + cl1_total

    group_keys = ["ref_area", "year", "indicator_id"]
    max_pri    = df.groupby(group_keys)["_priority"].transform("max")

    agg = (
        df[df["_priority"] == max_pri]
        .groupby(group_keys)["value"]
        .mean()
        .reset_index()
    )

    wide = agg.pivot_table(
        index=["ref_area", "year"],
        columns="indicator_id",
        values="value",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    wide = wide.sort_values(["ref_area", "year"]).reset_index(drop=True)

    logger.info(
        "[Wide panel] %d rows × %d cols  (%d countries, %d indicator columns)",
        len(wide), len(wide.columns),
        wide["ref_area"].nunique(),
        len(wide.columns) - 2,
    )
    return wide


def _parse_time_period(time_series: pd.Series) -> pd.DataFrame:
    """Parse ILOSTAT time strings into (year, month) integer columns.

    Handles:
      - Annual   "2023"       → year=2023, month=NaN
      - Monthly  "2023M01"    → year=2023, month=1
      - Quarterly"2023Q2"     → year=2023, month=4  (first month of quarter)
    Returns a DataFrame with columns [year, month, freq_rank] where
    freq_rank: 3=monthly, 2=quarterly, 1=annual (higher = more granular).
    """
    s = time_series.astype(str).str.strip()

    monthly_mask   = s.str.match(r"^\d{4}M\d{2}$", na=False)
    quarterly_mask = s.str.match(r"^\d{4}Q[1-4]$",  na=False)
    annual_mask    = s.str.match(r"^\d{4}$",          na=False)

    year  = pd.Series(pd.NA, index=s.index, dtype="Int64")
    month = pd.Series(pd.NA, index=s.index, dtype="Int64")
    rank  = pd.Series(0,     index=s.index, dtype="Int64")

    # Monthly
    year[monthly_mask]  = s[monthly_mask].str[:4].astype(int)
    month[monthly_mask] = s[monthly_mask].str[5:].astype(int)
    rank[monthly_mask]  = 3

    # Quarterly — map Q1→1, Q2→4, Q3→7, Q4→10
    q_map = {"1": 1, "2": 4, "3": 7, "4": 10}
    year[quarterly_mask]  = s[quarterly_mask].str[:4].astype(int)
    month[quarterly_mask] = s[quarterly_mask].str[5].map(q_map).astype("Int64")
    rank[quarterly_mask]  = 2

    # Annual
    year[annual_mask] = s[annual_mask].astype(int)
    rank[annual_mask] = 1

    return pd.DataFrame({"year": year, "month": month, "freq_rank": rank})


def _select_total_sex_age(df: pd.DataFrame) -> pd.DataFrame:
    """Keep SEX_T + broadest age group rows; fall back to whatever is available."""
    sex_col = _find_col(df, _COL_SEX)
    cl1_col = _find_col(df, _COL_CL1)

    sex_t = (df[sex_col].astype(str) == "SEX_T") if sex_col else pd.Series(True, index=df.index)
    age_broad = (
        df[cl1_col].astype(str).str.upper().str.contains("YGE15|TOTAL|AGE15-64", na=False)
    ) if cl1_col else pd.Series(True, index=df.index)

    mask = sex_t & age_broad
    return df[mask].copy() if mask.any() else df.copy()


def build_subannual_long_table(
    raw:        Dict[str, pd.DataFrame],
    indicator_ids: List[str],
    start_year: int,
    end_year:   int,
    countries:  Optional[Set[str]] = None,
) -> pd.DataFrame:
    """Like build_long_table but retains monthly and quarterly time resolution."""
    frames: List[pd.DataFrame] = []

    for iid in indicator_ids:
        df = raw.get(iid)
        if df is None or df.empty:
            continue

        area_col  = _find_col(df, _COL_AREA)
        time_col  = _find_col(df, _COL_TIME)
        value_col = _find_col(df, _COL_VALUE)

        if not (area_col and time_col and value_col):
            continue

        df = _select_total_sex_age(df.copy().reset_index(drop=True))

        parsed = _parse_time_period(df[time_col])
        df["_year"]      = parsed["year"].values
        df["_month"]     = parsed["month"].values
        df["_freq_rank"] = parsed["freq_rank"].values

        df = df[df["_freq_rank"] > 0].copy()
        df = df[(df["_year"] >= start_year) & (df["_year"] <= end_year)].copy()

        if countries:
            df = df[df[area_col].astype(str).isin(countries)].copy()

        df["_value"] = pd.to_numeric(df[value_col], errors="coerce")
        df = df.dropna(subset=["_value", "_year"]).copy()

        if df.empty:
            continue

        out = pd.DataFrame({
            "indicator_id": iid,
            "ref_area":     df[area_col].astype(str).values,
            "year":         df["_year"].values,
            "month":        df["_month"].values,
            "freq_rank":    df["_freq_rank"].values,
            "value":        df["_value"].values,
        })
        frames.append(out)

    if not frames:
        logger.warning("Subannual long table: no records.")
        return pd.DataFrame(columns=["indicator_id", "ref_area", "year", "month", "freq_rank", "value"])

    long = pd.concat(frames, ignore_index=True)
    logger.info("[Subannual long] %d rows  |  indicators: %s", len(long), long["indicator_id"].unique().tolist())
    return long


def build_country_month_panel(
    subannual_long: pd.DataFrame,
    indicator_map:  Dict[str, Tuple[Optional[str], Optional[str]]],
    start_year: int,
    end_year:   int,
) -> pd.DataFrame:
    """Build a country × month panel using best-available frequency per country.

    For each feature:
      1. Use monthly series where available per country.
      2. Fall back to quarterly (expand to monthly within each quarter).
      3. As final fallback, forward-fill annual value across all months of that year.

    Output: country_iso3, year, month, date, <feature_cols...>
    """
    if subannual_long.empty:
        return pd.DataFrame()

    # Full grid of country × year-month
    all_countries = sorted(subannual_long["ref_area"].unique())
    date_index = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq="MS",
    )
    grid = pd.MultiIndex.from_product(
        [all_countries, date_index], names=["country_iso3", "date"]
    )
    panel = pd.DataFrame(index=grid).reset_index()
    panel["year"]  = panel["date"].dt.year
    panel["month"] = panel["date"].dt.month

    for feat_name, (monthly_id, quarterly_id) in indicator_map.items():
        col = pd.Series(np.nan, index=panel.index, dtype=float)

        # ── monthly ──────────────────────────────────────────────────────────
        if monthly_id:
            m_data = subannual_long[
                (subannual_long["indicator_id"] == monthly_id) &
                (subannual_long["freq_rank"] == 3)
            ].copy()
            if not m_data.empty:
                m_data = m_data.dropna(subset=["month"])
                m_data["month"] = m_data["month"].astype(int)
                m_data["year"]  = m_data["year"].astype(int)
                # Deduplicate: one value per (country, year, month)
                m_data = (m_data.groupby(["ref_area", "year", "month"])["value"]
                          .mean().reset_index()
                          .rename(columns={"ref_area": "country_iso3", "value": f"_m_{feat_name}"}))
                m_merged = panel.merge(m_data, on=["country_iso3", "year", "month"], how="left")
                col = np.asarray(m_merged[f"_m_{feat_name}"].values, dtype=float)

        # ── quarterly fallback ────────────────────────────────────────────────
        if quarterly_id:
            q_data = subannual_long[
                (subannual_long["indicator_id"] == quarterly_id) &
                (subannual_long["freq_rank"] == 2)
            ].copy()
            if not q_data.empty:
                # Deduplicate quarterly source first
                q_data = (q_data.dropna(subset=["month"])
                          .groupby(["ref_area", "year", "month"])["value"]
                          .mean().reset_index())
                # Expand each quarterly row to 3 months
                q_rows = []
                for _, row in q_data.iterrows():
                    q_start = int(row["month"])
                    for m_offset in range(3):
                        q_rows.append({
                            "country_iso3": str(row["ref_area"]),
                            "year":  int(row["year"]),
                            "month": q_start + m_offset,
                            "_q_val": float(row["value"]),
                        })
                q_expand = (pd.DataFrame(q_rows)
                            .groupby(["country_iso3", "year", "month"])["_q_val"]
                            .mean().reset_index())
                q_merged = panel.merge(q_expand, on=["country_iso3", "year", "month"], how="left")
                q_vals = np.asarray(q_merged["_q_val"].values, dtype=float)

                # Only fill where monthly was missing
                col = np.array(col, dtype=float, copy=True)
                missing = np.isnan(col)
                col[missing] = q_vals[missing]

        panel[feat_name] = col

    # Forward-fill within each country (handles gaps from sparse quarterly/annual)
    feat_cols = list(indicator_map.keys())
    panel = panel.sort_values(["country_iso3", "date"])
    panel[feat_cols] = panel.groupby("country_iso3")[feat_cols].transform(
        lambda s: s.ffill(limit=13)  # don't fill more than 13 months
    )

    panel = panel.drop(columns=["year", "month"])
    logger.info(
        "[Country-month panel] %d rows × %d cols  |  %d countries",
        len(panel), len(panel.columns), panel["country_iso3"].nunique(),
    )
    return panel


def save_reference(toc: pd.DataFrame, selected: pd.DataFrame) -> None:
    toc.to_csv(     _REF_DIR / "ilostat_toc.csv",               index=False)
    selected.to_csv(_REF_DIR / "ilostat_selected_indicators.csv", index=False)
    logger.info("Reference files written to %s", _REF_DIR)


def save_outputs(
    long: pd.DataFrame,
    wide: pd.DataFrame,
    fmt:  str,
    tag:  str = "",
    country_month: Optional[pd.DataFrame] = None,
) -> None:
    sfx = f"_{tag}" if tag else ""

    def _write(df: pd.DataFrame, stem: str) -> None:
        if df is None or df.empty:
            logger.warning("Output empty — skipping: %s", stem)
            return
        if fmt == "parquet":
            path = _PROC_DIR / f"{stem}{sfx}.parquet"
            df.to_parquet(path, index=False)
            logger.info("Saved → %s", path)
        csv_path = _PROC_DIR / f"{stem}{sfx}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved → %s", csv_path)

    _write(long, "ilostat_long")
    _write(wide, "ilostat_wide")
    if country_month is not None:
        _write(country_month, "ilostat_country_month")


def _load_lines(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)
    lines: List[str] = []
    with open(p, "r", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            for cell in row:
                cell = cell.strip()
                if cell and not cell.startswith("#"):
                    lines.append(cell)
    return lines


def _parse_countries(arg: str) -> Optional[Set[str]]:
    if arg.strip().upper() == "ALL":
        return None
    p = Path(arg)
    if p.exists():
        return set(_load_lines(arg))
    return {c.strip() for c in arg.split(",") if c.strip()}


def main() -> None:
    setup_logging(LOG_LEVEL)

    logger.info("=" * 60)
    logger.info("ILOSTAT Pipeline")
    logger.info("Year range : %d → %d", START_YEAR, END_YEAR)
    logger.info("Countries  : %s",      COUNTRIES)
    logger.info("Format     : %s",      OUT_FORMAT)
    logger.info("Cache      : %s",      "enabled" if USE_CACHE else "DISABLED")
    logger.info("Workers    : %d",      MAX_WORKERS)
    logger.info("=" * 60)

    keywords  = _load_lines(KEYWORDS_FILE)  if KEYWORDS_FILE  else DEFAULT_KEYWORDS
    allowlist = set(_load_lines(ALLOWLIST_FILE)) if ALLOWLIST_FILE else set()
    denylist  = set(_load_lines(DENYLIST_FILE))  if DENYLIST_FILE  else set()
    countries = _parse_countries(COUNTRIES)

    cfg = ILOSTATClientConfig(cache_dir=_CACHE_DIR)

    with ILOSTATClient(cfg) as client:

        logger.info("Step 1 — Fetching indicator TOC")
        toc = client.fetch_toc(use_cache=USE_CACHE)
        logger.info("TOC: %d indicators (after SDG exclusion)", len(toc))

        logger.info("Step 2 — Filtering indicators by keyword")
        selected = filter_indicators(toc, keywords, allowlist, denylist)
        save_reference(toc, selected)

        if selected.empty:
            logger.error("No indicators matched — check KEYWORDS_FILE / ALLOWLIST_FILE.")
            sys.exit(1)

        logger.info("Selected: %d indicators", len(selected))

        logger.info("Step 3 — Downloading %d datasets", len(selected))
        raw = download_all(
            client      = client,
            dataset_ids = selected["id"].astype(str).tolist(),
            max_workers = MAX_WORKERS,
            use_cache   = USE_CACHE,
        )

    logger.info("Step 4 — Building long table")
    long = build_long_table(
        raw        = raw,
        toc        = toc,
        start_year = START_YEAR,
        end_year   = END_YEAR,
        countries  = countries,
    )

    logger.info("Step 5 — Building wide panel")
    wide = build_wide_panel(long)

    # ── Sub-annual pipeline ───────────────────────────────────────────────────
    all_subannual_ids: List[str] = []
    for m_id, q_id in SUBANNUAL_INDICATORS.values():
        if m_id:
            all_subannual_ids.append(m_id)
        if q_id:
            all_subannual_ids.append(q_id)
    all_subannual_ids = list(dict.fromkeys(all_subannual_ids))  # deduplicate, preserve order

    logger.info("Step 7 — Downloading %d sub-annual indicator(s)", len(all_subannual_ids))
    with ILOSTATClient(cfg) as client:
        raw_subannual = download_all(
            client      = client,
            dataset_ids = all_subannual_ids,
            max_workers = MAX_WORKERS,
            use_cache   = USE_CACHE,
        )

    logger.info("Step 8 — Building sub-annual long table")
    subannual_long = build_subannual_long_table(
        raw         = raw_subannual,
        indicator_ids = all_subannual_ids,
        start_year  = START_YEAR,
        end_year    = END_YEAR,
        countries   = countries,
    )

    logger.info("Step 9 — Building country-month panel")
    country_month = build_country_month_panel(
        subannual_long = subannual_long,
        indicator_map  = SUBANNUAL_INDICATORS,
        start_year     = START_YEAR,
        end_year       = END_YEAR,
    )

    logger.info("Step 6 — Saving outputs")
    tag = f"{START_YEAR}_{END_YEAR}"
    save_outputs(long, wide, OUT_FORMAT, tag, country_month=country_month)

    logger.info("=" * 60)
    logger.info(
        "Done.  Long: %d rows  |  Wide: %d rows × %d cols  |  Country-month: %d rows × %d cols",
        len(long),
        len(wide),
        len(wide.columns) if not wide.empty else 0,
        len(country_month) if not country_month.empty else 0,
        len(country_month.columns) if not country_month.empty else 0,
    )


if __name__ == "__main__":
    main()
