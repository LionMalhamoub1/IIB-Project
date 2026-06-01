from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from wdi_client import WDIClient, WDIClientConfig


START_YEAR = 2017
END_YEAR = 2025

WDI_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = WDI_ROOT / "indicators.yaml"
OUT_PARQUET = WDI_ROOT / "data" / "processed" / f"wdi_country_year_{START_YEAR}_{END_YEAR}.parquet"
OUT_CSV     = WDI_ROOT / "data" / "processed" / f"wdi_country_year_{START_YEAR}_{END_YEAR}.csv"
USE_CACHE = True


def load_indicator_registry(path: Path) -> Tuple[int, Dict[str, str]]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    source_id = int(cfg.get("source_id", 2))
    indicators = cfg["indicators"]
    return source_id, indicators


def tidy_rows(rows: List[dict], indicator_alias: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["country_iso3", "year", "value", "indicator"])

    out = []
    for r in rows:
        iso3 = r.get("countryiso3code")
        year = r.get("date")
        val = r.get("value")

        if not iso3 or not year:
            continue

        try:
            year_i = int(year)
        except Exception:
            continue

        out.append(
            {
                "country_iso3": iso3,
                "year": year_i,
                "value": val,
                "indicator": indicator_alias,
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    source_id, indicators = load_indicator_registry(REGISTRY_PATH)

    cache_dir = WDI_ROOT / "data" / "raw" / "api_cache"
    client = WDIClient(WDIClientConfig(cache_dir=cache_dir))

    countries = ("all",)

    frames: List[pd.DataFrame] = []
    for alias, code in indicators.items():
        rows = client.fetch_indicator_series(
            indicator_code=code,
            countries=countries,
            source_id=source_id,
            start_year=START_YEAR,
            end_year=END_YEAR,
            use_cache=USE_CACHE,
        )
        frames.append(tidy_rows(rows, alias))

    long = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    wide = (
        long.pivot_table(
            index=["country_iso3", "year"],
            columns="indicator",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        if not long.empty
        else pd.DataFrame()
    )

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    wide.to_parquet(OUT_PARQUET, index=False)
    wide.to_csv(OUT_CSV, index=False)

    print(f"Wrote {len(wide):,} rows to:\n- {OUT_PARQUET}\n- {OUT_CSV}")


if __name__ == "__main__":
    main()
