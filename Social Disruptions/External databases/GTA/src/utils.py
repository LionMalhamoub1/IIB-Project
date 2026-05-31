from __future__ import annotations

import logging
import sys
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def missingness_report(df: pd.DataFrame) -> str:
    n = len(df)
    if n == 0:
        return "  (empty dataframe)"
    lines = [
        "Missing-value report:",
        f"  {'Column':<45} {'Missing':>8}  {'%':>6}",
        "  " + "─" * 63,
    ]
    for col in df.columns:
        missing = int(df[col].isna().sum())
        pct = 100.0 * missing / n
        lines.append(f"  {col:<45} {missing:>8}  {pct:>5.1f}%")
    return "\n".join(lines)


_HARMFUL_VALS      = {"red",   "1", "harmful"}
_LIBERALISING_VALS = {"green", "2", "liberalising", "liberalizing"}


def eval_to_harmful(val: str) -> int:
    return int(str(val).lower().strip() in _HARMFUL_VALS)


def eval_to_liberalising(val: str) -> int:
    return int(str(val).lower().strip() in _LIBERALISING_VALS)


_ISO3_TO_UN: Dict[str, int] = {
    "ARG": 32,   "BOL": 68,   "BRA": 76,   "CHL": 152,  "COL": 170,
    "CRI": 188,  "CUB": 192,  "DOM": 214,  "ECU": 218,  "GTM": 320,
    "HND": 340,  "HTI": 332,  "JAM": 388,  "MEX": 484,  "NIC": 558,
    "PAN": 591,  "PER": 604,  "PRY": 600,  "SLV": 222,  "TTO": 780,
    "URY": 858,  "VEN": 862,
    "AGO": 24,   "BDI": 108,  "BEN": 204,  "BFA": 854,  "BWA": 72,
    "CAF": 140,  "CIV": 384,  "CMR": 120,  "COD": 180,  "COG": 178,
    "DJI": 262,  "ERI": 232,  "ETH": 231,  "GAB": 266,  "GHA": 288,
    "GIN": 324,  "GMB": 270,  "GNB": 624,  "KEN": 404,  "LBR": 430,
    "LSO": 426,  "MDG": 450,  "MLI": 466,  "MOZ": 508,  "MRT": 478,
    "MUS": 480,  "MWI": 454,  "NAM": 516,  "NER": 562,  "NGA": 566,
    "RWA": 646,  "SDN": 729,  "SEN": 686,  "SLE": 694,  "SOM": 706,
    "SSD": 728,  "SWZ": 748,  "TCD": 148,  "TGO": 768,  "TZA": 834,
    "UGA": 800,  "ZAF": 710,  "ZMB": 894,  "ZWE": 716,
    "DZA": 12,   "EGY": 818,  "IRQ": 368,  "JOR": 400,  "LBN": 422,
    "LBY": 434,  "MAR": 504,  "SAU": 682,  "SYR": 760,  "TUN": 788,
    "TUR": 792,  "YEM": 887,
    "AFG": 4,    "BGD": 50,   "CHN": 156,  "IDN": 360,  "IND": 356,
    "IRN": 364,  "KGZ": 417,  "KHM": 116,  "LAO": 418,  "LKA": 144,
    "MMR": 104,  "MNG": 496,  "NPL": 524,  "PAK": 586,  "PHL": 608,
    "PRK": 408,  "SGP": 702,  "THA": 764,  "TJK": 762,  "TKM": 795,
    "TLS": 626,  "VNM": 704,
    "ALB": 8,    "ARM": 51,   "AZE": 31,   "BLR": 112,  "BIH": 70,
    "GEO": 268,  "KAZ": 398,  "MDA": 498,  "MKD": 807,  "MNE": 499,
    "SRB": 688,  "UKR": 804,  "UZB": 860,
    "AUS": 36,   "BEL": 56,   "CAN": 124,  "CHE": 756,  "DEU": 276,
    "ESP": 724,  "FRA": 250,  "GBR": 826,  "ITA": 380,  "JPN": 392,
    "KOR": 410,  "NLD": 528,  "POL": 616,  "ROU": 642,  "RUS": 643,
    "SWE": 752,  "USA": 840,
}


def iso3_to_un_codes(iso3_list: List[str]) -> List[int]:
    codes: List[int] = []
    seen: set = set()
    for iso3 in iso3_list:
        key = iso3.upper().strip()
        code = _ISO3_TO_UN.get(key)
        if code is None:
            logger.warning("[GTA] ISO3 '%s' not in UN code mapping — skipped.", iso3)
        elif code not in seen:
            codes.append(code)
            seen.add(code)
    return codes
