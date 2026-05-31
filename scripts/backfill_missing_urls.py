from datetime import date, timedelta
from pathlib import Path
from Relevant_News_Retrieval.pipeline import run_one_date

URLS_DIR = Path("data/urls")
START = date(2017, 1, 1)
END   = date(2021, 12, 31)

existing = {f.stem for f in URLS_DIR.glob("????????.csv")}
missing = []
d = START
while d <= END:
    key = d.strftime("%Y%m%d")
    if key not in existing:
        missing.append(key)
    d += timedelta(days=1)

print(f"Found {len(missing)} missing dates.")

for i, d in enumerate(missing, start=1):
    print(f"\n[{i}/{len(missing)}] {d}")
    try:
        run_one_date(d)
    except Exception as e:
        print(f"!!! Failed for {d}: {repr(e)}")
