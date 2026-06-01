"""Quick diagnostic: test downloading one indicator with various format params."""
import gzip
import io
import requests
import pandas as pd

BASE_URL = "https://rplumber.ilo.org/data/indicator/"
TEST_ID  = "EMP_PTER_SEX_RT_A"

tests = [
    ("parquet",  ".parquet", "parquet"),
    ("csv.gz",   ".csv.gz",  "csv.gz"),
    ("csv",      ".csv",     "csv"),
]

for label, fmt, kind in tests:
    params = {"id": TEST_ID, "format": fmt}
    print(f"\n--- format={fmt!r} ---")
    try:
        r = requests.get(BASE_URL, params=params, timeout=60)
        print(f"  Status: {r.status_code}  Content-Type: {r.headers.get('Content-Type','?')}  Bytes: {len(r.content)}")
        if r.status_code != 200:
            print(f"  Body: {r.text[:200]}")
            continue
        if kind == "parquet":
            df = pd.read_parquet(io.BytesIO(r.content))
            print(f"  Parquet OK — shape: {df.shape}")
        elif kind == "csv.gz":
            text = gzip.decompress(r.content).decode()
            lines = text.splitlines()
            print(f"  CSV.GZ OK — {len(lines)} rows, header: {lines[0][:120]}")
        else:
            lines = r.text.splitlines()
            print(f"  CSV OK — {len(lines)} rows, header: {lines[0][:120]}")
    except Exception as e:
        print(f"  Failed: {e}")
