import re
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = Path("data/interim/gdelt_event_context_daily")

# Inclusive range (YYYYMMDD)
START_DAY = "20180101"
END_DAY   = "20180122"

# Analyse deduped files
INPUT_SUFFIX = "_deduped.csv"

# Output Excel
OUT_XLSX = Path(f"data/interim/url_path_token_stats_{START_DAY}_{END_DAY}_deduped.xlsx")

MIN_TOKEN_LENGTH = 3
DROP_NUMERIC = True

# Optional: ignore very common generic tokens (add as needed)
STOP_TOKENS = set([
    # "news", "article", "index", "amp"
])

# -----------------------------
# HELPERS
# -----------------------------
def parse_day_from_filename(name: str) -> str | None:
    """Extract YYYYMMDD from filename."""
    m = re.search(r"(\d{8})", name)
    return m.group(1) if m else None

def day_in_range(day: str, start: str, end: str) -> bool:
    return start <= day <= end

def extract_tokens(url: str):
    """
    Extract tokens from URL path.
    e.g. /world/chile/lithium-mine-strike -> ["world","chile","lithium","mine","strike"]
    """
    try:
        p = urlparse(url)
        path = (p.path or "").lower()
        tokens = re.split(r"[\/\-\_\.]+", path)

        out = []
        for t in tokens:
            t = t.strip()
            if not t:
                continue
            if DROP_NUMERIC and t.isdigit():
                continue
            if len(t) < MIN_TOKEN_LENGTH:
                continue
            if t in STOP_TOKENS:
                continue
            out.append(t)
        return out
    except Exception:
        return []

# -----------------------------
# FIND FILES
# -----------------------------
files = []
for f in BASE_DIR.rglob(f"*{INPUT_SUFFIX}"):
    # Exclude filtered files (they end with "_deduped_filtered.csv" which also contains "_deduped.csv")
    if f.name.endswith("_deduped_filtered.csv"):
        continue

    day = parse_day_from_filename(f.name)
    if not day:
        continue
    if day_in_range(day, START_DAY, END_DAY):
        files.append((day, f))

files.sort(key=lambda x: x[0])

if not files:
    raise SystemExit(f"No deduped files found in range {START_DAY}..{END_DAY} under {BASE_DIR}")

print(f"Found {len(files)} file(s) from {START_DAY} to {END_DAY}")

# -----------------------------
# LOAD + TOKENISE
# -----------------------------
rows = []
for day, f in files:
    print("Reading:", f)
    df = pd.read_csv(f, dtype=str, on_bad_lines="skip")

    if "sourceurl" not in df.columns:
        print(f"WARNING: {f} has no sourceurl column")
        continue

    for url in df["sourceurl"].dropna():
        tokens = extract_tokens(str(url))
        for t in tokens:
            rows.append({"token": t, "day": day, "file": f.name})

df_tokens = pd.DataFrame(rows)
if df_tokens.empty:
    raise SystemExit("No tokens extracted (check sourceurl column and tokenisation settings).")

# -----------------------------
# AGGREGATE
# -----------------------------
# Overall frequencies
freq = (
    df_tokens.groupby("token")
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

# How many distinct days each token appears in
n_days = (
    df_tokens.groupby("token")["day"]
    .nunique()
    .reset_index(name="n_days")
)

stats = freq.merge(n_days, on="token", how="left")

# Token frequency by day (wide table)
by_day = (
    df_tokens.groupby(["token", "day"])
    .size()
    .reset_index(name="count")
    .pivot(index="token", columns="day", values="count")
    .fillna(0)
    .astype(int)
    .reset_index()
)

# Helpful: total count column in by_day
day_cols = [c for c in by_day.columns if c != "token"]
by_day["total"] = by_day[day_cols].sum(axis=1)
by_day = by_day.sort_values("total", ascending=False)

# -----------------------------
# WRITE EXCEL
# -----------------------------
OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
    stats.to_excel(writer, sheet_name="token_frequencies", index=False)
    by_day.to_excel(writer, sheet_name="token_by_day", index=False)

print(f"\nWROTE: {OUT_XLSX}")
print("Top 20 tokens overall:")
print(stats.head(20))
