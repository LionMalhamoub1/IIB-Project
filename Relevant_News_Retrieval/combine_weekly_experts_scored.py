# combine_week_manual_filtered.py

from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# ---- SET WEEK RANGE ----
START_DATE = "20260101"
END_DATE   = "20260107"
THRESHOLD  = 0.4
# ------------------------

PATTERN = "*_experts_scored.csv"

project_root = Path(__file__).resolve().parent.parent
base = project_root / "data" / "processed" / "model_scored_daily"

start = datetime.strptime(START_DATE, "%Y%m%d")
end   = datetime.strptime(END_DATE, "%Y%m%d")

if start > end:
    raise ValueError("START_DATE must be <= END_DATE")

# Generate dates in range
dates = []
current = start
while current <= end:
    dates.append(current)
    current += timedelta(days=1)

files = []
for dt in dates:
    folder = base / dt.strftime("%Y") / dt.strftime("%m") / dt.strftime("%d")
    if folder.exists():
        files.extend(folder.glob(PATTERN))

if not files:
    raise FileNotFoundError("No experts_scored files found in given range.")

# Combine all daily files
df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

# Keep only required columns
cols_to_keep = [
    "url_normalized",
    "title",
    "meta_description",
    "top_expert",
    "top_expert_p",
]
df = df[cols_to_keep]

# Ensure numeric
df["top_expert_p"] = pd.to_numeric(df["top_expert_p"], errors="coerce")

# Apply threshold filter
df = df[df["top_expert_p"] > THRESHOLD]

# Save
out_dir = project_root / "data" / "processed" / "model_scored_daily" / end.strftime("%Y") / end.strftime("%m")
out_dir.mkdir(parents=True, exist_ok=True)

out_path = out_dir / f"weekly_experts_{START_DATE}_{END_DATE}.csv"
df.to_csv(out_path, index=False)

print(f"Combined files: {len(files)}")
print(f"Rows after threshold > {THRESHOLD}: {len(df):,}")
print(f"Saved to: {out_path}")

