import pandas as pd

csv_path = "data/interim/_state/url_title_meta_cache_20240115.csv"

df = pd.read_csv(csv_path)

# 13th row in human terms = index 12 (0-based)
title = df.loc[11, "title"]

print("Visible title:")
print(title)
print("\nUnicode codepoints:")
print([hex(ord(c)) for c in str(title)])
