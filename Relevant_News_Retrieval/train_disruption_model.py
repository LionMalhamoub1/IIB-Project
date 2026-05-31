
from pathlib import Path
import re
from urllib.parse import urlparse

import pandas as pd
import numpy as np
from joblib import dump

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score



TRAINING_XLSX = r"data/interim/labelled_disruption.xlsx"
SHEET_NAME = 0  # 0 = first sheet; or put the sheet name string
OUTPUT_DIR = "models/disruption_v1"
USE_URL_FALLBACK = True
REQUIRE_TEXT = True

THRESHOLD = 0.4  # <-- prediction threshold for class 1


BAD_TEXT_PATTERNS = [
    "your privacy", "privacy choices", "cookie", "consent", "gdpr",
    "subscribe", "sign in", "login", "access denied",
    "captcha", "#value", "value!", "Your Privacy Choices", "MSN", "bot"
]


def to_bool(x) -> int:
    """Map TRUE/FALSE/1/0/yes/no to {0,1}."""
    if pd.isna(x):
        return 0
    s = str(x).strip().lower()
    return int(s in {"true", "1", "yes", "y", "t"})


def looks_like_garbage(s: str) -> bool:
    if not isinstance(s, str):
        return True
    s = s.lower().strip()
    if len(s) < 15:
        return True
    return any(p in s for p in BAD_TEXT_PATTERNS)


def url_to_text(url: str) -> str:
    """Convert URL path into text tokens (slug -> words)."""
    if not isinstance(url, str) or not url.strip():
        return ""
    try:
        path = urlparse(url).path
    except Exception:
        return ""

    path = path.replace("/", " ")
    path = re.sub(r"[-_]+", " ", path)
    path = re.sub(r"\.(html|htm|php|aspx|jsp)$", "", path, flags=re.IGNORECASE)
    path = re.sub(r"\b\d+\b", " ", path)
    path = re.sub(r"\s+", " ", path).strip().lower()
    return path


def build_text(row: pd.Series) -> str:
    title = "" if pd.isna(row.get("title")) else str(row.get("title"))
    desc = "" if pd.isna(row.get("meta_description")) else str(row.get("meta_description"))
    url = "" if pd.isna(row.get("url_normalized")) else str(row.get("url_normalized"))

    main = f"{title}. {desc}".strip()
    main = " ".join(main.split())

    if USE_URL_FALLBACK and looks_like_garbage(main):
        return url_to_text(url)

    return main


print("Loading Excel:", TRAINING_XLSX)
df = pd.read_excel(TRAINING_XLSX, sheet_name=SHEET_NAME, engine="openpyxl")
print("Rows loaded:", len(df))
print("Columns:", list(df.columns))

# Clean column names (Excel sometimes adds spaces)
df.columns = df.columns.astype(str).str.strip()

# Check required columns
required = ["url_normalized", "title", "meta_description", "disruption"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")

# Build model text
df["text"] = df.apply(build_text, axis=1)
print("Non-empty text rows:", (df["text"].astype(str).str.len() > 0).sum())
print("Example texts:", df["text"].head(5).tolist())

if REQUIRE_TEXT:
    before = len(df)
    df = df[df["text"].astype(str).str.len() > 0].copy()
    df = df.reset_index(drop=True)  # <-- IMPORTANT: make index 0..N-1
    print(f"Dropped empty-text rows: {before} -> {len(df)}")
else:
    df = df.reset_index(drop=True)  # keep it consistent either way


# Labels
df["label"] = df["disruption"].map(to_bool).astype(int)
print("Label counts:", df["label"].value_counts().to_dict())

if df["label"].nunique() < 2:
    raise ValueError(
        "Need at least 2 classes in 'disruption' to train (some TRUE and some FALSE).\n"
        "Right now you only have one class in this file."
    )

# -------------------------
# Train/test split (INDEX-BASED so we can export exact FP/FN rows)
# -------------------------
idx = np.arange(len(df))
idx_train, idx_test = train_test_split(
    idx,
    test_size=0.2,
    stratify=df["label"].values,
    random_state=42,
)

X_train = df.loc[idx_train, "text"].astype(str).tolist()
X_test = df.loc[idx_test, "text"].astype(str).tolist()
y_train = df.loc[idx_train, "label"].values
y_test = df.loc[idx_test, "label"].values

# Embeddings
print("Embedding...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_train_emb = embedder.encode(X_train, normalize_embeddings=True, show_progress_bar=True)
X_test_emb = embedder.encode(X_test, normalize_embeddings=True, show_progress_bar=True)

# Classifier (linear SVM + probability calibration)
clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5)
clf.fit(X_train_emb, y_train)

probs = clf.predict_proba(X_test_emb)[:, 1]
preds = (probs > THRESHOLD).astype(int)

print("\nAverage precision:", average_precision_score(y_test, probs))
print(f"\nClassification report @ threshold={THRESHOLD}:")
print(classification_report(y_test, preds))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, preds))

# -------------------------
# Write review Excel (all_test + false_positives + false_negatives)
# -------------------------
out_dir = Path(OUTPUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

review_df = df.loc[idx_test].copy()
review_df["p_disruption"] = probs
review_df["y_true"] = y_test
review_df["y_pred"] = preds
review_df["error_type"] = ""

review_df.loc[(review_df["y_true"] == 0) & (review_df["y_pred"] == 1), "error_type"] = "FALSE_POSITIVE"
review_df.loc[(review_df["y_true"] == 1) & (review_df["y_pred"] == 0), "error_type"] = "FALSE_NEGATIVE"

false_pos = review_df[review_df["error_type"] == "FALSE_POSITIVE"].sort_values("p_disruption", ascending=False)
false_neg = review_df[review_df["error_type"] == "FALSE_NEGATIVE"].sort_values("p_disruption", ascending=True)

review_path = out_dir / "error_review.xlsx"
with pd.ExcelWriter(review_path, engine="openpyxl") as writer:
    review_df.sort_values("p_disruption", ascending=False).to_excel(writer, index=False, sheet_name="all_test")
    false_pos.to_excel(writer, index=False, sheet_name="false_positives")
    false_neg.to_excel(writer, index=False, sheet_name="false_negatives")

print(f"\n Wrote error review Excel to: {review_path}")
print("False positives:", len(false_pos), "| False negatives:", len(false_neg))

# -------------------------
# Save model bundle
# -------------------------
dump(
    {
        "embed_model": "all-MiniLM-L6-v2",
        "classifier": clf,
        "threshold": THRESHOLD,
        "use_url_fallback": USE_URL_FALLBACK,
    },
    out_dir / "disruption_model.joblib",
)

print("\n Model saved to:", out_dir / "disruption_model.joblib")
