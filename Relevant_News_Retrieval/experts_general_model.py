from pathlib import Path
import re
import time
from urllib.parse import urlparse

import pandas as pd
import numpy as np
from joblib import dump, load as joblib_load

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from tqdm import tqdm


# ============================
# Config
# ============================
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TRAINING_XLSX = BASE_DIR / "data" / "interim" / "disruption_master_10k_multiexpert_labelled.xlsx"
SHEET_NAME = "data"  # master sheet name
OUTPUT_DIR = BASE_DIR.parent / "models" / "disruption_v2_experts"

USE_URL_FALLBACK = True
REQUIRE_TEXT = True

THRESHOLD_GENERAL = 0.40
THRESHOLD_EXPERT = 0.40

# If True, skip retraining when a .joblib already exists and just re-evaluate
SKIP_IF_EXISTS = True

ROW_ORIGIN_COL = "row_origin"
GOLD_ORIGIN_VALUE = "gold_manual"

URL_COL = "url_normalized"
TITLE_COL = "title"
META_COL = "meta_description"

# Gold columns (NO prefix)
GOLD_TYPES = [
    "flood",
    "drought",
    "cyclone_huricane",
    "extreme_heat",
    "landslide",
    "earthquake",
    "mine_accident",
    "labour_strike",
    "protests",
    "trade_embargo",
    "country_relations",
    "tariffs",
]
GOLD_GENERAL_COL = "disruption"

# Weak labels (ChatGPT columns WITH prefix)
WEAK_PREFIX = "chatgpt_"  # as requested

BAD_TEXT_PATTERNS = [
    "your privacy", "privacy choices", "cookie", "consent", "gdpr",
    "subscribe", "sign in", "login", "access denied",
    "captcha", "#value", "value!", "msn", "bot"
]


# ============================
# Utilities
# ============================
def to_int01(x) -> int:
    """Map mixed (0/1, '0'/'1', booleans, true/false) to {0,1}."""
    if pd.isna(x):
        return 0
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)) and x in (0, 1):
        return int(x)
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return 1
    if s in {"0", "false", "no", "n", "f"}:
        return 0
    return 0


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
    title = "" if pd.isna(row.get(TITLE_COL)) else str(row.get(TITLE_COL))
    desc = "" if pd.isna(row.get(META_COL)) else str(row.get(META_COL))
    url = "" if pd.isna(row.get(URL_COL)) else str(row.get(URL_COL))

    main = f"{title}. {desc}".strip()
    main = " ".join(main.split())

    if USE_URL_FALLBACK and looks_like_garbage(main):
        return url_to_text(url)
    return main


def ensure_columns(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nFound columns: {list(df.columns)}")


def format_seconds(secs: float) -> str:
    secs = max(0.0, float(secs))
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m:d}m {s:02d}s"
    return f"{s:d}s"


# ============================
# Label construction (gold + chatgpt)
# ============================
def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gold rows (row_origin == gold_manual):
      - experts from gold columns (e.g., 'flood', ...)
      - general from gold 'disruption'
    Non-gold rows:
      - experts from chatgpt_* columns
      - general from chatgpt_disruption IF present, else OR over chatgpt experts
    """
    is_gold = (df[ROW_ORIGIN_COL].fillna("") == GOLD_ORIGIN_VALUE).values

    # Ensure weak label columns exist for all types
    weak_cols = [f"{WEAK_PREFIX}{t}" for t in GOLD_TYPES]
    ensure_columns(df, weak_cols)

    # General weak label: prefer chatgpt_disruption if present
    weak_general_col = f"{WEAK_PREFIX}{GOLD_GENERAL_COL}"
    has_weak_general = weak_general_col in df.columns

    y = pd.DataFrame(index=df.index)

    # Expert targets
    for t in GOLD_TYPES:
        gold_col = t                # NO prefix
        weak_col = f"{WEAK_PREFIX}{t}"

        if gold_col not in df.columns:
            raise ValueError(f"Gold column missing: '{gold_col}'")
        if weak_col not in df.columns:
            raise ValueError(f"Weak label column missing: '{weak_col}'")

        gold_vals = df[gold_col].map(to_int01).values
        weak_vals = df[weak_col].map(to_int01).values
        y[t] = np.where(is_gold, gold_vals, weak_vals).astype(int)

    # General target
    if GOLD_GENERAL_COL not in df.columns:
        raise ValueError(f"Gold general column missing: '{GOLD_GENERAL_COL}'")

    gold_general = df[GOLD_GENERAL_COL].map(to_int01).values

    or_experts = (df[weak_cols].map(to_int01).sum(axis=1).values > 0).astype(int)
    if has_weak_general:
        weak_general = df[weak_general_col].map(to_int01).values
        # Fall back to OR over experts where chatgpt_disruption is null
        weak_general = np.where(pd.isna(df[weak_general_col]).values, or_experts, weak_general)
    else:
        weak_general = or_experts

    y[GOLD_GENERAL_COL] = np.where(is_gold, gold_general, weak_general).astype(int)

    return y


# ============================
# Training util
# ============================
def train_one_binary(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    model_name: str,
    threshold: float,
    idx_train: np.ndarray,
    idx_test: np.ndarray,
):
    """
    Train a calibrated linear SVM on precomputed embeddings.
    Writes:
      - error_review.xlsx (all_test / false_pos / false_neg) with original columns
      - model bundle joblib
    Returns:
      - (metrics_dict, probs_series) where probs_series is indexed by df.index[idx_test]
      - None if only one class is present
    """
    if len(np.unique(y)) < 2:
        print(f"Skipping {model_name}: only one class present.")
        return None

    X_train_emb = embeddings[idx_train]
    X_test_emb = embeddings[idx_test]
    y_train = y[idx_train]
    y_test = y[idx_test]

    clf = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=5)
    clf.fit(X_train_emb, y_train)

    probs = clf.predict_proba(X_test_emb)[:, 1]
    preds = (probs > threshold).astype(int)

    ap = float(average_precision_score(y_test, probs))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    n_fp = int(((y_test == 0) & (preds == 1)).sum())
    n_fn = int(((y_test == 1) & (preds == 0)).sum())

    metrics = {
        "model": model_name,
        "n_train": len(idx_train),
        "n_test": len(idx_test),
        "n_pos_test": int(y_test.sum()),
        "avg_precision": round(ap, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "n_fp": n_fp,
        "n_fn": n_fn,
        "threshold": threshold,
    }

    print(f"\n==== {model_name} ====")
    print("Average precision:", round(ap, 4))
    print(f"Classification report @ threshold={threshold}:")
    print(classification_report(y_test, preds))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    out_dir.mkdir(parents=True, exist_ok=True)

    review_df = df.loc[df.index[idx_test]].copy()
    review_df["p_disruption"] = probs
    review_df["y_true"] = y_test
    review_df["y_pred"] = preds
    review_df["error_type"] = ""
    review_df.loc[(review_df["y_true"] == 0) & (review_df["y_pred"] == 1), "error_type"] = "FALSE_POSITIVE"
    review_df.loc[(review_df["y_true"] == 1) & (review_df["y_pred"] == 0), "error_type"] = "FALSE_NEGATIVE"

    false_pos = review_df[review_df["error_type"] == "FALSE_POSITIVE"].sort_values("p_disruption", ascending=False)
    false_neg = review_df[review_df["error_type"] == "FALSE_NEGATIVE"].sort_values("p_disruption", ascending=True)

    review_path = out_dir / f"{model_name}_error_review.xlsx"
    with pd.ExcelWriter(review_path, engine="openpyxl") as writer:
        review_df.sort_values("p_disruption", ascending=False).to_excel(writer, index=False, sheet_name="all_test")
        false_pos.to_excel(writer, index=False, sheet_name="false_positives")
        false_neg.to_excel(writer, index=False, sheet_name="false_negatives")

    dump(
        {
            "embed_model": "all-MiniLM-L6-v2",
            "classifier": clf,
            "threshold": threshold,
            "label": model_name,
            "use_url_fallback": USE_URL_FALLBACK,
        },
        out_dir / f"{model_name}.joblib",
    )

    print(f"Wrote: {review_path}")
    print(f"Saved model: {out_dir / f'{model_name}.joblib'}")
    print("False positives:", n_fp, "| False negatives:", n_fn)

    probs_series = pd.Series(probs, index=df.index[idx_test], name=model_name)
    return metrics, probs_series


def evaluate_existing(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
    model_name: str,
    threshold: float,
    idx_test: np.ndarray,
):
    """Load an existing .joblib and evaluate it on the test set (no retraining)."""
    joblib_path = out_dir / f"{model_name}.joblib"
    bundle = joblib_load(joblib_path)
    clf = bundle["classifier"]

    X_test_emb = embeddings[idx_test]
    y_test = y[idx_test]

    probs = clf.predict_proba(X_test_emb)[:, 1]
    preds = (probs > threshold).astype(int)

    ap   = float(average_precision_score(y_test, probs))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec  = float(recall_score(y_test, preds, zero_division=0))
    f1   = float(f1_score(y_test, preds, zero_division=0))
    n_fp = int(((y_test == 0) & (preds == 1)).sum())
    n_fn = int(((y_test == 1) & (preds == 0)).sum())

    metrics = {
        "model": model_name,
        "n_train": len(df) - len(idx_test),
        "n_test": len(idx_test),
        "n_pos_test": int(y_test.sum()),
        "avg_precision": round(ap, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "n_fp": n_fp,
        "n_fn": n_fn,
        "threshold": threshold,
    }

    print(f"\n==== {model_name} [loaded from disk] ====")
    print("Average precision:", round(ap, 4))
    print(f"Classification report @ threshold={threshold}:")
    print(classification_report(y_test, preds))

    probs_series = pd.Series(probs, index=df.index[idx_test], name=model_name)
    return metrics, probs_series


# ============================
# Main
# ============================
print("Loading Excel:", TRAINING_XLSX)
df = pd.read_excel(TRAINING_XLSX, sheet_name=SHEET_NAME, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()
print("Rows loaded:", len(df))

# Core required columns
ensure_columns(df, [ROW_ORIGIN_COL, URL_COL, TITLE_COL, META_COL])
ensure_columns(df, GOLD_TYPES + [GOLD_GENERAL_COL])

# Build text
df["text"] = df.apply(build_text, axis=1)
if REQUIRE_TEXT:
    before = len(df)
    df = df[df["text"].astype(str).str.len() > 0].copy().reset_index(drop=True)
    print(f"Dropped empty-text rows: {before} -> {len(df)}")
else:
    df = df.reset_index(drop=True)

# Build targets (gold + chatgpt)
y_df = build_targets(df)
print("Gold rows:", int((df[ROW_ORIGIN_COL].fillna("") == GOLD_ORIGIN_VALUE).sum()))
print("General label counts:", y_df[GOLD_GENERAL_COL].value_counts().to_dict())

# Embed once (shared across all models) + visible progress bar
print("Loading embedder...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding all texts (once)...")
X_text = df["text"].astype(str).tolist()
embeddings = embedder.encode(X_text, normalize_embeddings=True, show_progress_bar=True)

# Train 13 models (1 general + 12 experts) with overall progress + ETA
root = Path(OUTPUT_DIR)
root.mkdir(parents=True, exist_ok=True)

# Pre-compute a single shared train/test split (stratified on the general label)
# so all models are evaluated on exactly the same articles — enabling fair comparison.
idx_all = np.arange(len(df))
idx_train_shared, idx_test_shared = train_test_split(
    idx_all,
    test_size=0.2,
    stratify=y_df[GOLD_GENERAL_COL].values,
    random_state=42,
)
print(f"Shared split: {len(idx_train_shared)} train / {len(idx_test_shared)} test")

tasks = [("general", GOLD_GENERAL_COL, THRESHOLD_GENERAL)] + [(f"expert_{t}", t, THRESHOLD_EXPERT) for t in GOLD_TYPES]

all_metrics = []
all_probs = []

ema_per_model = None
alpha = 0.25  # EMA smoothing
start_all = time.time()

pbar = tqdm(tasks, desc="Training models", unit="model")
for i, (subdir, label_col, thr) in enumerate(pbar, start=1):
    t0 = time.time()

    model_name = "disruption_general" if label_col == GOLD_GENERAL_COL else f"disruption_{label_col}"
    out_dir = root / subdir

    y = y_df[label_col].values.astype(int)
    joblib_path = out_dir / f"{model_name}.joblib"
    if SKIP_IF_EXISTS and joblib_path.exists():
        result = evaluate_existing(
            df=df, embeddings=embeddings, y=y, out_dir=out_dir, model_name=model_name,
            threshold=thr, idx_test=idx_test_shared,
        )
    else:
        result = train_one_binary(
            df=df, embeddings=embeddings, y=y, out_dir=out_dir, model_name=model_name,
            threshold=thr, idx_train=idx_train_shared, idx_test=idx_test_shared,
        )
    if result is not None:
        metrics, probs_series = result
        all_metrics.append(metrics)
        all_probs.append(probs_series)

    dt = time.time() - t0
    ema_per_model = dt if ema_per_model is None else (alpha * dt + (1 - alpha) * ema_per_model)

    remaining = (len(tasks) - i) * (ema_per_model if ema_per_model is not None else 0.0)
    elapsed = time.time() - start_all

    pbar.set_postfix_str(f"last={format_seconds(dt)} | elapsed={format_seconds(elapsed)} | ETA={format_seconds(remaining)}")

pbar.close()

# ============================
# Comparison outputs
# ============================
if all_metrics:
    # 1. Metrics summary CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(root / "comparison_metrics.csv", index=False)
    print(f"\nSaved comparison metrics -> {root / 'comparison_metrics.csv'}")

    # 2. Per-article predictions on the shared test set
    test_df = df.loc[df.index[idx_test_shared], [URL_COL, TITLE_COL, ROW_ORIGIN_COL]].copy()
    for col in [GOLD_GENERAL_COL] + GOLD_TYPES:
        test_df[f"y_true_{col}"] = y_df[col].iloc[idx_test_shared].values
    for probs_series in all_probs:
        test_df[f"p_{probs_series.name}"] = probs_series
    test_df = test_df.reset_index(drop=True)
    test_df.to_excel(root / "test_predictions.xlsx", index=False)
    print(f"Saved test predictions  -> {root / 'test_predictions.xlsx'}")

    # 3. Gold-only metrics
    gold_mask = test_df[ROW_ORIGIN_COL].fillna("") == GOLD_ORIGIN_VALUE
    gold_df = test_df[gold_mask]
    print(f"\nGold rows in test set: {gold_mask.sum()} / {len(test_df)}")

    gold_metrics = []
    for m in all_metrics:
        model_name = m["model"]
        # Map model name back to label column
        if model_name == "disruption_general":
            label_col = GOLD_GENERAL_COL
        else:
            label_col = model_name.replace("disruption_", "")

        p_col = f"p_{model_name}"
        y_col = f"y_true_{label_col}"
        if p_col not in gold_df.columns or y_col not in gold_df.columns:
            continue

        y_g = gold_df[y_col].values.astype(int)
        p_g = gold_df[p_col].values
        n_pos = int(y_g.sum())

        if len(np.unique(y_g)) < 2:
            gold_metrics.append({
                "model": model_name, "n_test_gold": len(gold_df),
                "n_pos_test_gold": n_pos, "avg_precision": float("nan"),
                "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "n_fp": None, "n_fn": None,
                "note": "single class in gold test set",
            })
            continue

        thr = THRESHOLD_GENERAL if model_name == "disruption_general" else THRESHOLD_EXPERT
        preds_g = (p_g > thr).astype(int)
        gold_metrics.append({
            "model": model_name,
            "n_test_gold": len(gold_df),
            "n_pos_test_gold": n_pos,
            "avg_precision": round(float(average_precision_score(y_g, p_g)), 4),
            "precision": round(float(precision_score(y_g, preds_g, zero_division=0)), 4),
            "recall": round(float(recall_score(y_g, preds_g, zero_division=0)), 4),
            "f1": round(float(f1_score(y_g, preds_g, zero_division=0)), 4),
            "n_fp": int(((y_g == 0) & (preds_g == 1)).sum()),
            "n_fn": int(((y_g == 1) & (preds_g == 0)).sum()),
            "note": "low sample" if n_pos < 10 else "",
        })

    gold_metrics_df = pd.DataFrame(gold_metrics)
    gold_metrics_df.to_csv(root / "comparison_metrics_gold.csv", index=False)
    print(f"Saved gold-only metrics -> {root / 'comparison_metrics_gold.csv'}")

    # 4. Comparison figure: overall (top) and gold-only (bottom)
    names = [m["model"] for m in all_metrics]
    aps   = [m["avg_precision"] for m in all_metrics]
    f1s   = [m["f1"] for m in all_metrics]
    colours = ["#1f77b4" if "general" in n else "#ff7f0e" for n in names]

    gold_lookup = {m["model"]: m for m in gold_metrics}
    aps_gold = [gold_lookup.get(n, {}).get("avg_precision", float("nan")) for n in names]
    f1s_gold = [gold_lookup.get(n, {}).get("f1", float("nan")) for n in names]
    low_sample = [gold_lookup.get(n, {}).get("note", "") == "low sample" for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(14, max(8, len(all_metrics) * 0.7)),
                             sharex="col", sharey="row")

    for col_idx, (overall_vals, gold_vals, xlabel) in enumerate([
        (aps, aps_gold, "Average Precision"),
        (f1s, f1s_gold, f"F1 Score (threshold={THRESHOLD_EXPERT})"),
    ]):
        for row_idx, (values, title_suffix) in enumerate([
            (overall_vals, "Overall (all test rows)"),
            (gold_vals, "Gold labels only"),
        ]):
            ax = axes[row_idx][col_idx]
            bars = ax.barh(names, values, color=colours, edgecolor="white", linewidth=0.5)
            # Hatch bars with low gold sample count
            if row_idx == 1:
                for bar, is_low in zip(bars, low_sample):
                    if is_low:
                        bar.set_hatch("//")
                        bar.set_alpha(0.6)
            ax.axvline(overall_vals[0] if row_idx == 0 else gold_vals[0],
                       color="#1f77b4", linestyle="--", alpha=0.5, linewidth=1.0)
            ax.set_xlabel(xlabel if row_idx == 1 else "")
            ax.set_title(f"{xlabel} — {title_suffix}")
            ax.set_xlim(0, 1)

    legend_handles = [
        Patch(color="#1f77b4", label="General model"),
        Patch(color="#ff7f0e", label="Expert models"),
        Patch(facecolor="white", edgecolor="grey", hatch="//", label="<10 gold positives (unreliable)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(root / "comparison_figure.png", dpi=150)
    plt.close(fig)
    print(f"Saved comparison figure -> {root / 'comparison_figure.png'}")

print("\nAll done.")
print("Models saved under:", root)
