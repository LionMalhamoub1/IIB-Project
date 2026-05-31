import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix)
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from pathlib import Path
import re
from urllib.parse import urlparse

BAD_TEXT_PATTERNS = ['your privacy','privacy choices','cookie','consent','gdpr',
    'subscribe','sign in','login','access denied','captcha','#value','value!',
    'Your Privacy Choices','MSN','bot']

def looks_like_garbage(s):
    if not isinstance(s, str): return True
    s = s.lower().strip()
    if len(s) < 15: return True
    return any(p in s for p in BAD_TEXT_PATTERNS)

def url_to_text(url):
    if not isinstance(url, str) or not url.strip(): return ''
    try: path = urlparse(url).path
    except: return ''
    path = path.replace('/', ' ')
    path = re.sub(r'[-_]+', ' ', path)
    path = re.sub(r'\.(html|htm|php|aspx|jsp)$', '', path, flags=re.IGNORECASE)
    path = re.sub(r'\b\d+\b', ' ', path)
    return re.sub(r'\s+', ' ', path).strip().lower()

def build_text(row):
    title = '' if pd.isna(row.get('title')) else str(row.get('title'))
    desc  = '' if pd.isna(row.get('meta_description')) else str(row.get('meta_description'))
    url   = '' if pd.isna(row.get('url_normalized')) else str(row.get('url_normalized'))
    main = ' '.join((title + '. ' + desc).split())
    if looks_like_garbage(main): return url_to_text(url)
    return main

TYPES = ['flood','drought','cyclone_huricane','extreme_heat','landslide','earthquake',
         'mine_accident','labour_strike','protests','trade_embargo','country_relations','tariffs']

df = pd.read_excel('data/interim/disruption_master_10k_multiexpert_labelled.xlsx', sheet_name='data')

gold = df[df['row_origin']=='gold_manual'].copy()
gold['label'] = gold['disruption'].fillna(0).astype(int)
for t in TYPES:
    gold['label_'+t] = gold[t].fillna(0).astype(int)

syn = df[df['row_origin']=='synthetic'].copy()
for t in TYPES:
    cgpt = (syn['chatgpt_'+t].fillna(0) > 0)
    gem  = (syn['gemini_'+t].fillna(0)  > 0)
    syn['label_'+t] = (cgpt & gem).astype(int)
syn['label'] = syn[['label_'+t for t in TYPES]].max(axis=1)

full = pd.concat([gold, syn], ignore_index=True)
full['text'] = full.apply(build_text, axis=1)
full = full[full['text'].str.len() > 0].copy().reset_index(drop=True)

idx = np.arange(len(full))
idx_train, idx_test = train_test_split(idx, test_size=0.2, stratify=full['label'].values, random_state=42)

embedder = SentenceTransformer('all-MiniLM-L6-v2')
print('Embedding...')
X_all = embedder.encode(full['text'].tolist(), normalize_embeddings=True, show_progress_bar=False, batch_size=256)
X_train = X_all[idx_train]; X_test = X_all[idx_test]
y_train = full.loc[idx_train,'label'].values
y_test  = full.loc[idx_test, 'label'].values

clf = CalibratedClassifierCV(LinearSVC(class_weight='balanced'), cv=5)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:, 1]

# threshold sweep
thresholds = np.arange(0.05, 0.96, 0.05)
rows = []
for thr in thresholds:
    preds = (probs > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    rows.append({
        'threshold': round(thr, 2),
        'precision': round(precision_score(y_test, preds, zero_division=0), 3),
        'recall':    round(recall_score(y_test, preds), 3),
        'f1':        round(f1_score(y_test, preds, zero_division=0), 3),
        'fp': fp, 'fn': fn, 'tn': tn, 'tp': tp,
    })

res = pd.DataFrame(rows)
best_f1 = res.loc[res['f1'].idxmax()]

print()
print(res[['threshold','precision','recall','f1','fp','fn']].to_string(index=False))
print()
print('Best F1 threshold: {}  ->  F1={} Prec={} Recall={}'.format(
    best_f1['threshold'], best_f1['f1'], best_f1['precision'], best_f1['recall']))

# per-type sweep for key types
KEY_TYPES = ['protests', 'labour_strike']
neg_mask = full.loc[idx_test, 'label'].values == 0

print()
type_curves = {}
for typ in KEY_TYPES:
    yt_type = full.loc[idx_test, 'label_'+typ].values
    pos_mask = yt_type == 1
    mask = neg_mask | pos_mask
    yt = yt_type[mask]
    pp = probs[mask]
    f1s, precs, recs = [], [], []
    for thr in thresholds:
        pd_ = (pp > thr).astype(int)
        f1s.append(f1_score(yt, pd_, zero_division=0))
        precs.append(precision_score(yt, pd_, zero_division=0))
        recs.append(recall_score(yt, pd_))
    type_curves[typ] = {'f1': f1s, 'prec': precs, 'rec': recs}
    best_idx = np.argmax(f1s)
    print('{} -- best F1 @ {:.2f}  F1={:.3f}  Prec={:.3f}  Recall={:.3f}'.format(
        typ, thresholds[best_idx], f1s[best_idx], precs[best_idx], recs[best_idx]))

# plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Threshold Sensitivity Analysis (10k dataset, 2000-row test set)', fontsize=12, fontweight='bold')

# Panel 1 — overall prec/recall/F1
ax = axes[0]
ax.plot(res['threshold'], res['precision'], 'b-o', ms=4, label='Precision')
ax.plot(res['threshold'], res['recall'],    'r-o', ms=4, label='Recall')
ax.plot(res['threshold'], res['f1'],        'g-o', ms=4, label='F1')
ax.axvline(0.4,  color='grey',  ls='--', lw=1.2, label='Current (0.40)')
ax.axvline(float(best_f1['threshold']), color='green', ls=':', lw=1.5,
           label='Best F1 ({})'.format(best_f1['threshold']))
ax.set_xlabel('Threshold'); ax.set_ylabel('Score')
ax.set_title('Overall binary disruption')
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# Panel 2 — F1 by type
ax2 = axes[1]
colors = {'protests': 'steelblue', 'labour_strike': 'tomato'}
for typ in KEY_TYPES:
    ax2.plot(thresholds, type_curves[typ]['f1'], '-o', ms=4,
             color=colors[typ], label=typ)
ax2.axvline(0.4, color='grey', ls='--', lw=1.2, label='Current (0.40)')
ax2.set_xlabel('Threshold'); ax2.set_ylabel('F1')
ax2.set_title('F1 by threshold — protests & strikes')
ax2.legend(fontsize=8); ax2.grid(alpha=0.3); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

# Panel 3 — recall vs FP trade-off
ax3 = axes[2]
ax3.plot(res['fp'], res['recall'], 'purple', marker='o', ms=4)
for _, row in res[res['threshold'].isin([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])].iterrows():
    ax3.annotate(str(row['threshold']), (row['fp'], row['recall']),
                 textcoords='offset points', xytext=(5, 3), fontsize=8)
ax3.set_xlabel('False Positives (volume passed downstream)')
ax3.set_ylabel('Recall (fraction of positives caught)')
ax3.set_title('Recall vs False Positives trade-off')
ax3.grid(alpha=0.3)

plt.tight_layout()
out = Path('models/disruption_v2/threshold_sensitivity.png')
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight')
print('\nPlot saved to', out)
