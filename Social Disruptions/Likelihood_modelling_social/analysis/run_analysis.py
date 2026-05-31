from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

_HERE    = Path(__file__).resolve().parent
MOD_ROOT = _HERE.parent
PROC_DIR = MOD_ROOT / "data" / "processed"
FIG_DIR       = _HERE / "figures"
FIG_DIR_LAG   = FIG_DIR / "lag"
FIG_DIR_NOLAG = FIG_DIR / "no_lag"
TAB_DIR       = _HERE / "tables"

for _d in (FIG_DIR, FIG_DIR_LAG, FIG_DIR_NOLAG, TAB_DIR):
    _d.mkdir(exist_ok=True, parents=True)

HORIZONS = [7, 30]

TARGETS = {
    "target_original": "Any protest (next N days)",
    "target_elevated": "Elevated spike",
}

# Models that use ACLED lag features
LAG_MODELS: dict[str, str] = {
    "model0_persistence":  "Persistence",
    "model1_markets":      "+ Markets",
    "model2_full":         "Full LR",
    "model4_fao":          "+ FAO",
    "model5_xgboost_tuned": "XGBoost (global)",
    "model5_income_group": "XGBoost (income group)",
}

# Models that do NOT use ACLED lag features
NOLAG_MODELS: dict[str, str] = {
    "model6_xgb_nolag":          "XGBoost no lags (global)",
    "model6_income_group_nolag": "XGBoost no lags (income group)",
}

MODEL_LABELS = {**LAG_MODELS, **NOLAG_MODELS}

MODEL_COLORS = {
    "model0_persistence":        "#4393c3",
    "model1_markets":            "#74add1",
    "model2_full":               "#4dac26",
    "model4_fao":                "#e08214",
    "model5_xgboost_tuned":      "#969696",
    "model5_income_group":       "#b2182b",
    "model6_xgb_nolag":          "#c7c7c7",
    "model6_income_group_nolag": "#762a83",
}

TARGET_COLORS = {
    "target_original": "#4393c3",
    "target_elevated": "#b2182b",
}

# Primary single model used for per-model figures in each group
PRIMARY_LAG_MODEL   = "model5_xgboost_tuned"
PRIMARY_NOLAG_MODEL = "model6_xgb_nolag"

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     ":",
    "figure.dpi":         130,
    "axes.titlelocation": "left",
    "axes.titlesize":     12,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_target(target_key: str) -> dict:
    tdir = PROC_DIR / target_key
    out = {}
    for H in HORIZONS:
        out[H] = {
            "preds":   pd.read_parquet(tdir / f"preds_h{H}.parquet"),
            "metrics": pd.read_csv(tdir    / f"metrics_h{H}.csv"),
            "coefs":   pd.read_csv(tdir    / f"coefs_h{H}.csv"),
        }
        xgb_path = tdir / f"coefs_xgb_h{H}.csv"
        out[H]["coefs_xgb"] = pd.read_csv(xgb_path) if xgb_path.exists() else pd.DataFrame()
        out[H]["preds"]["date"] = pd.to_datetime(out[H]["preds"]["date"])
    return out


def load_all() -> dict[str, dict]:
    return {key: load_target(key) for key in TARGETS}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _numeric_coefs(coefs: pd.DataFrame) -> pd.DataFrame:
    coefs = coefs.copy()
    coefs["feature"] = coefs["feature"].str.replace("num__", "", regex=False)
    return coefs[~coefs["feature"].str.startswith("fe__")]


def _calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 8):
    bins = np.linspace(0, 1, n_bins + 1)
    idx  = np.digitize(y_prob, bins[1:-1])
    means_pred, means_act, ns = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        means_pred.append(y_prob[mask].mean())
        means_act.append(y_true[mask].mean())
        ns.append(mask.sum())
    return np.array(means_pred), np.array(means_act), np.array(ns)


def _pooled_metrics(preds: pd.DataFrame, model_name: str) -> dict:
    sub = preds[preds["model_name"] == model_name]
    if sub.empty:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan, "pos_rate": np.nan}
    yt = sub["y_true"].values.astype(float)
    yp = sub["y_pred"].values
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]
    if len(np.unique(yt)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan, "pos_rate": float(yt.mean())}
    return {
        "roc_auc":  round(roc_auc_score(yt, yp), 4),
        "pr_auc":   round(average_precision_score(yt, yp), 4),
        "brier":    round(brier_score_loss(yt, yp), 4),
        "pos_rate": round(float(yt.mean()), 4),
    }


def _optimal_threshold_metrics(yt: np.ndarray, yp: np.ndarray) -> dict:
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(yt, yp)
    f1_arr = np.where(
        (prec_arr[:-1] + rec_arr[:-1]) > 0,
        2 * prec_arr[:-1] * rec_arr[:-1] / (prec_arr[:-1] + rec_arr[:-1]),
        0.0,
    )
    best_idx    = int(np.argmax(f1_arr))
    best_thresh = float(thresh_arr[best_idx])
    yp_bin      = (yp >= best_thresh).astype(int)
    return {
        "threshold": round(best_thresh, 4),
        "precision": round(float(precision_score(yt, yp_bin, zero_division=0)), 4),
        "recall":    round(float(recall_score(yt, yp_bin, zero_division=0)), 4),
        "f1":        round(float(f1_score(yt, yp_bin, zero_division=0)), 4),
    }


def _find_episodes(yt: np.ndarray, dates: pd.DatetimeIndex) -> list[tuple[int, int]]:
    episodes = []
    in_ep    = False
    start    = 0
    for i, v in enumerate(yt):
        if v == 1 and not in_ep:
            in_ep = True
            start = i
        elif v != 1 and in_ep:
            episodes.append((start, i - 1))
            in_ep = False
    if in_ep:
        episodes.append((start, len(yt) - 1))
    return episodes


def _event_level_stats(preds: pd.DataFrame, model_name: str, threshold: float) -> dict:
    sub = preds[preds["model_name"] == model_name].copy()
    sub = sub.sort_values(["country_iso3", "date"])

    total_episodes  = 0
    caught_episodes = 0
    lead_times      = []

    for _, grp in sub.groupby("country_iso3"):
        grp    = grp.reset_index(drop=True)
        yt     = grp["y_true"].values.astype(float)
        yp     = grp["y_pred"].values
        valid  = ~np.isnan(yt)
        yt_int = np.where(valid, yt.astype(int), 0)

        for s, e in _find_episodes(yt_int, pd.DatetimeIndex(grp["date"])):
            total_episodes += 1
            ep_preds = yp[s:e + 1]
            if np.any(ep_preds >= threshold):
                caught_episodes += 1
                lead_times.append(int(np.argmax(ep_preds >= threshold)))

    hit_rate = caught_episodes / total_episodes if total_episodes > 0 else np.nan
    return {
        "total_episodes":  total_episodes,
        "caught_episodes": caught_episodes,
        "hit_rate":        round(hit_rate, 4),
        "lead_times":      lead_times,
    }


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def make_table1(all_data: dict) -> None:
    rows = []
    for tkey, tdata in all_data.items():
        for H in HORIZONS:
            preds = tdata[H]["preds"]
            for mn, label in MODEL_LABELS.items():
                m = _pooled_metrics(preds, mn)
                rows.append({"target": TARGETS[tkey], "horizon": H, "model": label, **m})
    table = pd.DataFrame(rows)
    table.to_csv(TAB_DIR / "table1_overall_metrics.csv", index=False)
    print("Table 1 — overall metrics:")
    print(table.to_string(index=False))
    print()


def make_table2(all_data: dict) -> None:
    frames = []
    for tkey, tdata in all_data.items():
        for H in HORIZONS:
            m = tdata[H]["metrics"].copy()
            m["target"]      = TARGETS[tkey]
            m["model_label"] = m["model_name"].map(MODEL_LABELS)
            frames.append(m)
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(TAB_DIR / "table2_fold_metrics.csv", index=False)
    print("Table 2 — per-fold metrics saved.")


def make_table3_xgb_importance(all_data: dict) -> None:
    rows = []
    for tkey, tdata in all_data.items():
        for H in HORIZONS:
            xgb = tdata[H]["coefs_xgb"]
            if xgb.empty:
                continue
            xgb = xgb.copy()
            xgb["feature"] = xgb["feature"].str.replace("num__", "", regex=False)
            xgb = xgb[~xgb["feature"].str.startswith("fe__")]
            mean_imp = (
                xgb.groupby("feature")["importance"]
                .mean()
                .reset_index()
                .rename(columns={"importance": "mean_importance"})
                .sort_values("mean_importance", ascending=False)
            )
            mean_imp["target"]  = TARGETS[tkey]
            mean_imp["horizon"] = H
            rows.append(mean_imp)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            TAB_DIR / "table3_xgb_importance.csv", index=False
        )
        print("Table 3 — XGBoost mean importance saved.")


# ---------------------------------------------------------------------------
# Figures — all accept out_dir and models dict
# ---------------------------------------------------------------------------

def fig1_model_comparison(all_data: dict, models: dict[str, str], out_dir: Path, label: str) -> None:
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(12, 5 * len(TARGETS)))
    if len(TARGETS) == 1:
        axes = [axes]

    for row_ax, (tkey, tlabel) in zip(axes, TARGETS.items()):
        tdata      = all_data[tkey]
        model_keys = list(models.keys())
        mlabels    = [models[m] for m in model_keys]

        for ax, H in zip(row_ax, HORIZONS):
            roc_vals = [_pooled_metrics(tdata[H]["preds"], mn)["roc_auc"] for mn in model_keys]
            pr_vals  = [_pooled_metrics(tdata[H]["preds"], mn)["pr_auc"]  for mn in model_keys]

            y = np.arange(len(model_keys))
            ax.plot(roc_vals, y, "o", color="#2166ac", markersize=8, label="ROC-AUC", zorder=3)
            ax.plot(pr_vals,  y, "s", color="#b2182b", markersize=8, label="PR-AUC",  zorder=3)
            for i, (r, p) in enumerate(zip(roc_vals, pr_vals)):
                ax.plot([r, p], [i, i], color="#cccccc", linewidth=1.5, zorder=2)

            ax.set_yticks(y)
            ax.set_yticklabels(mlabels)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Score")
            ax.set_title(f"{tlabel}  |  H = {H} days")
            ax.axvline(0.5, color="#cccccc", linewidth=1, linestyle="--")
            ax.legend(frameon=False, loc="lower right")

    fig.suptitle(f"Model performance — {label}", x=0.05, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig1_model_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig1_model_comparison -> {out_dir.name}")


def fig2_fold_roc_auc(all_data: dict, models: dict[str, str], out_dir: Path, label: str) -> None:
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(12, 4.5 * len(TARGETS)), sharey=False)
    if len(TARGETS) == 1:
        axes = [axes]

    for row_ax, (tkey, tlabel) in zip(axes, TARGETS.items()):
        tdata = all_data[tkey]
        for ax, H in zip(row_ax, HORIZONS):
            metrics   = tdata[H]["metrics"]
            fold_ids  = sorted(metrics["fold_id"].unique())
            test_years = [fid + 2019 for fid in fold_ids]

            for mn, mlabel in models.items():
                vals = []
                for fid in fold_ids:
                    row = metrics[(metrics["model_name"] == mn) & (metrics["fold_id"] == fid)]
                    vals.append(row["roc_auc"].values[0] if not row.empty else np.nan)
                ax.plot(test_years, vals, marker="o", label=mlabel,
                        color=MODEL_COLORS[mn], linewidth=1.8, markersize=5)

            ax.set_title(f"{tlabel}  |  H = {H} days")
            ax.set_xlabel("Test year")
            ax.set_xticks(test_years)
            ax.set_ylabel("ROC-AUC")
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"ROC-AUC by test year — {label}", x=0.05, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_fold_roc_auc.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig2_fold_roc_auc -> {out_dir.name}")


def fig3_pr_curves(all_data: dict, models: dict[str, str], out_dir: Path, label: str) -> None:
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(12, 5 * len(TARGETS)))
    if len(TARGETS) == 1:
        axes = [axes]

    for row_ax, (tkey, tlabel) in zip(axes, TARGETS.items()):
        tdata = all_data[tkey]
        for ax, H in zip(row_ax, HORIZONS):
            preds = tdata[H]["preds"]
            for mn, mlabel in models.items():
                sub = preds[preds["model_name"] == mn]
                if sub.empty:
                    continue
                yt = sub["y_true"].values.astype(float)
                yp = sub["y_pred"].values
                mask = ~(np.isnan(yt) | np.isnan(yp))
                yt, yp = yt[mask], yp[mask]
                if len(np.unique(yt)) < 2:
                    continue
                prec, rec, _ = precision_recall_curve(yt, yp)
                ap = average_precision_score(yt, yp)
                ax.plot(rec, prec, label=f"{mlabel}  (AP={ap:.2f})",
                        color=MODEL_COLORS[mn], linewidth=1.8)

            baseline = preds["y_true"].mean()
            ax.axhline(baseline, color="#aaaaaa", linewidth=1, linestyle="--",
                       label=f"No skill  ({baseline:.2f})")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title(f"{tlabel}  |  H = {H} days")
            ax.legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.suptitle(f"Precision-recall curves — {label}", x=0.05, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_pr_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig3_pr_curves -> {out_dir.name}")


def fig4_xgb_importance(all_data: dict, model_name: str, out_dir: Path, label: str) -> None:
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(13, 7 * len(TARGETS)))
    if len(TARGETS) == 1:
        axes = [axes]

    for row_ax, (tkey, tlabel) in zip(axes, TARGETS.items()):
        tdata = all_data[tkey]
        for ax, H in zip(row_ax, HORIZONS):
            xgb = tdata[H]["coefs_xgb"]
            if xgb.empty:
                ax.set_visible(False)
                continue

            xgb = xgb[xgb["model_name"] == model_name].copy() if "model_name" in xgb.columns else xgb.copy()
            xgb["feature"] = xgb["feature"].str.replace("num__", "", regex=False)
            xgb = xgb[~xgb["feature"].str.startswith("fe__")]

            mean_imp = (
                xgb.groupby("feature")["importance"]
                .mean()
                .nlargest(20)
                .sort_values()
            )

            colors = [
                "#e08214" if "fao" in f else
                "#4dac26" if f.startswith("month") else
                "#4393c3"
                for f in mean_imp.index
            ]
            ax.barh(mean_imp.index, mean_imp.values, color=colors, edgecolor="white", height=0.7)
            ax.set_xlabel("Mean importance (gain)")
            ax.set_title(f"{tlabel}  |  H = {H} days")
            ax.tick_params(axis="y", labelsize=9)

    legend_elements = [
        mpatches.Patch(facecolor="#4393c3", label="Markets / macro / governance"),
        mpatches.Patch(facecolor="#e08214", label="Food prices (FAO)"),
        mpatches.Patch(facecolor="#4dac26", label="Seasonality"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"XGBoost feature importance — {label}", x=0.05, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_xgb_importance.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig4_xgb_importance -> {out_dir.name}")


def fig5_calibration(all_data: dict, model_name: str, out_dir: Path, label: str) -> None:
    fig, axes = plt.subplots(len(TARGETS), 2, figsize=(10, 4.5 * len(TARGETS)))
    if len(TARGETS) == 1:
        axes = [axes]

    for row_ax, (tkey, tlabel) in zip(axes, TARGETS.items()):
        tdata = all_data[tkey]
        for ax, H in zip(row_ax, HORIZONS):
            preds = tdata[H]["preds"]
            sub   = preds[preds["model_name"] == model_name]
            if sub.empty:
                ax.set_visible(False)
                continue

            yt = sub["y_true"].values.astype(float)
            for pred_col, col_label, color, marker in [
                ("y_pred_raw", "Before calibration", "#969696", "s"),
                ("y_pred",     "After calibration",  "#b2182b", "o"),
            ]:
                if pred_col not in sub.columns:
                    continue
                yp = sub[pred_col].values
                mp, ma, ns = _calibration_bins(yt, yp, n_bins=8)
                sizes = (ns / ns.max()) * 180 + 30
                ax.scatter(mp, ma, s=sizes, color=color, alpha=0.85, zorder=3,
                           edgecolors="white", linewidths=0.5, marker=marker, label=col_label)

            ax.plot([0, 1], [0, 1], "--", color="#aaaaaa", linewidth=1, label="Perfect")
            ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed rate")
            ax.set_title(f"{tlabel}  |  H = {H} days")
            ax.legend(frameon=False, fontsize=9)

    fig.suptitle(f"Calibration — {label}", x=0.05, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "fig5_calibration.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  fig5_calibration -> {out_dir.name}")


def fig6_country_roc(all_data: dict, model_name: str, out_dir: Path, label: str) -> pd.DataFrame:
    all_rows = []
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(8 * len(TARGETS), 1))  # placeholder size

    # Compute per-target, then build a combined figure
    target_cdfs = {}
    for tkey, tlabel in TARGETS.items():
        preds = all_data[tkey][7]["preds"]
        sub   = preds[preds["model_name"] == model_name]
        rows  = []
        for iso3, grp in sub.groupby("country_iso3"):
            yt = grp["y_true"].values.astype(float)
            yp = grp["y_pred"].values
            mask = ~(np.isnan(yt) | np.isnan(yp))
            yt, yp = yt[mask], yp[mask]
            if len(np.unique(yt)) < 2 or len(yt) < 20:
                continue
            rows.append({
                "country":  iso3,
                "roc_auc":  round(roc_auc_score(yt, yp), 4),
                "pos_rate": round(float(yt.mean()), 4),
                "n":        len(yt),
                "target":   tlabel,
            })
        cdf = pd.DataFrame(rows).sort_values("roc_auc", ascending=True)
        target_cdfs[tkey] = cdf
        all_rows.append(cdf)

    plt.close(fig)

    pd.concat(all_rows, ignore_index=True).to_csv(
        TAB_DIR / f"table4_country_roc_auc_{out_dir.name}.csv", index=False
    )

    for tkey, tlabel in TARGETS.items():
        cdf = target_cdfs[tkey]
        if cdf.empty:
            continue
        fig2, ax = plt.subplots(figsize=(8, max(5, len(cdf) * 0.27)))
        colors = [
            "#2166ac" if v >= 0.70 else
            "#e08214" if v >= 0.60 else
            "#b2182b"
            for v in cdf["roc_auc"]
        ]
        ax.barh(cdf["country"], cdf["roc_auc"], color=colors, edgecolor="white", height=0.7)
        ax.axvline(0.70, color="#333333", linewidth=0.9, linestyle="--")
        ax.set_xlim(0.40, 1.0)
        ax.set_xlabel("ROC-AUC")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.text(0.705, ax.get_ylim()[0] - 0.8, "0.70", fontsize=8, color="#333333")
        fig2.suptitle(
            f"Country-level ROC-AUC — {label}, {tlabel}, H=7 days",
            x=0.05, ha="left", fontsize=13,
        )
        fig2.tight_layout()
        fig2.savefig(out_dir / f"fig6_country_roc_{tkey}.png", bbox_inches="tight")
        plt.close(fig2)

    print(f"  fig6_country_roc -> {out_dir.name}")
    return pd.concat(all_rows, ignore_index=True)


def fig7_prob_timeseries(all_data: dict, model_name: str, out_dir: Path, label: str) -> None:
    for tkey, tlabel in TARGETS.items():
        preds = all_data[tkey][7]["preds"]
        sub   = preds[preds["model_name"] == model_name].copy()

        if sub.empty:
            print(f"  fig7 skipped for {tkey} — no predictions for {model_name}.")
            continue

        country_stats = sub.groupby("country_iso3").agg(
            std_pred=("y_pred", "std"),
            pos_rate=("y_true", "mean"),
        )
        varied = country_stats[
            (country_stats["std_pred"] > 0.03) &
            (country_stats["pos_rate"] > 0.01) &
            (country_stats["pos_rate"] < 0.99)
        ].sort_values("pos_rate")
        if len(varied) < 6:
            varied = country_stats.sort_values("pos_rate")

        n        = len(varied)
        step     = max(1, n // 6)
        selected = [varied.index[min(i * step, n - 1)] for i in range(6)]

        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=False)
        axes_flat = axes.flatten()

        for ax, iso3 in zip(axes_flat, selected):
            c = sub[sub["country_iso3"] == iso3].sort_values("date")
            c = c.set_index("date").resample("ME").agg(
                y_pred=("y_pred", "mean"),
                y_true=("y_true", "mean"),
            ).reset_index()
            ax.fill_between(c["date"], c["y_true"], alpha=0.15, color="#4393c3", label="Observed rate")
            ax.plot(c["date"], c["y_pred"], color="#b2182b", linewidth=1.5, label="Predicted")
            ax.set_ylim(-0.05, 1.10)
            ax.set_title(iso3, fontsize=11)
            ax.set_ylabel("Probability")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
            ax.tick_params(axis="x", labelsize=9)

        handles, lbls = axes_flat[0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="lower center", ncol=2,
                   frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            f"Predicted probability — {tlabel} — {label} (next 7 days)",
            x=0.05, ha="left", fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"fig7_prob_timeseries_{tkey}.png", bbox_inches="tight")
        plt.close(fig)

    print(f"  fig7_prob_timeseries -> {out_dir.name}")


def fig8_lead_time(all_data: dict, model_name: str, out_dir: Path, label: str) -> None:
    for tkey, tlabel in TARGETS.items():
        tdata = all_data[tkey]
        H     = 7

        sub  = tdata[H]["preds"]
        sub5 = sub[sub["model_name"] == model_name]
        if sub5.empty:
            continue

        yt   = sub5["y_true"].values.astype(float)
        yp   = sub5["y_pred"].values
        mask = ~(np.isnan(yt) | np.isnan(yp))
        opt  = _optimal_threshold_metrics(yt[mask], yp[mask])
        threshold = opt["threshold"]

        stats      = _event_level_stats(tdata[H]["preds"], model_name, threshold)
        lead_times = np.array(stats["lead_times"])

        if len(lead_times) == 0:
            print(f"  fig8 skipped for {tkey} — no caught episodes.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        ax = axes[0]
        max_lt = int(lead_times.max()) + 1
        bins   = np.arange(-0.5, max_lt + 0.5, 1)
        ax.hist(lead_times, bins=bins, color="#4393c3", edgecolor="white", alpha=0.9)
        ax.axvline(np.median(lead_times), color="#333333", linewidth=1.2, linestyle="--",
                   label=f"Median = {np.median(lead_times):.0f} days")
        ax.set_xlabel("Days from episode start to first alert")
        ax.set_ylabel("Episodes")
        ax.set_title("Lead time distribution")
        ax.legend(frameon=False)

        ax2 = axes[1]
        max_days = min(H, max_lt)
        days     = np.arange(0, max_days + 1)
        cum_frac = [np.mean(lead_times <= d) * stats["hit_rate"] for d in days]
        ax2.plot(days, cum_frac, color="#4393c3", linewidth=2, marker="o", markersize=5)
        ax2.axhline(stats["hit_rate"], color="#aaaaaa", linewidth=1, linestyle="--",
                    label=f"Overall hit rate = {stats['hit_rate']:.0%}")
        ax2.set_xlabel("Days from episode start")
        ax2.set_ylabel("Fraction of episodes caught")
        ax2.set_ylim(0, 1.05)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax2.set_title("Cumulative catch rate")
        ax2.legend(frameon=False)

        fig.suptitle(
            f"{label} — {tlabel} — {stats['caught_episodes']} of {stats['total_episodes']} caught "
            f"({stats['hit_rate']:.0%})",
            x=0.05, ha="left", fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"fig8_lead_time_{tkey}.png", bbox_inches="tight")
        plt.close(fig)

    print(f"  fig8_lead_time -> {out_dir.name}")


def fig9_weekly_timeseries(all_data: dict, model_name: str, out_dir: Path, label: str) -> None:
    panel_path = MOD_ROOT / "data" / "interim" / "modelling_panel.parquet"
    panel = pd.read_parquet(panel_path, columns=["country_iso3", "date", "acled_events"])
    panel["date"] = pd.to_datetime(panel["date"])

    for tkey, tlabel in TARGETS.items():
        preds = all_data[tkey][7]["preds"]
        sub   = preds[preds["model_name"] == model_name].copy()

        if sub.empty:
            print(f"  fig9 skipped for {tkey} — no predictions for {model_name}.")
            continue

        sub = sub.merge(panel, on=["country_iso3", "date"], how="left")

        country_stats = sub.groupby("country_iso3").agg(
            std_pred=("y_pred", "std"),
            pos_rate=("y_true", "mean"),
        )
        varied = country_stats[
            (country_stats["std_pred"] > 0.03) &
            (country_stats["pos_rate"] > 0.01) &
            (country_stats["pos_rate"] < 0.99)
        ].sort_values("pos_rate")
        if len(varied) < 6:
            varied = country_stats.sort_values("pos_rate")

        n        = len(varied)
        step     = max(1, n // 6)
        selected = [varied.index[min(i * step, n - 1)] for i in range(6)]

        fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=False)
        axes_flat = axes.flatten()

        for ax, iso3 in zip(axes_flat, selected):
            c = sub[sub["country_iso3"] == iso3].sort_values("date")

            ax2 = ax.twinx()
            ax2.bar(c["date"], c["acled_events"], width=5, color="#969696", alpha=0.4,
                    label="ACLED event count")
            ax2.set_ylabel("Event count", fontsize=9, color="#969696")
            ax2.tick_params(axis="y", labelcolor="#969696", labelsize=8)
            ax2.spines["right"].set_visible(True)

            ax.plot(c["date"], c["y_pred"], color="#b2182b", linewidth=1.2,
                    label="Predicted probability", zorder=3)
            ax.set_ylim(-0.05, 1.10)
            ax.set_title(iso3, fontsize=11)
            ax.set_ylabel("Probability")
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))
            ax.tick_params(axis="x", labelsize=9)
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)

        from matplotlib.patches import Patch
        handles = [
            Patch(facecolor="#969696", alpha=0.4, label="ACLED event count"),
            plt.Line2D([0], [0], color="#b2182b", linewidth=1.5, label="Predicted probability"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            f"Predicted probability vs ACLED event counts — {tlabel} — {label} (next 7 days)",
            x=0.05, ha="left", fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"fig9_weekly_timeseries_{tkey}.png", bbox_inches="tight")
        plt.close(fig)

    print(f"  fig9_weekly_timeseries -> {out_dir.name}")


def fig10_lr_vs_xgb(all_data: dict, lr_model: str, xgb_model: str, out_dir: Path, label: str) -> None:
    TOP_N = 20

    for tkey, tlabel in TARGETS.items():
        tdata = all_data[tkey]
        fig, axes = plt.subplots(len(HORIZONS), 2, figsize=(14, 6 * len(HORIZONS)))

        for row_idx, H in enumerate(HORIZONS):
            # --- LR coefficients ---
            coefs = tdata[H]["coefs"]
            lr = coefs[coefs["model_name"] == lr_model].copy() if "model_name" in coefs.columns else coefs.copy()
            lr["feature"] = lr["feature"].str.replace("num__", "", regex=False)
            lr = lr[~lr["feature"].str.startswith("fe__")]
            lr_mean = (
                lr.groupby("feature")["coefficient"]
                .mean()
                .reindex(lr.groupby("feature")["coefficient"].mean().abs().nlargest(TOP_N).index)
                .sort_values()
            )

            ax_lr = axes[row_idx, 0]
            colors_lr = ["#b2182b" if v > 0 else "#2166ac" for v in lr_mean.values]
            ax_lr.barh(lr_mean.index, lr_mean.values, color=colors_lr, edgecolor="white", height=0.7)
            ax_lr.axvline(0, color="#333333", linewidth=0.8)
            ax_lr.set_xlabel("Mean coefficient (standardised)")
            ax_lr.set_title(f"LR coefficients  |  {tlabel}  |  H = {H}d")
            ax_lr.tick_params(axis="y", labelsize=8)

            # --- XGBoost importances ---
            xgb_df = tdata[H]["coefs_xgb"]
            if xgb_df.empty:
                axes[row_idx, 1].set_visible(False)
                continue
            xgb_df = xgb_df[xgb_df["model_name"] == xgb_model].copy() if "model_name" in xgb_df.columns else xgb_df.copy()
            xgb_df["feature"] = xgb_df["feature"].str.replace("num__", "", regex=False)
            xgb_df = xgb_df[~xgb_df["feature"].str.startswith("fe__")]
            xgb_mean = (
                xgb_df.groupby("feature")["importance"]
                .mean()
                .nlargest(TOP_N)
                .sort_values()
            )

            ax_xgb = axes[row_idx, 1]
            colors_xgb = [
                "#e08214" if "fao" in f else
                "#4dac26" if f.startswith("month") else
                "#4393c3"
                for f in xgb_mean.index
            ]
            ax_xgb.barh(xgb_mean.index, xgb_mean.values, color=colors_xgb, edgecolor="white", height=0.7)
            ax_xgb.set_xlabel("Mean importance (gain)")
            ax_xgb.set_title(f"XGBoost importance  |  {tlabel}  |  H = {H}d")
            ax_xgb.tick_params(axis="y", labelsize=8)

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#b2182b", label="LR: positive coefficient"),
            Patch(facecolor="#2166ac", label="LR: negative coefficient"),
            Patch(facecolor="#4393c3", label="XGB: markets / macro / governance"),
            Patch(facecolor="#e08214", label="XGB: food prices (FAO)"),
            Patch(facecolor="#4dac26", label="XGB: seasonality"),
        ]
        fig.legend(handles=legend_elements, loc="lower center", ncol=3,
                   frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle(
            f"LR coefficients vs XGBoost importance — {tlabel} — {label}",
            x=0.05, ha="left", fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"fig10_lr_vs_xgb_{tkey}.png", bbox_inches="tight")
        plt.close(fig)

    print(f"  fig10_lr_vs_xgb -> {out_dir.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data...")
    all_data = load_all()

    print("Writing tables...")
    make_table1(all_data)
    make_table2(all_data)
    make_table3_xgb_importance(all_data)

    for out_dir, models, primary_model, label in [
        (FIG_DIR_LAG,   LAG_MODELS,   PRIMARY_LAG_MODEL,   "with lags"),
        (FIG_DIR_NOLAG, NOLAG_MODELS, PRIMARY_NOLAG_MODEL, "no lags"),
    ]:
        print(f"\nGenerating figures — {label} -> {out_dir.name}/")
        fig1_model_comparison(all_data, models, out_dir, label)
        fig2_fold_roc_auc(all_data, models, out_dir, label)
        fig3_pr_curves(all_data, models, out_dir, label)
        fig4_xgb_importance(all_data, primary_model, out_dir, label)
        fig5_calibration(all_data, primary_model, out_dir, label)
        fig6_country_roc(all_data, primary_model, out_dir, label)
        fig7_prob_timeseries(all_data, primary_model, out_dir, label)
        fig8_lead_time(all_data, primary_model, out_dir, label)
        fig9_weekly_timeseries(all_data, primary_model, out_dir, label)
        fig10_lr_vs_xgb(all_data, "model4_fao", primary_model, out_dir, label)

    print(f"\nFigures: {FIG_DIR_LAG}  |  {FIG_DIR_NOLAG}")
    print(f"Tables:  {TAB_DIR}")


if __name__ == "__main__":
    main()
