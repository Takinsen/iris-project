import os
import glob
import pickle
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


# ------------------------------------------------------------------
# Pickle helper — resolves HammingThresholdClassifier pickled from
# __main__ (when train.py was run directly) into train.module scope.
# ------------------------------------------------------------------

class _TrainUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            import train as _train
            if hasattr(_train, name):
                return getattr(_train, name)
        return super().find_class(module, name)


def _pickle_load(path: str):
    with open(path, "rb") as f:
        return _TrainUnpickler(f).load()


# ------------------------------------------------------------------
# NaN-safe statistics helpers
# ------------------------------------------------------------------

def _safe_mean(vals: list) -> float:
    finite = [float(v) for v in vals
              if v is not None and not np.isnan(float(v))]
    return float(np.mean(finite)) if finite else float("nan")


def _safe_std(vals: list) -> float:
    finite = [float(v) for v in vals
              if v is not None and not np.isnan(float(v))]
    return float(np.std(finite)) if len(finite) > 1 else 0.0


# ------------------------------------------------------------------
# Configuration  (must match train.py)
# ------------------------------------------------------------------

TARGET_EYE_SIDE = "L"
OUTPUT_ROOT    = rf".\out_CASIA_Iris_Thousand_MultiSet_{TARGET_EYE_SIDE}"
MODEL_SAVE_DIR = os.path.join(".", "model")
VIZ_DIR        = os.path.join(OUTPUT_ROOT, "visualization")
FEATURE_COLS   = ["hamming", "jaccard", "cosine", "pearson"]

MODEL_DIRS = {
    "Baseline (Hamming)": "baseline_hamming",
    "Random Forest":      "random_forest",
    "Logistic Regression": "logistic_regression",
    "XGBoost":            "xgboost",
    "MLP":                "mlp",
}

_PALETTE = {
    "genuine":  "#2196F3",
    "impostor": "#F44336",
    "far":      "#E91E63",
    "frr":      "#009688",
    "eer":      "#FF9800",
    "roc":      "#673AB7",
}

METRIC_LABELS = {
    "accuracy":          "Accuracy",
    "balanced_accuracy": "Balanced Acc.",
    "precision":         "Precision",
    "recall":            "Recall / TPR",
    "f1":                "F1 Score",
    "eer":               "EER (lower=better)",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _model_slug(name: str) -> str:
    return (name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .strip("_"))


def _compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """Returns (eer, eer_threshold). y_score must be genuine-ascending."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def _genuine_score(b: dict) -> np.ndarray:
    """
    Returns y_score in genuine-ascending order (higher = more genuine).
    For distance models (score = Hamming distance), flips to 1 - score.
    """
    s = b["all_y_score"]
    return 1.0 - s if b.get("score_is_distance") else s


def _save(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved → {path}")


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_all_multi_score_features() -> list:
    frames = []
    for fpath in sorted(glob.glob(
            os.path.join(OUTPUT_ROOT, "**", "multi_score_features.csv"),
            recursive=True)):
        df = pd.read_csv(fpath)
        if len(df) > 0:
            frames.append(df)
    return frames


def load_model_results(model_name: str) -> dict | None:
    """
    Loads fold_*.csv + fold_metrics.csv + final_model.pkl for a model.
    Returns a bucket dict, or None if no fold CSVs are found.
    """
    slug      = _model_slug(model_name)
    model_dir = os.path.join(MODEL_SAVE_DIR, slug)
    fold_paths = sorted(glob.glob(os.path.join(model_dir, "fold_[0-9]*.csv")))
    if not fold_paths:
        return None

    all_dfs  = [pd.read_csv(p) for p in fold_paths]
    combined = pd.concat(all_dfs, ignore_index=True)

    all_y_true    = combined["y_true"].values.astype(int)
    all_y_pred    = combined["y_pred"].values.astype(int)
    all_y_score   = combined["y_score"].values.astype(float)   # raw (Hamming for baseline)
    all_distances = combined["hamming"].values.astype(float)
    all_pair_types = (combined["pair_type"].tolist()
                      if "pair_type" in combined.columns else
                      ["genuine" if y else "impostor" for y in all_y_true])

    fm_path      = os.path.join(model_dir, "fold_metrics.csv")
    fold_metrics = []
    set_ids      = []
    if os.path.exists(fm_path):
        fm_df        = pd.read_csv(fm_path)
        fold_metrics = fm_df.to_dict("records")
        set_ids      = [str(m.get("set_id", f"fold_{i+1:02d}"))
                        for i, m in enumerate(fold_metrics)]

    metric_keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "eer",
                   "threshold", "mean_genuine_distance", "mean_impostor_distance"]
    avg_metrics = {
        k: _safe_mean([m.get(k) for m in fold_metrics])
        for k in metric_keys
    }

    # Baseline stores raw Hamming distance as y_score (lower = more genuine)
    score_is_distance = (slug == "baseline_hamming")

    # Keep individual fold DataFrames for per-set FAR/FRR and EER-threshold CM
    fold_dfs = all_dfs

    final_model = None
    final_path  = os.path.join(model_dir, "final_model.pkl")
    if os.path.exists(final_path):
        try:
            final_model = _pickle_load(final_path)
        except Exception as e:
            print(f"  [warn] could not load final_model.pkl: {e}")

    return {
        "fold_metrics":      fold_metrics,
        "fold_dfs":          fold_dfs,
        "set_ids":           set_ids,
        "all_y_true":        all_y_true,
        "all_y_pred":        all_y_pred,
        "all_y_score":       all_y_score,     # raw score (Hamming for baseline)
        "all_distances":     all_distances,
        "all_pair_types":    all_pair_types,
        "final_model":       final_model,
        "avg_metrics":       avg_metrics,
        "score_is_distance": score_is_distance,
    }


# ==================================================================
# Baseline-specific plots
# ==================================================================

def plot_distance_distributions(b: dict, out_dir: str) -> None:
    """Hamming distance density — genuine vs impostor."""
    y_true = b["all_y_true"]
    gen    = b["all_distances"][y_true == 1]
    imp    = b["all_distances"][y_true == 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 80)
    ax.hist(gen, bins=bins, density=True, alpha=0.55,
            color=_PALETTE["genuine"],  label=f"Genuine  (n={len(gen):,})")
    ax.hist(imp, bins=bins, density=True, alpha=0.55,
            color=_PALETTE["impostor"], label=f"Impostor (n={len(imp):,})")
    thr = b["avg_metrics"].get("threshold", float("nan"))
    if not np.isnan(thr):
        ax.axvline(thr, color="black", ls="--", lw=1.5,
                   label=f"Avg threshold = {thr:.4f}")
    ax.set_xlabel("Hamming Distance")
    ax.set_ylabel("Density")
    ax.set_title("Hamming Distance Distribution — Genuine vs Impostor")
    ax.legend()
    _save(fig, os.path.join(out_dir, "distance_distribution.png"))


def plot_loso_threshold_per_set(b: dict, out_dir: str) -> None:
    """Per-fold fitted Hamming threshold (Baseline only)."""
    thresholds = [m.get("threshold", float("nan")) for m in b["fold_metrics"]]
    set_ids    = b["set_ids"]
    fig, ax = plt.subplots(figsize=(max(8, len(set_ids) * 0.7), 5))
    ax.plot(set_ids, thresholds, marker="o", color=_PALETTE["genuine"], lw=2)
    avg = float(np.nanmean(thresholds))
    ax.axhline(avg, color="gray", ls="--", lw=1.2, label=f"Mean = {avg:.4f}")
    ax.set_xlabel("Set")
    ax.set_ylabel("Threshold")
    ax.set_title("LOSO Hamming Threshold per Set")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    _save(fig, os.path.join(out_dir, "loso_threshold_per_set.png"))


# ==================================================================
# Per-model plots
# ==================================================================

def plot_model_confusion_matrix(b: dict, model_name: str, out_dir: str) -> None:
    """Aggregate confusion matrix at the per-fold EER threshold."""
    is_dist  = b.get("score_is_distance", False)
    tp = fp = fn = tn = 0

    for df in b.get("fold_dfs", []):
        y_true = df["y_true"].values.astype(int)
        y_raw  = df["y_score"].values.astype(float)
        y_roc  = 1.0 - y_raw if is_dist else y_raw
        try:
            _, eer_thr = _compute_eer(y_true, y_roc)
        except Exception:
            continue
        y_pred = (y_roc >= eer_thr).astype(int)
        tp += int(((y_true == 1) & (y_pred == 1)).sum())
        fp += int(((y_true == 0) & (y_pred == 1)).sum())
        fn += int(((y_true == 1) & (y_pred == 0)).sum())
        tn += int(((y_true == 0) & (y_pred == 0)).sum())

    cm = np.array([[tp, fn], [fp, tn]])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    labels = [["TP", "FN"], ["FP", "TN"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{labels[i][j]}\n{cm[i, j]:,}",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Genuine", "Pred Impostor"])
    ax.set_yticklabels(["Actual Genuine", "Actual Impostor"])
    ax.set_title(f"Aggregate Confusion Matrix — {model_name}")
    _save(fig, os.path.join(out_dir, "confusion_matrix.png"))


def plot_model_roc_curve(b: dict, model_name: str, out_dir: str) -> None:
    y_sc        = _genuine_score(b)   # genuine-ascending (flip if distance model)
    fpr, tpr, _ = roc_curve(b["all_y_true"], y_sc)
    roc_auc     = auc(fpr, tpr)
    eer, _      = _compute_eer(b["all_y_true"], y_sc)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color=_PALETTE["roc"], lw=2,
            label=f"AUC = {roc_auc:.4f}  |  EER = {eer:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (1 − FRR)")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    _save(fig, os.path.join(out_dir, "roc_curve.png"))


def plot_model_metrics_per_set(b: dict, model_name: str, out_dir: str) -> None:
    """One subplot per metric (accuracy, precision, recall, f1, eer) across folds."""
    keys    = ["accuracy", "precision", "recall", "f1", "eer"]
    n_folds = len(b["fold_metrics"])
    if n_folds == 0:
        return

    set_ids = b["set_ids"]
    fig, axes = plt.subplots(len(keys), 1,
                              figsize=(max(10, n_folds * 0.8), 3.2 * len(keys)),
                              sharex=True)
    for ax, key in zip(axes, keys):
        vals = [m.get(key) for m in b["fold_metrics"]]
        avg  = _safe_mean(vals)
        ax.bar(set_ids, vals, color="#90CAF9", edgecolor="white")
        ax.axhline(avg, color="#1565C0", ls="--", lw=1.5,
                   label=f"Mean = {avg:.4f}")
        ax.set_ylabel(METRIC_LABELS.get(key, key), fontsize=9)
        ax.legend(fontsize=8)

    axes[-1].tick_params(axis="x", rotation=45)
    fig.suptitle(f"Per-Fold Metrics — {model_name}", y=1.01, fontsize=13)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "metrics_per_fold.png"))


def plot_far_frr_curve(b: dict, model_name: str, out_dir: str,
                       is_distance: bool = False) -> None:
    """
    FAR / FRR per test set (thin lines) + aggregate (bold), with EER markers.

    is_distance=True  — x-axis is Hamming distance threshold (baseline).
    is_distance=False — x-axis is probability / score threshold.
    """
    is_dist  = b.get("score_is_distance", False)
    fold_dfs = b.get("fold_dfs", [])
    set_ids  = b.get("set_ids", [f"fold_{i+1:02d}" for i in range(len(fold_dfs))])
    xlabel   = "Hamming Distance Threshold" if is_dist else "Score Threshold"

    fig, ax  = plt.subplots(figsize=(9, 6))
    per_eers = []

    # ---- Per-set thin curves ----
    for df in fold_dfs:
        y_true = df["y_true"].values.astype(int)
        y_raw  = df["y_score"].values.astype(float)
        y_roc  = 1.0 - y_raw if is_dist else y_raw
        try:
            fpr, tpr, thr = roc_curve(y_true, y_roc)
        except Exception:
            continue
        x_s = (1.0 - thr[1:]) if is_dist else thr[1:]
        ax.plot(x_s, fpr[1:],           color=_PALETTE["far"], lw=0.7, alpha=0.25)
        ax.plot(x_s, 1.0 - tpr[1:],    color=_PALETTE["frr"], lw=0.7, alpha=0.25, ls="--")
        try:
            eer_s, eer_t_s = _compute_eer(y_true, y_roc)
            eer_x_s = float(1.0 - eer_t_s) if is_dist else float(eer_t_s)
            ax.scatter([eer_x_s], [eer_s], color="gray", s=18, zorder=4)
            per_eers.append(eer_s)
        except Exception:
            pass

    # ---- Aggregate bold curves ----
    y_roc_all        = _genuine_score(b)
    fpr_a, tpr_a, thr_a = roc_curve(b["all_y_true"], y_roc_all)
    x_a   = (1.0 - thr_a[1:]) if is_dist else thr_a[1:]
    far_a = fpr_a[1:]
    frr_a = 1.0 - tpr_a[1:]
    eer_a, eer_t_a = _compute_eer(b["all_y_true"], y_roc_all)
    eer_x_a  = float(1.0 - eer_t_a) if is_dist else float(eer_t_a)
    mean_eer = float(np.mean(per_eers)) if per_eers else eer_a

    ax.plot(x_a, far_a, color=_PALETTE["far"], lw=2.5)
    ax.plot(x_a, frr_a, color=_PALETTE["frr"], lw=2.5, ls="--")
    ax.axvline(eer_x_a, color=_PALETTE["eer"], ls="--", lw=1.5)
    ax.scatter([eer_x_a], [eer_a], color=_PALETTE["eer"], zorder=5, s=70)

    # ---- Legend (manual handles) ----
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], color=_PALETTE["far"], lw=2.5,
               label="FAR — aggregate"),
        Line2D([0],[0], color=_PALETTE["frr"], lw=2.5, ls="--",
               label="FRR — aggregate"),
        Line2D([0],[0], color=_PALETTE["far"], lw=0.8, alpha=0.5,
               label=f"FAR — per set  (n={len(fold_dfs)})"),
        Line2D([0],[0], color=_PALETTE["frr"], lw=0.8, alpha=0.5, ls="--",
               label=f"FRR — per set  (n={len(fold_dfs)})"),
        Line2D([0],[0], color=_PALETTE["eer"], ls="--", lw=1.5,
               label=f"EER = {eer_a:.4f}  (per-set mean = {mean_eer:.4f})"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
               markersize=5, label="EER per set"),
    ]
    ax.legend(handles=handles, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"FAR / FRR per Set — {model_name}")
    _save(fig, os.path.join(out_dir, "far_frr_curve.png"))


def plot_feature_importance(b: dict, model_name: str, out_dir: str) -> None:
    """Horizontal bar chart — feature_importances_ (RF, XGBoost, Baseline)."""
    model = b.get("final_model")
    if model is None:
        return
    try:
        imps = model.feature_importances_
    except Exception:
        return
    indices = np.argsort(imps)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([FEATURE_COLS[i] for i in indices], imps[indices], color="#42A5F5")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}")
    _save(fig, os.path.join(out_dir, "feature_importance.png"))


def plot_model_coefficients(b: dict, model_name: str, out_dir: str) -> None:
    """Bar chart of LR coefficients from the final model."""
    model = b.get("final_model")
    if model is None:
        return
    try:
        coefs = model.coef_.ravel()
    except Exception:
        return
    colors = ["#42A5F5" if c >= 0 else "#EF5350" for c in coefs]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(FEATURE_COLS, coefs, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Coefficient")
    ax.set_title(f"Logistic Regression Coefficients — {model_name}")
    _save(fig, os.path.join(out_dir, "coefficients.png"))


# ==================================================================
# Per-model dashboard
# ==================================================================

def plot_model_dashboard(b: dict, model_name: str, out_dir: str) -> None:
    """
    Summary dashboard: metric cards (avg ± std) on top,
    per-fold bar charts on the bottom.
    Metrics: Accuracy, Balanced Accuracy, Precision, Recall, F1.
    """
    from matplotlib.gridspec import GridSpec

    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]
    card_labels = {
        "accuracy":          "Accuracy",
        "balanced_accuracy": "Balanced\nAccuracy",
        "precision":         "Precision",
        "recall":            "Recall",
        "f1":                "F1 Score",
    }
    folds = b["fold_metrics"]
    sids  = b["set_ids"]
    n_f   = len(folds)
    if n_f == 0:
        return

    n   = len(keys)
    fig = plt.figure(figsize=(4 * n, 8))
    gs  = GridSpec(2, n, figure=fig,
                   height_ratios=[1, 2.5], hspace=0.45, wspace=0.3)

    for j, key in enumerate(keys):
        vals = [m.get(key) for m in folds]
        avg  = _safe_mean(vals)
        std  = _safe_std(vals)

        # ---- Summary card (row 0) ----
        ax_c = fig.add_subplot(gs[0, j])
        bg   = "#C8E6C9" if avg >= 0.8 else ("#FFF9C4" if avg >= 0.6 else "#FFCDD2")
        ax_c.set_facecolor(bg)
        ax_c.tick_params(left=False, bottom=False,
                         labelleft=False, labelbottom=False)
        for sp in ax_c.spines.values():
            sp.set_edgecolor("#BDBDBD")
            sp.set_linewidth(0.8)
        ax_c.text(0.5, 0.70, f"{avg:.4f}",
                  ha="center", va="center", fontsize=20, fontweight="bold",
                  transform=ax_c.transAxes)
        ax_c.text(0.5, 0.36, f"± {std:.4f}",
                  ha="center", va="center", fontsize=10, color="#555555",
                  transform=ax_c.transAxes)
        ax_c.text(0.5, 0.93, card_labels[key],
                  ha="center", va="top", fontsize=10, fontweight="bold",
                  transform=ax_c.transAxes)

        # ---- Per-fold bar chart (row 1) ----
        ax_b  = fig.add_subplot(gs[1, j])
        x_pos = np.arange(n_f)
        plot_vals = [v if v is not None else float("nan") for v in vals]
        ax_b.bar(x_pos, plot_vals, color="#90CAF9", edgecolor="white", width=0.8)
        ax_b.axhline(avg, color="#1565C0", ls="--", lw=1.2,
                     label=f"Mean = {avg:.3f}")
        ax_b.set_ylim(0, 1.05)

        if n_f <= 20:
            ax_b.set_xticks(x_pos)
            ax_b.set_xticklabels(sids, rotation=90, fontsize=6)
        else:
            step = max(1, n_f // 10)
            ax_b.set_xticks(x_pos[::step])
            ax_b.set_xticklabels(
                [sids[i] for i in range(0, n_f, step)],
                rotation=90, fontsize=6)

        ax_b.set_title(card_labels[key], fontsize=9)
        ax_b.tick_params(axis="y", labelsize=8)
        ax_b.legend(fontsize=7, loc="lower right")

    fig.suptitle(f"Evaluation Dashboard — {model_name}",
                 fontsize=14, fontweight="bold", y=1.01)
    _save(fig, os.path.join(out_dir, "dashboard.png"))


# ==================================================================
# Root / comparison visualizations
# ==================================================================

def plot_global_distance_distribution(per_set_dfs: list, out_dir: str) -> None:
    """2×2 panel: Hamming, Jaccard, Cosine, Pearson — genuine vs impostor."""
    if not per_set_dfs:
        return
    all_df = pd.concat(per_set_dfs, ignore_index=True)
    if "pair_type" in all_df.columns:
        is_gen = all_df["pair_type"] == "genuine"
    elif "true_label" in all_df.columns:
        is_gen = all_df["true_label"] == 1
    else:
        print("  [warn] no pair_type or true_label column — skipping global distribution")
        return

    col_labels = {
        "hamming": "Hamming Distance",
        "jaccard": "Jaccard Distance",
        "cosine":  "Cosine Distance",
        "pearson": "Pearson Distance",
    }
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, col in zip(axes.flat, FEATURE_COLS):
        if col not in all_df.columns:
            ax.set_visible(False)
            continue
        gen_vals = all_df.loc[is_gen,  col].dropna().values
        imp_vals = all_df.loc[~is_gen, col].dropna().values
        if len(gen_vals) == 0 or len(imp_vals) == 0:
            ax.set_visible(False)
            continue
        lo   = min(gen_vals.min(), imp_vals.min())
        hi   = max(gen_vals.max(), imp_vals.max())
        bins = np.linspace(lo, hi, 80)
        ax.hist(gen_vals, bins=bins, density=True, alpha=0.55,
                color=_PALETTE["genuine"],  label=f"Genuine  (n={len(gen_vals):,})")
        ax.hist(imp_vals, bins=bins, density=True, alpha=0.55,
                color=_PALETTE["impostor"], label=f"Impostor (n={len(imp_vals):,})")
        label = col_labels.get(col, col)
        ax.set_xlabel(label)
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle("All-Set Distance Distributions — Genuine vs Impostor",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "global_distance_distribution.png"))


def plot_metrics_summary_table(results: dict, out_dir: str) -> None:
    """PNG table: rows = models, cols = avg metrics."""
    metric_keys = ["accuracy", "precision", "recall", "f1", "eer"]
    rows = []
    for name, b in results.items():
        if b is None:
            continue
        am = b["avg_metrics"]
        rows.append([name] + [
            f"{am.get(k, float('nan')):.4f}" for k in metric_keys
        ])
    if not rows:
        return

    col_headers = ["Model"] + [METRIC_LABELS.get(k, k) for k in metric_keys]
    fig_h = max(2.5, len(rows) * 0.65 + 1.5)
    fig_w = len(col_headers) * 2.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.9)
    for j in range(len(col_headers)):
        tbl[0, j].set_facecolor("#1565C0")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        bg = "#E3F2FD" if i % 2 == 0 else "white"
        for j in range(len(col_headers)):
            tbl[i, j].set_facecolor(bg)
    ax.set_title("Model Evaluation Summary (LOSO avg)",
                 pad=20, fontsize=13, fontweight="bold")
    _save(fig, os.path.join(out_dir, "metrics_summary_table.png"))


def plot_comparison_report(results: dict, out_dir: str) -> None:
    """Side-by-side bar charts comparing all models on key metrics."""
    metric_keys = ["accuracy", "precision", "recall", "f1", "eer"]
    model_names = [n for n, b in results.items() if b is not None]
    if not model_names:
        return

    colors = plt.cm.tab10(np.linspace(0, 0.8, len(model_names)))
    fig, axes = plt.subplots(1, len(metric_keys),
                              figsize=(4.5 * len(metric_keys), 5))
    if len(metric_keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, metric_keys):
        vals = [results[n]["avg_metrics"].get(key, float("nan"))
                for n in model_names]
        bars = ax.bar(range(len(model_names)), vals, color=colors)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=8)
        ax.set_title(METRIC_LABELS.get(key, key), fontsize=10)
        finite = [v for v in vals if not np.isnan(v)]
        ax.set_ylim(0, max(1.0, max(finite) * 1.15) if finite else 1.0)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Model Comparison Report", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "comparison_report.png"))


# ==================================================================
# Main
# ==================================================================

def main() -> None:
    ensure_dir(VIZ_DIR)

    print(f"Loading distance features from: {OUTPUT_ROOT}")
    per_set_dfs = load_all_multi_score_features()
    print(f"  {len(per_set_dfs)} set(s) found")

    print("\n[Root] global distance distribution (2×2 panel)")
    plot_global_distance_distribution(per_set_dfs, VIZ_DIR)

    results = {}
    for model_name in MODEL_DIRS:
        print(f"\n[{model_name}] loading results ...")
        b = load_model_results(model_name)
        results[model_name] = b
        if b is None:
            print("  no fold CSVs found — skipping")
            continue

        n_folds = len(b["fold_metrics"])
        eer_avg = b["avg_metrics"].get("eer", float("nan"))
        print(f"  {n_folds} fold(s) | avg EER = {eer_avg:.4f}")

        slug    = _model_slug(model_name)
        out_dir = ensure_dir(os.path.join(OUTPUT_ROOT, slug))
        is_base = b.get("score_is_distance", False)

        if is_base:
            plot_distance_distributions(b, out_dir)
            plot_loso_threshold_per_set(b, out_dir)

        plot_model_dashboard(b, model_name, out_dir)
        plot_model_confusion_matrix(b, model_name, out_dir)
        plot_model_roc_curve(b, model_name, out_dir)
        plot_model_metrics_per_set(b, model_name, out_dir)
        plot_far_frr_curve(b, model_name, out_dir, is_distance=is_base)

        if b.get("final_model") is not None:
            plot_feature_importance(b, model_name, out_dir)
            plot_model_coefficients(b, model_name, out_dir)

    print("\n[Root] metrics summary table")
    plot_metrics_summary_table(results, VIZ_DIR)

    print("\n[Root] comparison report")
    plot_comparison_report(results, VIZ_DIR)

    print("\nDone. Outputs:")
    print(f"  {VIZ_DIR}/")
    for model_name, b in results.items():
        if b is not None:
            slug = _model_slug(model_name)
            print(f"  {OUTPUT_ROOT}/{slug}/")


if __name__ == "__main__":
    main()
