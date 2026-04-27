import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (roc_curve, auc, accuracy_score, precision_score,
                             recall_score, f1_score, balanced_accuracy_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# ------------------------------------------------------------------
# Output paths
# ------------------------------------------------------------------
OUTPUT_ROOT  = r".\out_CASIA_Iris_Thousand_MultiSet_L"
VIZ_DIR      = os.path.join(OUTPUT_ROOT, "visualizations")
BASELINE_DIR = os.path.join(VIZ_DIR, "baseline")
RF_DIR       = os.path.join(VIZ_DIR, "random_forest")
LR_DIR       = os.path.join(VIZ_DIR, "logistic_regression")
GNB_DIR      = os.path.join(VIZ_DIR, "gaussian_naive_bayes")
LDA_DIR      = os.path.join(VIZ_DIR, "linear_discriminant_analysis")

FEATURE_COLS = ["hamming", "jaccard", "cosine", "pearson"]
MODEL_DIRS   = {
    "Random Forest":                RF_DIR,
    "Logistic Regression":          LR_DIR,
    "Gaussian Naive Bayes":         GNB_DIR,
    "Linear Discriminant Analysis": LDA_DIR,
}

# ------------------------------------------------------------------
# Model Configuration
# Toggle `enabled` to include/exclude a model.
# Edit `params` to change hyperparameters without touching training code.
# ------------------------------------------------------------------
MODELS_CONFIG = {
    "Random Forest": {
        "enabled": True,
        "params": {
            "n_estimators": 200,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs":       -1,
        },
    },
    "Logistic Regression": {
        "enabled": True,
        "params": {
            "C":            1.0,
            "class_weight": "balanced",
            "max_iter":     1000,
            "random_state": 42,
        },
    },
    "Gaussian Naive Bayes": {
        "enabled": True,
        "params": {},
    },
    "Linear Discriminant Analysis": {
        "enabled": True,
        "params": {
            "solver": "svd",
        },
    },
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _instantiate_model(name: str):
    params = MODELS_CONFIG[name]["params"]
    constructors = {
        "Random Forest":                RandomForestClassifier,
        "Logistic Regression":          LogisticRegression,
        "Gaussian Naive Bayes":         GaussianNB,
        "Linear Discriminant Analysis": LinearDiscriminantAnalysis,
    }
    return constructors[name](**params)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_per_set_pair_records() -> list:
    """Return a list of DataFrames, one per set, sorted by set folder name."""
    frames = []
    for fpath in sorted(glob.glob(os.path.join(OUTPUT_ROOT, "**", "pair_records.csv"), recursive=True)):
        df = pd.read_csv(fpath)
        if len(df) == 0:
            continue
        if "set_id" not in df.columns:
            set_id = os.path.basename(os.path.dirname(fpath))
            df["set_id"] = set_id
        if "pair_type" not in df.columns and "true_label" in df.columns:
            df["pair_type"] = df["true_label"].map({1: "genuine", 0: "impostor"})
        frames.append(df)
    return frames


def load_all_multi_score_features() -> list:
    frames = []
    for fpath in sorted(glob.glob(os.path.join(OUTPUT_ROOT, "**", "multi_score_features.csv"), recursive=True)):
        df = pd.read_csv(fpath)
        if len(df) > 0:
            frames.append(df)
    return frames


# ------------------------------------------------------------------
# Baseline — LOSO with optimised threshold
# ------------------------------------------------------------------

def _find_best_threshold(distances: np.ndarray, y_true: np.ndarray,
                         n_steps: int = 500) -> float:
    """Sweep thresholds on training data; return the one that maximises balanced accuracy."""
    thresholds = np.linspace(distances.min(), distances.max(), n_steps)
    best_thr, best_score = thresholds[0], -1.0
    for thr in thresholds:
        pred  = (distances <= thr).astype(int)
        score = float(balanced_accuracy_score(y_true, pred))
        if score > best_score:
            best_score, best_thr = score, thr
    return float(best_thr)


def run_baseline_loso_evaluation(per_set_pair_records: list) -> dict:
    """
    Leave-one-set-out cross-validation for the Hamming distance baseline.
    For each fold:
      - Concatenate n-1 sets → find the Hamming threshold that maximises
        balanced accuracy on those pairs.
      - Apply that threshold to the held-out set and record metrics.
    Returns:
      summary  — DataFrame with one row per fold (metrics, TP/FP/FN/TN, threshold)
      pairs    — DataFrame of all held-out pairs (for ROC / DET curves)
    """
    n = len(per_set_pair_records)
    rows          = []
    all_distances = []
    all_y_true    = []
    all_y_pred    = []
    all_pair_type = []

    for test_idx in range(n):
        print(f"  [baseline LOSO] fold {test_idx + 1}/{n} ...", end="\r", flush=True)

        train_df = pd.concat(
            [df for i, df in enumerate(per_set_pair_records) if i != test_idx],
            ignore_index=True,
        ).dropna(subset=["distance", "true_label"])
        test_df = per_set_pair_records[test_idx].dropna(subset=["distance", "true_label"])

        best_thr = _find_best_threshold(train_df["distance"].values,
                                        train_df["true_label"].values)

        d_te   = test_df["distance"].values
        y_te   = test_df["true_label"].values
        pred   = (d_te <= best_thr).astype(int)

        tp = int(((y_te == 1) & (pred == 1)).sum())
        fp = int(((y_te == 0) & (pred == 1)).sum())
        fn = int(((y_te == 1) & (pred == 0)).sum())
        tn = int(((y_te == 0) & (pred == 0)).sum())

        gen_d = test_df[test_df["pair_type"] == "genuine"]["distance"].dropna()
        imp_d = test_df[test_df["pair_type"] == "impostor"]["distance"].dropna()
        set_id = test_df["set_id"].iloc[0] if "set_id" in test_df.columns else f"set_{test_idx + 1:02d}"

        rows.append({
            "set_id":                 set_id,
            "threshold":              best_thr,
            "accuracy":               float(accuracy_score(y_te, pred)),
            "precision":              float(precision_score(y_te, pred, zero_division=0)),
            "recall":                 float(recall_score(y_te, pred, zero_division=0)),
            "f1":                     float(f1_score(y_te, pred, zero_division=0)),
            "balanced_accuracy":      float(balanced_accuracy_score(y_te, pred)),
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "mean_genuine_distance":  float(gen_d.mean()) if len(gen_d) else np.nan,
            "std_genuine_distance":   float(gen_d.std())  if len(gen_d) else np.nan,
            "mean_impostor_distance": float(imp_d.mean()) if len(imp_d) else np.nan,
            "std_impostor_distance":  float(imp_d.std())  if len(imp_d) else np.nan,
            "total_unique_pairs":     len(test_df),
        })

        all_distances.extend(d_te.tolist())
        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(pred.tolist())
        all_pair_type.extend(test_df["pair_type"].tolist() if "pair_type" in test_df.columns
                             else (["genuine" if v == 1 else "impostor" for v in y_te]))

    print()

    summary = pd.DataFrame(rows)
    pairs   = pd.DataFrame({
        "distance":   all_distances,
        "true_label": all_y_true,
        "pred_label": all_y_pred,
        "pair_type":  all_pair_type,
    })

    metric_keys = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]
    avg_metrics = {k: float(summary[k].mean()) for k in metric_keys}

    return {"summary": summary, "pairs": pairs, "avg_metrics": avg_metrics}


# ------------------------------------------------------------------
# Score-Level Fusion — LOSO (unchanged logic, model-driven)
# ------------------------------------------------------------------

def _eval(model, X: np.ndarray, y: np.ndarray) -> dict:
    yp = model.predict(X)
    return {
        "accuracy":          float(accuracy_score(y, yp)),
        "precision":         float(precision_score(y, yp, zero_division=0)),
        "recall":            float(recall_score(y, yp, zero_division=0)),
        "f1":                float(f1_score(y, yp, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
    }


def run_fusion_evaluation(per_set_dfs: list) -> dict:
    """
    LOSO cross-validation for all enabled models in MODELS_CONFIG.
    Returns a dict keyed by model name, each value containing:
      avg_metrics, fold_metrics, set_ids, all_y_true, all_y_pred, all_y_score, final_model
    """
    n = len(per_set_dfs)
    metric_keys   = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]
    enabled_names = [name for name, cfg in MODELS_CONFIG.items() if cfg["enabled"]]

    buckets: dict = {
        name: {"fold_metrics": [], "set_ids": [],
               "all_y_true": [], "all_y_pred": [], "all_y_score": []}
        for name in enabled_names
    }

    for test_idx in range(n):
        print(f"  [fusion] fold {test_idx + 1}/{n} ...", end="\r", flush=True)

        train_df = pd.concat(
            [df for i, df in enumerate(per_set_dfs) if i != test_idx],
            ignore_index=True,
        ).dropna(subset=FEATURE_COLS)
        test_df = per_set_dfs[test_idx].dropna(subset=FEATURE_COLS)

        X_tr, y_tr = train_df[FEATURE_COLS].values, train_df["true_label"].values
        X_te, y_te = test_df[FEATURE_COLS].values,  test_df["true_label"].values
        set_id = test_df["set_id"].iloc[0] if "set_id" in test_df.columns else f"set_{test_idx + 1:02d}"

        for name in enabled_names:
            model = _instantiate_model(name)
            model.fit(X_tr, y_tr)
            b = buckets[name]
            b["fold_metrics"].append(_eval(model, X_te, y_te))
            b["set_ids"].append(set_id)
            b["all_y_true"].extend(y_te.tolist())
            b["all_y_pred"].extend(model.predict(X_te).tolist())
            b["all_y_score"].extend(model.predict_proba(X_te)[:, 1].tolist())

    print()

    # Final models trained on all data (for plots that need the fitted model)
    all_df = pd.concat(per_set_dfs, ignore_index=True).dropna(subset=FEATURE_COLS)
    X_all, y_all = all_df[FEATURE_COLS].values, all_df["true_label"].values

    for name in enabled_names:
        final = _instantiate_model(name)
        final.fit(X_all, y_all)
        buckets[name]["final_model"] = final

    for b in buckets.values():
        b["all_y_true"]  = np.array(b["all_y_true"])
        b["all_y_pred"]  = np.array(b["all_y_pred"])
        b["all_y_score"] = np.array(b["all_y_score"])
        b["avg_metrics"] = {k: float(np.mean([m[k] for m in b["fold_metrics"]])) for k in metric_keys}

    return buckets


# ------------------------------------------------------------------
# Baseline plots  (saved to BASELINE_DIR)
# ------------------------------------------------------------------

def plot_metrics_per_set(summary: pd.DataFrame, out_dir: str) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]
    labels  = summary["set_id"].tolist()
    x, width = np.arange(len(labels)), 0.15
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.7), 5))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, summary[m].fillna(0), width, label=m.replace("_", " ").title(), color=c)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics per Set  (LOSO, threshold optimised on training folds)")
    ax.legend(loc="lower right", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metrics_per_set.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [baseline] saved: metrics_per_set.png")


def plot_distance_distributions(pairs: pd.DataFrame, out_dir: str) -> None:
    genuine  = pairs[pairs["pair_type"] == "genuine"]["distance"].dropna()
    impostor = pairs[pairs["pair_type"] == "impostor"]["distance"].dropna()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(genuine,  bins=80, density=True, alpha=0.65, color="#4C72B0", label=f"Genuine (n={len(genuine):,})")
    ax.hist(impostor, bins=80, density=True, alpha=0.55, color="#DD8452", label=f"Impostor (n={len(impostor):,})")
    ax.set_xlabel("Hamming Distance")
    ax.set_ylabel("Density")
    ax.set_title("Genuine vs Impostor Distance Distribution  (all LOSO held-out sets)")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "distance_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [baseline] saved: distance_distribution.png")


def plot_roc_curve(pairs: pd.DataFrame, out_dir: str) -> None:
    fpr, tpr, _ = roc_curve(pairs["true_label"].values, -pairs["distance"].values)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (TAR)")
    ax.set_title("ROC Curve — Baseline LOSO (all held-out sets)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [baseline] saved: roc_curve.png")


def plot_det_curve(pairs: pd.DataFrame, out_dir: str) -> None:
    y_true, distances = pairs["true_label"].values, pairs["distance"].values
    thresholds = np.linspace(distances.min(), distances.max(), 500)
    far_list, frr_list = [], []
    for t in thresholds:
        pred = (distances <= t).astype(int)
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        tn = (y_true == 0).sum()
        tp = (y_true == 1).sum()
        far_list.append(fp / tn if tn > 0 else np.nan)
        frr_list.append(fn / tp if tp > 0 else np.nan)
    far_arr, frr_arr = np.array(far_list), np.array(frr_list)
    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer = (far_arr[eer_idx] + frr_arr[eer_idx]) / 2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(far_arr * 100, frr_arr * 100, color="#55A868", lw=2)
    ax.scatter([far_arr[eer_idx] * 100], [frr_arr[eer_idx] * 100],
               color="crimson", zorder=5, label=f"EER ≈ {eer * 100:.2f}%")
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("FRR (%)")
    ax.set_title("DET Curve — Baseline LOSO (all held-out sets)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "det_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [baseline] saved: det_curve.png")


def plot_distance_per_set(summary: pd.DataFrame, out_dir: str) -> None:
    sets = summary["set_id"].tolist()
    x    = np.arange(len(sets))

    fig, ax = plt.subplots(figsize=(max(12, len(sets) * 0.7), 5))
    ax.errorbar(x - 0.15, summary["mean_genuine_distance"],  yerr=summary["std_genuine_distance"],
                fmt="o", color="#4C72B0", capsize=4, label="Genuine (mean ± std)")
    ax.errorbar(x + 0.15, summary["mean_impostor_distance"], yerr=summary["std_impostor_distance"],
                fmt="s", color="#DD8452", capsize=4, label="Impostor (mean ± std)")

    if "threshold" in summary.columns:
        thresholds = summary["threshold"].values
        if np.ptp(thresholds) < 1e-6:
            ax.axhline(thresholds[0], color="crimson", linestyle="--", linewidth=1.2,
                       label=f"Threshold = {thresholds[0]:.4f}")
        else:
            ax.step(np.append(x - 0.5, x[-1] + 0.5),
                    np.append(thresholds, thresholds[-1]),
                    color="crimson", linestyle="--", linewidth=1.2, where="post",
                    label="Best threshold (per fold)")

    ax.set_xticks(x)
    ax.set_xticklabels(sets, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Hamming Distance")
    ax.set_title("Mean Genuine vs Impostor Distance per Set  (LOSO)")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "distance_per_set.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [baseline] saved: distance_per_set.png")


def plot_loso_threshold_per_set(summary: pd.DataFrame, out_dir: str) -> None:
    sets       = summary["set_id"].tolist()
    thresholds = summary["threshold"].tolist()
    x          = np.arange(len(sets))
    mean_thr   = float(np.mean(thresholds))

    fig, ax = plt.subplots(figsize=(max(12, len(sets) * 0.7), 4))
    ax.bar(x, thresholds, color="#55A868", alpha=0.8, label="Best threshold (fold)")
    ax.axhline(mean_thr, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Mean = {mean_thr:.4f}")
    ax.set_xticks(x)
    ax.set_xticklabels(sets, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Hamming Threshold")
    ax.set_title("Optimised Hamming Threshold per LOSO Fold\n"
                 "(maximises balanced accuracy on the n-1 training sets)")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "loso_threshold_per_set.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [baseline] saved: loso_threshold_per_set.png")


def compute_far_frr(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    df["far"] = df["FP"] / (df["FP"] + df["TN"])
    df["frr"] = df["FN"] / (df["FN"] + df["TP"])
    return df


def plot_far_frr_per_set(summary: pd.DataFrame, out_dir: str) -> None:
    df = compute_far_frr(summary)
    sets = df["set_id"].tolist()
    x, width = np.arange(len(sets)), 0.35
    eer_line = ((df["far"] + df["frr"]) / 2 * 100).mean()

    fig, ax = plt.subplots(figsize=(max(12, len(sets) * 0.7), 5))
    ax.bar(x - width / 2, df["far"] * 100, width, color="#C44E52", label="FAR")
    ax.bar(x + width / 2, df["frr"] * 100, width, color="#8172B2", label="FRR")
    ax.axhline(eer_line, color="gray", linestyle=":", linewidth=1,
               label=f"Mean (FAR+FRR)/2 ≈ {eer_line:.3f}%")
    ax.set_xticks(x)
    ax.set_xticklabels(sets, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("FAR & FRR per Set  (LOSO, threshold optimised on training folds)")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "far_frr_per_set.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [baseline] saved: far_frr_per_set.png")


def plot_aggregate_confusion_matrix(summary: pd.DataFrame, out_dir: str, title_prefix: str = "Baseline") -> None:
    tp, fp = int(summary["TP"].sum()), int(summary["FP"].sum())
    fn, tn = int(summary["FN"].sum()), int(summary["TN"].sum())
    values = np.array([[tp, fn], [fp, tn]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(values, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Match", "Predicted Non-Match"])
    ax.set_yticklabels(["Actual Match", "Actual Non-Match"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"Aggregate Confusion Matrix — {title_prefix} ({len(summary)} sets)")
    thresh = values.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{values[i, j]:,}", ha="center", va="center",
                    color="white" if values[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "aggregate_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [baseline] saved: aggregate_confusion_matrix.png")


def plot_dashboard(summary: pd.DataFrame, pairs: pd.DataFrame, out_dir: str) -> None:
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Baseline Evaluation Dashboard\nCASIA-Iris-Thousand (Hamming Distance, LOSO)",
                 fontsize=14, fontweight="bold")
    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    sets = summary["set_id"].tolist()
    x    = np.arange(len(sets))
    mean_thr = float(summary["threshold"].mean()) if "threshold" in summary.columns else 0.38

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, summary["f1"].fillna(0), marker="o", color="#4C72B0", label="F1")
    ax1.plot(x, summary["balanced_accuracy"].fillna(0), marker="s", color="#DD8452", label="Bal. Acc.")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sets, rotation=90, fontsize=6)
    ax1.set_ylim(0, 1.08)
    ax1.set_title("F1 & Balanced Accuracy per Set", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = fig.add_subplot(gs[0, 1])
    genuine  = pairs[pairs["pair_type"] == "genuine"]["distance"].dropna()
    impostor = pairs[pairs["pair_type"] == "impostor"]["distance"].dropna()
    ax2.hist(genuine,  bins=60, density=True, alpha=0.65, color="#4C72B0", label="Genuine")
    ax2.hist(impostor, bins=60, density=True, alpha=0.55, color="#DD8452", label="Impostor")
    ax2.axvline(mean_thr, color="crimson", linestyle="--", linewidth=1.2,
                label=f"Mean thr={mean_thr:.4f}")
    ax2.set_title("Distance Distribution", fontsize=9)
    ax2.set_xlabel("Hamming Distance", fontsize=8)
    ax2.legend(fontsize=7)
    ax2.grid(True, linestyle="--", alpha=0.4)

    ax3 = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(pairs["true_label"].values, -pairs["distance"].values)
    ax3.plot(fpr, tpr, color="#55A868", lw=1.5, label=f"AUC={auc(fpr, tpr):.4f}")
    ax3.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax3.set_title("ROC Curve", fontsize=9)
    ax3.set_xlabel("FAR", fontsize=8)
    ax3.set_ylabel("TAR", fontsize=8)
    ax3.legend(fontsize=7)
    ax3.grid(True, linestyle="--", alpha=0.4)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.errorbar(x, summary["mean_genuine_distance"],  yerr=summary["std_genuine_distance"],
                 fmt="o-", color="#4C72B0", capsize=3, markersize=4, label="Genuine")
    ax4.errorbar(x, summary["mean_impostor_distance"], yerr=summary["std_impostor_distance"],
                 fmt="s-", color="#DD8452", capsize=3, markersize=4, label="Impostor")
    if "threshold" in summary.columns:
        ax4.plot(x, summary["threshold"].values, color="crimson", linestyle="--",
                 linewidth=1, marker="x", markersize=5, label="Best threshold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(sets, rotation=90, fontsize=6)
    ax4.set_title("Mean Distance + Threshold per Set", fontsize=9)
    ax4.legend(fontsize=7)
    ax4.grid(True, linestyle="--", alpha=0.4)

    ax5 = fig.add_subplot(gs[1, 1])
    tp, fp = int(summary["TP"].sum()), int(summary["FP"].sum())
    fn, tn = int(summary["FN"].sum()), int(summary["TN"].sum())
    vals   = np.array([[tp, fn], [fp, tn]])
    im     = ax5.imshow(vals, cmap="Blues")
    ax5.set_xticks([0, 1])
    ax5.set_yticks([0, 1])
    ax5.set_xticklabels(["Pred Match", "Pred Non-Match"], fontsize=7)
    ax5.set_yticklabels(["Act Match", "Act Non-Match"], fontsize=7)
    thresh_cm = vals.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax5.text(j, i, f"{vals[i, j]:,}", ha="center", va="center",
                     color="white" if vals[i, j] > thresh_cm else "black", fontsize=10, fontweight="bold")
    ax5.set_title("Aggregate Confusion Matrix", fontsize=9)

    df_ff = compute_far_frr(summary)
    ax6   = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    rows_tbl = []
    for col, label in zip(["accuracy", "f1", "balanced_accuracy", "far", "frr"],
                           ["Accuracy", "F1", "Bal. Acc.", "FAR", "FRR"]):
        v = df_ff[col].dropna()
        rows_tbl.append([label, f"{v.mean():.4f}", f"{v.std():.4f}", f"{v.min():.4f}", f"{v.max():.4f}"])
    if "threshold" in summary.columns:
        v = summary["threshold"].dropna()
        rows_tbl.append(["Threshold", f"{v.mean():.4f}", f"{v.std():.4f}", f"{v.min():.4f}", f"{v.max():.4f}"])
    tbl = ax6.table(cellText=rows_tbl, colLabels=["Metric", "Mean", "Std", "Min", "Max"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.4)
    ax6.set_title("Aggregate Metrics Summary", fontsize=9)

    fig.savefig(os.path.join(out_dir, "dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [baseline] saved: dashboard.png")


# ------------------------------------------------------------------
# Model report plots  (saved to per-model directories)
# ------------------------------------------------------------------

def plot_model_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str, out_dir: str) -> None:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    values = np.array([[tp, fn], [fp, tn]])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(values, cmap="Oranges")
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Match", "Predicted Non-Match"])
    ax.set_yticklabels(["Actual Match", "Actual Non-Match"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(f"Aggregate Confusion Matrix — {model_name}\n(LOSO predictions)")
    thresh = values.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{values[i, j]:,}", ha="center", va="center",
                    color="white" if values[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{model_name}] saved: confusion_matrix.png")


def plot_model_roc_curve(y_true: np.ndarray, y_score: np.ndarray,
                         model_name: str, out_dir: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#DD8452", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}\n(LOSO predictions)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{model_name}] saved: roc_curve.png")


def plot_model_metrics_per_set(fold_metrics: list, set_ids: list,
                               model_name: str, out_dir: str) -> None:
    metric_keys   = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "Bal. Acc."]
    x      = np.arange(len(set_ids))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    width  = 0.15

    fig, ax = plt.subplots(figsize=(max(12, len(set_ids) * 0.7), 5))
    for i, (k, label, c) in enumerate(zip(metric_keys, metric_labels, colors)):
        vals = [m[k] for m in fold_metrics]
        ax.bar(x + i * width, vals, width, label=label, color=c)

    ax.set_xticks(x + width * (len(metric_keys) - 1) / 2)
    ax.set_xticklabels(set_ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"Evaluation Metrics per Set — {model_name}")
    ax.legend(loc="lower right", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metrics_per_set.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{model_name}] saved: metrics_per_set.png")


def plot_feature_importance(model, model_name: str, out_dir: str) -> None:
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]
    sorted_feat = [FEATURE_COLS[i] for i in sorted_idx]
    sorted_imp  = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(sorted_feat, sorted_imp, color="#4C72B0", alpha=0.85)
    for bar, val in zip(bars, sorted_imp):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Importance")
    ax.set_title(f"Feature Importance — {model_name}\n(trained on all sets)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{model_name}] saved: feature_importance.png")


def plot_model_coefficients(model, model_name: str, out_dir: str) -> None:
    coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
    sorted_idx  = np.argsort(np.abs(coef))[::-1]
    sorted_feat = [FEATURE_COLS[i] for i in sorted_idx]
    sorted_coef = coef[sorted_idx]
    colors = ["#4C72B0" if v >= 0 else "#DD8452" for v in sorted_coef]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(sorted_feat, sorted_coef, color=colors, alpha=0.85)
    for bar, val in zip(bars, sorted_coef):
        offset = 0.002 if val >= 0 else -0.012
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Coefficient")
    ax.set_title(f"Feature Coefficients — {model_name}\n"
                 "(trained on all sets, blue=positive, orange=negative)")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "coefficients.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{model_name}] saved: coefficients.png")


# ------------------------------------------------------------------
# Comparison plot  (saved to VIZ_DIR root)
# ------------------------------------------------------------------

def plot_comparison_report(loso_summary: pd.DataFrame, fusion_results: dict, out_dir: str) -> None:
    metric_keys   = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "Bal. Accuracy"]

    baseline_avg    = {k: float(loso_summary[k].mean()) for k in metric_keys}
    approach_names  = ["Baseline\n(Hamming LOSO)"] + [f"Fusion\n{n}" for n in fusion_results]
    approach_values = [baseline_avg] + [r["avg_metrics"] for r in fusion_results.values()]
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    colors  = (palette * ((len(approach_names) // len(palette)) + 1))[:len(approach_names)]

    n_metrics    = len(metric_keys)
    n_approaches = len(approach_names)
    x     = np.arange(n_metrics)
    width = 0.75 / n_approaches

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (name, metrics, color) in enumerate(zip(approach_names, approach_values, colors)):
        offsets = x + (i - n_approaches / 2 + 0.5) * width
        vals = [metrics[k] for k in metric_keys]
        bars = ax.bar(offsets, vals, width, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0015,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Score-Level Fusion vs Baseline  (Leave-One-Set-Out CV)", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_report.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [root] saved: comparison_report.png")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    enabled_dirs = [MODEL_DIRS[name] for name, cfg in MODELS_CONFIG.items() if cfg["enabled"]]
    for d in [VIZ_DIR, BASELINE_DIR] + enabled_dirs:
        ensure_dir(d)

    print(f"Loading data from: {OUTPUT_ROOT}")
    per_set_pairs = load_per_set_pair_records()
    per_set_multi = load_all_multi_score_features()
    total_pairs   = sum(len(df) for df in per_set_pairs)
    print(f"  {len(per_set_pairs)} sets | {total_pairs:,} pair records | {len(per_set_multi)} multi-score sets")

    print("\nEnabled models:")
    for name, cfg in MODELS_CONFIG.items():
        status = "ON " if cfg["enabled"] else "OFF"
        print(f"  [{status}] {name}  params={cfg['params']}")

    # ---- Baseline LOSO → visualizations/baseline/ ----
    print("\n[baseline — LOSO cross-validation]")
    baseline     = run_baseline_loso_evaluation(per_set_pairs)
    loso_summary = baseline["summary"]
    loso_pairs   = baseline["pairs"]

    plot_metrics_per_set(loso_summary, BASELINE_DIR)
    plot_distance_distributions(loso_pairs, BASELINE_DIR)
    plot_roc_curve(loso_pairs, BASELINE_DIR)
    plot_det_curve(loso_pairs, BASELINE_DIR)
    plot_distance_per_set(loso_summary, BASELINE_DIR)
    plot_loso_threshold_per_set(loso_summary, BASELINE_DIR)
    plot_far_frr_per_set(loso_summary, BASELINE_DIR)
    plot_aggregate_confusion_matrix(loso_summary, BASELINE_DIR, title_prefix="Baseline LOSO")
    plot_dashboard(loso_summary, loso_pairs, BASELINE_DIR)

    # ---- Fusion evaluation ----
    if not per_set_multi:
        print("\n  [skip] no multi_score_features.csv found — re-run baseline script first.")
    else:
        print("\n[fusion evaluation]")
        fusion_results = run_fusion_evaluation(per_set_multi)

        for model_name, res in fusion_results.items():
            out_dir = MODEL_DIRS.get(model_name, VIZ_DIR)
            print(f"\n[{model_name}]")
            plot_model_confusion_matrix(res["all_y_true"], res["all_y_pred"], model_name, out_dir)
            plot_model_roc_curve(res["all_y_true"], res["all_y_score"], model_name, out_dir)
            plot_model_metrics_per_set(res["fold_metrics"], res["set_ids"], model_name, out_dir)
            final = res["final_model"]
            if hasattr(final, "feature_importances_"):
                plot_feature_importance(final, model_name, out_dir)
            elif hasattr(final, "coef_"):
                plot_model_coefficients(final, model_name, out_dir)
            else:
                print(f"  [{model_name}] skipped: no coefficient/importance attribute")

        # ---- Comparison → visualizations/ (root) ----
        print("\n[comparison]")
        plot_comparison_report(loso_summary, fusion_results, VIZ_DIR)

    print("\nDone.")
    print(f"  baseline/              → {BASELINE_DIR}")
    for name, cfg in MODELS_CONFIG.items():
        if cfg["enabled"]:
            print(f"  {MODEL_DIRS[name].split(os.sep)[-1]}/  → {MODEL_DIRS[name]}")
    print(f"  comparison_report.png  → {VIZ_DIR}")


if __name__ == "__main__":
    main()
