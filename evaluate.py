import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score,
)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False

OUTPUT_ROOT = r".\out_CASIA_Iris_Thousand_MultiSet_L"
VIZ_DIR = os.path.join(OUTPUT_ROOT, "visualizations")

FEATURE_COLS = ["hamming", "jaccard", "cosine", "pearson", "tanimoto"]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_all_sets_summary() -> pd.DataFrame:
    path = os.path.join(OUTPUT_ROOT, "all_sets_summary.csv")
    df = pd.read_csv(path)
    return df[df["total_unique_pairs"] > 0].reset_index(drop=True)


def load_all_pair_records() -> pd.DataFrame:
    pattern = os.path.join(OUTPUT_ROOT, "**", "pair_records.csv")
    frames = []
    for fpath in sorted(glob.glob(pattern, recursive=True)):
        df = pd.read_csv(fpath)
        if len(df) > 0:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_all_multi_score_features() -> list:
    """Return a list of per-set DataFrames from multi_score_features.csv files."""
    pattern = os.path.join(OUTPUT_ROOT, "**", "multi_score_features.csv")
    frames = []
    for fpath in sorted(glob.glob(pattern, recursive=True)):
        df = pd.read_csv(fpath)
        if len(df) > 0:
            frames.append(df)
    return frames


# ------------------------------------------------------------------
# Score-Level Fusion evaluation (leave-one-set-out)
# ------------------------------------------------------------------

def _eval_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy":          float(accuracy_score(y_test, y_pred)),
        "precision":         float(precision_score(y_test, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_test, y_pred, zero_division=0)),
        "f1":                float(f1_score(y_test, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    }


def run_fusion_evaluation(per_set_dfs: list) -> dict:
    """Leave-one-set-out CV → averaged metrics per model."""
    n = len(per_set_dfs)
    metric_keys = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]
    dt_results, xgb_results = [], []

    for test_idx in range(n):
        print(f"  [fusion] fold {test_idx + 1}/{n} ...", end="\r", flush=True)
        train_df = pd.concat(
            [df for i, df in enumerate(per_set_dfs) if i != test_idx],
            ignore_index=True,
        ).dropna(subset=FEATURE_COLS)
        test_df = per_set_dfs[test_idx].dropna(subset=FEATURE_COLS)

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df["true_label"].values
        X_test  = test_df[FEATURE_COLS].values
        y_test  = test_df["true_label"].values

        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())

        dt = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
        dt.fit(X_train, y_train)
        dt_results.append(_eval_model(dt, X_test, y_test))

        if XGBOOST_AVAILABLE:
            spw = neg / pos if pos > 0 else 1.0
            xgb = XGBClassifier(
                n_estimators=100, max_depth=4, scale_pos_weight=spw,
                random_state=42, eval_metric="logloss", verbosity=0,
            )
            xgb.fit(X_train, y_train)
            xgb_results.append(_eval_model(xgb, X_test, y_test))

    print()

    def _avg(results):
        return {k: float(np.mean([r[k] for r in results])) for k in metric_keys}

    out = {"Decision Tree": _avg(dt_results)}
    if XGBOOST_AVAILABLE and xgb_results:
        out["XGBoost"] = _avg(xgb_results)
    return out


# ------------------------------------------------------------------
# Plot 1 — Per-set metrics bar chart
# ------------------------------------------------------------------

def plot_metrics_per_set(summary: pd.DataFrame) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]
    labels = summary["set_id"].tolist()
    x = np.arange(len(labels))
    width = 0.15
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.7), 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = summary[metric].fillna(0).tolist()
        ax.bar(x + i * width, vals, width, label=metric.replace("_", " ").title(), color=color)

    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics per Set")
    ax.legend(loc="lower right", fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "metrics_per_set.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved: metrics_per_set.png")


# ------------------------------------------------------------------
# Plot 2 — Genuine vs impostor distance distribution (aggregated)
# ------------------------------------------------------------------

def plot_distance_distributions(pairs: pd.DataFrame) -> None:
    genuine = pairs[pairs["pair_type"] == "genuine"]["distance"].dropna()
    impostor = pairs[pairs["pair_type"] == "impostor"]["distance"].dropna()

    threshold = pairs["distance"].min()
    if "pred_label" in pairs.columns:
        matched = pairs[pairs["pred_label"] == 1]["distance"]
        unmatched = pairs[pairs["pred_label"] == 0]["distance"]
        if len(matched) > 0 and len(unmatched) > 0:
            threshold = (matched.max() + unmatched.min()) / 2

    bins = 80
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(genuine, bins=bins, density=True, alpha=0.65, color="#4C72B0", label=f"Genuine (n={len(genuine):,})")
    ax.hist(impostor, bins=bins, density=True, alpha=0.55, color="#DD8452", label=f"Impostor (n={len(impostor):,})")
    ax.axvline(threshold, color="crimson", linestyle="--", linewidth=1.5, label=f"Threshold ≈ {threshold:.3f}")
    ax.set_xlabel("Hamming Distance")
    ax.set_ylabel("Density")
    ax.set_title("Genuine vs Impostor Distance Distribution (all sets)")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "distance_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved: distance_distribution.png")


# ------------------------------------------------------------------
# Plot 3 — ROC curve with AUC
# ------------------------------------------------------------------

def plot_roc_curve(pairs: pd.DataFrame) -> None:
    y_true = pairs["true_label"].values
    scores = -pairs["distance"].values

    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlim([-0.01, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (TAR)")
    ax.set_title("ROC Curve (all sets)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "roc_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved: roc_curve.png")


# ------------------------------------------------------------------
# Plot 4 — DET curve (FAR vs FRR)
# ------------------------------------------------------------------

def plot_det_curve(pairs: pd.DataFrame) -> None:
    y_true = pairs["true_label"].values
    distances = pairs["distance"].values
    thresholds = np.linspace(distances.min(), distances.max(), 500)

    far_list, frr_list = [], []
    for t in thresholds:
        pred = (distances <= t).astype(int)
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        total_neg = (y_true == 0).sum()
        total_pos = (y_true == 1).sum()
        far_list.append(fp / total_neg if total_neg > 0 else np.nan)
        frr_list.append(fn / total_pos if total_pos > 0 else np.nan)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer = (far_arr[eer_idx] + frr_arr[eer_idx]) / 2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(far_arr * 100, frr_arr * 100, color="#55A868", lw=2)
    ax.scatter([far_arr[eer_idx] * 100], [frr_arr[eer_idx] * 100],
               color="crimson", zorder=5, label=f"EER ≈ {eer * 100:.2f}%")
    ax.set_xlabel("FAR (%)")
    ax.set_ylabel("FRR (%)")
    ax.set_title("DET Curve (all sets)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "det_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved: det_curve.png")


# ------------------------------------------------------------------
# Plot 5 — Genuine / impostor distance per set
# ------------------------------------------------------------------

def plot_distance_boxplots(summary: pd.DataFrame) -> None:
    sets = summary["set_id"].tolist()
    x = np.arange(len(sets))

    fig, ax = plt.subplots(figsize=(max(12, len(sets) * 0.7), 5))

    genuine_means = summary["mean_genuine_distance"].tolist()
    genuine_stds = summary["std_genuine_distance"].tolist()
    impostor_means = summary["mean_impostor_distance"].tolist()
    impostor_stds = summary["std_impostor_distance"].tolist()
    thresholds = summary["threshold"].tolist()

    ax.errorbar(x - 0.15, genuine_means, yerr=genuine_stds, fmt="o", color="#4C72B0",
                capsize=4, label="Genuine (mean ± std)")
    ax.errorbar(x + 0.15, impostor_means, yerr=impostor_stds, fmt="s", color="#DD8452",
                capsize=4, label="Impostor (mean ± std)")
    ax.step([-0.5, len(sets) - 0.5], [thresholds[0], thresholds[0]],
            color="crimson", linestyle="--", linewidth=1.2, label=f"Threshold = {thresholds[0]:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(sets, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Hamming Distance")
    ax.set_title("Mean Genuine vs Impostor Distance per Set")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "distance_per_set.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved: distance_per_set.png")


# ------------------------------------------------------------------
# Plot 6 — FAR / FRR per set
# ------------------------------------------------------------------

def compute_far_frr(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    df["far"] = df["FP"] / (df["FP"] + df["TN"])
    df["frr"] = df["FN"] / (df["FN"] + df["TP"])
    return df


def plot_far_frr_per_set(summary: pd.DataFrame) -> None:
    df = compute_far_frr(summary)
    sets = df["set_id"].tolist()
    x = np.arange(len(sets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(sets) * 0.7), 5))
    ax.bar(x - width / 2, df["far"] * 100, width, color="#C44E52", label="FAR (False Acceptance Rate)")
    ax.bar(x + width / 2, df["frr"] * 100, width, color="#8172B2", label="FRR (False Rejection Rate)")

    eer_line = ((df["far"] + df["frr"]) / 2 * 100).mean()
    ax.axhline(eer_line, color="gray", linestyle=":", linewidth=1, label=f"Mean (FAR+FRR)/2 ≈ {eer_line:.3f}%")

    ax.set_xticks(x)
    ax.set_xticklabels(sets, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("FAR & FRR per Set (at fixed threshold)")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "far_frr_per_set.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved: far_frr_per_set.png")


# ------------------------------------------------------------------
# Plot 7 — Aggregate confusion matrix
# ------------------------------------------------------------------

def plot_aggregate_confusion_matrix(summary: pd.DataFrame) -> None:
    tp = int(summary["TP"].sum())
    fp = int(summary["FP"].sum())
    fn = int(summary["FN"].sum())
    tn = int(summary["TN"].sum())

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
    ax.set_title(f"Aggregate Confusion Matrix ({len(summary)} sets)")
    thresh = values.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{values[i, j]:,}", ha="center", va="center",
                    color="white" if values[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "aggregate_confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved: aggregate_confusion_matrix.png")


# ------------------------------------------------------------------
# Plot 8 — Summary dashboard
# ------------------------------------------------------------------

def plot_dashboard(summary: pd.DataFrame, pairs: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Iris Verification Evaluation Dashboard\nCASIA-Iris-Thousand", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    sets = summary["set_id"].tolist()
    x = np.arange(len(sets))

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, summary["f1"].fillna(0), marker="o", color="#4C72B0", label="F1")
    ax1.plot(x, summary["balanced_accuracy"].fillna(0), marker="s", color="#DD8452", label="Bal. Acc.")
    ax1.set_xticks(x)
    ax1.set_xticklabels(sets, rotation=90, fontsize=6)
    ax1.set_ylim(0.9, 1.02)
    ax1.set_title("F1 & Balanced Accuracy per Set", fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = fig.add_subplot(gs[0, 1])
    genuine = pairs[pairs["pair_type"] == "genuine"]["distance"].dropna()
    impostor = pairs[pairs["pair_type"] == "impostor"]["distance"].dropna()
    ax2.hist(genuine, bins=60, density=True, alpha=0.65, color="#4C72B0", label="Genuine")
    ax2.hist(impostor, bins=60, density=True, alpha=0.55, color="#DD8452", label="Impostor")
    thr = summary["threshold"].iloc[0]
    ax2.axvline(thr, color="crimson", linestyle="--", linewidth=1.2, label=f"thr={thr:.3f}")
    ax2.set_title("Distance Distribution", fontsize=9)
    ax2.set_xlabel("Hamming Distance", fontsize=8)
    ax2.legend(fontsize=7)
    ax2.grid(True, linestyle="--", alpha=0.4)

    ax3 = fig.add_subplot(gs[0, 2])
    scores = -pairs["distance"].values
    fpr, tpr, _ = roc_curve(pairs["true_label"].values, scores)
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, color="#55A868", lw=1.5, label=f"AUC={roc_auc:.4f}")
    ax3.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax3.set_title("ROC Curve", fontsize=9)
    ax3.set_xlabel("FAR", fontsize=8)
    ax3.set_ylabel("TAR", fontsize=8)
    ax3.legend(fontsize=7)
    ax3.grid(True, linestyle="--", alpha=0.4)

    ax4 = fig.add_subplot(gs[1, 0])
    ax4.errorbar(x, summary["mean_genuine_distance"], yerr=summary["std_genuine_distance"],
                 fmt="o-", color="#4C72B0", capsize=3, markersize=4, label="Genuine")
    ax4.errorbar(x, summary["mean_impostor_distance"], yerr=summary["std_impostor_distance"],
                 fmt="s-", color="#DD8452", capsize=3, markersize=4, label="Impostor")
    ax4.axhline(thr, color="crimson", linestyle="--", linewidth=1, label=f"thr={thr:.3f}")
    ax4.set_xticks(x)
    ax4.set_xticklabels(sets, rotation=90, fontsize=6)
    ax4.set_title("Mean Distance per Set", fontsize=9)
    ax4.legend(fontsize=7)
    ax4.grid(True, linestyle="--", alpha=0.4)

    ax5 = fig.add_subplot(gs[1, 1])
    tp = int(summary["TP"].sum())
    fp = int(summary["FP"].sum())
    fn = int(summary["FN"].sum())
    tn = int(summary["TN"].sum())
    vals = np.array([[tp, fn], [fp, tn]])
    im = ax5.imshow(vals, cmap="Blues")
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

    df_far_frr = compute_far_frr(summary)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    metric_cols   = ["accuracy", "f1", "balanced_accuracy", "far", "frr"]
    metric_labels = ["Accuracy", "F1", "Bal. Acc.", "FAR", "FRR"]
    table_data = []
    for col, label in zip(metric_cols, metric_labels):
        vals_col = df_far_frr[col].dropna()
        table_data.append([label,
                            f"{vals_col.mean():.4f}",
                            f"{vals_col.std():.4f}",
                            f"{vals_col.min():.4f}",
                            f"{vals_col.max():.4f}"])
    tbl = ax6.table(cellText=table_data,
                    colLabels=["Metric", "Mean", "Std", "Min", "Max"],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.4)
    ax6.set_title("Aggregate Metrics Summary", fontsize=9)

    fig.savefig(os.path.join(VIZ_DIR, "dashboard.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved: dashboard.png")


# ------------------------------------------------------------------
# Plot 9 — Score-Level Fusion vs Baseline comparison
# ------------------------------------------------------------------

def plot_comparison_report(summary: pd.DataFrame, fusion_results: dict) -> None:
    metric_keys   = ["accuracy", "precision", "recall", "f1", "balanced_accuracy"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "Bal. Accuracy"]

    baseline_avg = {k: float(summary[k].mean()) for k in metric_keys}

    approach_names  = ["Baseline\n(Hamming 0.38)"] + [f"Fusion\n{n}" for n in fusion_results]
    approach_values = [baseline_avg] + list(fusion_results.values())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(approach_names)]

    n_metrics    = len(metric_keys)
    n_approaches = len(approach_names)
    x     = np.arange(n_metrics)
    width = 0.75 / n_approaches

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (name, metrics, color) in enumerate(zip(approach_names, approach_values, colors)):
        offsets = x + (i - n_approaches / 2 + 0.5) * width
        vals = [metrics[k] for k in metric_keys]
        bars = ax.bar(offsets, vals, width, label=name, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.0015,
                    f"{val:.4f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Score-Level Fusion vs Baseline  (Leave-One-Set-Out CV)", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ_DIR, "comparison_report.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  saved: comparison_report.png")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    ensure_dir(VIZ_DIR)
    print(f"Loading data from: {OUTPUT_ROOT}")

    summary = load_all_sets_summary()
    pairs   = load_all_pair_records()
    per_set_multi = load_all_multi_score_features()

    print(f"  {len(summary)} sets | {len(pairs):,} pair records | {len(per_set_multi)} multi-score sets")
    print(f"Saving plots to: {VIZ_DIR}")

    plot_metrics_per_set(summary)
    plot_distance_distributions(pairs)
    plot_roc_curve(pairs)
    plot_det_curve(pairs)
    plot_distance_boxplots(summary)
    plot_far_frr_per_set(summary)
    plot_aggregate_confusion_matrix(summary)
    plot_dashboard(summary, pairs)

    if per_set_multi:
        if not XGBOOST_AVAILABLE:
            print("  [warning] xgboost not installed — only Decision Tree will run.")
        fusion_results = run_fusion_evaluation(per_set_multi)
        plot_comparison_report(summary, fusion_results)
    else:
        print("  [skip] no multi_score_features.csv found — re-run baseline script first.")

    print("\nDone.")


if __name__ == "__main__":
    main()
