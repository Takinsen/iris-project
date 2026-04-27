# Improvement: LOSO Evaluation and Score-Level Fusion

This document describes the end-to-end working steps of `evaluate.py`.

**Prerequisites:** `baseline_casia_thousand_multiset.py` must be run first so that `pair_records.csv` and `multi_score_features.csv` exist in each set folder.

---

## Overview

`evaluate.py` runs two complementary evaluations on top of the baseline outputs and produces `.png` reports for each:

| Phase | Input | Method |
|---|---|---|
| **Baseline LOSO** | `pair_records.csv` (Hamming distances) | Threshold swept on n-1 sets, applied to test set |
| **Score-Level Fusion** | `multi_score_features.csv` (4 distance features) | Classifier trained on n-1 sets, predicts on test set |

Both use **Leave-One-Set-Out (LOSO)** cross-validation — each set takes one turn as the held-out test set while the remaining n-1 sets form the training data. This mirrors real deployment: a model trained on known subjects is applied to an unseen group.

---

## Terminology

| Term | Meaning |
|---|---|
| **LOSO fold** | One iteration of cross-validation: n-1 sets for training, 1 set as the test set. |
| **Hamming threshold** | A cut-off value in [0, 1]. Pairs with Hamming distance ≤ threshold are classified as genuine (match). |
| **Score-level fusion** | Combining multiple similarity scores (Hamming, Jaccard, Cosine, Pearson) into a single classifier decision, instead of relying on Hamming alone. |
| **Feature columns** | `hamming`, `jaccard`, `cosine`, `pearson` — four distance metrics computed per pair by the baseline script and stored in `multi_score_features.csv`. Lower values mean more similar templates. |
| **Genuine / Impostor** | Same as in the baseline: genuine = same subject + same eye side; impostor = different subjects. |

---

## Configuration

All model settings live in the `MODELS_CONFIG` dict at the top of `evaluate.py`. Edit this block to enable/disable models or tune hyperparameters — no changes to training code needed.

```python
MODELS_CONFIG = {
    "Random Forest": {
        "enabled": True,            # set False to skip this model entirely
        "params": {
            "n_estimators": 200,    # number of trees
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs":       -1,
        },
    },
    "Logistic Regression": {
        "enabled": True,
        "params": {
            "C":            1.0,    # inverse regularisation strength
            "class_weight": "balanced",
            "max_iter":     1000,
            "random_state": 42,
        },
    },
    "Gaussian Naive Bayes": {
        "enabled": True,
        "params": {},               # no hyperparameters required
    },
    "Linear Discriminant Analysis": {
        "enabled": True,
        "params": {
            "solver": "svd",
        },
    },
}
```

**Output path** is controlled by `OUTPUT_ROOT` at the top of the file (must match the baseline script's `OUTPUT_ROOT`).

---

## Step-by-Step Process

### Step 1 — Load Per-Set Data

`load_per_set_pair_records()` scans all `pair_records.csv` files, one per set folder:

```
<OUTPUT_ROOT>/set_01_L_<subjects>/pair_records.csv
<OUTPUT_ROOT>/set_02_L_<subjects>/pair_records.csv
...
```

Each CSV is loaded as a separate DataFrame. Two columns are added automatically if absent:

- `set_id` — derived from the folder name (e.g. `set_01_L_001-009`)
- `pair_type` — derived from `true_label` (`1 → "genuine"`, `0 → "impostor"`)

`load_all_multi_score_features()` does the same for `multi_score_features.csv` files, which additionally contain the `hamming`, `jaccard`, `cosine`, and `pearson` feature columns used by the fusion models.

---

### Step 2 — Baseline LOSO Evaluation

`run_baseline_loso_evaluation(per_set_pair_records)` runs the threshold-based baseline under LOSO.

#### 2a — Split

For each fold `k` (where `k` goes from 0 to n-1):

```
Train: all sets except set k   →  concatenated pair_records
Test:  set k                   →  held-out pair_records
```

#### 2b — Find Best Threshold on Training Data

`_find_best_threshold` sweeps 500 evenly-spaced threshold candidates over the range `[min_distance, max_distance]` of the training pairs. For each candidate:

```python
pred  = (distances <= threshold).astype(int)   # 1 = match, 0 = non-match
score = balanced_accuracy_score(y_true, pred)
```

The threshold that maximises balanced accuracy on the n-1 training sets is selected. This avoids bias from any fixed value (e.g. 0.38) and adapts to the actual score distribution of the data.

#### 2c — Evaluate on Test Set

The selected threshold is applied to the held-out test set:

```python
pred = (test_distances <= best_threshold).astype(int)
```

Metrics recorded per fold: `accuracy`, `precision`, `recall`, `f1`, `balanced_accuracy`, `TP`, `FP`, `FN`, `TN`, `mean_genuine_distance`, `std_genuine_distance`, `mean_impostor_distance`, `std_impostor_distance`, `threshold`.

#### 2d — Aggregate

After all n folds, results are collected into:

- `loso_summary` — one row per fold, used for all baseline plots
- `loso_pairs` — all held-out test pairs concatenated, used for ROC and DET curves

---

### Step 3 — Baseline Plots

All plots are saved to `visualizations/baseline/`.

| File | What it shows |
|---|---|
| `metrics_per_set.png` | Bar chart of accuracy, precision, recall, F1, balanced accuracy per LOSO fold |
| `distance_distribution.png` | Overlapping histograms of genuine vs impostor Hamming distances across all held-out pairs |
| `roc_curve.png` | ROC curve (AUC) across all held-out pairs |
| `det_curve.png` | DET curve with EER marked |
| `distance_per_set.png` | Mean genuine and impostor distance per fold with per-fold threshold step line |
| `loso_threshold_per_set.png` | Bar chart of the optimised threshold selected per fold and the mean |
| `far_frr_per_set.png` | FAR and FRR per fold at each fold's selected threshold |
| `aggregate_confusion_matrix.png` | Summed TP/FP/FN/TN across all folds |
| `dashboard.png` | 6-panel summary: F1/balanced-acc trends, distance distribution, ROC, distance + threshold per set, confusion matrix, aggregate metrics table |

---

### Step 4 — Score-Level Fusion LOSO Evaluation

`run_fusion_evaluation(per_set_multi_score_dfs)` trains and evaluates classifiers using the same LOSO protocol.

#### 4a — Feature Matrix

Each row in `multi_score_features.csv` is one pair with four input features:

| Feature | Description |
|---|---|
| `hamming` | Hamming distance between the two IrisCodes (primary biometric score) |
| `jaccard` | Jaccard distance on binarised template bits |
| `cosine` | Cosine distance on raw template bits |
| `pearson` | Pearson correlation distance on raw template bits |

The target label is `true_label` (1 = genuine, 0 = impostor).

#### 4b — Split and Train

For each fold `k`:

```
X_train, y_train = feature rows from n-1 sets
X_test,  y_test  = feature rows from set k
```

Each enabled model in `MODELS_CONFIG` is independently instantiated with its configured `params` and trained on `(X_train, y_train)`:

```python
model = _instantiate_model(name)   # constructs from MODELS_CONFIG params
model.fit(X_train, y_train)
```

#### 4c — Predict and Record

```python
y_pred  = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]   # probability of being genuine
```

Metrics recorded per fold (same set as baseline): accuracy, precision, recall, F1, balanced accuracy.

#### 4d — Final Model

After all folds, each model is retrained on **all** sets combined. This final model is used only for feature importance and coefficient plots, not for the reported metrics.

---

### Step 5 — Per-Model Plots

Each enabled model gets its own output directory under `visualizations/`:

| Directory | Model |
|---|---|
| `random_forest/` | Random Forest |
| `logistic_regression/` | Logistic Regression |
| `gaussian_naive_bayes/` | Gaussian Naive Bayes |
| `linear_discriminant_analysis/` | Linear Discriminant Analysis |

Each directory contains:

| File | What it shows | Models |
|---|---|---|
| `confusion_matrix.png` | Aggregate TP/FP/FN/TN across all LOSO held-out predictions | All |
| `roc_curve.png` | ROC curve (AUC) across all LOSO held-out predictions | All |
| `metrics_per_set.png` | Bar chart of per-fold metrics | All |
| `feature_importance.png` | Feature importances (`feature_importances_`) from final model | Random Forest only |
| `coefficients.png` | Feature coefficients (`coef_`) from final model | Logistic Regression, LDA only |

Gaussian Naive Bayes does not expose feature importances or coefficients; it produces the first three plots only.

---

### Step 6 — Comparison Report

`plot_comparison_report` generates `visualizations/comparison_report.png` — a grouped bar chart placing the Baseline LOSO result side-by-side with each fusion model across all five metrics.

```
Baseline LOSO  |  Fusion RF  |  Fusion LR  |  Fusion GNB  |  Fusion LDA
─────────────────────────────────────────────────────────────────────────
Accuracy  Precision  Recall  F1  Balanced Accuracy
```

This is the primary summary for comparing whether score-level fusion improves on the single-score Hamming baseline.

---

## Output Structure

```
<OUTPUT_ROOT>/
  visualizations/
    comparison_report.png               — all models vs baseline side-by-side
    baseline/
      metrics_per_set.png
      distance_distribution.png
      roc_curve.png
      det_curve.png
      distance_per_set.png
      loso_threshold_per_set.png        — optimised threshold per fold
      far_frr_per_set.png
      aggregate_confusion_matrix.png
      dashboard.png
    random_forest/
      confusion_matrix.png
      roc_curve.png
      metrics_per_set.png
      feature_importance.png
    logistic_regression/
      confusion_matrix.png
      roc_curve.png
      metrics_per_set.png
      coefficients.png
    gaussian_naive_bayes/
      confusion_matrix.png
      roc_curve.png
      metrics_per_set.png
    linear_discriminant_analysis/
      confusion_matrix.png
      roc_curve.png
      metrics_per_set.png
      coefficients.png
```

---

## Data Flow Diagram

```
baseline_casia_thousand_multiset.py
  └─ pair_records.csv          (one file per set)
  └─ multi_score_features.csv  (one file per set)
          │
          ▼
evaluate.py
          │
          ├─ load_per_set_pair_records()
          │         │
          │         ▼
          │  run_baseline_loso_evaluation()
          │     │
          │     ├─ for each fold k  (k = 0 … n-1):
          │     │     train = concat(all sets except k)
          │     │     _find_best_threshold(train distances, labels)
          │     │         └─ sweep 500 thresholds → pick max balanced_accuracy
          │     │     apply best_threshold to test set k
          │     │     record: metrics, TP/FP/FN/TN, threshold, distances
          │     │
          │     └─ loso_summary (n rows)  +  loso_pairs (all held-out pairs)
          │              │
          │              ▼
          │         baseline plots → visualizations/baseline/
          │
          ├─ load_all_multi_score_features()
          │         │
          │         ▼
          │  run_fusion_evaluation()
          │     │
          │     ├─ for each fold k  (k = 0 … n-1):
          │     │     for each enabled model in MODELS_CONFIG:
          │     │         _instantiate_model(name)   ← params from MODELS_CONFIG
          │     │         model.fit(X_train, y_train)
          │     │         model.predict / predict_proba on test set k
          │     │         record: metrics, y_pred, y_score
          │     │
          │     ├─ retrain each model on all sets → final_model
          │     │
          │     └─ buckets[model_name] = {fold_metrics, all predictions, final_model}
          │              │
          │              ▼
          │         per-model plots → visualizations/<model_dir>/
          │
          └─ plot_comparison_report(loso_summary, fusion_results)
                     │
                     ▼
               visualizations/comparison_report.png
```

---

## Running

```bash
conda activate iris-dev

# Step 1 — generate pair_records.csv and multi_score_features.csv per set
python baseline_casia_thousand_multiset.py

# Step 2 — run LOSO evaluation and produce all plots
python evaluate.py
```

To disable a model without deleting code, set `"enabled": False` in `MODELS_CONFIG`.  
To change a hyperparameter, edit its value in `"params"`.
