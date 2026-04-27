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

All model settings live in the `MODELS_CONFIG` dict at the top of `evaluate.py`. Edit this block to enable/disable models, toggle Optuna tuning, or adjust trial budgets — no changes to training code needed.

```python
MODELS_CONFIG = {
    "Baseline (Hamming)": {
        "enabled":        True,
        "hp_tuning":      False,   # self-tunes via threshold sweep in fit()
        "params":         {"n_steps": 1000},
        "param_space":    None,
        "n_trials":       0,
        "study_n_jobs":   1,
        "tune_subsample": None,
    },
    "Random Forest": {
        "enabled":        True,
        "hp_tuning":      True,
        "params": {
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs":       -1,
        },
        "param_space":    _rf_space,
        "n_trials":       25,
        "study_n_jobs":   1,        # RF uses n_jobs=-1 internally; don't nest
        "tune_subsample": 50_000,   # cap rows for Optuna; final fit uses all
    },
    "Logistic Regression": {
        "enabled":        True,
        "hp_tuning":      True,
        "params": {
            "class_weight": "balanced",
            "max_iter":     2000,
            "random_state": 42,
        },
        "param_space":    _lr_space,
        "n_trials":       20,
        "study_n_jobs":   4,
        "tune_subsample": None,
    },
    "Gaussian Naive Bayes": {
        "enabled":        True,
        "hp_tuning":      True,
        "params":         {},
        "param_space":    _gnb_space,
        "n_trials":       15,
        "study_n_jobs":   4,
        "tune_subsample": None,
    },
    "Linear Discriminant Analysis": {
        "enabled":        True,
        "hp_tuning":      True,
        "params":         {"solver": "svd"},
        "param_space":    _lda_space,
        "n_trials":       10,
        "study_n_jobs":   4,
        "tune_subsample": None,
    },
}
```

### MODELS_CONFIG field reference

| Field | Type | Description |
|---|---|---|
| `enabled` | bool | Set `False` to skip this model entirely |
| `hp_tuning` | bool | Set `True` to run Optuna search on each LOSO training fold |
| `params` | dict | Fixed constructor parameters always passed to the model |
| `param_space` | callable or None | Function `(trial) → dict` that defines the Optuna search space |
| `n_trials` | int | Number of Optuna TPE trials per LOSO fold |
| `study_n_jobs` | int | Parallel Optuna trials (thread-based; set to 1 for RF to avoid CPU over-subscription) |
| `tune_subsample` | int or None | Max rows passed to the Optuna objective; final model always refits on all training data |

**Output path** is controlled by `OUTPUT_ROOT` at the top of the file (must match the baseline script's `OUTPUT_ROOT`).

---

## Hyperparameter Tuning (Optuna)

When `hp_tuning: True` and `param_space` is provided, each LOSO training fold runs an **Optuna TPE study** before fitting the final model.

### Search space functions

Each model has a dedicated `_*_space(trial)` function defined above `MODELS_CONFIG`:

| Function | Model | Parameters searched |
|---|---|---|
| `_rf_space` | Random Forest | `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` |
| `_lr_space` | Logistic Regression | `C` (regularisation strength, log-uniform in [1e-4, 1e2]) |
| `_gnb_space` | Gaussian Naive Bayes | `var_smoothing` (log-uniform in [1e-12, 1e-1]) |
| `_lda_space` | Linear Discriminant Analysis | `solver` (`"svd"` or `"lsqr"`); `shrinkage` only when `solver="lsqr"` |

LDA uses **conditional parameters**: `shrinkage` is only valid for `solver="lsqr"` (sklearn raises an error otherwise), so `_lda_space` only includes it in the returned dict when `solver != "svd"`.

### `_tune_model` flow

```
_tune_model(name, X_tr, y_tr)
    │
    ├─ hp_tuning=False or param_space=None
    │       └─ instantiate with base params → fit → return (model, None)
    │
    └─ hp_tuning=True
            │
            ├─ subsample X_tr if tune_subsample is set (RF: 50,000 rows)
            │
            ├─ create Optuna TPE study
            │       for each trial (up to n_trials, study_n_jobs parallel):
            │           sample params from param_space(trial)
            │           cross_val_score(3-fold StratifiedKFold, scoring="balanced_accuracy")
            │           return mean CV score
            │
            ├─ select best_params from study.best_params
            │
            └─ refit model on full X_tr (not subsampled) → return (model, best_params)
```

### Parallelism and CPU over-subscription

`cv_jobs` (parallelism inside `cross_val_score`) is derived automatically:

```python
model_is_parallel = "n_jobs" in cfg["params"]
cv_jobs = 1 if (model_is_parallel or study_jobs != 1) else -1
```

- **RF** (`n_jobs=-1` internally, `study_n_jobs=1`): `cv_jobs=1` — no nesting.
- **LR / GNB / LDA** (`study_n_jobs=4`): `cv_jobs=1` — study parallelism avoids nested threading.
- A hypothetical single-threaded, single-trial model would get `cv_jobs=-1` (parallel CV folds).

### Best params logging

When Optuna finds best params, they are printed per fold and stored in `fold_metrics`:

```
  [LOSO] fold 3/20 ...
    [Logistic Regression] best params: {'C': 0.04231}
    [Linear Discriminant Analysis] best params: {'solver': 'lsqr', 'shrinkage': 'auto'}
```

---

## Step-by-Step Process

### Step 1 — Load Per-Set Data

`load_all_multi_score_features()` scans all `multi_score_features.csv` files, one per set folder:

```
<OUTPUT_ROOT>/set_01_L_<subjects>/multi_score_features.csv
<OUTPUT_ROOT>/set_02_L_<subjects>/multi_score_features.csv
...
```

Each CSV contains the four feature columns (`hamming`, `jaccard`, `cosine`, `pearson`) plus `true_label`, `set_id`, and `pair_type` for every pair in that set.

---

### Step 2 — Baseline LOSO Evaluation

The Baseline (Hamming) model runs inside the same unified LOSO loop as all fusion models (see Step 4). It uses `HammingThresholdClassifier`, a sklearn-compatible wrapper that self-tunes its threshold during `fit()` by sweeping 1000 candidates and maximising balanced accuracy — no Optuna needed.

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

`run_loso_evaluation(per_set_multi_score_dfs)` trains and evaluates all enabled models (including the Baseline) in a single unified loop.

#### 4a — Feature Matrix

Each row in `multi_score_features.csv` is one pair with four input features:

| Feature | Description |
|---|---|
| `hamming` | Hamming distance between the two IrisCodes (primary biometric score) |
| `jaccard` | Jaccard distance on binarised template bits |
| `cosine` | Cosine distance on raw template bits |
| `pearson` | Pearson correlation distance on raw template bits |

The target label is `true_label` (1 = genuine, 0 = impostor).

#### 4b — Split, Tune, and Train

For each fold `k`:

```
X_train, y_train = feature rows from n-1 sets
X_test,  y_test  = feature rows from set k
```

Each enabled model is tuned and fitted via `_tune_model`:

```python
model, best_params = _tune_model(name, X_tr, y_tr)
# If hp_tuning=True: runs Optuna TPE study, refits on full X_tr with best params
# If hp_tuning=False: fits directly with base params (e.g. Baseline threshold sweep)
if best_params:
    print(f"  [{name}] best params: {best_params}")
```

#### 4c — Predict and Record

```python
y_pred  = model.predict(X_test)
y_score = model.predict_proba(X_test)[:, 1]   # probability of being genuine
```

Metrics recorded per fold (same set as baseline): accuracy, precision, recall, F1, balanced accuracy, TP, FP, FN, TN. If `best_params` is not None, it is also stored under `metrics["best_params"]`.

#### 4d — Final Model

After all folds, each model is re-tuned and retrained on **all** sets combined:

```python
final, best_params = _tune_model(name, X_all, y_all)
```

This final model is used only for feature importance and coefficient plots, not for the reported LOSO metrics.

---

### Step 5 — Per-Model Plots

Each enabled model gets its own output directory under `visualizations/`:

| Directory | Model |
|---|---|
| `baseline/` | Baseline (Hamming) |
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
  └─ multi_score_features.csv  (one file per set — hamming/jaccard/cosine/pearson + true_label)
          │
          ▼
evaluate.py
          │
          ├─ load_all_multi_score_features()
          │         │
          │         ▼
          │  run_loso_evaluation()          ← single loop over ALL enabled models
          │     │
          │     ├─ for each fold k  (k = 0 … n-1):
          │     │     train = concat(all sets except k)
          │     │     test  = set k
          │     │     │
          │     │     for each enabled model in MODELS_CONFIG:
          │     │         _tune_model(name, X_tr, y_tr)
          │     │             ├─ hp_tuning=False → fit directly
          │     │             └─ hp_tuning=True  → Optuna TPE study
          │     │                   n_trials × 3-fold CV → best_params
          │     │                   refit on full X_tr with best_params
          │     │         model.predict / predict_proba on test set k
          │     │         record: metrics, best_params, y_pred, y_score
          │     │
          │     ├─ for each enabled model:
          │     │     _tune_model(name, X_all, y_all)  → final_model (all data)
          │     │
          │     └─ buckets[model_name] = {fold_metrics, all predictions, final_model}
          │              │
          │              ├─ Baseline → baseline plots  → visualizations/baseline/
          │              └─ Fusion   → per-model plots → visualizations/<model_dir>/
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

# Step 1 — generate multi_score_features.csv (and pair_records.csv) per set
python baseline_casia_thousand_multiset.py

# Step 2 — run LOSO evaluation and produce all plots
python evaluate.py
```

To disable a model without deleting code, set `"enabled": False` in `MODELS_CONFIG`.  
To disable Optuna tuning for a model, set `"hp_tuning": False` — the model will use its `"params"` directly.  
To reduce tuning time, lower `n_trials` or set `tune_subsample` to cap the training rows seen by Optuna.
