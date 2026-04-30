# Improvement: LOSO Evaluation and Score-Level Fusion

This document describes the core concepts, configuration, hyperparameter tuning, evaluation logic, and output of `evaluate.py`.

**Prerequisites:** `baseline_casia_thousand_multiset.py` must be run first to generate `pair_records.csv` and `multi_score_features.csv` in each set folder.

---

## Overview

`evaluate.py` runs two complementary evaluations on top of the baseline outputs:

| Phase | Input | Method |
|---|---|---|
| **Baseline LOSO** | `pair_records.csv` (Hamming distances) | Threshold swept on n-1 sets, applied to held-out set |
| **Score-Level Fusion** | `multi_score_features.csv` (4 distance features) | Classifier trained on n-1 sets, predicts on held-out set |

Both phases use **Leave-One-Set-Out (LOSO)** cross-validation inside a single unified training loop. The results are compared in a final side-by-side report.

---

## Terminology

| Term | Meaning |
|---|---|
| **LOSO** | Leave-One-Set-Out — a cross-validation strategy where each set takes one turn as the held-out test fold while all others form the training data. |
| **LOSO fold** | One iteration: n-1 sets are used for training, 1 set is held out for evaluation. |
| **Score-level fusion** | Combining multiple distance scores (Hamming, Jaccard, Cosine, Pearson) into a single classification decision, replacing the single Hamming threshold. |
| **Feature columns** | `hamming`, `jaccard`, `cosine`, `pearson` — four distance metrics per pair, computed by the baseline script. Lower = more similar templates. |
| **Genuine / Impostor** | Same as in the baseline: genuine = same subject + same eye side; impostor = different subjects. |
| **Hyperparameter tuning** | Searching for model parameters that maximise cross-validated balanced accuracy on the training folds before committing to predictions on the test fold. |
| **TPE sampler** | Tree-structured Parzen Estimator — an Optuna Bayesian search strategy that models the objective function to suggest promising parameter regions. |

---

## Core Concept

### Why LOSO?

A standard train/test split would mix subjects between train and test. LOSO ensures **zero subject overlap** between training and test in any fold — the model must generalise to completely unseen identities, exactly like deployment.

```
Sets:  [set_01] [set_02] [set_03] ... [set_10]
                                               all different subjects

Fold 1:  Train = set_02 + set_03 + … + set_10   Test = set_01
Fold 2:  Train = set_01 + set_03 + … + set_10   Test = set_02
...
Fold 10: Train = set_01 + … + set_09             Test = set_10
```

### Why Score-Level Fusion?

The baseline uses only Hamming distance with a single threshold. Score-level fusion trains a classifier on four distance features (`hamming`, `jaccard`, `cosine`, `pearson`) to learn a more expressive decision boundary. The fusion models may capture complementary information not captured by Hamming alone.

### Why Optuna?

Fixed hyperparameters (`C=1.0`, `n_estimators=200`, etc.) may not be optimal for the specific training fold at hand. Optuna runs a Bayesian search over the hyperparameter space, spending more trials in promising regions. The best parameters found on the training fold are then used to fit the final model before predicting on the test fold.

---

## Configuration

### Eye Side

```python
TARGET_EYE_SIDE = "L"   # "L" or "R"
OUTPUT_ROOT = rf".\out_CASIA_Iris_Thousand_MultiSet_{TARGET_EYE_SIDE}"
```

Change `TARGET_EYE_SIDE` to match the eye side used in the baseline script. All output and visualization paths are derived automatically from this value.

### MODELS_CONFIG

All model settings live in `MODELS_CONFIG` at the top of `evaluate.py`.

```python
MODELS_CONFIG = {
    "Baseline (Hamming)": {
        "enabled":        False,
        "hp_tuning":      False,   # self-tunes via threshold sweep in fit()
        "params":         {"n_steps": 500},
        "param_space":    None,
        "n_trials":       0,
        "study_n_jobs":   1,
        "tune_subsample": None,
    },
    "Random Forest": {
        "enabled":        False,
        "hp_tuning":      True,
        "params": {
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs":       -1,
        },
        "param_space":    _rf_space,
        "n_trials":       25,
        "study_n_jobs":   1,
        "tune_subsample": 50_000,
    },
    "Logistic Regression": {
        "enabled":        False,
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
    "XGBoost": {
        "enabled":        True,
        "hp_tuning":      True,
        "params": {
            "scale_pos_weight": 50,
            "random_state":     42,
            "n_jobs":           -1,
            "verbosity":        0,
            "tree_method":      "hist",
        },
        "param_space":    _xgb_space,
        "n_trials":       25,
        "study_n_jobs":   1,
        "tune_subsample": 50_000,
    },
    "MLP": {
        "enabled":        False,
        "hp_tuning":      False,
        "params": {
            "hidden_layer_sizes": (128, 64),
            "alpha":              1e-4,
            "learning_rate_init": 1e-3,
            "max_iter":           1000,
            "random_state":       42,
        },
        "param_space":    None,
        "n_trials":       0,
        "study_n_jobs":   1,
        "tune_subsample": None,
    },
}
```

### Field Reference

| Field | Type | Description |
|---|---|---|
| `enabled` | bool | `False` = skip this model entirely |
| `hp_tuning` | bool | `True` = run Optuna search on each LOSO training fold |
| `params` | dict | Fixed constructor parameters always passed to the model |
| `param_space` | callable or `None` | Function `(trial) → dict` defining the Optuna search space |
| `n_trials` | int | Number of Optuna TPE trials per LOSO fold |
| `study_n_jobs` | int | Parallel Optuna trials (thread-based). Set to `1` for models with internal `n_jobs` to avoid CPU over-subscription. |
| `tune_subsample` | int or `None` | Max rows passed to the Optuna objective. Final model always refits on all training data. |

---

## Hyperparameter Tuning (Optuna)

### Search Space Functions

Each tuned model has a dedicated `_*_space(trial)` function that maps an Optuna trial to a parameter dict:

| Function | Model | Parameters searched |
|---|---|---|
| `_rf_space` | Random Forest | `n_estimators` [100–300], `max_depth` {None,5,10}, `min_samples_split` [2–20], `min_samples_leaf` [1–10] |
| `_lr_space` | Logistic Regression | `C` log-uniform in [1e-4, 1e2] |
| `_xgb_space` | XGBoost | `n_estimators` [50–500], `max_depth` [2–8], `learning_rate` log-uniform [1e-5, 0.5], `subsample` [0.5–1.0], `colsample_bytree` [0.5–1.0], `min_child_weight` [1–30], `gamma` [0–20], `reg_alpha` log-uniform [1e-6, 10], `reg_lambda` log-uniform [1e-6, 10] |

MLP does not use Optuna — it runs with fixed `params` directly.

### `_tune_model` Flow

```
_tune_model(name, X_tr, y_tr)
    │
    ├─ hp_tuning=False or param_space=None
    │       └─ instantiate with base params → fit(X_tr, y_tr)
    │          return (model, None)
    │
    └─ hp_tuning=True
            │
            ├─ if tune_subsample set and len(X_tr) > subsample:
            │       sample subsample rows from X_tr  →  X_opt, y_opt
            │  else:
            │       X_opt, y_opt = X_tr, y_tr
            │
            ├─ create Optuna TPE study (direction="maximize", seed=42)
            │
            ├─ study.optimize(objective, n_trials=n_trials, n_jobs=study_n_jobs)
            │       objective(trial):
            │           params = param_space(trial)           ← sample from search space
            │           model  = instantiate with base_params + params
            │           scores = cross_val_score(model, X_opt, y_opt,
            │                        cv=StratifiedKFold(3), scoring="balanced_accuracy",
            │                        n_jobs=cv_jobs)
            │           return scores.mean()
            │
            ├─ best_params = study.best_params
            │
            └─ best_model = instantiate with base_params + best_params
               best_model.fit(X_tr, y_tr)     ← always refit on FULL training slice
               return (best_model, best_params)
```

### Parallelism and CPU Over-Subscription

`cv_jobs` (parallelism inside `cross_val_score`) is derived automatically to avoid nesting:

```python
model_is_parallel = "n_jobs" in cfg["params"]
cv_jobs = 1 if (model_is_parallel or study_jobs != 1) else -1
```

| Model | `study_n_jobs` | `cv_jobs` | Reason |
|---|---|---|---|
| Random Forest | 1 | 1 | RF uses `n_jobs=-1` internally; no nesting |
| Logistic Regression | 4 | 1 | Study is parallel; avoid nested threads |
| XGBoost | 1 | 1 | XGBoost uses `n_jobs=-1` internally; no nesting |
| MLP | 1 | — | No Optuna; fits directly on full training slice |

### Best Params Logging

After each fold, found parameters are printed and stored in `fold_metrics["best_params"]`:

```
  [LOSO] fold 3/10 ...
    [XGBoost] best params: {'n_estimators': 312, 'max_depth': 4, 'learning_rate': 0.042, ...}
```

---

## LOSO Evaluation Loop

All enabled models (including the Baseline) run inside a single unified loop in `run_loso_evaluation`. For each fold `k`:

```
train_df = concat(all sets except set k)   |  dropna on feature columns
test_df  = set k                           |

X_tr, y_tr = train_df[feature_cols], train_df["true_label"]
X_te, y_te = test_df[feature_cols],  test_df["true_label"]

for each enabled model:
    model, best_params = _tune_model(name, X_tr, y_tr)
    y_pred  = model.predict(X_te)
    y_score = model.predict_proba(X_te)[:, 1]
    record: accuracy, precision, recall, f1, balanced_accuracy, TP, FP, FN, TN
    (Baseline also records: threshold, mean/std genuine distance, mean/std impostor distance)
```

After all folds, each model is retrained on **all** sets combined. This final model is used only for feature importance / coefficient plots — the LOSO metrics are not affected.

---

## Models

### Baseline (Hamming Threshold Classifier)

`HammingThresholdClassifier` is a sklearn-compatible wrapper around the threshold approach:

- `fit()`: sweeps `n_steps` threshold candidates over `[min_distance, max_distance]` of the training pairs and selects the one that maximises `balanced_accuracy_score`
- `predict()`: applies `distance <= threshold_`
- `predict_proba()`: returns `1 - hamming` as the genuine probability

This is the same logic as the baseline script but made adaptive — the threshold is optimised on the training folds rather than fixed at 0.38.

### Random Forest

Ensemble of decision trees. `class_weight="balanced"` compensates for the ~54:1 impostor/genuine imbalance. `n_jobs=-1` uses all CPU cores. `tune_subsample=50_000` caps the rows seen during Optuna search (full dataset per fold may exceed 1M pairs).

### Logistic Regression

Linear classifier. Tuned parameter `C` (inverse regularisation strength). `class_weight="balanced"` and `max_iter=2000` ensure convergence on large imbalanced datasets.

### XGBoost

Gradient-boosted tree ensemble. `scale_pos_weight=50` compensates for the ~54:1 impostor/genuine imbalance (equivalent to `class_weight="balanced"` for XGBoost). `tree_method="hist"` uses histogram-based splits for speed. `n_jobs=-1` uses all CPU cores. `tune_subsample=50_000` caps the Optuna search rows. The search space covers 9 hyperparameters including regularisation terms `reg_alpha` and `reg_lambda`.

### MLP (Multi-Layer Perceptron)

Neural network classifier with fixed architecture `(128, 64)` hidden layers. Hyperparameter tuning is disabled (`hp_tuning=False`) — the model uses fixed `alpha=1e-4` and `learning_rate_init=1e-3`. MLP does not expose `feature_importances_` or `coef_`, so feature importance plots are skipped for this model.

---

## Evaluation Metrics

The same metric set is used for all models:

| Metric | Formula | Notes |
|---|---|---|
| **Accuracy** | (TP + TN) / total | Can be misleading with imbalanced pairs (~54:1) |
| **Precision** | TP / (TP + FP) | Of predicted matches, how many are genuine |
| **Recall (TAR)** | TP / (TP + FN) | Of genuine pairs, how many were detected |
| **F1-score** | 2·P·R / (P+R) | Harmonic mean; robust to imbalance |
| **Balanced Accuracy** | (TPR + TNR) / 2 | Primary metric for tuning; equal weight to both classes |

---

## Plots and Output

### Baseline plots — `visualizations/baseline/`

| File | Content |
|---|---|
| `metrics_per_set.png` | Bar chart of all metrics per LOSO fold |
| `distance_distribution.png` | Genuine vs impostor Hamming distance histograms (all held-out pairs) |
| `roc_curve.png` | ROC curve with AUC |
| `det_curve.png` | DET curve with Equal Error Rate (EER) marked |
| `distance_per_set.png` | Mean genuine / impostor distance per fold with threshold step line |
| `loso_threshold_per_set.png` | Optimised threshold value per fold and mean |
| `far_frr_per_set.png` | FAR and FRR per fold |
| `aggregate_confusion_matrix.png` | Summed TP/FP/FN/TN across all folds |
| `dashboard.png` | 6-panel summary: F1/balanced-acc trends, distance distribution, ROC, distance+threshold, confusion matrix, aggregate table |

### Per-model plots — `visualizations/<model_dir>/`

| Directory | Model |
|---|---|
| `baseline/` | Baseline (Hamming) |
| `random_forest/` | Random Forest |
| `logistic_regression/` | Logistic Regression |
| `xgboost/` | XGBoost |
| `mlp/` | MLP |

Each model directory contains:

| File | Content | Models |
|---|---|---|
| `confusion_matrix.png` | Aggregate TP/FP/FN/TN across all LOSO predictions | All |
| `roc_curve.png` | ROC curve (AUC) across all LOSO predictions | All |
| `metrics_per_set.png` | Bar chart of per-fold metrics | All |
| `feature_importance.png` | Feature importances (`feature_importances_`) from final model | Random Forest, XGBoost |
| `coefficients.png` | Feature coefficients (`coef_`) from final model | Logistic Regression |

### Comparison report — `visualizations/comparison_report.png`

Grouped bar chart placing the Baseline LOSO result side-by-side with each fusion model across all five metrics. Primary summary for deciding whether score-level fusion improves on single-score Hamming.

---

## Output Structure

```
<OUTPUT_ROOT>/                          e.g. out_CASIA_Iris_Thousand_MultiSet_L/
  visualizations/
    comparison_report.png
    baseline/
      metrics_per_set.png
      distance_distribution.png
      roc_curve.png
      det_curve.png
      distance_per_set.png
      loso_threshold_per_set.png
      far_frr_per_set.png
      aggregate_confusion_matrix.png
      dashboard.png
    random_forest/
      confusion_matrix.png  roc_curve.png  metrics_per_set.png  feature_importance.png
    logistic_regression/
      confusion_matrix.png  roc_curve.png  metrics_per_set.png  coefficients.png
    xgboost/
      confusion_matrix.png  roc_curve.png  metrics_per_set.png  feature_importance.png
    mlp/
      confusion_matrix.png  roc_curve.png  metrics_per_set.png
```

---

## Data Flow

```
baseline_casia_thousand_multiset.py
  └─ multi_score_features.csv  (per set — hamming/jaccard/cosine/pearson + true_label)
          │
          ▼
evaluate.py
  load_all_multi_score_features()
          │
          ▼
  run_loso_evaluation()
      │
      ├─ for each fold k  (k = 0 … n-1):
      │       train = concat(all sets except k)
      │       test  = set k
      │       │
      │       for each enabled model:
      │           _tune_model(name, X_tr, y_tr)
      │               ├─ hp_tuning=False → fit directly with base params
      │               └─ hp_tuning=True  →
      │                     Optuna TPE  (n_trials × 3-fold CV on X_opt)
      │                         → best_params
      │                     refit on full X_tr with best_params
      │           model.predict / predict_proba on test fold k
      │           record: metrics, best_params, y_pred, y_score
      │
      ├─ for each enabled model:
      │       _tune_model(name, X_all, y_all)   →  final_model (all data combined)
      │
      └─ buckets[name] = {fold_metrics, all_predictions, final_model}
              │
              ├─ Baseline  →  baseline plots  →  visualizations/baseline/
              └─ Fusion    →  per-model plots →  visualizations/<model_dir>/
          │
          ▼
  plot_comparison_report()
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

# Step 2 — run LOSO evaluation and produce all visualizations
python evaluate.py
```

**Common adjustments:**

| Goal | Action |
|---|---|
| Skip a model | Set `"enabled": False` in `MODELS_CONFIG` |
| Disable Optuna for a model | Set `"hp_tuning": False` — uses `"params"` directly |
| Reduce tuning time | Lower `n_trials` or set `tune_subsample` to cap training rows |
| More parallel trials | Increase `study_n_jobs` (only for models without internal `n_jobs`) |
| Switch eye side | Change `TARGET_EYE_SIDE` to `"L"` or `"R"` (must match baseline script) |
