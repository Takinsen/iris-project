# Improvement: LOSO Evaluation and Score-Level Fusion

This document describes the core concepts, configuration, hyperparameter tuning, evaluation logic, and output of `train.py` and `evaluate.py`.

**Prerequisites:** `baseline_casia_thousand_multiset.py` must be run first to generate `multi_score_features.csv` in each set folder.

---

## Overview

`train.py` runs LOSO training for all enabled models and saves per-fold predictions and metrics.  
`evaluate.py` loads those results and produces all visualizations.

| Phase | Input | Method |
|---|---|---|
| **Baseline LOSO** | `multi_score_features.csv` (hamming feature only) | Threshold found at FAR ≤ `target_far` on n-1 training folds |
| **Score-Level Fusion** | `multi_score_features.csv` (4 distance features) | Classifier trained on n-1 sets, predicts on held-out set |

Both phases use **Leave-One-Set-Out (LOSO)** cross-validation inside a single unified training loop.

---

## Terminology

| Term | Meaning |
|---|---|
| **LOSO** | Leave-One-Set-Out — each set takes one turn as the held-out test fold while all others form the training data. |
| **LOSO fold** | One iteration: n-1 sets used for training, 1 set held out for evaluation. |
| **Score-level fusion** | Combining multiple distance scores (Hamming, Jaccard, Weighted Euclidean, Pearson) into a single classification decision. |
| **Feature columns** | `hamming`, `jaccard`, `weighted_euclidean`, `pearson` — four distance metrics per pair. Lower = more similar templates. |
| **Genuine / Impostor** | Genuine = same subject + same eye side; impostor = different subjects. |
| **FAR (False Accept Rate)** | Rate of impostor pairs incorrectly classified as genuine. Equivalent to FPR in sklearn. |
| **FRR (False Reject Rate)** | Rate of genuine pairs incorrectly classified as impostor. Equal to `1 - TPR`. |
| **EER** | Equal Error Rate — the operating point where FAR = FRR. |
| **FAR operating point** | A target FAR used to select the classification threshold. The threshold is set to the highest score cut-off that keeps FAR ≤ the target value. |
| **TPE sampler** | Tree-structured Parzen Estimator — an Optuna Bayesian search strategy. |

---

## Core Concept

### Why LOSO?

A standard train/test split would mix subjects between train and test. LOSO ensures **zero subject overlap** between training and test in any fold — the model must generalise to completely unseen identities.

```
Sets:  [set_01] [set_02] [set_03] ... [set_10]
                                               all different subjects

Fold 1:  Train = set_02 + set_03 + … + set_10   Test = set_01
Fold 2:  Train = set_01 + set_03 + … + set_10   Test = set_02
...
Fold 10: Train = set_01 + … + set_09             Test = set_10
```

### Why Score-Level Fusion?

The baseline uses only Hamming distance with a single threshold. Score-level fusion trains a classifier on four distance features to learn a more expressive decision boundary. The fusion models may capture complementary information not captured by Hamming alone.

### Why Optuna?

Fixed hyperparameters may not be optimal for each training fold. Optuna runs a Bayesian search over the hyperparameter space using TPE, spending more trials in promising regions.

---

## Configuration

### Eye Side

```python
TARGET_EYE_SIDE = "L"   # "L" or "R"
OUTPUT_ROOT = rf".\out_CASIA_Iris_Thousand_MultiSet_{TARGET_EYE_SIDE}"
```

### MODELS_CONFIG

All model settings live in `MODELS_CONFIG` at the top of `train.py`.

```python
MODELS_CONFIG = {
    "Baseline (Hamming)": {
        "enabled":        True,
        "hp_tuning":      False,
        "params":         {"target_far": 1e-5},
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
        "study_n_jobs":   1,
        "tune_subsample": 50_000,
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
    "MLP": {
        "enabled":        True,
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
| `study_n_jobs` | int | Parallel Optuna trials. Set to `1` for models with internal `n_jobs` to avoid CPU over-subscription. |
| `tune_subsample` | int or `None` | Max rows passed to the Optuna objective. Final model always refits on all training data. |

---

## Hyperparameter Tuning (Optuna)

### Search Space Functions

| Function | Model | Parameters searched |
|---|---|---|
| `_rf_space` | Random Forest | `n_estimators` [100–300], `max_depth` {None,5,10}, `min_samples_split` [2–20], `min_samples_leaf` [1–10] |
| `_lr_space` | Logistic Regression | `C` log-uniform in [1e-4, 1e2] |

MLP and Baseline do not use Optuna — they run with fixed `params` directly.

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
            │           params = param_space(trial)
            │           model  = instantiate with base_params + params
            │           scores = cross_val_score(model, X_opt, y_opt,
            │                        cv=StratifiedKFold(3), scoring="roc_auc",
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

```python
model_is_parallel = "n_jobs" in cfg["params"]
cv_jobs = 1 if (model_is_parallel or study_jobs != 1) else -1
```

| Model | `study_n_jobs` | `cv_jobs` | Reason |
|---|---|---|---|
| Random Forest | 1 | 1 | RF uses `n_jobs=-1` internally; no nesting |
| Logistic Regression | 4 | 1 | Study is parallel; avoid nested threads |
| MLP | 1 | — | No Optuna; fits directly on full training slice |

### Best Params Logging

```
  [LOSO] fold 3/10 ...
    [Random Forest] best params: {'n_estimators': 250, 'max_depth': 10, ...}
```

---

## LOSO Evaluation Loop

All enabled models run inside a single unified loop in `run_loso_training`. For each fold `k`:

```
train_df = concat(all sets except set k)   |  dropna on feature columns
test_df  = set k                           |

X_tr, y_tr = train_df[feature_cols], train_df["true_label"]
X_te, y_te = test_df[feature_cols],  test_df["true_label"]

for each enabled model:
    model, best_params = _tune_model(name, X_tr, y_tr)
    y_pred  = model.predict(X_te)
    y_score = model.predict_proba(X_te)[:, 1]
    record: accuracy, precision, recall, f1, balanced_accuracy, EER, TP, FP, FN, TN
    (Baseline also records: threshold, mean/std genuine distance, mean/std impostor distance)
```

After all folds, each model is retrained on **all** sets combined. This final model is used only for feature importance / coefficient plots.

---

## Models

### Baseline (Hamming Threshold Classifier)

`HammingThresholdClassifier` is a sklearn-compatible wrapper:

- `fit()`: scans the ROC curve on the training pairs and selects the **highest Hamming threshold where FAR ≤ `target_far`** (default `1e-5`). This maximises genuine recall while capping the false accept rate. The fitted threshold is stored in `threshold_` and varies per fold.
- `predict()`: applies `hamming_distance ≤ threshold_`
- `predict_proba()`: returns raw Hamming distance as column 1 (`score_is_distance = True`)
- `feature_importances_`: returns `[1, 0, 0, 0]` — only the Hamming feature is used

The threshold is adaptive — optimised on each training fold rather than fixed at a global value.

### Random Forest

Ensemble of decision trees. `class_weight="balanced"` compensates for the ~54:1 impostor/genuine imbalance. `n_jobs=-1` uses all CPU cores. `tune_subsample=50_000` caps the rows seen during Optuna search.

### Logistic Regression

Linear classifier. Tuned parameter `C` (inverse regularisation strength). `class_weight="balanced"` and `max_iter=2000` ensure convergence on large imbalanced datasets.

### MLP (Multi-Layer Perceptron)

Neural network classifier with fixed architecture `(128, 64)` hidden layers. Hyperparameter tuning is disabled — the model uses fixed `alpha=1e-4` and `learning_rate_init=1e-3`. MLP does not expose `feature_importances_` or `coef_`, so those plots are skipped.

---

## Feature Columns

All models (except Baseline, which uses only `hamming`) train on four distance features:

| Column | Description |
|---|---|
| `hamming` | Hamming distance between iris code bits at unmasked positions |
| `jaccard` | Jaccard distance on binarised bits: `1 − (intersection / union)` |
| `weighted_euclidean` | L2 distance weighted by per-position activation magnitude: `√(Σ max(aᵢ,bᵢ)·(aᵢ−bᵢ)² / Σ max(aᵢ,bᵢ))`. Positions where both bits are zero contribute negligible weight. |
| `pearson` | Pearson correlation distance: `1 − corr(a, b)` on continuous code values |

All four distances are in `[0, 1]` with lower values indicating more similar templates.

---

## Evaluation Metrics

| Metric | Formula | Notes |
|---|---|---|
| **Accuracy** | (TP + TN) / total | Can be misleading with imbalanced pairs (~54:1) |
| **Precision** | TP / (TP + FP) | Of predicted matches, how many are genuine |
| **Recall (TAR)** | TP / (TP + FN) | Of genuine pairs, how many were detected |
| **F1-score** | 2·P·R / (P+R) | Harmonic mean; robust to imbalance |
| **Balanced Accuracy** | (TPR + TNR) / 2 | Equal weight to both classes |
| **EER** | FAR = FRR operating point | Lower is better |

---

## Plots and Output

### Per-model plots — `<OUTPUT_ROOT>/<model_slug>/`

| File | Content | Models |
|---|---|---|
| `global_distance_distribution.png` | 2×2 panel: Hamming, Jaccard, Weighted Euclidean, Pearson — genuine vs impostor | All |
| `dashboard.png` | Summary cards (avg ± std) + per-fold bar charts for Accuracy, Balanced Acc., Precision, Recall, F1 | All |
| `confusion_matrix.png` | Aggregate TP/FP/FN/TN at the threshold where FAR ≤ 1e-5 per fold | All |
| `roc_curve.png` | ROC curve with AUC and EER | All |
| `metrics_per_fold.png` | Bar chart of per-fold accuracy, precision, recall, F1, EER | All |
| `far_frr_curve.png` | FAR/FRR curves: one thin line per test fold + bold aggregate, with EER markers | All |
| `distance_distribution.png` | Hamming distance density — genuine vs impostor (from fold predictions) | Baseline only |
| `loso_threshold_per_set.png` | Per-fold fitted Hamming threshold and mean | Baseline only |
| `feature_importance.png` | Feature importances (`feature_importances_`) from final model | RF, Baseline |
| `coefficients.png` | Feature coefficients (`coef_`) from final model | Logistic Regression |

### Root visualizations — `<OUTPUT_ROOT>/visualization/`

| File | Content |
|---|---|
| `global_distance_distribution.png` | Same 2×2 distance distribution panel (global copy) |
| `metrics_summary_table.png` | Table: rows = models, cols = avg metrics |
| `comparison_report.png` | Side-by-side bar chart comparing all models on key metrics |

---

## Confusion Matrix Operating Point

The confusion matrix is evaluated at a **per-fold FAR ≤ 1e-5 threshold**, not at the EER. For each test fold:

1. Compute `roc_curve(y_true, y_score_genuine_ascending)`
2. Find the last index where `fpr ≤ 1e-5` — this gives the highest TPR that still satisfies the FAR constraint
3. Re-predict at that threshold and accumulate TP/FP/FN/TN across all folds

For the Baseline, `y_score` is raw Hamming distance (lower = more genuine), so it is flipped to `1 − hamming` before the ROC curve computation, and the resulting threshold is converted back to Hamming space.

---

## Output Structure

```
<OUTPUT_ROOT>/                          e.g. out_CASIA_Iris_Thousand_MultiSet_L/
  visualization/
    global_distance_distribution.png
    metrics_summary_table.png
    comparison_report.png
  baseline_hamming/
    global_distance_distribution.png
    dashboard.png
    confusion_matrix.png
    roc_curve.png
    metrics_per_fold.png
    far_frr_curve.png
    distance_distribution.png
    loso_threshold_per_set.png
    feature_importance.png
  random_forest/
    global_distance_distribution.png
    dashboard.png  confusion_matrix.png  roc_curve.png
    metrics_per_fold.png  far_frr_curve.png  feature_importance.png
  logistic_regression/
    global_distance_distribution.png
    dashboard.png  confusion_matrix.png  roc_curve.png
    metrics_per_fold.png  far_frr_curve.png  coefficients.png
  mlp/
    global_distance_distribution.png
    dashboard.png  confusion_matrix.png  roc_curve.png
    metrics_per_fold.png  far_frr_curve.png

model/
  baseline_hamming/   fold_01.csv … fold_NN.csv   fold_metrics.csv   final_model.pkl
  random_forest/      fold_01.csv … fold_NN.csv   fold_metrics.csv   final_model.pkl
  logistic_regression/ fold_01.csv … fold_NN.csv  fold_metrics.csv   final_model.pkl
  mlp/                fold_01.csv … fold_NN.csv   fold_metrics.csv   final_model.pkl
```

---

## Data Flow

```
baseline_casia_thousand_multiset.py
  └─ multi_score_features.csv  (per set — hamming/jaccard/weighted_euclidean/pearson + true_label)
          │
          ▼
train.py
  load_all_multi_score_features()
          │
          ▼
  run_loso_training()
      │
      ├─ for each fold k  (k = 1 … n):
      │       train = concat(all sets except k)
      │       test  = set k
      │       │
      │       for each enabled model:
      │           _tune_model(name, X_tr, y_tr)
      │               ├─ Baseline  → fit() finds threshold at FAR ≤ 1e-5 on training pairs
      │               ├─ hp_tuning=False → fit directly with base params
      │               └─ hp_tuning=True  →
      │                     Optuna TPE  (n_trials × 3-fold CV on X_opt, scoring=roc_auc)
      │                         → best_params
      │                     refit on full X_tr with best_params
      │           model.predict / predict_proba on test fold k
      │           save: fold_kk.csv  (y_true, y_pred, y_score, hamming, pair_type)
      │
      ├─ for each enabled model:
      │       refit on all data → final_model.pkl
      │       write fold_metrics.csv
      │
          ▼
evaluate.py
  load_model_results()   ←  reads fold_[0-9]*.csv + fold_metrics.csv + final_model.pkl
          │
          ▼
  per-model plots  →  <OUTPUT_ROOT>/<model_slug>/
  comparison plots →  <OUTPUT_ROOT>/visualization/
```

---

## Running

```bash
conda activate iris-dev

# Step 1 — generate multi_score_features.csv per set
python baseline_casia_thousand_multiset.py

# Step 2 — run LOSO training (saves fold CSVs and models)
python train.py

# Step 3 — generate all visualizations
python evaluate.py
```

**Common adjustments:**

| Goal | Action |
|---|---|
| Skip a model | Set `"enabled": False` in `MODELS_CONFIG` |
| Change FAR operating point | Set `"target_far"` in Baseline `params`; set `TARGET_FAR` in `evaluate.py` |
| Disable Optuna for a model | Set `"hp_tuning": False` — uses `"params"` directly |
| Reduce tuning time | Lower `n_trials` or set `tune_subsample` to cap training rows |
| More parallel trials | Increase `study_n_jobs` (only for models without internal `n_jobs`) |
| Switch eye side | Change `TARGET_EYE_SIDE` to `"L"` or `"R"` in both scripts |
| Regenerate cached folds | Delete `model/<slug>/fold_*.csv` or set `USE_CACHE = False` |
