import os
import glob
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (roc_curve, accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------
# Baseline classifier — sklearn-compatible wrapper
# ------------------------------------------------------------------

class HammingThresholdClassifier:
    """
    Fixed Hamming-distance threshold classifier.
    fit()           → no-op; threshold stays at the value set in __init__.
    predict_proba() → column 1 = raw Hamming distance (score_is_distance=True).
    feature_importances_ → [1, 0, 0, 0] for plot compatibility.
    """

    score_is_distance = True   # lower score = more genuine

    def __init__(self, threshold: float = 0.38):
        self.threshold_ = threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HammingThresholdClassifier":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (X[:, 0] <= self.threshold_).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ham = np.clip(X[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - ham, ham])

    @property
    def feature_importances_(self) -> np.ndarray:
        imp    = np.zeros(4)   # hamming, jaccard, cosine, pearson
        imp[0] = 1.0
        return imp


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

TARGET_EYE_SIDE = "L"   # "L" or "R"
OUTPUT_ROOT    = rf".\out_CASIA_Iris_Thousand_MultiSet_{TARGET_EYE_SIDE}"
MODEL_SAVE_DIR = os.path.join(".", "model")
FEATURE_COLS   = ["hamming", "jaccard", "cosine", "pearson"]
USE_CACHE      = True   # skip fold training when fold CSV already exists


# ------------------------------------------------------------------
# Optuna search-space definitions
# ------------------------------------------------------------------

def _rf_space(trial):
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 300),
        "max_depth":         trial.suggest_categorical("max_depth", [None, 5, 10]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
    }


def _lr_space(trial):
    return {
        "C": trial.suggest_float("C", 1e-4, 1e2, log=True),
    }


def _xgb_space(trial):
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 50, 500),
        "max_depth":        trial.suggest_int("max_depth", 2, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-5, 0.5, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
        "gamma":            trial.suggest_float("gamma", 0.0, 20.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-6, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-6, 10.0, log=True),
    }


# ------------------------------------------------------------------
# Model Configuration
# Toggle `enabled` to include/exclude a model.
# Set `hp_tuning: True` to run Optuna search on each LOSO training fold.
# ------------------------------------------------------------------

MODELS_CONFIG = {
    "Baseline (Hamming)": {
        "enabled":        True,
        "hp_tuning":      False,
        "params":         {"threshold": 0.38},
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
        "enabled":        False,
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


# ------------------------------------------------------------------
# Model utilities
# ------------------------------------------------------------------

_CONSTRUCTORS = {
    "Baseline (Hamming)": HammingThresholdClassifier,
    "Random Forest":      RandomForestClassifier,
    "Logistic Regression": LogisticRegression,
    "XGBoost":            XGBClassifier,
    "MLP":                MLPClassifier,
}


def _instantiate_model(name: str, extra_params: dict = None):
    params = {**MODELS_CONFIG[name]["params"], **(extra_params or {})}
    return _CONSTRUCTORS[name](**params)


def _model_slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").strip("_")


def _fold_csv_path(name: str, fold_idx: int) -> str:
    return os.path.join(MODEL_SAVE_DIR, _model_slug(name), f"fold_{fold_idx:02d}.csv")


def _fold_metrics_path(name: str) -> str:
    return os.path.join(MODEL_SAVE_DIR, _model_slug(name), "fold_metrics.csv")


def _final_model_path(name: str) -> str:
    return os.path.join(MODEL_SAVE_DIR, _model_slug(name), "final_model.pkl")


# ------------------------------------------------------------------
# Hyperparameter tuning
# ------------------------------------------------------------------

def _tune_model(name: str, X_tr: np.ndarray, y_tr: np.ndarray):
    """
    Fits a model for `name` on (X_tr, y_tr).
    If hp_tuning=True, runs an Optuna TPE study (optimises roc_auc).
    Returns (fitted_model, best_params_or_None).
    """
    cfg         = MODELS_CONFIG[name]
    param_space = cfg.get("param_space")
    if not cfg.get("hp_tuning", False) or param_space is None:
        model = _instantiate_model(name)
        model.fit(X_tr, y_tr)
        return model, None

    n_trials   = cfg.get("n_trials", 30)
    study_jobs = cfg.get("study_n_jobs", 1)
    subsample  = cfg.get("tune_subsample")

    model_is_parallel = "n_jobs" in cfg["params"]
    cv_jobs = 1 if (model_is_parallel or study_jobs != 1) else -1
    cv      = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    if subsample and len(X_tr) > subsample:
        rng          = np.random.default_rng(42)
        idx          = rng.choice(len(X_tr), size=subsample, replace=False)
        X_opt, y_opt = X_tr[idx], y_tr[idx]
    else:
        X_opt, y_opt = X_tr, y_tr

    def objective(trial):
        scores = cross_val_score(
            _instantiate_model(name, param_space(trial)),
            X_opt, y_opt,
            cv=cv, scoring="roc_auc", n_jobs=cv_jobs,
        )
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=study_jobs, show_progress_bar=False)

    best_model = _instantiate_model(name, study.best_params)
    best_model.fit(X_tr, y_tr)
    return best_model, study.best_params


# ------------------------------------------------------------------
# Evaluation helpers
# ------------------------------------------------------------------

def _compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> tuple:
    """Returns (eer, eer_threshold). y_score: higher = more likely genuine."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def _eval(model, X: np.ndarray, y: np.ndarray, y_score: np.ndarray) -> dict:
    yp  = model.predict(X)
    tp  = int(((y == 1) & (yp == 1)).sum())
    fp  = int(((y == 0) & (yp == 1)).sum())
    fn  = int(((y == 1) & (yp == 0)).sum())
    tn  = int(((y == 0) & (yp == 0)).sum())
    eer, eer_thr = _compute_eer(y, y_score)
    return {
        "accuracy":          float(accuracy_score(y, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
        "precision":         float(precision_score(y, yp, zero_division=0)),
        "recall":            float(recall_score(y, yp, zero_division=0)),
        "f1":                float(f1_score(y, yp, zero_division=0)),
        "eer":               eer,
        "eer_threshold":     eer_thr,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
    }


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------

def load_all_multi_score_features() -> list:
    frames = []
    for fpath in sorted(glob.glob(
            os.path.join(OUTPUT_ROOT, "**", "multi_score_features.csv"), recursive=True)):
        df = pd.read_csv(fpath)
        if len(df) > 0:
            frames.append(df)
    return frames


# ------------------------------------------------------------------
# LOSO training
# Saves per fold:  model/<slug>/fold_NN.csv
# Saves per model: model/<slug>/fold_metrics.csv
#                  model/<slug>/final_model.pkl
# ------------------------------------------------------------------

def run_loso_training(per_set_dfs: list) -> None:
    """
    Runs Leave-One-Set-Out training for all enabled models and persists results.

    fold_NN.csv      — one row per pair: set_id, y_true, y_pred, y_score,
                       hamming, pair_type, [hamming_threshold for Baseline]
    fold_metrics.csv — one row per fold with all scalar metrics
    final_model.pkl  — model refitted on ALL data (for feature importance plots)

    USE_CACHE=True skips training for folds whose CSV already exists.
    """
    n             = len(per_set_dfs)
    enabled_names = [name for name, cfg in MODELS_CONFIG.items() if cfg["enabled"]]

    for name in enabled_names:
        os.makedirs(os.path.join(MODEL_SAVE_DIR, _model_slug(name)), exist_ok=True)

    all_fold_metrics = {name: [] for name in enabled_names}

    for fold_idx in range(1, n + 1):
        test_idx = fold_idx - 1
        print(f"\n  [LOSO] fold {fold_idx}/{n}", flush=True)

        train_df = pd.concat(
            [df for i, df in enumerate(per_set_dfs) if i != test_idx],
            ignore_index=True,
        ).dropna(subset=FEATURE_COLS)
        test_df  = per_set_dfs[test_idx].dropna(subset=FEATURE_COLS)

        X_tr, y_tr = train_df[FEATURE_COLS].values, train_df["true_label"].values
        X_te, y_te = test_df[FEATURE_COLS].values,  test_df["true_label"].values
        set_id     = (test_df["set_id"].iloc[0] if "set_id" in test_df.columns
                      else f"set_{fold_idx:02d}")
        d_te       = test_df["hamming"].values
        pt_te      = (test_df["pair_type"].tolist() if "pair_type" in test_df.columns
                      else np.where(y_te == 1, "genuine", "impostor").tolist())

        for name in enabled_names:
            fold_path = _fold_csv_path(name, fold_idx)

            # ---- Cache hit: recompute metrics from saved predictions ----
            if USE_CACHE and os.path.exists(fold_path):
                print(f"    [{name}] fold {fold_idx} — loaded from cache", flush=True)
                cached   = pd.read_csv(fold_path)
                y_true_c = cached["y_true"].values
                y_pred_c = cached["y_pred"].values
                y_scr_c  = cached["y_score"].values
                ham_c    = cached["hamming"].values

                y_scr_eval = (1.0 - y_scr_c) if name == "Baseline (Hamming)" else y_scr_c
                eer, eer_thr = _compute_eer(y_true_c, y_scr_eval)
                tp = int(((y_true_c == 1) & (y_pred_c == 1)).sum())
                fp = int(((y_true_c == 0) & (y_pred_c == 1)).sum())
                fn = int(((y_true_c == 1) & (y_pred_c == 0)).sum())
                tn = int(((y_true_c == 0) & (y_pred_c == 0)).sum())
                metrics = {
                    "fold":              fold_idx,
                    "set_id":            set_id,
                    "accuracy":          float(accuracy_score(y_true_c, y_pred_c)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_true_c, y_pred_c)),
                    "precision":         float(precision_score(y_true_c, y_pred_c, zero_division=0)),
                    "recall":            float(recall_score(y_true_c, y_pred_c, zero_division=0)),
                    "f1":                float(f1_score(y_true_c, y_pred_c, zero_division=0)),
                    "eer":               eer,
                    "eer_threshold":     eer_thr,
                    "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                }
                if "hamming_threshold" in cached.columns:
                    metrics["threshold"] = float(cached["hamming_threshold"].iloc[0])
                    gen_d = ham_c[y_true_c == 1]
                    imp_d = ham_c[y_true_c == 0]
                    metrics["mean_genuine_distance"]  = float(gen_d.mean()) if len(gen_d) else np.nan
                    metrics["std_genuine_distance"]   = float(gen_d.std())  if len(gen_d) else np.nan
                    metrics["mean_impostor_distance"] = float(imp_d.mean()) if len(imp_d) else np.nan
                    metrics["std_impostor_distance"]  = float(imp_d.std())  if len(imp_d) else np.nan
                all_fold_metrics[name].append(metrics)
                continue

            # ---- Train ----
            model, best_params = _tune_model(name, X_tr, y_tr)
            if best_params:
                print(f"    [{name}] best params: {best_params}", flush=True)

            y_score_te   = model.predict_proba(X_te)[:, 1]   # raw (hamming for Baseline)
            y_score_eval = (1.0 - y_score_te) if getattr(model, "score_is_distance", False) else y_score_te
            metrics      = _eval(model, X_te, y_te, y_score_eval)
            metrics["fold"]   = fold_idx
            metrics["set_id"] = set_id
            if best_params:
                metrics["best_params"] = str(best_params)

            if hasattr(model, "threshold_"):
                metrics["threshold"] = float(model.threshold_)
                gen_d = d_te[y_te == 1]
                imp_d = d_te[y_te == 0]
                metrics["mean_genuine_distance"]  = float(gen_d.mean()) if len(gen_d) else np.nan
                metrics["std_genuine_distance"]   = float(gen_d.std())  if len(gen_d) else np.nan
                metrics["mean_impostor_distance"] = float(imp_d.mean()) if len(imp_d) else np.nan
                metrics["std_impostor_distance"]  = float(imp_d.std())  if len(imp_d) else np.nan

            all_fold_metrics[name].append(metrics)

            # Save fold predictions CSV
            fold_df = pd.DataFrame({
                "set_id":    [set_id] * len(y_te),
                "y_true":    y_te,
                "y_pred":    model.predict(X_te),
                "y_score":   y_score_te,
                "hamming":   d_te,
                "pair_type": pt_te,
            })
            if hasattr(model, "threshold_"):
                fold_df["hamming_threshold"] = metrics["threshold"]
            fold_df.to_csv(fold_path, index=False)

    # Save fold_metrics.csv + final_model.pkl per model
    all_df = pd.concat(per_set_dfs, ignore_index=True).dropna(subset=FEATURE_COLS)
    X_all, y_all = all_df[FEATURE_COLS].values, all_df["true_label"].values

    print("\n  [LOSO] fitting final models on all data ...")
    for name in enabled_names:
        pd.DataFrame(all_fold_metrics[name]).to_csv(_fold_metrics_path(name), index=False)
        print(f"    [{name}] saved fold_metrics.csv")

        final_path = _final_model_path(name)
        if USE_CACHE and os.path.exists(final_path):
            print(f"    [{name}] final_model.pkl loaded from cache")
        else:
            final, best_params = _tune_model(name, X_all, y_all)
            if best_params:
                print(f"    [{name}] final best params: {best_params}", flush=True)
            with open(final_path, "wb") as f:
                pickle.dump(final, f)
            print(f"    [{name}] saved final_model.pkl")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    print(f"Loading data from: {OUTPUT_ROOT}")
    per_set_multi = load_all_multi_score_features()
    total_pairs   = sum(len(df) for df in per_set_multi)
    print(f"  {len(per_set_multi)} sets | {total_pairs:,} pair records")

    print("\nEnabled models:")
    for name, cfg in MODELS_CONFIG.items():
        status = "ON " if cfg["enabled"] else "OFF"
        if cfg.get("hp_tuning") and cfg.get("param_space"):
            sub  = f" sub={cfg.get('tune_subsample','all')}" if cfg.get("tune_subsample") else ""
            tune = f" [Optuna n={cfg.get('n_trials',0)} jobs={cfg.get('study_n_jobs',1)}{sub}]"
        else:
            tune = ""
        print(f"  [{status}] {name}{tune}")

    if not per_set_multi:
        print("\n  [skip] no multi_score_features.csv found — run baseline script first.")
        return

    print("\n[LOSO training — all enabled models]")
    run_loso_training(per_set_multi)

    print("\nDone. Run evaluate.py to generate visualizations.")
    for name, cfg in MODELS_CONFIG.items():
        if cfg["enabled"]:
            print(f"  model/{_model_slug(name)}/")


if __name__ == "__main__":
    main()
