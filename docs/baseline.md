# Baseline: CASIA-Iris-Thousand Multi-Set Evaluation

This document describes the core concepts, configuration, evaluation logic, and output of `baseline_casia_thousand_multiset.py`.

---

## Overview

The script runs a multi-set iris **verification** experiment on the CASIA-Iris-Thousand dataset using the Open-Iris / Worldcoin pipeline. It:

1. Downloads the dataset via `kagglehub`
2. Partitions subjects into non-overlapping **sets**
3. Generates an `IrisTemplate` (IrisCode) for each valid image via the IRISPipeline
4. Computes pairwise Hamming distances within each set
5. Applies a fixed threshold to classify each pair as genuine (match) or impostor (non-match)
6. Records per-set and aggregate evaluation metrics and confusion matrices

The output of this script is the **input** to `evaluate.py` — specifically `pair_records.csv` and `multi_score_features.csv` in each set folder.

---

## Terminology

| Term | Meaning |
|---|---|
| **Iris verification** | A 1-vs-1 task: given two iris images, decide if they come from the same person. |
| **IrisCode / IrisTemplate** | A binary array encoding the texture pattern of one iris, produced by the Open-Iris pipeline. Stored as `iris.IrisTemplate`. |
| **Hamming distance** | The fraction of bit positions where two IrisCodes differ, computed only at positions that are **unmasked in both** templates. Range: [0, 1]. Lower = more similar. |
| **Threshold** | A Hamming distance cut-off. Pairs with distance ≤ threshold → predicted as genuine (match). Pairs above → predicted as impostor (non-match). |
| **Genuine pair** | Two images from the same subject and same eye side. Expected Hamming distance ≈ 0.0–0.35 in practice. |
| **Impostor pair** | Two images from different subjects. Expected Hamming distance ≈ 0.45–0.50 (close to random bit strings). |
| **Set** | A self-contained evaluation batch of `SUBJECTS_PER_SET` subjects × `IMAGES_PER_SUBJECT` valid images each. Sets are non-overlapping. |
| **Mask code** | `True` at each bit position means the bit is valid (not occluded by eyelid or reflection). Masked bits are excluded from all distance computations. |

---

## Core Concept

Iris biometrics work because two IrisCodes from the **same** iris have a Hamming distance close to 0 (nearly identical codes), while two IrisCodes from **different** irises have a Hamming distance close to 0.5 (random overlap). A single threshold separates the two distributions:

```
Genuine distribution:  ──────▓▓▓▓▓▓──────────────────────────
Impostor distribution: ──────────────────────▓▓▓▓▓▓▓▓▓────────
                       0.0        0.38 (threshold)         1.0
                                   │
                          distance ≤ 0.38 → match
                          distance >  0.38 → non-match
```

The quality of separation (the gap between the two distributions) determines how accurately a threshold can distinguish genuine pairs from impostors.

---

## Configuration

All parameters are defined at the top of the script. Edit these to change experiment scope.

| Parameter | Default | Description |
|---|---|---|
| `DATASET_ROOT` | *(auto from kagglehub)* | Path to the downloaded dataset root |
| `OUTPUT_ROOT` | `.\out_CASIA_Iris_Thousand_MultiSet_L` | Root folder for all experiment outputs |
| `CACHE_DIR` | `.\cache_templates_CASIA_Iris_Thousand_Baseline` | Folder for cached IrisTemplate `.pkl` files |
| `TARGET_EYE_SIDE` | `"L"` | Eye side to use — `"L"` (left) or `"R"` (right). One side per run. |
| `SUBJECTS_PER_SET` | `50` | Number of subjects in each evaluation set |
| `IMAGES_PER_SUBJECT` | `10` | Number of valid images required per subject |
| `MAX_SETS` | `10` | Cap on sets to run; `None` = all possible non-overlapping sets |
| `THRESHOLD` | `0.38` | Hamming distance cut-off: ≤ threshold → match |
| `USE_CACHE` | `True` | Reuse previously generated `.pkl` templates across runs |
| `OVERWRITE_CACHE` | `False` | Force template regeneration even if cache exists |

---

## Step-by-Step Process

### Step 1 — Dataset Download and Root Resolution

`kagglehub` downloads and caches the dataset on first run. The returned path may be wrapped in packaging folders. `resolve_dataset_root` descends up to 8 levels until it finds a directory whose children contain `L/` or `R/` subfolders (i.e. actual subject folders).

---

### Step 2 — Set Construction

`build_sets_from_dataset` iterates through subject folders in sorted order and fills sets:

```
For each subject directory:
  1. Attempt to collect IMAGES_PER_SUBJECT valid templates
  2. If enough templates → add subject to the current set
  3. If not enough → record in skipped_subjects, try next subject
  4. Once SUBJECTS_PER_SET subjects collected → finalise set, start new set
  5. Stop when MAX_SETS reached or subjects exhausted
```

Each subject appears in **at most one set** — a `used_subject_ids` set prevents overlap.

---

### Step 3 — Template Generation and Caching

For each image, `get_or_create_cached_template` either loads a cached template or runs the full pipeline:

```
Cache hit  (USE_CACHE=True, file exists, OVERWRITE_CACHE=False):
    → load .pkl from CACHE_DIR/<md5(image_path)>.pkl
    → return CachedTemplate

Cache miss / forced regeneration:
    → cv2.imread(image_path, IMREAD_GRAYSCALE)
    → iris.IRISPipeline()(IRImage(...))
    → extract iris_template from output
    → pickle.dumps(template) → save .pkl
    → return CachedTemplate
```

The cache key is the MD5 hash of the full image path, so renaming or moving the dataset invalidates the cache.

---

### Step 4 — Distance Matrix

`build_distance_matrix` computes a pairwise Hamming distance for every unique pair `(i, j)` in the set using `HammingDistanceMatcher`. The result is an `N × N` symmetric `DataFrame` with `NaN` on the diagonal.

With defaults (50 subjects × 10 images = 500 images): **C(500, 2) = 124,750 unique pairs**.

---

### Step 5 — Pair Records and Labelling

`extract_unique_pair_records` reads the upper triangle of the distance matrix and labels each pair:

```python
pair_type  = "genuine"   if same subject_id AND same eye_side
true_label = 1           (genuine)
true_label = 0           (impostor)
```

With 50 subjects × 10 images per subject:
- **Genuine pairs:** 50 × C(10, 2) = 50 × 45 = **2,250**
- **Impostor pairs:** 124,750 − 2,250 = **122,500**

---

### Step 6 — Multi-Score Features

`build_multi_score_pair_df` computes four distance metrics per pair and writes `multi_score_features.csv`. These features are later used by the fusion classifiers in `evaluate.py`.

| Feature | How computed |
|---|---|
| `hamming` | `HammingDistanceMatcher.run(tpl_a, tpl_b)` — fraction of differing bits at jointly unmasked positions |
| `jaccard` | `1 − (intersection / union)` on binarised bits at jointly unmasked positions |
| `cosine` | `1 − (dot(a, b) / (‖a‖ · ‖b‖))` on raw float bit values at jointly unmasked positions |
| `pearson` | `1 − corrcoef(a, b)` on raw float bit values at jointly unmasked positions |

Only bits where `mask_codes == True` in **both** templates are included. If no valid bits overlap, all metrics are `NaN`.

---

### Step 7 — Classification and Metrics

`compute_comparison_summary` applies the fixed `THRESHOLD` to every pair:

```python
pred_label = 1  if distance <= THRESHOLD   (match / genuine)
pred_label = 0  if distance >  THRESHOLD   (non-match / impostor)
```

Confusion matrix:

| | Predicted Match | Predicted Non-Match |
|---|---|---|
| **Actual Match** (genuine) | TP | FN |
| **Actual Non-Match** (impostor) | FP | TN |

---

## Evaluation Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | (TP + TN) / (TP + FP + FN + TN) | Proportion of all pairs correctly classified |
| **Precision** | TP / (TP + FP) | Of all predicted matches, how many are actually genuine |
| **Recall (TAR)** | TP / (TP + FN) | Of all genuine pairs, how many are correctly detected |
| **F1-score** | 2 · precision · recall / (precision + recall) | Harmonic mean of precision and recall |
| **Balanced Accuracy** | (TPR + TNR) / 2 | Average of recall and specificity; robust to class imbalance |
| **FAR** | FP / (FP + TN) | False Accept Rate — fraction of impostors incorrectly accepted |
| **FRR** | FN / (FN + TP) | False Reject Rate — fraction of genuines incorrectly rejected |

**Class imbalance note:** with defaults, impostor pairs outnumber genuine pairs ~54:1 (122,500 vs 2,250). Accuracy alone is misleading — a classifier that always predicts non-match scores ~98% accuracy while being useless. Balanced accuracy and F1-score are more informative.

---

## Output Files

### Per set — written to `<OUTPUT_ROOT>/set_<N>_<EYE>_<subject_range>/`

| File | Contents |
|---|---|
| `selected_images.csv` | Images successfully included in the set |
| `failed_images.csv` | Images that failed template creation with stage and reason |
| `skipped_subjects.csv` | Subjects excluded for insufficient valid images |
| `set_manifest.csv` | Subject list and set configuration |
| `distance_matrix.csv` | N×N pairwise Hamming distance matrix |
| `pair_records.csv` | One row per pair: distance, pair_type, true_label, pred_label — **read by evaluate.py** |
| `multi_score_features.csv` | One row per pair: hamming + jaccard + cosine + pearson — **read by evaluate.py** |
| `comparison_summary.csv` | TP, FP, FN, TN and all metrics at the configured threshold |
| `confusion_matrix.csv` | 2×2 confusion matrix |
| `confusion_matrix.png` | Confusion matrix heatmap |

### Aggregate — written to `<OUTPUT_ROOT>/`

| File | Contents |
|---|---|
| `all_sets_summary.csv` | One row per set with all metrics |
| `aggregate_summary.csv` | mean / std / min / max across all sets for every numeric column |

---

## Data Flow

```
kagglehub.dataset_download()
        │
        ▼
resolve_dataset_root()          ← unwrap packaging folders
        │
        ▼
list_subject_dirs()             ← sorted subject folder list
        │
        ▼
┌─── for each set ───────────────────────────────────────────────────────┐
│                                                                        │
│   for each subject:                                                    │
│       list_eye_images()         ← sorted images in L/ or R/           │
│       │                                                                │
│       for each image:                                                  │
│           get_or_create_cached_template()                              │
│               ├─ cache hit  → load .pkl                                │
│               └─ cache miss → IRISPipeline() → save .pkl               │
│       │                                                                │
│       └─ collect IMAGES_PER_SUBJECT valid images                       │
│          (skip subject if insufficient, log failures)                  │
│                                                                        │
│   collect SUBJECTS_PER_SET subjects                                    │
│   build_distance_matrix()       ← pairwise HammingDistanceMatcher     │
│   extract_unique_pair_records() ← label genuine / impostor            │
│   build_multi_score_pair_df()   ← hamming + jaccard + cosine + pearson│
│   compute_comparison_summary()  ← apply THRESHOLD → TP/FP/FN/TN      │
│   save_set_outputs()            ← write all CSVs + confusion PNG       │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
        │
        ▼
concat all per-set summaries
save all_sets_summary.csv
save aggregate_summary.csv
```

---

## Limitations

- Only one eye side (`TARGET_EYE_SIDE`) per run — run twice (L and R) for both sides.
- `THRESHOLD` is fixed for classification within this script. `evaluate.py` learns the optimal threshold adaptively.
- Currently designed for the CASIA-Iris-Thousand folder structure (`<subject>/L/<images>` and `<subject>/R/<images>`).
- Template caching uses the image **path** as the key — moving or renaming files invalidates the cache without error.
