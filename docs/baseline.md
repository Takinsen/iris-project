# Baseline: CASIA-Iris-Thousand Multi-Set Evaluation

This document describes the end-to-end working steps of `baseline_casia_thousand_multiset.py`.

---

## Overview

The script runs a multi-set iris verification experiment on the CASIA-Iris-Thousand dataset using the Open-Iris / Worldcoin pipeline. It partitions subjects into non-overlapping sets, generates IrisCode templates for each image, computes pairwise Hamming distances, applies a fixed threshold to classify genuine/impostor pairs, and records evaluation metrics per set and in aggregate.

---

## Configuration (Input)

All parameters are defined at the top of the script.

| Parameter | Default | Description |
|---|---|---|
| `DATASET_ROOT` | *(auto from kagglehub)* | Path to the downloaded dataset |
| `OUTPUT_ROOT` | `.\out_CASIA_Iris_Thousand_MultiSet_L` | Root folder for all experiment outputs |
| `CACHE_DIR` | `.\cache_templates_CASIA_Iris_Thousand_Baseline` | Folder for cached iris templates |
| `TARGET_EYE_SIDE` | `"L"` | Eye side to use: `"L"` (left) or `"R"` (right) |
| `SUBJECTS_PER_SET` | `9` | Number of subjects in each evaluation set |
| `IMAGES_PER_SUBJECT` | `10` | Number of valid images required per subject |
| `MAX_SETS` | `20` | Max sets to run; `None` = run all possible non-overlapping sets |
| `THRESHOLD` | `0.38` | Hamming distance cut-off: ≤ threshold → match |
| `USE_CACHE` | `True` | Reuse previously generated `.pkl` templates |
| `OVERWRITE_CACHE` | `False` | Force template regeneration even if cache exists |

**Dataset folder structure expected:**
```
<DATASET_ROOT>/
  <subject_id>/        e.g. 001/
    L/                 left eye images
      image1.jpg
      image2.jpg
      ...
    R/                 right eye images
      ...
```

---

## Step-by-Step Process

### Step 1 — Dataset Download

```python
path = kagglehub.dataset_download("sondosaabed/casia-iris-thousand")
DATASET_ROOT = path
```

`kagglehub` automatically downloads and caches the CASIA-Iris-Thousand dataset from Kaggle on the first run. Subsequent runs reuse the cached download. A Kaggle account and API token must be configured beforehand.

---

### Step 2 — Resolve Dataset Root

**Function:** `resolve_dataset_root(download_root)`

The downloaded archive may be wrapped in extra top-level folders. This step walks up to 8 directory levels deep to find the first folder that contains subject-like subdirectories (folders that have an `L/` or `R/` child).

```
Input:  raw kagglehub download path
Output: actual subject-level root directory
```

**Logic:**
1. Count child folders that contain `L/` or `R/` subfolder — these are subject folders.
2. If count > 0, return the current directory as the subject root.
3. If exactly one child directory exists (packaging wrapper), descend into it and repeat.
4. Stop after 8 levels.

---

### Step 3 — Build Experiment Sets

**Function:** `build_sets_from_dataset(...)`

Subjects are partitioned sequentially into non-overlapping sets of `SUBJECTS_PER_SET` subjects each.

```
Input:  dataset root, SUBJECTS_PER_SET, IMAGES_PER_SUBJECT, TARGET_EYE_SIDE
Output: list of set dicts, each containing selected images, failed images, skipped subjects, and templates
```

**For each set:**

#### 3a — Select Images per Subject

**Function:** `select_subject_images_for_set(...)`

For each candidate subject:
1. Locate the eye directory: `<subject_dir>/<TARGET_EYE_SIDE>/`
2. List all valid image files (`.jpg`, `.jpeg`, `.png`, `.bmp`), sorted alphabetically.
3. Iterate images until `IMAGES_PER_SUBJECT` valid templates have been collected.
4. For each image, call `get_or_create_cached_template()` (see Step 3b).
5. If an image fails template creation → log a `FailedImageRecord` and continue.
6. If the subject cannot provide `IMAGES_PER_SUBJECT` valid images → log a `SkippedSubjectRecord` and skip the subject entirely.

```
Output per subject:
  - List[SelectedImageRecord]   — images successfully included
  - List[FailedImageRecord]     — images that failed
  - SkippedSubjectRecord | None — subject skipped if insufficient valid images
  - Dict[label → CachedTemplate]
```

#### 3b — Template Creation and Caching

**Function:** `get_or_create_cached_template(...)`

```
Input:  image path, subject_id, eye_side
Output: CachedTemplate | (failure_stage, failure_reason)
```

**Cache key:** MD5 hash of the absolute image path → `<CACHE_DIR>/<hash>.pkl`

**Cache hit path** (`USE_CACHE=True`, file exists, `OVERWRITE_CACHE=False`):
- Load and return the pickled `CachedTemplate` directly. Skip pipeline entirely.

**Cache miss path:**
1. Read image as grayscale using `cv2.imread`.
2. Wrap in `iris.IRImage(img_data, image_id, eye_side={"L":"left","R":"right"})`.
3. Run `iris.IRISPipeline()(image)` — this performs segmentation, normalization, and IrisCode encoding.
4. Check pipeline output for errors.
5. Extract `iris_template` (`iris.IrisTemplate`) from output.
6. Serialize template with `pickle.dumps` and store in a `CachedTemplate` dataclass.
7. Save the dataclass to `<CACHE_DIR>/<hash>.pkl` if `USE_CACHE=True`.

**Failure stages captured:**
| Stage | Cause |
|---|---|
| `image_read_failed` | `cv2.imread` returned `None` |
| `pipeline_exception` | Unhandled exception inside the pipeline |
| `pipeline_failed` | Pipeline returned `None` or an error dict |
| `template_missing` | Pipeline succeeded but no `iris_template` key in output |
| `serialization_failed` | `pickle.dumps` failed |
| `cache_load_failed` | Cached `.pkl` is corrupt |
| `cache_save_failed` | Could not write `.pkl` to disk |

#### 3c — Set Completion Check

After collecting subjects:
- If `len(set_subjects) == SUBJECTS_PER_SET` → set is complete; append to `all_sets`.
- Otherwise → not enough subjects remain; stop processing cleanly.

Each subject appears in at most one set (`used_subject_ids` tracks this).

---

### Step 4 — Per-Set Evaluation

**Function:** `save_set_outputs(set_data, set_output_dir, threshold, eye_side, matcher)`

Runs for each completed set. Output folder: `<OUTPUT_ROOT>/set_<N>_<EYE>_<subject_ids>/`

#### 4a — Build Distance Matrix

**Function:** `build_distance_matrix(selected_images, templates_by_label, matcher)`

```
Input:  N selected image records and their templates
Output: N×N symmetric DataFrame of pairwise Hamming distances
```

- Iterates all unique pairs `(i, j)` where `i < j`.
- Calls `HammingDistanceMatcher.run(template_i, template_j)` for each pair.
- Matrix is symmetric; diagonal is `NaN`.
- With 9 subjects × 10 images = 90 images → 4,005 unique pairs.

#### 4b — Extract Pair Records

**Function:** `extract_unique_pair_records(distance_matrix_df, selected_images)`

For every unique pair, records:

| Column | Description |
|---|---|
| `img1_label`, `img2_label` | `<subject>_<eye>_<stem>` labels |
| `subject1`, `subject2` | Subject IDs |
| `distance` | Hamming distance |
| `pair_type` | `"genuine"` if same subject + same eye side, else `"impostor"` |
| `true_label` | `1` for genuine, `0` for impostor |

**Genuine pairs:** same subject, same eye side (within-subject comparisons).
**Impostor pairs:** different subjects (cross-subject comparisons).

With 9 subjects × 10 images:
- Genuine pairs: 9 × C(10,2) = 9 × 45 = **405**
- Impostor pairs: 4,005 − 405 = **3,600**

#### 4c — Compute Comparison Summary

**Function:** `compute_comparison_summary(pair_df, selected_images, threshold, eye_side)`

```
Input:  pair records, threshold
Output: summary DataFrame, pair records with pred_label column
```

Classification rule:
- `pred_label = 1` (match) if `distance ≤ THRESHOLD`
- `pred_label = 0` (non-match) if `distance > THRESHOLD`

Confusion matrix cells:
| | Predicted Match | Predicted Non-Match |
|---|---|---|
| **Actual Match** (genuine) | TP | FN |
| **Actual Non-Match** (impostor) | FP | TN |

Metrics computed:

| Metric | Formula |
|---|---|
| Accuracy | (TP + TN) / total |
| Precision | TP / (TP + FP) |
| Recall (TAR) | TP / (TP + FN) |
| F1 | 2 × Precision × Recall / (Precision + Recall) |
| Balanced Accuracy | (Recall + TNR) / 2, where TNR = TN / (TN + FP) |

Also records distance statistics (mean, std, min, max) for genuine and impostor distributions separately.

---

### Step 5 — Save Per-Set Outputs

Written to `<OUTPUT_ROOT>/set_<N>_<EYE>_<subjects>/`:

| File | Contents |
|---|---|
| `selected_images.csv` | All images successfully included in the set |
| `failed_images.csv` | Images that failed template creation |
| `skipped_subjects.csv` | Subjects excluded due to insufficient valid images |
| `set_manifest.csv` | Subject list and set configuration |
| `distance_matrix.csv` | N×N pairwise Hamming distance matrix (index = image labels) |
| `pair_records.csv` | One row per unique pair: distances, pair type, true/predicted labels |
| `comparison_summary.csv` | TP, FP, FN, TN, and all metrics for this set |
| `confusion_matrix.csv` | 2×2 confusion matrix table |
| `confusion_matrix.png` | Confusion matrix heatmap visualization |

---

### Step 6 — Aggregate Across All Sets

After all sets are processed:

1. Concatenate all per-set `comparison_summary.csv` data into **`all_sets_summary.csv`** — one row per set.
2. Compute `mean`, `std`, `min`, `max` across sets for all numeric columns → **`aggregate_summary.csv`**.

Both files are written to `OUTPUT_ROOT`.

---

## Output Summary

```
<OUTPUT_ROOT>/
  all_sets_summary.csv          — per-set metrics (one row per set)
  aggregate_summary.csv         — mean/std/min/max across all sets
  set_01_L_<subjects>/
    selected_images.csv
    failed_images.csv
    skipped_subjects.csv
    set_manifest.csv
    distance_matrix.csv
    pair_records.csv
    comparison_summary.csv
    confusion_matrix.csv
    confusion_matrix.png
  set_02_L_<subjects>/
    ...
  visualizations/               — produced by evaluate.py
    metrics_per_set.png
    distance_distribution.png
    roc_curve.png
    det_curve.png
    distance_per_set.png
    aggregate_confusion_matrix.png
    dashboard.png
```

---

## Data Flow Diagram

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
┌─── for each set ───────────────────────────────────────────────┐
│                                                                │
│   for each subject                                             │
│       list_eye_images()         ← sorted images in L/ or R/   │
│       │                                                        │
│       for each image                                           │
│           get_or_create_cached_template()                      │
│               ├─ cache hit  → load .pkl                        │
│               └─ cache miss → IRISPipeline() → save .pkl       │
│       │                                                        │
│       └─ collect IMAGES_PER_SUBJECT valid images               │
│          (skip subject if insufficient)                        │
│                                                                │
│   collect SUBJECTS_PER_SET subjects                            │
│                                                                │
│   build_distance_matrix()      ← pairwise HammingDistanceMatcher│
│   extract_unique_pair_records() ← label genuine / impostor     │
│   compute_comparison_summary() ← apply threshold → TP/FP/FN/TN│
│   save_set_outputs()           ← write CSVs + confusion PNG    │
└────────────────────────────────────────────────────────────────┘
        │
        ▼
concat all summaries
save all_sets_summary.csv
save aggregate_summary.csv
```
