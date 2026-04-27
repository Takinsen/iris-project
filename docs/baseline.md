# Baseline: CASIA-Iris-Thousand Multi-Set Evaluation

This document describes the end-to-end working steps of `baseline_casia_thousand_multiset.py`.

---

## Overview

The script runs a multi-set iris verification experiment on the CASIA-Iris-Thousand dataset using the Open-Iris / Worldcoin pipeline. It partitions subjects into non-overlapping sets, generates IrisCode templates for each image, computes pairwise Hamming distances, applies a fixed threshold to classify genuine/impostor pairs, and records evaluation metrics per set and in aggregate.

---

## Terminology

| Term | Meaning |
|---|---|
| **Subject** | A single person in the dataset. Each subject has a folder (e.g. `001/`) containing their iris images. One subject = one identity. Controlled by `SUBJECTS_PER_SET`. |
| **Image** | One iris photograph of a subject's eye (e.g. `S5001L00.jpg`). Multiple photos are taken per subject per eye side. Controlled by `IMAGES_PER_SUBJECT`. |
| **Set** | A self-contained evaluation group of `SUBJECTS_PER_SET` subjects × `IMAGES_PER_SUBJECT` images. Sets are non-overlapping — each subject appears in at most one set. Controlled by `MAX_SETS`. |
| **Genuine pair** | A comparison between two images from the **same subject and same eye side**. Expected to have a low Hamming distance (similar irises). |
| **Impostor pair** | A comparison between two images from **different subjects**. Expected to have a high Hamming distance (different irises). |
| **Hamming distance** | A value in [0, 1] measuring how different two IrisCode templates are. Lower = more similar. |
| **Threshold** | The Hamming distance cut-off. Pairs with distance ≤ threshold are classified as a match (genuine); pairs above are classified as non-match (impostor). |
| **IrisTemplate / IrisCode** | The binary biometric code produced by the Open-Iris pipeline for one iris image. Used as the unit of comparison. |

**Example with defaults** (`SUBJECTS_PER_SET=9`, `IMAGES_PER_SUBJECT=10`, `MAX_SETS=20`):

```
Set 01
├── Subject 001 → 10 images
├── Subject 002 → 10 images
├── ...
└── Subject 009 → 10 images
      ↓
  90 images total
  Genuine pairs:  9 × C(10,2) = 9 × 45 =   405
  Impostor pairs: C(90,2) − 405         = 3,600
  Total pairs:                             4,005

Full run: 20 sets × 9 subjects × 10 images = 1,800 images, 80,100 pairs
```

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

`kagglehub` downloads and caches the dataset automatically on first run. The returned `path` is assigned as `DATASET_ROOT`.

```python
path = kagglehub.dataset_download("sondosaabed/casia-iris-thousand")
DATASET_ROOT = path
```

---

### Step 2 — Resolve Dataset Root

The downloaded archive may have extra wrapper folders. `resolve_dataset_root` walks down directory levels until it finds a folder whose children contain `L/` or `R/` subfolders (i.e. actual subject folders).

```python
def resolve_dataset_root(download_root: str) -> str:
    current = download_root
    for _ in range(8):
        score = count_subject_like_dirs(current)
        if score > 0:
            return current                  # found subject-level root

        child_dirs = [
            os.path.join(current, name)
            for name in sorted(os.listdir(current))
            if os.path.isdir(os.path.join(current, name))
        ]
        if len(child_dirs) == 1:
            current = child_dirs[0]         # descend one packaging wrapper
            continue
        break
    return current
```

`count_subject_like_dirs` counts folders that have an `L/` or `R/` child:

```python
def count_subject_like_dirs(root_dir: str) -> int:
    count = 0
    for name in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, name)
        if os.path.isdir(os.path.join(subject_path, "L")) or \
           os.path.isdir(os.path.join(subject_path, "R")):
            count += 1
    return count
```

**Input:** raw kagglehub download path  
**Output:** actual subject-level root (e.g. `.../CASIA-Iris-Thousand/`)

---

### Step 3 — Build Experiment Sets

`build_sets_from_dataset` iterates through subject folders sequentially and fills sets of `SUBJECTS_PER_SET` subjects. Each subject is used in at most one set (`used_subject_ids` prevents overlap).

```python
def build_sets_from_dataset(...):
    subject_dirs = list_subject_dirs(dataset_root)  # sorted list of subject folders
    used_subject_ids = set()
    all_sets = []
    set_counter = 1
    idx = 0

    while idx < len(subject_dirs):
        if max_sets is not None and len(all_sets) >= max_sets:
            break

        set_id = f"set_{set_counter:02d}"
        set_subjects = []
        ...

        while idx < len(subject_dirs) and len(set_subjects) < subjects_per_set:
            subject_dir = subject_dirs[idx]
            idx += 1

            subject_id = get_subject_id_from_dir(subject_dir)
            if subject_id in used_subject_ids:
                continue

            selected_records, failed_images, skipped_subject, templates = \
                select_subject_images_for_set(set_id, subject_dir, ...)

            if skipped_subject is not None:
                continue                    # not enough valid images → skip

            set_subjects.append(subject_id)
            used_subject_ids.add(subject_id)

        if len(set_subjects) == subjects_per_set:
            all_sets.append({...})          # complete set → keep
            set_counter += 1
        else:
            break                           # ran out of subjects → stop
```

**Input:** dataset root, `SUBJECTS_PER_SET`, `IMAGES_PER_SUBJECT`, `TARGET_EYE_SIDE`  
**Output:** list of set dicts, each holding selected images, failed images, skipped subjects, and templates

---

#### 3a — Select Images per Subject

For each subject, `select_subject_images_for_set` locates the eye folder and collects images until `IMAGES_PER_SUBJECT` valid templates are found.

```python
def select_subject_images_for_set(set_id, subject_dir, images_per_subject, eye_side, pipeline):
    subject_id = get_subject_id_from_dir(subject_dir)
    eye_dir = get_eye_dir(subject_dir, eye_side)    # e.g. 001/L/
    if eye_dir is None:
        return None, [], SkippedSubjectRecord(..., "Missing eye folder"), {}

    candidate_images = list_eye_images(eye_dir)     # sorted .jpg/.png/.bmp files
    selected_records = []
    valid_count = 0

    for image_path in candidate_images:
        if valid_count >= images_per_subject:
            break

        cached_tpl, failure_stage, failure_reason = \
            get_or_create_cached_template(pipeline, subject_id, eye_side, image_path)

        if cached_tpl is None:
            failed_images.append(FailedImageRecord(..., failure_stage, failure_reason))
            continue                                # try next image

        valid_count += 1
        selected_records.append(SelectedImageRecord(...))
        selected_templates[cached_tpl.image_label] = cached_tpl

    if valid_count < images_per_subject:
        return None, failed_images, SkippedSubjectRecord(..., "Fewer than required"), {}

    return selected_records, failed_images, None, selected_templates
```

---

#### 3b — Template Creation and Caching

`get_or_create_cached_template` either loads a cached template or runs the full pipeline and saves the result.

**Cache key:** MD5 hash of the image path → `<CACHE_DIR>/<hash>.pkl`

```python
def get_or_create_cached_template(pipeline, subject_id, eye_side, image_path):
    cache_path = cache_path_for_image(image_path)   # MD5(image_path) + .pkl

    # --- Cache hit ---
    if USE_CACHE and os.path.exists(cache_path) and not OVERWRITE_CACHE:
        rec = load_template_cache(cache_path)        # pickle.load
        return rec, None, None

    # --- Cache miss: run pipeline ---
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, "image_read_failed", f"Could not read: {image_path}"

    out = pipeline(
        iris.IRImage(
            img_data=img,
            image_id=image_path,
            eye_side="left" if eye_side.upper() == "L" else "right",
        )
    )

    if out.get("error") is not None:
        return None, "pipeline_failed", safe_error_message(out["error"])

    template = out.get("iris_template")             # iris.IrisTemplate (IrisCode)
    if template is None:
        return None, "template_missing", "No iris_template in output"

    # --- Serialize and cache ---
    rec = CachedTemplate(
        subject_id=subject_id,
        iris_template_bytes=pickle.dumps(template),
        ...
    )
    save_template_cache(rec, cache_path)            # pickle.dump
    return rec, None, None
```

Failure stages returned when `cached_tpl is None`:

| Stage | Cause |
|---|---|
| `image_read_failed` | `cv2.imread` returned `None` |
| `pipeline_exception` | Unhandled exception inside the pipeline |
| `pipeline_failed` | Pipeline returned an error dict |
| `template_missing` | No `iris_template` key in pipeline output |
| `serialization_failed` | `pickle.dumps` raised an exception |
| `cache_load_failed` | Existing `.pkl` is corrupt |
| `cache_save_failed` | Could not write `.pkl` to disk |

---

### Step 4 — Per-Set Evaluation

`save_set_outputs` runs the full evaluation for one set and writes all output files.

```python
def save_set_outputs(set_data, set_output_dir, threshold, eye_side, matcher):
    selected_images    = set_data["selected_images"]
    templates_by_label = set_data["templates_by_label"]

    distance_matrix_df   = build_distance_matrix(selected_images, templates_by_label, matcher)
    pair_df              = extract_unique_pair_records(distance_matrix_df, selected_images)
    summary_df, pair_df_with_pred = compute_comparison_summary(pair_df, selected_images, threshold, eye_side)

    # write all CSVs and confusion matrix PNG
    save_dataframe(distance_matrix_df, ".../distance_matrix.csv", index=True)
    save_dataframe(pair_df_with_pred,  ".../pair_records.csv")
    save_dataframe(summary_df,         ".../comparison_summary.csv")
    save_confusion_matrix_plot(confusion_df, ".../confusion_matrix.png")
```

---

#### 4a — Build Distance Matrix

`build_distance_matrix` computes a pairwise Hamming distance for every unique pair of images in the set and stores the result as a symmetric N×N DataFrame (diagonal = `NaN`).

```python
def build_distance_matrix(selected_images, templates_by_label, matcher):
    labels = [rec.image_label for rec in selected_images]   # e.g. "001_L_S5001L00"
    n = len(labels)
    matrix = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            tpl_a = pickle.loads(templates_by_label[labels[i]].iris_template_bytes)
            tpl_b = pickle.loads(templates_by_label[labels[j]].iris_template_bytes)
            distance = float(matcher.run(tpl_a, tpl_b))    # HammingDistanceMatcher
            matrix[i, j] = distance
            matrix[j, i] = distance                         # symmetric

    return pd.DataFrame(matrix, index=labels, columns=labels)
```

With 90 images (9 subjects × 10): **4,005 unique pairs** computed.

---

#### 4b — Extract Pair Records

`extract_unique_pair_records` reads the upper triangle of the distance matrix and labels each pair as genuine or impostor.

```python
def extract_unique_pair_records(distance_matrix_df, selected_images):
    by_label = {rec.image_label: rec for rec in selected_images}
    labels   = list(distance_matrix_df.index)
    records  = []

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rec_i, rec_j = by_label[labels[i]], by_label[labels[j]]

            # genuine = same subject AND same eye side
            pair_type = "genuine" if (rec_i.subject_id == rec_j.subject_id and
                                      rec_i.eye_side   == rec_j.eye_side) else "impostor"
            records.append({
                "distance":   float(distance_matrix_df.iloc[i, j]),
                "pair_type":  pair_type,
                "true_label": 1 if pair_type == "genuine" else 0,
                ...
            })
    return pd.DataFrame(records)
```

With 9 subjects × 10 images:
- **Genuine:** 9 × C(10,2) = 9 × 45 = **405 pairs**
- **Impostor:** 4,005 − 405 = **3,600 pairs**

---

#### 4c — Compute Comparison Summary

`compute_comparison_summary` applies the threshold to every pair and computes the confusion matrix and all metrics.

```python
def compute_comparison_summary(pair_df, selected_images, threshold, eye_side):
    pair_df = pair_df.copy()
    pair_df["pred_label"] = (pair_df["distance"] <= threshold).astype(int)
    #  distance ≤ 0.38  →  pred_label = 1  (match / genuine)
    #  distance >  0.38  →  pred_label = 0  (non-match / impostor)

    tp = int(((pair_df["true_label"] == 1) & (pair_df["pred_label"] == 1)).sum())
    fp = int(((pair_df["true_label"] == 0) & (pair_df["pred_label"] == 1)).sum())
    fn = int(((pair_df["true_label"] == 1) & (pair_df["pred_label"] == 0)).sum())
    tn = int(((pair_df["true_label"] == 0) & (pair_df["pred_label"] == 0)).sum())

    accuracy         = (tp + tn) / (tp + fp + fn + tn)
    precision        = tp / (tp + fp)
    recall           = tp / (tp + fn)
    f1               = 2 * precision * recall / (precision + recall)
    balanced_accuracy = (recall + tn / (tn + fp)) / 2
```

Confusion matrix layout:

| | Predicted Match | Predicted Non-Match |
|---|---|---|
| **Actual Match** (genuine) | TP | FN |
| **Actual Non-Match** (impostor) | FP | TN |

---

### Step 5 — Save Per-Set Outputs

Written to `<OUTPUT_ROOT>/set_<N>_<EYE>_<subjects>/`:

```python
# Logs
save_dataframe(pd.DataFrame([asdict(x) for x in selected_images]),  ".../selected_images.csv")
save_dataframe(pd.DataFrame([asdict(x) for x in failed_images]),    ".../failed_images.csv")
save_dataframe(pd.DataFrame([asdict(x) for x in skipped_subjects]), ".../skipped_subjects.csv")
save_dataframe(manifest_df,                                          ".../set_manifest.csv")

# Evaluation
save_dataframe(distance_matrix_df,  ".../distance_matrix.csv", index=True)
save_dataframe(pair_df_with_pred,   ".../pair_records.csv")
save_dataframe(summary_df,          ".../comparison_summary.csv")
save_dataframe(confusion_df,        ".../confusion_matrix.csv", index=True)
save_confusion_matrix_plot(confusion_df, ".../confusion_matrix.png")
```

| File | Contents |
|---|---|
| `selected_images.csv` | Images successfully included in the set |
| `failed_images.csv` | Images that failed template creation |
| `skipped_subjects.csv` | Subjects excluded due to insufficient valid images |
| `set_manifest.csv` | Subject list and set configuration |
| `distance_matrix.csv` | N×N pairwise Hamming distance matrix |
| `pair_records.csv` | One row per pair: distance, pair type, true/predicted labels — **read by `evaluate.py` for LOSO baseline** |
| `comparison_summary.csv` | TP, FP, FN, TN and all metrics |
| `confusion_matrix.csv` | 2×2 confusion matrix table |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `multi_score_features.csv` | One row per pair: hamming + jaccard + cosine + pearson features + true_label — **read by `evaluate.py` for score-level fusion** |

---

### Step 6 — Aggregate Across All Sets

After all sets are processed, per-set summaries are concatenated and statistics are computed across sets.

```python
all_sets_summary_df = pd.concat(all_summaries, ignore_index=True)
save_dataframe(all_sets_summary_df, os.path.join(OUTPUT_ROOT, "all_sets_summary.csv"))

numeric_cols = [
    "total_images", "total_unique_pairs", "genuine_pairs", "impostor_pairs",
    "mean_genuine_distance", "mean_impostor_distance",
    "TP", "FP", "FN", "TN",
    "accuracy", "precision", "recall", "f1", "balanced_accuracy",
]
aggregate_df = all_sets_summary_df[numeric_cols].agg(["mean", "std", "min", "max"])
save_dataframe(aggregate_df.reset_index(), os.path.join(OUTPUT_ROOT, "aggregate_summary.csv"))
```

**Output:**
- `all_sets_summary.csv` — one row per set with all metrics
- `aggregate_summary.csv` — mean / std / min / max of every numeric column across all sets

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
    pair_records.csv            ← read by evaluate.py (LOSO baseline)
    comparison_summary.csv
    confusion_matrix.csv
    confusion_matrix.png
    multi_score_features.csv    ← read by evaluate.py (fusion models)
  set_02_L_<subjects>/
    ...
  visualizations/               — produced by evaluate.py (see improvement.md)
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
    logistic_regression/
    gaussian_naive_bayes/
    linear_discriminant_analysis/
    comparison_report.png
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
