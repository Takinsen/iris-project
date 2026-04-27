# Dataset: CASIA-Iris-Thousand

This document describes the dataset used by this project, how it is organized, how images are selected and preprocessed, how pairs are constructed, and how sets are used in cross-validation.

---

## What Is the Dataset

**CASIA-Iris-Thousand** is a large-scale iris biometric dataset published by the Chinese Academy of Sciences (CASIA). It contains iris photographs from 1,000 subjects, each captured from both the left (`L`) and right (`R`) eye. The dataset is downloaded automatically on first run via `kagglehub`:

```python
path = kagglehub.dataset_download("sondosaabed/casia-iris-thousand")
```

| Property | Value |
|---|---|
| Total subjects | 1,000 |
| Eye sides per subject | 2 (L and R) |
| Images per eye side | ~20 per subject |
| Image format | JPEG, PNG, BMP (grayscale) |
| Use case | Iris verification — matching a probe image to a gallery |

---

## Folder Structure

The dataset follows a fixed hierarchy. Each subject has its own folder, containing one subfolder per eye side:

```
<DATASET_ROOT>/
  001/
    L/
      S5001L00.jpg
      S5001L01.jpg
      ...
    R/
      S5001R00.jpg
      ...
  002/
    L/
      ...
    R/
      ...
  ...
  1000/
    L/  R/
```

The script resolves the dataset root automatically — if `kagglehub` wraps the download in extra packaging folders, `resolve_dataset_root` walks down until it finds the actual subject-level directory (detected by the presence of `L/` or `R/` child folders).

---

## Terminology

| Term | Meaning |
|---|---|
| **Subject** | One person in the dataset. Identified by their folder name (e.g. `001`). Each subject has a unique identity. |
| **Eye side** | `L` (left) or `R` (right). One side is selected per run via `TARGET_EYE_SIDE`. |
| **Image** | A single grayscale iris photograph (e.g. `S5001L00.jpg`). One subject has multiple images per eye side. |
| **Image label** | A unique string identifying one image: `<subject_id>_<eye_side>_<filename_stem>` (e.g. `001_L_S5001L00`). |
| **IrisTemplate / IrisCode** | The binary biometric code produced by the Open-Iris pipeline for one image. This is the unit of comparison, not the raw image. |
| **Set** | A self-contained evaluation group of `SUBJECTS_PER_SET` subjects, each contributing `IMAGES_PER_SUBJECT` valid images. |
| **Genuine pair** | Two images from the **same subject** and **same eye side**. Expected to have a low distance (similar irises). |
| **Impostor pair** | Two images from **different subjects**. Expected to have a high distance (different irises). |
| **Template cache** | A `.pkl` file storing a serialized `IrisTemplate` so the pipeline does not need to re-run for repeated experiments. |

---

## Image Selection

For each subject, images are scanned in sorted filename order from the eye-side folder. Images are tried one by one until `IMAGES_PER_SUBJECT` valid templates have been produced. If the pipeline fails on an image, that image is recorded in `failed_images.csv` and skipped — the next image is tried. If a subject cannot produce enough valid images, the subject is recorded in `skipped_subjects.csv` and excluded from the set.

```
Subject folder (e.g. 001/L/)
  S5001L00.jpg  → pipeline → IrisTemplate  ✓  (valid_count = 1)
  S5001L01.jpg  → pipeline → failed         ✗  (logged, try next)
  S5001L02.jpg  → pipeline → IrisTemplate  ✓  (valid_count = 2)
  ...
  until valid_count == IMAGES_PER_SUBJECT
```

Failure reasons are categorized:

| Stage | Cause |
|---|---|
| `image_read_failed` | `cv2.imread` returned `None` (corrupt or unreadable file) |
| `pipeline_exception` | Unhandled exception inside `iris.IRISPipeline` |
| `pipeline_failed` | Pipeline returned an error dict |
| `template_missing` | No `iris_template` key in pipeline output |
| `serialization_failed` | `pickle.dumps` raised an exception on the template |
| `cache_load_failed` | Existing `.pkl` cache file is corrupt |
| `cache_save_failed` | Could not write `.pkl` to disk |

---

## Preprocessing: Template Generation

Each valid image is processed by the **Open-Iris / Worldcoin IRISPipeline** to produce an `IrisTemplate` (also called an IrisCode). This pipeline performs:

1. **Iris segmentation** — locates the iris region and its inner/outer boundaries
2. **Normalisation** — unwraps the iris annulus into a rectangular strip
3. **Feature encoding** — applies Gabor filters or equivalent to produce a binary code
4. **Masking** — produces a mask array marking bits that are valid (not occluded by eyelids/reflections)

The resulting `IrisTemplate` contains:
- `iris_codes` — binary code array (the biometric fingerprint)
- `mask_codes` — boolean validity mask (`True` = bit is valid)

```python
out = pipeline(
    iris.IRImage(
        img_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE),
        image_id = image_path,
        eye_side = "left" or "right",
    )
)
template = out["iris_template"]   # iris.IrisTemplate
```

---

## Template Caching

Running the pipeline on every image on every run is slow. Templates are serialized with `pickle` and stored in `CACHE_DIR`. The cache key is the **MD5 hash of the image file path** → `<CACHE_DIR>/<md5hash>.pkl`.

```
USE_CACHE=True, OVERWRITE_CACHE=False  →  load .pkl if it exists, else generate and save
USE_CACHE=True, OVERWRITE_CACHE=True   →  always regenerate and overwrite
USE_CACHE=False                        →  always regenerate, never save
```

On a cache hit the pipeline is skipped entirely — only the cached `CachedTemplate` (which contains `iris_template_bytes`) is loaded from disk.

---

## Set Construction

Subjects are partitioned into **non-overlapping sets**. Each set contains exactly `SUBJECTS_PER_SET` subjects, each contributing `IMAGES_PER_SUBJECT` valid images.

```
Dataset subjects (sorted): 001, 002, 003, ..., 1000
                              │
          ┌───────────────────┤
          ▼                   │
        set_01                ▼
   subjects 001–0??         set_02
   (first SUBJECTS_PER_SET  (next SUBJECTS_PER_SET
    that produce enough      valid subjects)
    valid images)
          │
          ▼
        set_03 ...
```

A subject that cannot provide `IMAGES_PER_SUBJECT` valid templates is skipped and the next subject is tried. This means consecutive subject IDs are not guaranteed — skipped subjects create gaps. `used_subject_ids` prevents any subject from appearing in more than one set.

**With default config** (`SUBJECTS_PER_SET=50`, `IMAGES_PER_SUBJECT=10`, `MAX_SETS=10`):

```
Per set:
  50 subjects × 10 images = 500 images
  Genuine pairs:  50 × C(10,2) = 50 × 45 = 2,250
  Impostor pairs: C(500,2) − 2,250 = 124,750 − 2,250 = 122,500
  Total pairs:    C(500,2) = 124,750

Full run (10 sets):
  5,000 images
  1,247,500 total pairs
```

---

## Pair Construction

For each set, every unique pair of images `(i, j)` where `i < j` is evaluated. A pair is labelled:

```python
pair_type = "genuine"   if same subject_id AND same eye_side
pair_type = "impostor"  otherwise

true_label = 1  (genuine)
true_label = 0  (impostor)
```

Pairs are stored in `pair_records.csv` and `multi_score_features.csv`.

---

## Distance Scores

For each pair, four distance metrics are computed:

| Metric | Description | Range | Identical templates |
|---|---|---|---|
| **Hamming** | Fraction of differing bits at unmasked positions (via `HammingDistanceMatcher`) | [0, 1] | ≈ 0 |
| **Jaccard** | 1 − (intersection / union) on binarised bits | [0, 1] | 0 |
| **Cosine** | 1 − cosine similarity on raw bit values | [0, 2] | 0 |
| **Pearson** | 1 − Pearson correlation on raw bit values | [0, 2] | 0 |

All metrics treat **lower = more similar** (distance convention, not similarity). Hamming is the primary biometric score; Jaccard, Cosine, and Pearson are auxiliary features used by the fusion models in `evaluate.py`.

Only bits that are **valid in both templates** (i.e. `mask_codes[i] == True` in both) are used for Jaccard, Cosine, and Pearson — occluded bits are excluded.

---

## Cross-Validation (LOSO)

`evaluate.py` consumes the set outputs from this script to run **Leave-One-Set-Out (LOSO)** cross-validation. Each set becomes the held-out test fold once, while all other sets form the training data:

```
Sets: [set_01, set_02, set_03, ..., set_10]

Fold 1:  Train = [set_02 … set_10]   Test = set_01
Fold 2:  Train = [set_01, set_03 … set_10]   Test = set_02
...
Fold 10: Train = [set_01 … set_09]   Test = set_10
```

Because each set contains entirely different subjects than every other set (enforced by `used_subject_ids`), there is **zero subject overlap** between train and test in any fold. This matches real deployment: the model sees no identity it will encounter at test time.

---

## Output Files (per set)

Written to `<OUTPUT_ROOT>/set_<N>_<EYE>_<subject_range>/`:

| File | Contents |
|---|---|
| `selected_images.csv` | Images successfully included (set_id, subject_id, eye_side, image_name, image_label, …) |
| `failed_images.csv` | Images that could not produce a valid template (failure_stage, failure_reason) |
| `skipped_subjects.csv` | Subjects excluded for insufficient valid images |
| `set_manifest.csv` | Subject list and set configuration metadata |
| `distance_matrix.csv` | N×N symmetric Hamming distance matrix (N = total images in set) |
| `pair_records.csv` | One row per unique pair: Hamming distance, pair_type, true_label, pred_label |
| `multi_score_features.csv` | One row per pair: all four distance features + pair_type + true_label |
| `comparison_summary.csv` | TP, FP, FN, TN and all metrics for this set at the configured threshold |
| `confusion_matrix.csv` | 2×2 confusion matrix table |
| `confusion_matrix.png` | Confusion matrix heatmap |

Aggregate files written at `<OUTPUT_ROOT>/`:

| File | Contents |
|---|---|
| `all_sets_summary.csv` | One row per set with all metrics |
| `aggregate_summary.csv` | mean / std / min / max of every numeric column across all sets |
