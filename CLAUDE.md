# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Requires Anaconda with Python 3.10. Use the `iris-dev` conda environment.

```bash
# First time
conda env create -f environment.yml
conda activate iris-dev

# After dependency changes
conda env update -f environment.yml --prune
```

## Running the Script

```bash
conda activate iris-dev
python baseline_casia_thousand_multiset.py
```

The dataset is auto-downloaded via `kagglehub` (CASIA-Iris-Thousand from Kaggle). A Kaggle account and API token are required for first-time download.

## Architecture

The single script `baseline_casia_thousand_multiset.py` implements a full iris verification pipeline:

1. **Dataset resolution** — `resolve_dataset_root` unwraps the kagglehub download path to find the actual subject folders (which follow `<subject_id>/L/` and `<subject_id>/R/` structure).

2. **Template generation** — Each image is fed through `iris.IRISPipeline` to produce an `IrisTemplate`. Templates are serialized with `pickle` and cached to `CACHE_DIR` (keyed by MD5 hash of the image path) to avoid reprocessing.

3. **Set construction** — `build_sets_from_dataset` partitions subjects into non-overlapping sets of `SUBJECTS_PER_SET` subjects, each contributing `IMAGES_PER_SUBJECT` valid images. Subjects/images that fail template creation are logged and excluded.

4. **Matching & evaluation** — For each set, a full pairwise Hamming distance matrix is computed via `HammingDistanceMatcher`. Pairs are labeled genuine (same subject + eye side) or impostor, then classified against `THRESHOLD`. Metrics (accuracy, precision, recall, F1, balanced accuracy) and confusion matrices are saved per-set and aggregated across all sets.

## Key Configuration (top of script)

| Variable | Purpose |
|---|---|
| `TARGET_EYE_SIDE` | `"L"` or `"R"` — one side per run |
| `SUBJECTS_PER_SET` | Subjects per evaluation set (default 9) |
| `IMAGES_PER_SUBJECT` | Valid images required per subject (default 10) |
| `MAX_SETS` | Cap on sets to run; `None` = all possible |
| `THRESHOLD` | Hamming distance cut-off for match/non-match (default 0.38) |
| `USE_CACHE` | Reuse cached templates across runs |
| `OVERWRITE_CACHE` | Force regeneration of cached templates |
| `OUTPUT_ROOT` | Directory for per-set and aggregate CSVs/plots |

## Output Structure

Each set produces a subfolder under `OUTPUT_ROOT` named `set_<N>_<EYE>_<subjects>` containing CSVs for selected/failed/skipped images, pairwise distances, and a confusion matrix PNG. Two aggregate files (`all_sets_summary.csv`, `aggregate_summary.csv`) are written at the `OUTPUT_ROOT` level.
